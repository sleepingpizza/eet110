"""
model/bilstm_decoder.py
========================
Temporal encoder: takes the per-bus spatial embeddings Q(N, T, d_h)
produced by the GAT encoder and outputs per-generator commitment logits
of shape (N_GEN, T).

ARCHITECTURAL CHOICES AND RATIONALE
-------------------------------------

WHY BiLSTM (BIDIRECTIONAL LSTM)?
  Unit commitment decisions are inherently coupled across the entire 24h
  horizon — whether to start a steam turbine at hour 6 depends on whether
  it will still be needed at hour 10 (minimum uptime constraints).
  A unidirectional LSTM can only use *past* hours to inform the current
  decision.  BiLSTM processes the sequence in both forward and backward
  directions and concatenates the hidden states, so the prediction at
  hour t has access to context from BOTH earlier and later hours.
  This is exactly eq. (33)–(35) of sleeping_pizza.pdf:
      →h_t = tanh(W_c Q_t + W_→h →h_{t-1} + b_→h)   [forward]
      ←h_t = tanh(W_c Q_t + W_←h ←h_{t-1} + b_←h)   [backward]
      Ỹ_t  = tanh(V_→h →h_t + V_←h ←h_t + b_Y)

WHY PER-BUS BiLSTM (not one global LSTM)?
  The spatial embedding Q[:,t,:] already has per-bus representations.
  Running a *shared* BiLSTM over each bus's T-length sequence respects
  the locality of each bus while allowing temporal dependencies.
  An alternative is a global LSTM over the flattened (N*d_h, T) sequence,
  but that would mix spatial and temporal computation in an unstructured
  way and has N times more input features, making it harder to train on
  our dataset size.

  Implementation detail: we reshape (N, T, d_h) → (N, T, d_h) and pass
  it through nn.LSTM with batch_first=True, treating the N buses as a
  "batch" dimension.  This is valid because the BiLSTM's hidden state
  is independent across buses (no cross-bus temporal communication —
  that was handled by the GAT).

WHY LSTM_HIDDEN=64 (same as d_h)?
  Matching the LSTM hidden size to the GAT embedding dimension avoids
  any projection bottleneck.  The BiLSTM outputs 2×lstm_hidden (forward
  + backward) → 128 per bus per time-step.  We then project 128 → d_h=64
  via the output projection before the generator mapping.

GENERATOR-BUS INCIDENCE MATRIX M
  The incidence matrix M ∈ {0,1}^{N×G} maps bus embeddings to generators:
      Ỹ_g(t) = Σ_n M_{n,g} Ỹ_n(t)
  This is a sparse, *fixed*, *physics-grounded* projection.  We do NOT
  learn this mapping — the physical location of generators on the grid is
  known and should not be learned away.

  After this projection, we have per-generator logits of shape (G, T).
  The sigmoid of these logits gives the probability that generator g is
  committed at hour t; during training we use BCEWithLogitsLoss which
  numerically applies the sigmoid internally (more stable than explicit
  sigmoid + BCE).

WHY BCEWithLogitsLoss (NOT MSELoss FROM THE PAPER)?
  The paper (eq. 36) uses MSE for the MATLAB implementation, which works
  because their output layer has tanh activation (values in [-1,1]).
  BCEWithLogitsLoss is strictly better for binary targets because:
    1. It is the proper log-likelihood for Bernoulli random variables.
    2. Its gradient is O(1) away from 0/1 targets (MSE gradient vanishes
       near 0/1 with sigmoid output).
    3. It directly optimises what we measure: per-bit accuracy.
  We match the paper's intent (minimise prediction error of binary z)
  while using a numerically superior loss.

OUTPUT PROJECTION BEFORE GENERATOR MAPPING
  After the BiLSTM, we apply a linear+ELU layer to project from 2*hidden
  back to d_h before using M.  This ensures that the bus embeddings
  passed to M are in the same semantic space as the input features —
  without it, the raw BiLSTM output would be in LSTM hidden-state space,
  which is harder for M to interpret.
"""

import torch
import torch.nn as nn


class BiLSTMTemporalDecoder(nn.Module):
    """
    Per-bus BiLSTM temporal encoder → generator commitment logits.

    Parameters
    ----------
    d_h          : input embedding dimension from GAT (64)
    lstm_hidden  : hidden size of the LSTM (64; output is 2×lstm_hidden)
    n_gen        : number of generators (6)
    dropout      : dropout on LSTM output (0.1)
    """

    def __init__(
        self,
        d_h:         int = 64,
        lstm_hidden: int = 64,
        n_gen:       int = 6,
        dropout:     float = 0.10,
    ):
        super().__init__()
        self.d_h         = d_h
        self.lstm_hidden = lstm_hidden
        self.n_gen       = n_gen

        # ── BiLSTM ────────────────────────────────────────────────────────
        # input_size  = d_h  (per-bus GAT embedding at each time-step)
        # hidden_size = lstm_hidden
        # bidirectional=True → outputs have size 2*lstm_hidden
        # batch_first=True   → input shape is (batch, seq, features)
        #                       here batch=N_BUS, seq=T, features=d_h
        self.bilstm = nn.LSTM(
            input_size   = d_h,
            hidden_size  = lstm_hidden,
            num_layers   = 2,
            bidirectional= True,
            batch_first  = True,
            dropout      = 0.2,   # no dropout inside single-layer LSTM
        )

        # ── Output projection: 2*lstm_hidden → d_h ───────────────────────
        # Projects BiLSTM's combined forward+backward output back to d_h.
        # WHY: keeps the downstream generator mapping independent of the
        # LSTM hidden size choice.  Changing lstm_hidden doesn't require
        # touching any code beyond this decoder.
        self.out_proj = nn.Sequential(
            nn.Linear(2 * lstm_hidden, d_h),
            nn.LayerNorm(d_h),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # ── Generator logit head ─────────────────────────────────────────
        # After the M-projection we have (G, T) bus-aggregated embeddings.
        # A final linear layer maps each generator's d_h features → 1 logit.
        # WHY a learned head instead of a raw dot-product with M:
        # M is a hard binary map (generator → bus); the linear head learns
        # a soft, continuous transformation of the bus embedding that is
        # most predictive of that generator's commitment decision.
        # Without this head, the model would need the BiLSTM output to
        # already be in "commitment logit space", which is too much to ask
        # of a general-purpose embedding.
        self.gen_heads = nn.ModuleList([
    nn.Linear(d_h, 1) for _ in range(n_gen)
])

    def forward(
        self,
        Q: torch.Tensor,    # (N_BUS, T, d_h) — GAT spatial embeddings
        M: torch.Tensor,    # (N_BUS, N_GEN)  — generator-bus incidence
    ) -> torch.Tensor:
        """
        Returns commitment logits of shape (N_GEN, T).

        Steps
        -----
        1. BiLSTM over each bus's T-length embedding sequence.
           Input:  (N_BUS, T, d_h)
           Output: (N_BUS, T, 2*lstm_hidden)

        2. Project back to d_h.
           Output: (N_BUS, T, d_h)  — call this Ỹ_bus

        3. Map bus embeddings → generator embeddings via M.
           Ỹ_gen[g, t, :] = Σ_n M[n,g] * Ỹ_bus[n, t, :]
           Output: (N_GEN, T, d_h)

        4. Apply linear head per generator per time-step.
           Output: (N_GEN, T)
        """
        N_BUS, T, d_h = Q.shape

        # Step 1: BiLSTM
        # Q is (N_BUS, T, d_h); LSTM treats N_BUS as the batch axis.
        lstm_out, _ = self.bilstm(Q)    # (N_BUS, T, 2*lstm_hidden)

        # Step 2: Project to d_h
        Y_bus = self.out_proj(lstm_out) # (N_BUS, T, d_h)

        # Step 3: Bus → generator projection via M
        # M: (N_BUS, N_GEN)
        # Y_bus: (N_BUS, T, d_h)
        # We want Y_gen: (N_GEN, T, d_h)
        # Y_gen[g, t, :] = Σ_n M[n,g] * Y_bus[n, t, :]
        # Einsum notation: "ng, ntd -> gtd"
        Y_gen = torch.einsum('ng, ntd -> gtd', M, Y_bus)  # (N_GEN, T, d_h)

        # Step 4: Linear logit head
        # gen_head maps (d_h,) → (1,) for each (g, t)
        logits = torch.stack([
    self.gen_heads[g](Y_gen[g]).squeeze(-1)   # (T,)
    for g in range(self.n_gen)
], dim=0)   # (N_GEN, T)

        return logits