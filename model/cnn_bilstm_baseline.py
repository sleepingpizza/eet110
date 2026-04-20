"""
model/cnn_bilstm_baseline.py
=============================
Topology-blind CNN-BiLSTM baseline replicating the spatial/temporal
architecture of Venkatesh et al. (2025).

KEY STRUCTURAL DIFFERENCES from GNN_UC (intentional — this is the ablation):
  - Receives NO edge_index, NO edge_attr, NO contingency information
  - CNN pools over bus spatial dimension → loses bus-level locality
  - No dynamic adjacency matrix — topology changes are invisible to model
  - Trained on base-case scenarios only (no N-1 augmentation)

These are not implementation weaknesses — they are the exact properties
of the Venkatesh et al. architecture that this work argues are insufficient
for N-1 security-constrained UC.

Architecture (matching Venkatesh et al. Fig. 3):
  Input  : (N_BUS, T, F) node feature tensor
  CNN    : 1D conv over N_BUS spatial dimension at each timestep → (T, d_cnn)
  BiLSTM : processes (T, d_cnn) sequence → (T, 2*lstm_hidden)
  Head   : linear → (T, N_GEN) → transpose → (N_GEN, T) logits
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data


class CNN_BiLSTM_Baseline(nn.Module):
    """
    Topology-blind CNN-BiLSTM for unit commitment.

    Parameters
    ----------
    n_bus        : number of buses (30)
    n_features   : node features per timestep (4)
    t_horizon    : scheduling horizon (24)
    n_gen        : number of generators (6)
    cnn_filters  : CNN output channels (32, matching Venkatesh Table 3)
    cnn_kernel   : CNN kernel size (3)
    lstm_hidden  : BiLSTM hidden size (64, matching our GNN for fairness)
    dropout      : dropout rate (0.10)
    """

    def __init__(
        self,
        n_bus       : int   = 30,
        n_features  : int   = 4,
        t_horizon   : int   = 24,
        n_gen       : int   = 6,
        cnn_filters : int   = 32,
        cnn_kernel  : int   = 3,
        lstm_hidden : int   = 64,
        dropout     : float = 0.10,
    ):
        super().__init__()
        self.n_gen     = n_gen
        self.t_horizon = t_horizon

        # ── CNN spatial encoder ───────────────────────────────────────────
        # Processes each timestep independently.
        # Input shape for Conv1d: (batch=T, channels=n_features, length=n_bus)
        # This is the "flat Euclidean" treatment — buses are treated as
        # an ordered sequence, not a graph.
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_filters, kernel_size=cnn_kernel,
                      padding=cnn_kernel // 2),
            nn.Tanh(),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=cnn_kernel,
                      padding=cnn_kernel // 2),
            nn.Tanh(),
            nn.AdaptiveAvgPool1d(1),   # pool over bus dimension → scalar per filter
        )

        # ── BiLSTM temporal encoder ───────────────────────────────────────
        self.bilstm = nn.LSTM(
            input_size   = cnn_filters,
            hidden_size  = lstm_hidden,
            num_layers   = 1,
            bidirectional= True,
            batch_first  = True,
            dropout      = 0.0,
        )

        # ── Output projection ─────────────────────────────────────────────
        self.out_proj = nn.Sequential(
            nn.Linear(2 * lstm_hidden, lstm_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        # ── Generator head ────────────────────────────────────────────────
        self.gen_head = nn.Linear(lstm_hidden, n_gen)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : PyG Data object — only data.x is used.
               data.edge_index, data.edge_attr, data.contingency are
               deliberately ignored — this is the topology-blind design.

        Returns
        -------
        logits : (N_GEN, T)
        """
        x = data.x   # (N_BUS, T, F) — ignores all topology information

        N_BUS, T, F = x.shape

        # Reshape for CNN: process each timestep independently
        # (N_BUS, T, F) → (T, F, N_BUS)  so Conv1d sees F channels over N_BUS
        x_cnn = x.permute(1, 2, 0)          # (T, F, N_BUS)

        # CNN: (T, F, N_BUS) → (T, cnn_filters, 1) → (T, cnn_filters)
        cnn_out = self.cnn(x_cnn).squeeze(-1)   # (T, cnn_filters)

        # BiLSTM: (1, T, cnn_filters) → (1, T, 2*lstm_hidden)
        lstm_in  = cnn_out.unsqueeze(0)          # (1, T, cnn_filters)
        lstm_out, _ = self.bilstm(lstm_in)       # (1, T, 2*lstm_hidden)
        lstm_out = lstm_out.squeeze(0)            # (T, 2*lstm_hidden)

        # Project: (T, 2*lstm_hidden) → (T, lstm_hidden)
        proj = self.out_proj(lstm_out)            # (T, lstm_hidden)

        # Generator logits: (T, lstm_hidden) → (T, N_GEN) → (N_GEN, T)
        logits = self.gen_head(proj).T            # (N_GEN, T)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)