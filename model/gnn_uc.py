"""
model/gnn_uc.py
===============
Top-level GNN-UC model.  Wires together:
  1. GATSpatialEncoder  (model/gat_encoder.py)
  2. BiLSTMTemporalDecoder (model/bilstm_decoder.py)

and exposes a single forward() that takes a PyG Data object and returns
per-generator commitment logits.

DESIGN PRINCIPLE: KEEP THE FORWARD PASS CLEAN
  The model should not need to know about:
    - Normalisation (done in graph_dataset.py)
    - Loss computation (done in train.py)
    - Fuzzy constraint construction (done in solver/fuzzy_uc.py — Stage 3)
  This separation of concerns makes unit-testing each component easy
  and keeps the model swap-able (e.g., replace GAT with a GCN for
  ablation without touching training code).

SINGLE-SAMPLE vs BATCHED FORWARD
  PyG's DataLoader batches multiple graphs by offsetting node indices
  and concatenating edge_index tensors.  This means a batch of B graphs
  with N nodes each appears as one graph with B*N nodes.  Our forward()
  handles this transparently because:
    - GATv2Conv operates on the concatenated (B*N, T, d_h) node tensor.
    - The BiLSTM operates independently per node (no cross-graph temporal
      mixing) because we reshape to treat each node as a batch element.
    - M is block-diagonal across the batch (each sample's M is independent).
  PyG's `batch` attribute tells us which graph each node belongs to,
  but we do not need it explicitly here because M already encodes the
  generator-bus mapping per graph.

WEIGHT PARAMETER COUNT (approximate, d_h=64, n_heads=4, n_layers=2)
  Input projection:          4×64 + 64         =   320
  GATv2 layer 1 (concat):   2×64×16 + 64×1×16 = 2 048 + 1 024 + ... ≈ 5k
  GATv2 layer 2 (avg):       similar            ≈  5k
  FF blocks:                 2×(64×128 + 128×64)= 32k
  BiLSTM:                    4×(64+64+1)×64 × 2 ≈ 66k
  Out projection:            128×64             ≈  8k
  Gen head:                  64×1 × N_GEN=6    =  384
  LayerNorms:                small
  TOTAL                      ≈ 117k parameters

  This is intentionally modest.  The 30-bus system has 8 820 training
  samples × 24 time-steps = 211 680 training examples.  A model with
  ~100k parameters has a healthy 2:1 examples-to-parameters ratio.
  Doubling d_h to 128 would quadruple the parameter count to ~450k,
  risking overfitting.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from model.gat_encoder import GATSpatialEncoder
from model.bilstm_decoder import BiLSTMTemporalDecoder


class GNN_UC(nn.Module):
    """
    Topology-aware Graph Neural Network for Unit Commitment.

    Maps a power grid snapshot (node features + topology) to per-generator
    commitment decisions over a 24-hour horizon.

    Parameters
    ----------
    in_features  : number of raw node features (4)
    d_h          : shared embedding dimension (64)
    n_heads      : GAT attention heads (4)
    gat_layers   : number of GATv2 layers (2)
    lstm_hidden  : BiLSTM hidden size (64)
    n_gen        : number of generators (6)
    dropout      : shared dropout rate (0.1)
    """

    def __init__(
        self,
        in_features: int   = 4,
        d_h:         int   = 64,
        n_heads:     int   = 4,
        gat_layers:  int   = 2,
        lstm_hidden: int   = 64,
        n_gen:       int   = 6,
        dropout:     float = 0.10,
    ):
        super().__init__()

        self.spatial_encoder = GATSpatialEncoder(
            in_features = in_features,
            d_h         = d_h,
            n_heads     = n_heads,
            n_layers    = gat_layers,
            edge_dim    = 3,
            dropout     = dropout,
        )

        self.temporal_decoder = BiLSTMTemporalDecoder(
            d_h         = d_h,
            lstm_hidden = lstm_hidden,
            n_gen       = n_gen,
            dropout     = dropout,
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : PyG Data object with attributes:
            x          (N, T, 4)    normalised node features
            edge_index (2, E)       adjacency (contingency-aware)
            edge_attr  (E, 3)       normalised susceptances
            M          (N, N_GEN)   generator-bus incidence

        Returns
        -------
        logits : (N_GEN, T) — raw pre-sigmoid commitment scores.
            Positive logit → model predicts generator ON.
            Apply torch.sigmoid() to get probabilities.
            Apply (logits >= 0) to get binary predictions.
        """
        x          = data.x           # (N, T, F)
        edge_index = data.edge_index  # (2, E)
        edge_attr  = data.edge_attr   # (E, 1)
        M          = data.M           # (N, N_GEN)

        # Stage 1: Spatial encoding — same GAT weights for all T time-steps
        Q = self.spatial_encoder(x, edge_index, edge_attr)   # (N, T, d_h)

        # Stage 2: Temporal encoding + generator projection
        logits = self.temporal_decoder(Q, M)                 # (N_GEN, T)

        return logits

    def predict(self, data: Data) -> torch.Tensor:
        """
        Convenience method: returns hard binary commitment decisions.
        Used during evaluation and fuzzy MILP construction (Stage 3).

        Returns
        -------
        z_pred : (N_GEN, T) boolean tensor
        """
        with torch.no_grad():
            logits = self.forward(data)
        return (logits >= 0.0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)