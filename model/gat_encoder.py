"""
model/gat_encoder.py  (fixed v2)
=================================
Performance fix (BUG 12):
  The original forward() rebuilt ei_tiled and ea_tiled on every forward
  call using a Python loop over T=24, allocating and concatenating 24
  tensors each time.  With 6174 samples per epoch this caused ~148k
  tensor allocations per epoch, most on CPU before transfer to GPU.
  Fix: cache the tiled edge tensors on first call and reuse them.
  The topology (edge_index, edge_attr) is identical for all samples
  with the same contingency — for batch_size=1 it is always the same
  object shape.  We cache by (E, T, device) key.

Bug fix (BUG 13):
  forward() had broken indentation — the body was at module level
  rather than method level, which Python accepted due to the def
  being syntactically valid but caused silent incorrect behaviour
  when self.training was referenced.  Fixed all indentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATSpatialEncoder(nn.Module):
    """
    K-layer GATv2 spatial encoder.

    Parameters
    ----------
    in_features : raw node features per time-step (4)
    d_h         : hidden / output embedding dimension (64)
    n_heads     : attention heads in intermediate layers (4)
    n_layers    : total GATv2 layers (2)
    edge_dim    : edge feature dimension (3)
    dropout     : dropout probability (0.10)
    """

    def __init__(
        self,
        in_features: int   = 4,
        d_h:         int   = 64,
        n_heads:     int   = 4,
        n_layers:    int   = 2,
        edge_dim:    int   = 3,
        dropout:     float = 0.10,
    ):
        super().__init__()
        self.d_h      = d_h
        self.n_layers = n_layers
        self.dropout  = dropout

        # Cache for tiled edge tensors — keyed by (E, N, device_str)
        self._edge_cache: dict = {}

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_h),
            nn.LayerNorm(d_h),
            nn.ELU(),
        )

        self.gat_layers  = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.ff_layers   = nn.ModuleList()

        for k in range(n_layers):
            is_last = (k == n_layers - 1)

            if is_last:
                self.gat_layers.append(
                    GATv2Conv(
                        in_channels    = d_h,
                        out_channels   = d_h,
                        heads          = 1,
                        concat         = False,
                        edge_dim       = edge_dim,
                        dropout        = dropout,
                        add_self_loops = True,
                    )
                )
                gat_out_dim = d_h
            else:
                head_dim = d_h // n_heads
                self.gat_layers.append(
                    GATv2Conv(
                        in_channels    = d_h,
                        out_channels   = head_dim,
                        heads          = n_heads,
                        concat         = True,
                        edge_dim       = edge_dim,
                        dropout        = dropout,
                        add_self_loops = True,
                    )
                )
                gat_out_dim = d_h

            self.norm_layers.append(nn.LayerNorm(gat_out_dim))
            self.ff_layers.append(nn.Sequential(
                nn.Linear(gat_out_dim, gat_out_dim * 2),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(gat_out_dim * 2, d_h),
                nn.LayerNorm(d_h),
            ))

        self.out_dim = d_h

    def _get_tiled_edges(
        self,
        edge_index: torch.Tensor,   # (2, E)
        edge_attr:  torch.Tensor,   # (E, 3)
        N:          int,
        T:          int,
    ):
        """
        Build tiled edge tensors for all T timesteps at once.
        Cached by (E, N, T, device) so we only build once per unique topology.
        For batch_size=1 this means once per contingency type (41 unique
        topologies), not once per sample.
        """
        E = edge_index.shape[1]
        cache_key = (E, N, T, str(edge_index.device))

        if cache_key not in self._edge_cache:
            # Build offset list on the same device as edge_index
            offsets = torch.arange(T, device=edge_index.device) * N  # (T,)
            # edge_index: (2, E) → add offset per timestep → (2, E*T)
            ei_tiled = (
                edge_index.unsqueeze(2)          # (2, E, 1)
                + offsets.view(1, 1, T)          # (1, 1, T)
            ).reshape(2, E * T)                  # (2, E*T)

            # ea_tiled: (E, 3) → (E*T, 3) via repeat
            ea_tiled = edge_attr.repeat(T, 1)    # (E*T, 3)

            self._edge_cache[cache_key] = (ei_tiled, ea_tiled)

        return self._edge_cache[cache_key]

    def forward(
        self,
        x:          torch.Tensor,   # (N, T, F)
        edge_index: torch.Tensor,   # (2, E)
        edge_attr:  torch.Tensor,   # (E, 3)
    ) -> torch.Tensor:              # (N, T, d_h)
        N, T, n_feat = x.shape

        # Flatten: (N, T, F) → (N*T, F)
        x_flat = x.reshape(N * T, n_feat)

        # Get cached (or freshly built) tiled edge tensors
        ei_tiled, ea_tiled = self._get_tiled_edges(edge_index, edge_attr, N, T)

        # GAT forward pass over all T timesteps simultaneously
        h = self.input_proj(x_flat)    # (N*T, d_h)

        for k in range(self.n_layers):
            h_new = self.gat_layers[k](h, ei_tiled, edge_attr=ea_tiled)
            h_new = self.norm_layers[k](h_new)
            h_new = F.elu(h_new)
            h_new = self.ff_layers[k](h_new)
            h     = h + h_new
            h     = F.dropout(h, p=self.dropout, training=self.training)

        # Unflatten: (N*T, d_h) → (N, T, d_h)
        return h.reshape(N, T, self.d_h)