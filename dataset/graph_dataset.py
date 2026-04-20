

import json
import os
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


# ─────────────────────────────────────────────────────────────────────────────
# Helper: one numpy sample → one PyG Data object
# ─────────────────────────────────────────────────────────────────────────────

def _make_data(
    X_raw:       np.ndarray,   # (N_BUS, T, 4)   already z-scored by augmented_dataset
    edge_index:  np.ndarray,   # (2, E)
    edge_attr:   np.ndarray,   # (E, 3)  [susceptance_pu, rate_pu, is_removed]
    z:           np.ndarray,   # (N_GEN, T)
    M:           np.ndarray,   # (N_BUS, N_GEN)
    contingency: int,
) -> Data:
    """
    Convert one raw sample → one PyG Data object.

    X_raw is already normalised by normalise_dataset() inside
    augmented_dataset.py before being written to the .npz file.
    We do NOT re-normalise here.

    edge_attr shape is (E, 3) — three columns:
      col 0  susceptance 1/x_l  (pu)     range ~2–10
      col 1  thermal rate / S_BASE (pu)  range ~0.16–2
      col 2  is_removed flag            0 or 1
    All three are in physically meaningful ranges.  The is_removed flag
    must NOT be rescaled — it is the topology-change signal for the GAT.
    """
    x_t          = torch.from_numpy(X_raw.astype(np.float32))            # (N, T, 4)
    edge_index_t = torch.from_numpy(edge_index.astype(np.int64))         # (2, E)
    edge_attr_t  = torch.from_numpy(edge_attr.astype(np.float32))        # (E, 3)
    z_t          = torch.from_numpy(z.astype(np.float32))                # (N_GEN, T)
    M_t          = torch.from_numpy(M.astype(np.float32))                # (N_BUS, N_GEN)

    return Data(
        x           = x_t,
        edge_index  = edge_index_t,
        edge_attr   = edge_attr_t,
        y           = z_t,
        M           = M_t,
        contingency = contingency,
        num_nodes   = X_raw.shape[0],   # N_BUS
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

# Bump this string whenever process() logic changes to force cache invalidation.
_CACHE_VERSION = 'v3'


class UCGraphDataset(InMemoryDataset):
    """
    Wraps one {train, val, test} .npz split into a PyG InMemoryDataset.

    Parameters
    ----------
    root        : path to dataset_output/ directory
    split       : 'train', 'val', or 'test'
    feat_stats  : path to feat_stats.json  (default: root/feat_stats.json)
    transform   : optional PyG transform applied per __getitem__
    """

    def __init__(
        self,
        root:       str,
        split:      str = 'train',
        feat_stats: Optional[str] = None,
        transform=None,
    ):
        assert split in ('train', 'val', 'test'), \
            f"split must be train/val/test, got {split!r}"
        self.split             = split
        self._root             = root
        self._feat_stats_path  = feat_stats or os.path.join(root, 'feat_stats.json')

        # Load stats before super().__init__ because process() may be called
        # immediately inside it (if no cached .pt exists yet).
        self._load_feat_stats()

        super().__init__(root=root, transform=transform)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

    # ── InMemoryDataset protocol ──────────────────────────────────────────────

    @property
    def raw_file_names(self):
        return [f'{self.split}.npz']

    @property
    def processed_file_names(self):
        # Including _CACHE_VERSION forces re-processing when logic changes.
        return [f'{self.split}_{_CACHE_VERSION}_pyg.pt']

    def download(self):
        pass   # .npz files must be produced by pipeline.py --mode full

    def process(self):
        # ── BUG 1 FIX: S was never defined ───────────────────────────────
        npz_path = os.path.join(self._root, f'{self.split}.npz')
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Expected {npz_path}.\n"
                f"Run:  python pipeline.py --mode full --output_dir {self._root}"
            )

        raw = dict(np.load(npz_path, allow_pickle=True))

        # S = number of samples in this split — was missing in original!
        S = raw['z'].shape[0]   # ← FIX for BUG 1

        # M is stored once (no batch dim) — shape (N_BUS, N_GEN)
        # _pack() in augmented_dataset.py does: 'M': samples[0]['M']
        M = raw['M']   # (N_BUS, N_GEN)

        data_list = []
        for i in range(S):
            d = _make_data(
                X_raw      = raw['X'][i],                    # (N, T, 4)
                edge_index = raw['edge_index'][i],           # (2, E)
                edge_attr  = raw['edge_attr'][i],            # (E, 3)
                z          = raw['z'][i],                    # (N_GEN, T)
                M          = M,                              # (N, N_GEN) — no [i]
                contingency= int(raw['contingency_idx'][i]),
            )
            data_list.append(d)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_feat_stats(self):
        with open(self._feat_stats_path) as f:
            stats = json.load(f)
        # Keys in feat_stats.json are string integers "0","1","2","3"
        self._feat_mean = np.array(
            [stats[str(i)]['mean'] for i in range(3)], dtype=np.float32
        )
        self._feat_std = np.array(
            [stats[str(i)]['std']  for i in range(3)], dtype=np.float32
        )

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def n_bus(self) -> int:
        return self[0].x.shape[0]          # N_BUS = 30

    @property
    def n_gen(self) -> int:
        return self[0].y.shape[0]          # N_GEN = 6

    @property
    def t_horizon(self) -> int:
        return self[0].x.shape[1]          # T = 24

    @property
    def n_node_features(self) -> int:
        return self[0].x.shape[2]          # F = 4

    @property
    def n_edge_features(self) -> int:
        return self[0].edge_attr.shape[1]  # 3 — susceptance, rate, is_removed