import numpy as np
from data.ieee30_system import (
    GENERATORS, LINES, N_BUS, N_GEN, N_LINE, T_HORIZON, S_BASE
)


# ── Generator-bus incidence matrix (static, topology-independent) ────────────
def build_gen_bus_incidence() -> np.ndarray:
    """
    Build M ∈ {0,1}^{N_BUS × N_GEN}.
    M[n, g] = 1 iff generator g is located at bus n (0-indexed).
    """
    M = np.zeros((N_BUS, N_GEN), dtype=np.float32)
    for g, gen in enumerate(GENERATORS):
        n = gen['bus'] - 1    # 0-indexed
        M[n, g] = 1.0
    return M


GEN_BUS_INCIDENCE = build_gen_bus_incidence()

# ── Nominal generator parameters for feature construction ───────────────────
_GEN_PMAX = np.array([g['Pmax'] for g in GENERATORS], dtype=np.float32)
_GEN_CP0  = np.array([g['c_p'][0] for g in GENERATORS], dtype=np.float32)


def build_node_features(
    net_load_bus: np.ndarray,   # (N_BUS, T) net load, MW
    costs: np.ndarray,          # (N_GEN, N_SEG) cost coefficients, $/MWh
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Build node feature matrix X ∈ R^{N_BUS × T × 4}.

    Features per node n, time t:
      [0]  Pd_n(t)   : net load (MW)
      [1]  c̄p_n     : average fuel cost at bus n ($/MWh)
                       = (sum_g M[n,g] * c_p_g,0) / (sum_g M[n,g] + eps)
                       Using first-segment cost as representative.
      [2]  P̄U_n     : total installed capacity at bus n (MW)
      [3]  has_gen_n : 1 if bus has a generator, else 0

    c̄p and P̄U are time-invariant (within a scenario, cost varies
    per scenario but not per hour — per eq. 29, cost is drawn once
    per day).  We broadcast them across the T dimension.
    """
    X = np.zeros((N_BUS, T_HORIZON, 4), dtype=np.float32)

    # Feature 0: net load
    X[:, :, 0] = net_load_bus.astype(np.float32)

    # Feature 1: average fuel cost at each bus
    # c̄p_n = sum_g M[n,g]*c_p_g,0 / (sum_g M[n,g] + eps)
    # costs shape: (N_GEN, N_SEG); use segment-0 cost as representative
    cost_vec = costs[:, 0]                            # (N_GEN,)
    weighted_cost = GEN_BUS_INCIDENCE @ cost_vec      # (N_BUS,)
    gen_count     = GEN_BUS_INCIDENCE.sum(axis=1)     # (N_BUS,)
    avg_cost      = weighted_cost / (gen_count + eps) # (N_BUS,)
    X[:, :, 1] = avg_cost[:, np.newaxis]              # broadcast over T

    # Feature 2: total installed capacity at bus
    pmax_bus = GEN_BUS_INCIDENCE @ _GEN_PMAX          # (N_BUS,)
    X[:, :, 2] = pmax_bus[:, np.newaxis]

    # Feature 3: has-generator indicator
    has_gen = (gen_count > 0).astype(np.float32)
    X[:, :, 3] = has_gen[:, np.newaxis]

    return X


def build_adjacency(
    removed_line_idx: int | None = None,
    weighted: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build adjacency matrix A^(c) and edge index arrays.

    Parameters
    ----------
    removed_line_idx : index of tripped line (None = base case)
    weighted         : if True, edge weight = susceptance (1/x);
                       if False, edge weight = 1 (binary adjacency)

    Returns
    -------
    A           : (N_BUS, N_BUS) adjacency matrix (float32)
    edge_index  : (2, 2*N_LINE_active) COO edge indices (int64)
                  Both directions included (undirected graph).
    edge_attr   : (2*N_LINE_active, 3) edge features:
                  [susceptance_pu, thermal_rate_pu, is_removed]
    """
    A          = np.zeros((N_BUS, N_BUS), dtype=np.float32)
    src_list, dst_list = [], []
    attr_list          = []

    for l_idx, (fb, tb, _r, x, _b, rate) in enumerate(LINES):
        i = fb - 1    # 0-indexed
        j = tb - 1
        is_removed = float(l_idx == removed_line_idx)

        if abs(x) < 1e-8:
            x = 1e-4
        b_line = 1.0 / x              # susceptance (pu)
        weight = b_line if weighted else 1.0

        if l_idx != removed_line_idx:
            A[i, j] = weight
            A[j, i] = weight

        # Include both directions in edge_index regardless of removal
        # (is_removed feature signals the outage to the GAT)
        ea = [b_line, rate / S_BASE, is_removed]   # normalise rate to pu

        src_list += [i, j]
        dst_list += [j, i]
        attr_list += [ea, ea]

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr  = np.array(attr_list, dtype=np.float32)

    return A, edge_index, edge_attr


def build_sample(
    net_load_bus    : np.ndarray,    # (N_BUS, T)
    costs           : np.ndarray,    # (N_GEN, N_SEG)
    z_optimal       : np.ndarray,   
      # (N_GEN, T) binary — training label
    removed_line_idx: int | None,
    scenario_idx    : int,
    contingency_idx : int,           # -1 = base case, else line index
) -> dict:
    """
    Build one complete graph sample for GNN training.

    Returns a dict ready to be converted to a PyTorch Geometric Data object
    (or saved to disk as a numpy npz file for later loading).
    """
    X          = build_node_features(net_load_bus, costs)
    A, eidx, ea = build_adjacency(removed_line_idx)
    z_bin      = (z_optimal >= 0.5).astype(np.float32)  # binarise solver output

    return {
        # Graph structure
        'X'          : X,              # (N_BUS, T, 4) node features
        'A'          : A,              # (N_BUS, N_BUS) adjacency
        'edge_index' : eidx,           # (2, 2*N_LINE) COO indices
        'edge_attr'  : ea,             # (2*N_LINE, 3) edge features
        'M'          : GEN_BUS_INCIDENCE.copy(),   # (N_BUS, N_GEN)

        # Training target
        'z'          : z_bin,          # (N_GEN, T) binary commitment labels
        'net_load_bus': net_load_bus.astype(np.float32),

        # Metadata (useful for diagnostics and NFR/C-SOI evaluation later)
        'meta': {
            'scenario_idx'    : scenario_idx,
            'contingency_idx' : contingency_idx,   # line index or -1
            'removed_line'    : removed_line_idx,
            'is_base_case'    : removed_line_idx is None,
        }
    }


def normalise_dataset(samples: list[dict]) -> tuple[list[dict], dict]:
    """
    Compute dataset-wide statistics and normalise continuous node features.
    Binary features (feature indices 3) and edge is_removed (index 2) are
    skipped.

    Returns normalised samples and the stats dict (save for inference-time use).
    """
    # Collect all X arrays to compute stats
    all_X = np.concatenate([s['X'] for s in samples], axis=0)   # (N*S, T, 4)

    stats = {}
    for feat_idx in range(4):
        feat_data = all_X[:, :, feat_idx]
        if feat_idx == 3:    # binary has_gen — skip
            stats[feat_idx] = {'mean': 0.0, 'std': 1.0}
        else:
            mu  = float(feat_data.mean())
            sig = float(feat_data.std()) + 1e-8
            stats[feat_idx] = {'mean': mu, 'std': sig}

    normed = []
    for s in samples:
        s2 = dict(s)
        X  = s['X'].copy()
        for feat_idx in range(3):                # skip binary feature
            X[:, :, feat_idx] = (
                (X[:, :, feat_idx] - stats[feat_idx]['mean'])
                / stats[feat_idx]['std']
            )
        s2['X'] = X
        normed.append(s2)

    return normed, stats