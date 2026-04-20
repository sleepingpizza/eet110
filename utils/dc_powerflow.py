import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional


def build_susceptance_matrix(
    n_bus: int,
    lines: list[tuple],
    removed_line_idx: Optional[int] = None,
) -> np.ndarray:
    
    B = np.zeros((n_bus, n_bus), dtype=float)

    for idx, (fb, tb, _r, x, _b, _rate) in enumerate(lines):
        if idx == removed_line_idx:
            continue                  # N-1: skip this line
        if abs(x) < 1e-8:
            x = 1e-4                  # avoid divide-by-zero on zero-x transformers
        i = fb - 1                    # convert to 0-indexed
        j = tb - 1
        b_line = 1.0 / x
        B[i, i] += b_line
        B[j, j] += b_line
        B[i, j] -= b_line
        B[j, i] -= b_line

    return B


def build_reduced_B(
    n_bus: int,
    lines: list[tuple],
    ref_bus_idx: int = 0,
    removed_line_idx: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    
    B_full = build_susceptance_matrix(n_bus, lines, removed_line_idx)

    # Remove ref bus row and column
    idx = list(range(n_bus))
    idx.pop(ref_bus_idx)
    B_red = B_full[np.ix_(idx, idx)]

    return B_full, B_red


def compute_angles_and_flows(
    net_injection: np.ndarray,   # P_inject (MW), shape (n_bus,)
    lines: list[tuple],
    n_bus: int,
    ref_bus_idx: int = 0,
    removed_line_idx: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve DC power flow and return bus angles (radians) and
    line power flows (MW).

    DC power flow:
        B_red · θ_red = P_inject_red / S_base   [already in pu from caller]
        P_flow_l = (θ_i - θ_j) / x_l   [pu → multiply by S_base for MW]

    Parameters
    ----------
    net_injection : shape (n_bus,) net generation minus load (MW)
    lines         : line list from ieee30_system
    n_bus         : number of buses
    ref_bus_idx   : reference bus index (0-indexed)
    removed_line_idx : tripped line for N-1 (None = base case)

    Returns
    -------
    theta      : shape (n_bus,)   bus voltage angles (radians)
    line_flows : shape (n_lines,) power flows (MW, positive = from→to)
    """
    from data.ieee30_system import S_BASE

    B_full, B_red = build_reduced_B(n_bus, lines, ref_bus_idx, removed_line_idx)

    # Remove ref bus from injection vector
    idx_non_ref = list(range(n_bus))
    idx_non_ref.pop(ref_bus_idx)
    P_red = net_injection[idx_non_ref] / S_BASE   # pu

    # Solve: sparse system for stability even at 30-bus scale
    B_sp  = csr_matrix(B_red)
    try:
        theta_red = np.linalg.solve(B_red, P_red)
    except np.linalg.LinAlgError:
        # Island formed by contingency — power flow infeasible
        raise ValueError(
            f"Singular B matrix: contingency line {removed_line_idx} "
            "creates an island. Skip this contingency in dataset."
        )

    # Reconstruct full angle vector (θ_ref = 0)
    theta = np.zeros(n_bus)
    for k, bus in enumerate(idx_non_ref):
        theta[bus] = theta_red[k]

    # Compute line flows (MW)
    n_lines = len(lines)
    line_flows = np.zeros(n_lines)
    for idx, (fb, tb, _r, x, _b, _rate) in enumerate(lines):
        if idx == removed_line_idx:
            line_flows[idx] = 0.0
            continue
        if abs(x) < 1e-8:
            x = 1e-4
        i, j = fb - 1, tb - 1
        line_flows[idx] = (theta[i] - theta[j]) / x * S_BASE   # MW

    return theta, line_flows


def build_ptdf_matrix(
    n_bus: int,
    lines: list[tuple],
    ref_bus_idx: int = 0,
    removed_line_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Build the Power Transfer Distribution Factor (PTDF) matrix.
    Shape: (n_lines, n_bus).

    PTDF[l, n] = change in flow on line l per unit of power injection
                 at bus n (with withdrawal at the slack bus).

    Used in the MILP formulation as:
        p_br_l(t) = sum_n PTDF[l,n] * P_net_n(t)   [MW]

    This replaces explicit angle variables (delta_n) in the MILP,
    reducing the number of continuous variables by N and eliminating
    the need for angle-bound constraints (eq. 22 in Venkatesh et al.).

    Design note: PTDF avoids angle variables but couples all buses
    in every line-flow constraint.  For 30-bus this is fine.
    For 2383-bus, the original angle-variable formulation may be
    more sparse and faster — benchmark both.
    """
    from data.ieee30_system import S_BASE

    B_full, B_red = build_reduced_B(n_bus, lines, ref_bus_idx, removed_line_idx)

    n_lines = len(lines)
    PTDF = np.zeros((n_lines, n_bus))

    idx_non_ref = list(range(n_bus))
    idx_non_ref.pop(ref_bus_idx)

    try:
        B_inv_red = np.linalg.inv(B_red)
    except np.linalg.LinAlgError:
        raise ValueError(
            f"Singular reduced B: contingency {removed_line_idx} creates island."
        )

    # For each non-reference bus n, inject 1 pu and solve for angles
    for k, bus_n in enumerate(idx_non_ref):
        e_k = np.zeros(len(idx_non_ref))
        e_k[k] = 1.0
        theta_red = B_inv_red @ e_k         # shape (N-1,)

        theta = np.zeros(n_bus)
        for kk, bus in enumerate(idx_non_ref):
            theta[bus] = theta_red[kk]

        for l_idx, (fb, tb, _r, x, _b, _rate) in enumerate(lines):
            if l_idx == removed_line_idx:
                continue
            if abs(x) < 1e-8:
                x = 1e-4
            i, j = fb - 1, tb - 1
            PTDF[l_idx, bus_n] = (theta[i] - theta[j]) / x   # pu/pu

    return PTDF   # pu/pu; multiply by S_BASE (100) to get MW/MW=dimensionless