import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Optional

from data.ieee30_system import (
    GENERATORS, LINES, N_BUS, N_GEN, N_LINE, T_HORIZON,
    GAMMA, REF_BUS, N_SEG, S_BASE
)
from utils.dc_powerflow import build_ptdf_matrix


def solve_pm1_sc(
    net_load_bus    : np.ndarray,        # (N_BUS, T) net MW load
    costs           : np.ndarray,        # (N_GEN, N_SEG) fuel costs $/MWh
    z_pred          : np.ndarray,        # (N_GEN, T) GNN binary prediction
    removed_line_idx: Optional[int],     # None = base case
    K               : float = 1.0,       # K-factor for cost satisfaction region
    mip_gap         : float = 0.005,
    time_limit      : float = 120.0,
    verbose         : bool  = False,
) -> dict:
    """
    Solve the PM1-SC fuzzy MILP-UC.

    Returns
    -------
    dict with keys:
        status       : 'optimal' | 'feasible' | 'infeasible' | 'timeout'
        obj_lambda   : fuzzy objective λ (satisfaction level, 0-1)
        obj_cost     : UC operational cost C(x^F)
        z            : (N_GEN, T) optimal commitment schedule
        p            : (N_GEN, T) optimal dispatch (MW)
        flows        : (N_LINE, T) line flows (MW)
        soi          : suboptimality index vs MILP-UC benchmark
        solve_time   : wall-clock solve time (s)
        n_on_correct : number of ON predictions that were kept ON
        n_off_correct: number of OFF predictions that were kept OFF
    """
    G = N_GEN
    T = T_HORIZON
    M_seg = N_SEG
    N = N_BUS

    # ── Build PTDF matrix ─────────────────────────────────────────────────────
    try:
        PTDF = build_ptdf_matrix(N, LINES, ref_bus_idx=REF_BUS - 1,
                                  removed_line_idx=removed_line_idx)
    except ValueError as e:
        return {'status': 'infeasible', 'reason': str(e)}

    line_rates = np.array([ln[5] for ln in LINES], dtype=float)
    gen_bus    = [g['bus'] - 1 for g in GENERATORS]

    # ── Partition GNN prediction into ON/OFF index sets ───────────────────────
    # Υu = {(g,t) : z̃_g(t) = 1}   (eq. 39 of baseline, extended to SC)
    # Υd = {(g,t) : z̃_g(t) = 0}
    upsilon_u = [(g, t) for g in range(G) for t in range(T) if z_pred[g, t] == 1]
    upsilon_d = [(g, t) for g in range(G) for t in range(T) if z_pred[g, t] == 0]

    # ── Compute cost bounds for fuzzy objective constraint ────────────────────
    # C_bar: upper bound = all units ON at max generation
    C_bar = 0.0
    for g, gen in enumerate(GENERATORS):
        for t in range(T):
            C_bar += gen['c_noload']
            for seg in range(M_seg):
                C_bar += costs[g, seg] * gen['Pu_seg'][seg]
        C_bar += gen['c_startup'] * T  # worst-case startups

    # C_under: LP relaxation lower bound (solve with relaxed binaries)
    C_under = _compute_lp_lower_bound(
        net_load_bus, costs, removed_line_idx, PTDF, line_rates, gen_bus
    )
    if C_under is None:
        C_under = 0.0   # fallback if LP fails

    # ── Build Gurobi model ────────────────────────────────────────────────────
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 1 if verbose else 0)
    env.start()
    m = gp.Model(env=env)
    m.setParam('MIPGap',    mip_gap)
    m.setParam('TimeLimit', time_limit)
    m.setParam('Presolve',  2)
    m.setParam('Threads',   4)

    # ── Decision variables ────────────────────────────────────────────────────
    # Fuzzy objective
    lam    = m.addVar(lb=0.0, ub=1.0, name='lambda')

    # UC binary variables (same as base MILP-UC)
    z_f    = m.addVars(G, T, vtype=GRB.BINARY, name='z')
    v_f    = m.addVars(G, T, vtype=GRB.BINARY, name='v')
    y_f    = m.addVars(G, T, vtype=GRB.BINARY, name='y')

    # Continuous dispatch variables
    p_seg  = m.addVars(G, M_seg, T, lb=0.0, name='p_seg')
    p      = m.addVars(G, T,        lb=0.0, name='p')
    r      = m.addVars(G, T,        lb=0.0, name='r')
    s_r    = m.addVars(T,           lb=0.0, name='s_r')
    f_br   = m.addVars(N_LINE, T,   lb=-GRB.INFINITY, name='f_br')

    # ── Objective: maximise λ (PM1-SC) ───────────────────────────────────────
    m.setObjective(lam, GRB.MAXIMIZE)

    # ── PM1-SC fuzzy constraints (eqs PM1-SC-a, PM1-SC-b, PM1-SC-c) ─────────

    # (PM1-SC-a): λ ≤ z^F_g(t)  for all (g,t) ∈ Υu  [ON predictions]
    for (g, t) in upsilon_u:
        m.addConstr(lam <= z_f[g, t], name=f'pm1a_g{g}_t{t}')

    # (PM1-SC-b): λ + z^F_g(t) ≤ 1  for all (g,t) ∈ Υd  [OFF predictions]
    for (g, t) in upsilon_d:
        m.addConstr(lam + z_f[g, t] <= 1.0, name=f'pm1b_g{g}_t{t}')

    # (PM1-SC-c): K-factor cost satisfaction constraint
    # K * (C_bar - C(x^F)) / (C_bar - C_under) ≥ λ
    # Rearranged: (C_bar - C_under) * λ / K + C(x^F) ≤ C_bar
    obj_cost_expr = gp.LinExpr()
    for g, gen in enumerate(GENERATORS):
        c_z  = gen['c_noload']
        c_su = gen['c_startup']
        c_sd = gen['c_shutdown']
        c_res = 5.0
        for t in range(T):
            obj_cost_expr += c_z  * z_f[g, t]
            obj_cost_expr += c_su * v_f[g, t]
            obj_cost_expr += c_sd * y_f[g, t]
            obj_cost_expr += c_res * r[g, t]
            for seg in range(M_seg):
                obj_cost_expr += costs[g, seg] * p_seg[g, seg, t]

    cost_range = max(C_bar - C_under, 1.0)
    m.addConstr(
        (cost_range / K) * lam + obj_cost_expr <= C_bar,
        name='pm1c_cost_satisfaction'
    )

    # ── Standard UC constraints (G^SC — same as milp_uc.py) ──────────────────
    for g, gen in enumerate(GENERATORS):
        TU   = gen['TU'];   TD   = gen['TD']
        RU   = gen['RU'];   RD   = gen['RD']
        PD   = gen['PD'];   Pmax = gen['Pmax']
        Pmin = gen['Pmin']; Tint = gen['Tint']
        Pg0  = gen['Pg_int']; R10 = gen['R10']
        Pu   = gen['Pu_seg']

        # State transition (eqs 4-5)
        for t in range(T):
            m.addConstr(v_f[g,t] + y_f[g,t] <= 1)
        z_init = 1 if Tint > 0 else 0
        m.addConstr(z_f[g,0] - z_init == v_f[g,0] - y_f[g,0])
        for t in range(1, T):
            m.addConstr(z_f[g,t] - z_f[g,t-1] == v_f[g,t] - y_f[g,t])

        # Initial uptime/downtime
        if Tint > 0:
            min_on = max(0, TU - Tint)
            for t in range(min(min_on, T)):
                m.addConstr(z_f[g,t] == 1)
        else:
            min_off = max(0, TD - abs(Tint))
            for t in range(min(min_off, T)):
                m.addConstr(z_f[g,t] == 0)

        # Uptime constraints (eq 8)
        for t in range(T - 1):
            t_end = min(t + 1 + TU, T)
            for s in range(t + 2, t_end):
                m.addConstr(z_f[g,s] >= v_f[g,t+1])

        # Downtime constraints (eq 10)
        for t in range(T - 1):
            t_end = min(t + 1 + TD, T)
            for s in range(t + 1, t_end):
                m.addConstr(1 - z_f[g,s] >= y_f[g,t+1])

        # Generation = sum of segments (eq 11)
        for t in range(T):
            m.addConstr(p[g,t] == gp.quicksum(p_seg[g,seg,t] for seg in range(M_seg)))

        # Ramp constraints (eqs 12-15)
        for t in range(1, T):
            m.addConstr(p[g,t] - p[g,t-1] <= RU*(1-v_f[g,t]) + PD*v_f[g,t])
            m.addConstr(p[g,t-1] - p[g,t] <= PD*y_f[g,t] + RD*(1-y_f[g,t]))
        m.addConstr(p[g,0] - Pg0 <= RU*(1-v_f[g,0]) + PD*v_f[g,0])
        m.addConstr(Pg0 - p[g,0] <= PD*y_f[g,0] + RD*(1-y_f[g,0]))

        # Generation bounds (eqs 16-17)
        for t in range(T):
            for seg in range(M_seg):
                m.addConstr(p_seg[g,seg,t] <= Pu[seg] * z_f[g,t])
            m.addConstr(p[g,t] >= Pmin * z_f[g,t])

        # Reserve constraints (eqs 25-26)
        for t in range(T):
            m.addConstr(r[g,t] <= R10 * z_f[g,t])
            m.addConstr(r[g,t] <= Pmax * z_f[g,t] - p[g,t])

    # Power balance (eq 18)
    for t in range(T):
        total_load_t = float(net_load_bus[:, t].sum())
        m.addConstr(gp.quicksum(p[g,t] for g in range(G)) == total_load_t)

    # Line flows via PTDF + thermal limits (eqs 19-21)
    for l in range(N_LINE):
        if l == removed_line_idx:
            for t in range(T):
                m.addConstr(f_br[l,t] == 0.0)
            continue
        rate = line_rates[l]
        for t in range(T):
            flow_def = (
                gp.quicksum(
                    PTDF[l, gen_bus[g]] * p[g,t]
                    for g in range(G) if abs(PTDF[l, gen_bus[g]]) > 1e-8
                )
                - float(sum(PTDF[l,n] * net_load_bus[n,t] for n in range(N)))
            )
            m.addConstr(f_br[l,t] == flow_def)
            m.addConstr(f_br[l,t] <=  rate)
            m.addConstr(f_br[l,t] >= -rate)

    # Spinning reserve (eqs 23-24)
    for t in range(T):
        for g in range(G):
            m.addConstr(s_r[t] >= GENERATORS[g]['Pmax'] * z_f[g,t])
        m.addConstr(gp.quicksum(r[g,t] for g in range(G)) >= GAMMA * s_r[t])

    # ── Solve ─────────────────────────────────────────────────────────────────
    m.optimize()

    status_map = {
        GRB.OPTIMAL:    'optimal',
        GRB.SUBOPTIMAL: 'feasible',
        GRB.INFEASIBLE: 'infeasible',
        GRB.TIME_LIMIT: 'timeout',
    }
    sol_status = status_map.get(m.Status, f'unknown_{m.Status}')

    result = {
        'status'    : sol_status,
        'solve_time': m.Runtime,
        'C_bar'     : C_bar,
        'C_under'   : C_under,
    }

    if m.SolCount > 0:
        lam_val  = lam.X
        cost_val = sum(
            GENERATORS[g]['c_noload'] * z_f[g,t].X
            + GENERATORS[g]['c_startup'] * v_f[g,t].X
            + GENERATORS[g]['c_shutdown'] * y_f[g,t].X
            + 5.0 * r[g,t].X
            + sum(costs[g,seg] * p_seg[g,seg,t].X for seg in range(M_seg))
            for g in range(G) for t in range(T)
        )

        z_out = np.array([[z_f[g,t].X for t in range(T)] for g in range(G)])
        p_out = np.array([[p[g,t].X   for t in range(T)] for g in range(G)])
        f_out = np.array([[f_br[l,t].X for t in range(T)] for l in range(N_LINE)])

        # Count how many GNN decisions were respected
        n_on_correct  = sum(
            1 for (g,t) in upsilon_u if z_out[g,t] > 0.5
        )
        n_off_correct = sum(
            1 for (g,t) in upsilon_d if z_out[g,t] < 0.5
        )

        result.update({
            'obj_lambda'    : lam_val,
            'obj_cost'      : cost_val,
            'z'             : z_out,
            'p'             : p_out,
            'flows'         : f_out,
            'n_on_correct'  : n_on_correct,
            'n_off_correct' : n_off_correct,
            'n_upsilon_u'   : len(upsilon_u),
            'n_upsilon_d'   : len(upsilon_d),
        })

    m.dispose()
    env.dispose()
    return result


def _compute_lp_lower_bound(
    net_load_bus, costs, removed_line_idx, PTDF, line_rates, gen_bus
) -> Optional[float]:
    """
    Compute LP relaxation lower bound C_under by relaxing all binaries.
    Used in the fuzzy cost satisfaction constraint (PM1-SC-c).
    """
    G = N_GEN; T = T_HORIZON; M_seg = N_SEG; N = N_BUS

    try:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        m = gp.Model(env=env)

        z_r    = m.addVars(G, T, lb=0.0, ub=1.0, name='z')
        v_r    = m.addVars(G, T, lb=0.0, ub=1.0, name='v')
        y_r    = m.addVars(G, T, lb=0.0, ub=1.0, name='y')
        p_seg  = m.addVars(G, M_seg, T, lb=0.0, name='p_seg')
        p      = m.addVars(G, T, lb=0.0, name='p')
        r      = m.addVars(G, T, lb=0.0, name='r')
        s_r    = m.addVars(T, lb=0.0, name='s_r')

        obj = gp.LinExpr()
        for g, gen in enumerate(GENERATORS):
            for t in range(T):
                obj += gen['c_noload'] * z_r[g,t]
                obj += gen['c_startup'] * v_r[g,t]
                obj += gen['c_shutdown'] * y_r[g,t]
                obj += 5.0 * r[g,t]
                for seg in range(M_seg):
                    obj += costs[g,seg] * p_seg[g,seg,t]
        m.setObjective(obj, GRB.MINIMIZE)

        for g, gen in enumerate(GENERATORS):
            Pu = gen['Pu_seg']
            for t in range(T):
                m.addConstr(p[g,t] == gp.quicksum(p_seg[g,seg,t] for seg in range(M_seg)))
                for seg in range(M_seg):
                    m.addConstr(p_seg[g,seg,t] <= Pu[seg] * z_r[g,t])
                m.addConstr(p[g,t] >= gen['Pmin'] * z_r[g,t])
                m.addConstr(p[g,t] <= gen['Pmax'] * z_r[g,t])

        for t in range(T):
            m.addConstr(gp.quicksum(p[g,t] for g in range(G)) == float(net_load_bus[:,t].sum()))

        m.optimize()
        val = m.ObjVal if m.Status == GRB.OPTIMAL else None
        m.dispose(); env.dispose()
        return val
    except Exception:
        return None