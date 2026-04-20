import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import Optional

from data.ieee30_system import (
    GENERATORS, LINES, BUS_LOAD_MW, S_BASE,
    N_BUS, N_GEN, N_LINE, T_HORIZON, GAMMA, REF_BUS, N_SEG
)
from utils.dc_powerflow import build_ptdf_matrix


def solve_milp_uc(
    net_load_bus: np.ndarray,         # shape (N_BUS, T) — net MW per bus per hour
    costs: np.ndarray,                 # shape (N_GEN, T) — optional per-gen per-hour costs
    removed_line_idx: Optional[int],  # None = base case, int = N-1 contingency
    mip_gap: float = 0.001,
    time_limit: float = 300.0,
    verbose: bool = False,
) -> dict:
    
    G = N_GEN
    T = T_HORIZON
    M = N_SEG
    N = N_BUS

    try:
        PTDF = build_ptdf_matrix(N, LINES, ref_bus_idx=REF_BUS-1,
                                  removed_line_idx=removed_line_idx)
    except ValueError as e:
        return {'status': 'infeasible', 'reason': str(e)}

    # Thermal line limits (MW) — respect removed line
    line_rates = np.array([ln[5] for ln in LINES], dtype=float)

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 1 if verbose else 0)
    env.start()
    m = gp.Model(env=env)
    m.setParam('MIPGap',    mip_gap)
    m.setParam('TimeLimit', time_limit)
    m.setParam('Presolve',  2)       # aggressive presolve (eq. Table 2 context)
    m.setParam('Threads',   4)

    # ── Decision Variables ───────────────────────────────────────────────────
    # z[g,t] : binary ON/OFF (eq. 27)
    z = m.addVars(G, T, vtype=GRB.BINARY, name='z')
    # v[g,t] : binary startup   (eq. 4–5)
    v = m.addVars(G, T, vtype=GRB.BINARY, name='v')
    # y[g,t] : binary shutdown  (eq. 4–5)
    y = m.addVars(G, T, vtype=GRB.BINARY, name='y')
    # p_seg[g,m,t] : piecewise generation (MW)
    p_seg = m.addVars(G, M, T, lb=0.0, name='p_seg')
    # p[g,t] : total generation (MW)  — eq. 11
    p = m.addVars(G, T, lb=0.0, name='p')
    # r[g,t] : spinning reserve (MW)  — eq. 23–26
    r = m.addVars(G, T, lb=0.0, name='r')
    # s_r[t] : system spinning reserve (MW)  — eq. 23
    s_r = m.addVars(T, lb=0.0, name='s_r')
    # f_br[l,t] : line power flow (MW)  — replaced by PTDF expression
    #   We keep it as a variable so we can retrieve it post-solve
    f_br = m.addVars(N_LINE, T, lb=-GRB.INFINITY, name='f_br')

    # ── Objective Function: eq. (2)+(3) ─────────────────────────────────────
    # Minimize total cost:
    #   fuel (piecewise) + no-load + startup + shutdown + reserve
    obj = gp.LinExpr()
    for g, gen in enumerate(GENERATORS):
        c_z   = gen['c_noload']
        c_su  = gen['c_startup']
        c_sd  = gen['c_shutdown']
        c_res = 5.0                       # $/MWh reserve cost (typical)
        for t in range(T):
            obj += c_z * z[g, t]          # no-load cost (eq. 3 constant term)
            obj += c_su * v[g, t]         # startup cost
            obj += c_sd * y[g, t]         # shutdown cost
            obj += c_res * r[g, t]        # reserve cost
            for seg in range(M):
                obj += costs[g, seg] * p_seg[g, seg, t]   # fuel (eq. 3)
    m.setObjective(obj, GRB.MINIMIZE)

    # ── Constraints ──────────────────────────────────────────────────────────

    for g, gen in enumerate(GENERATORS):
        TU   = gen['TU']
        TD   = gen['TD']
        RU   = gen['RU']
        RD   = gen['RD']
        PD   = gen['PD']
        Pmax = gen['Pmax']
        Pmin = gen['Pmin']
        Tint = gen['Tint']
        Pg0  = gen['Pg_int']
        R10  = gen['R10']
        Pu   = gen['Pu_seg']   # segment MW widths

        # ── eq. (4): startup + shutdown ≤ 1 in each period
        for t in range(T):
            m.addConstr(v[g, t] + y[g, t] <= 1, name=f'c4_g{g}_t{t}')

        # ── eq. (5): z state transition
        for t in range(1, T):
            m.addConstr(
                z[g, t] - z[g, t-1] == v[g, t] - y[g, t],
                name=f'c5_g{g}_t{t}'
            )
        # t=0: transition from initial state
        z_init = 1 if Tint > 0 else 0
        m.addConstr(z[g, 0] - z_init == v[g, 0] - y[g, 0],
                    name=f'c5_g{g}_t0')

        # ── Initial uptime/downtime constraints (eq. 6–7)
        if Tint > 0:   # initially ON
            # Must stay ON for remaining uptime obligation
            min_on = max(0, TU - Tint)
            for t in range(min(min_on, T)):
                m.addConstr(z[g, t] == 1, name=f'c6_init_g{g}_t{t}')
        else:           # initially OFF
            min_off = max(0, TD - abs(Tint))
            for t in range(min(min_off, T)):
                m.addConstr(z[g, t] == 0, name=f'c7_init_g{g}_t{t}')

        # ── Uptime constraints (eq. 8–9): once ON, stay ON for TU hours
        for t in range(T - 1):
            # If unit starts up at t+1, must remain ON for TU hours
            t_end = min(t + 1 + TU, T)
            for s in range(t + 2, t_end):
                m.addConstr(
                    z[g, s] >= v[g, t + 1],
                    name=f'c8_g{g}_t{t}_s{s}'
                )

        # ── Downtime constraints (eq. 10): once OFF, stay OFF for TD hours
        for t in range(T - 1):
            t_end = min(t + 1 + TD, T)
            for s in range(t + 1, t_end):
                m.addConstr(
                    1 - z[g, s] >= y[g, t + 1],
                    name=f'c10_g{g}_t{t}_s{s}'
                )

        # ── eq. (11): total generation = sum of segments
        for t in range(T):
            m.addConstr(
                p[g, t] == gp.quicksum(p_seg[g, seg, t] for seg in range(M)),
                name=f'c11_g{g}_t{t}'
            )

        # ── eq. (12): ramp-up constraint
        for t in range(1, T):
            m.addConstr(
                p[g, t] - p[g, t-1] <= RU * (1 - v[g, t]) + PD * v[g, t],
                name=f'c12_g{g}_t{t}'
            )
        # t=0: from initial condition
        m.addConstr(
            p[g, 0] - Pg0 <= RU * (1 - v[g, 0]) + PD * v[g, 0],
            name=f'c14_g{g}'
        )

        # ── eq. (13): ramp-down constraint
        for t in range(1, T):
            m.addConstr(
                p[g, t-1] - p[g, t] <= PD * y[g, t] + RD * (1 - y[g, t]),
                name=f'c13_g{g}_t{t}'
            )
        # t=0: from initial condition
        m.addConstr(
            Pg0 - p[g, 0] <= PD * y[g, 0] + RD * (1 - y[g, 0]),
            name=f'c15_g{g}'
        )

        # ── eq. (16)+(17): generation bounds via segment limits
        # p_g,m ≤ Pu_seg[m] * z_g  (enforces p=0 when z=0 — no explicit BigM)
        for t in range(T):
            for seg in range(M):
                m.addConstr(
                    p_seg[g, seg, t] <= Pu[seg] * z[g, t],
                    name=f'c16_g{g}_s{seg}_t{t}'
                )
            # Pmin constraint: p_g ≥ Pmin * z_g
            m.addConstr(
                p[g, t] >= Pmin * z[g, t],
                name=f'c17lb_g{g}_t{t}'
            )

        # ── Reserve constraints (eq. 25–26)
        for t in range(T):
            # 10-min spinning reserve limited by R10
            m.addConstr(r[g, t] <= R10 * z[g, t], name=f'c25_g{g}_t{t}')
            # Reserve ≤ headroom: Pmax - p_g
            m.addConstr(
                r[g, t] <= Pmax * z[g, t] - p[g, t],
                name=f'c26_g{g}_t{t}'
            )

    # ── Power balance (eq. 18 adapted for DC)
    #    sum_g p_g(t) - total_load(t) = 0   (lossless DC, no P_w in 30-bus)
    for t in range(T):
        total_load_t = float(net_load_bus[:, t].sum())
        m.addConstr(
            gp.quicksum(p[g, t] for g in range(G)) == total_load_t,
            name=f'c18_t{t}'
        )

    # ── Line flows via PTDF (eq. 19–21)
    #    p_br_l(t) = sum_n PTDF[l,n] * P_net_n(t)
    #    P_net_n(t) = sum_{g at n} p_g(t) - load_n(t)
    #
    # Design: express f_br[l,t] as a linear function of p[g,t].
    # Then add thermal bound constraints.
    # PTDF approach keeps angles implicit — cleaner MILP structure.

    # Generator–bus incidence: which generators are at which bus
    gen_bus = [g['bus'] - 1 for g in GENERATORS]   # 0-indexed

    for l in range(N_LINE):
        if l == removed_line_idx:
            # Tripped line: flow forced to zero
            for t in range(T):
                m.addConstr(f_br[l, t] == 0.0, name=f'c_trip_l{l}_t{t}')
            continue

        rate = line_rates[l]
        for t in range(T):
            # f_br[l,t] = PTDF[l,:] · P_net(t)
            #           = sum_n PTDF[l,n] * (sum_{g@n} p[g,t] - load_n(t))
            flow_expr = gp.LinExpr()
            for g in range(G):
                n = gen_bus[g]
                if abs(PTDF[l, n]) > 1e-8:
                    flow_expr += PTDF[l, n] * p[g, t]

            # Subtract load contribution (constant)
            load_const = sum(
                PTDF[l, n] * net_load_bus[n, t] for n in range(N)
                if abs(PTDF[l, n]) > 1e-8
            )
            # Wait — net_load_bus already contains net_load so:
            # P_net_n = gen_n - net_load_n
            # But power balance: sum gen = sum load, so
            # f_br[l,t] = PTDF[l,:] · (gen - load)
            # Re-express using gen only and subtract PTDF·load as constant:
            flow_def = (
                gp.quicksum(
                    PTDF[l, gen_bus[g]] * p[g, t]
                    for g in range(G)
                    if abs(PTDF[l, gen_bus[g]]) > 1e-8
                )
                - float(sum(PTDF[l, n] * net_load_bus[n, t] for n in range(N)))
            )
            m.addConstr(f_br[l, t] == flow_def, name=f'cflow_l{l}_t{t}')
            m.addConstr(f_br[l, t] <=  rate,    name=f'c21p_l{l}_t{t}')
            m.addConstr(f_br[l, t] >= -rate,    name=f'c21n_l{l}_t{t}')

    # ── Spinning reserve requirements (eq. 23–24)
    for t in range(T):
        total_load_t = float(net_load_bus[:, t].sum())
        # System reserve requirement (eq. 23): at least GAMMA * total_load
        for g in range(G):
            m.addConstr(s_r[t] >= GENERATORS[g]['Pmax'] * z[g, t])
        # Total reserve must meet system requirement (eq. 24)
        m.addConstr(
            gp.quicksum(r[g, t] for g in range(G)) >= GAMMA * s_r[t],
            name=f'c24_t{t}'
        )

    # ── Solve ────────────────────────────────────────────────────────────────
    m.optimize()

    # ── Extract Results ──────────────────────────────────────────────────────
    status_map = {
        GRB.OPTIMAL:      'optimal',
        GRB.SUBOPTIMAL:   'feasible',
        GRB.INFEASIBLE:   'infeasible',
        GRB.TIME_LIMIT:   'timeout',
    }
    sol_status = status_map.get(m.Status, f'unknown_{m.Status}')

    result = {
        'status'    : sol_status,
        'obj_cost'  : m.ObjVal if m.SolCount > 0 else None,
        'mip_gap'   : m.MIPGap if m.SolCount > 0 else None,
        'solve_time': m.Runtime,
    }

    if m.SolCount > 0:
        result['z']     = np.array([[z[g,t].X for t in range(T)]
                                     for g in range(G)])
        result['v']     = np.array([[v[g,t].X for t in range(T)]
                                     for g in range(G)])
        result['y']     = np.array([[y[g,t].X for t in range(T)]
                                     for g in range(G)])
        result['p']     = np.array([[p[g,t].X for t in range(T)]
                                     for g in range(G)])
        result['p_seg'] = np.array([[[p_seg[g,s,t].X for t in range(T)]
                                      for s in range(M)] for g in range(G)])
        result['r']     = np.array([[r[g,t].X for t in range(T)]
                                     for g in range(G)])
        result['flows'] = np.array([[f_br[l,t].X for t in range(T)]
                                     for l in range(N_LINE)])

    m.dispose()
    env.dispose()
    return result