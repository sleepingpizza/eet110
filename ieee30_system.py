import numpy as np

# ─────────────────────────────────────────────
# System constants
# ─────────────────────────────────────────────
S_BASE     = 100.0   # MVA system base
N_BUS      = 30
T_HORIZON  = 24      # scheduling horizon (hours)
GAMMA      = 0.25   # spinning-reserve ratio (NPCC standard, eq. 24)
REF_BUS    = 1       # slack bus index (1-indexed, bus 1)

# ─────────────────────────────────────────────
# Bus base loads (MW, peak-day snapshot)
# IEEE 30-bus standard load distribution
# ─────────────────────────────────────────────
# Key: 1-indexed bus number
BUS_LOAD_MW = {
     1:  0.0,  2: 21.7,  3:  2.4,  4:  7.6,  5: 94.2,
     6:  0.0,  7: 22.8,  8: 30.0,  9:  0.0, 10:  5.8,
    11:  0.0, 12: 11.2, 13:  0.0, 14:  6.2, 15:  8.2,
    16:  3.5, 17:  9.0, 18:  3.2, 19:  9.5, 20:  2.2,
    21: 17.5, 22:  0.0, 23:  3.2, 24:  8.7, 25:  0.0,
    26:  3.5, 27:  0.0, 28:  0.0, 29:  2.4, 30: 10.6,
}
TOTAL_PEAK_LOAD = sum(BUS_LOAD_MW.values())   # ≈ 283.5 MW

# ─────────────────────────────────────────────
# Generators
# ─────────────────────────────────────────────
# Design choice — piecewise-linear (PWL) fuel cost with M=2 segments:
#   f_g(p) = c_z*z + c_p1*p1 + c_p2*p2
# Two segments approximate the quadratic curve well enough for
# DC-UC and keep the problem as a pure MILP (no quadratic terms).
# Segment breakpoints are at Pmax/2 so each segment is equal width.
#
# Ramp limits:  RU/RD chosen so a unit can go from 0→Pmax in 2 hours
# (typical for gas turbines in this size range).
# PD (startup/shutdown ramp) is set to 50 % of Pmax — conservative.
#
# Min uptime/downtime:
#   Large units (gens 0,1,5): TU=TD=3 h  (steam turbines need warmup)
#   Small/peaker units (2,3,4): TU=TD=1 h  (fast-start gas turbines)
#
# Tint > 0  → unit has been ON for Tint hours at t=0
# Tint < 0  → unit has been OFF for |Tint| hours at t=0

GENERATORS = [
    {  # Gen 0 — large base-load unit at bus 1 (slack)
        'bus'       : 1,
        'Pmin'      : 10.0,   # MW  (non-zero Pmin forces commitment diversity)
        'Pmax'      : 80.0,   # MW
        'TU'        : 3,      # min uptime  (hours)
        'TD'        : 3,      # min downtime (hours)
        'RU'        : 40.0,   # ramp up   MW/h
        'RD'        : 40.0,   # ramp down MW/h
        'PD'        : 30.0,   # startup/shutdown ramp limit MW
        'c_startup' : 200.0,  # $/start
        'c_shutdown':  80.0,  # $/stop
        'c_noload'  :  90.0,  # $/h  (no-load fixed cost when ON)
        'c_p'       : [2.0,  3.2],   # $/MWh per segment
        'Pu_seg'    : [40.0, 40.0],  # MW segment widths  (sum = Pmax)
        'Tint'      :  4,     # initially ON for 4 h
        'Pg_int'    : 40.0,   # initial power (MW)
        'R10'       : 25.0,   # 10-min spinning reserve capability (MW)
    },
    {  # Gen 1 — large mid-merit unit at bus 2
        'bus'       : 2,
        'Pmin'      : 10.0,
        'Pmax'      : 80.0,
        'TU'        : 3,
        'TD'        : 3,
        'RU'        : 40.0,
        'RD'        : 40.0,
        'PD'        : 25.0,
        'c_startup' : 180.0,
        'c_shutdown':  70.0,
        'c_noload'  :  75.0,
        'c_p'       : [2.4,  3.6],
        'Pu_seg'    : [40.0, 40.0],
        'Tint'      :  3,
        'Pg_int'    : 40.0,
        'R10'       :  25.0,
    },
    {  # Gen 2 — medium peaker at bus 13
        'bus'       : 13,
        'Pmin'      :  5.0,
        'Pmax'      : 40.0,
        'TU'        : 1,
        'TD'        : 1,
        'RU'        : 40.0,   # fast-start: full ramp in 1 h
        'RD'        : 40.0,
        'PD'        : 15.0,
        'c_startup' :  90.0,
        'c_shutdown':  40.0,
        'c_noload'  :  55.0,
        'c_p'       : [3.0,  4.8],
        'Pu_seg'    : [20.0, 20.0],
        'Tint'      :  2,
        'Pg_int'    : 20.0,
        'R10'       :  40.0,
    },
    {  # Gen 3 — medium unit at bus 22
        'bus'       : 22,
        'Pmin'      :  5.0,
        'Pmax'      : 50.0,
        'TU'        : 2,
        'TD'        : 2,
        'RU'        : 30.0,
        'RD'        : 30.0,
        'PD'        : 20.0,
        'c_startup' : 110.0,
        'c_shutdown':  50.0,
        'c_noload'  :  65.0,
        'c_p'       : [2.8,  4.2],
        'Pu_seg'    : [25.0, 25.0],
        'Tint'      : -2,     # initially OFF for 2 h
        'Pg_int'    :  0.0,
        'R10'       :  50.0,
    },
    {  # Gen 4 — small peaker at bus 23
        'bus'       : 23,
        'Pmin'      :  3.0,
        'Pmax'      : 30.0,
        'TU'        : 1,
        'TD'        : 1,
        'RU'        : 30.0,
        'RD'        : 30.0,
        'PD'        : 12.0,
        'c_startup' :  70.0,
        'c_shutdown':  30.0,
        'c_noload'  :  45.0,
        'c_p'       : [3.5,  5.5],
        'Pu_seg'    : [15.0, 15.0],
        'Tint'      : -1,
        'Pg_int'    :  0.0,
        'R10'       :  30.0,
    },
    {  # Gen 5 — medium base unit at bus 27
        'bus'       : 27,
        'Pmin'      :  8.0,
        'Pmax'      : 150.0,
        'TU'        : 3,
        'TD'        : 3,
        'RU'        : 28.0,
        'RD'        : 28.0,
        'PD'        : 20.0,
        'c_startup' : 130.0,
        'c_shutdown':  55.0,
        'c_noload'  :  70.0,
        'c_p'       : [2.6,  3.9],
        'Pu_seg'    : [75.0, 75.0],
        'Tint'      :  2,
        'Pg_int'    : 30.0,
        'R10'       :  50.0,
    },
]

N_GEN = len(GENERATORS)
N_SEG = 2          # piecewise segments per generator (M)

# ─────────────────────────────────────────────
# Transmission Lines
# ─────────────────────────────────────────────
# Format: (from_bus, to_bus, r_pu, x_pu, b_pu, rate_MW)
# Standard IEEE 30-bus branch data (100 MVA base).
# Design choice — DC power flow uses only x (reactance).
# We keep r and b for completeness (AC extension later).
# Thermal limits (rate_MW) set from standard IEEE data scaled
# to avoid trivial congestion while still creating meaningful N-1 stress.

LINES = [
    #  fb   tb     r        x        b     rate(MW)
    (  1,   2, 0.0192, 0.0575, 0.0264,  195),   # 0
    (  1,   3, 0.0452, 0.1652, 0.0204,   65*1.5),   # 1
    (  2,   4, 0.0570, 0.1737, 0.0184,   65*1.5),   # 2
    (  3,   4, 0.0132, 0.0379, 0.0042,  130*1.5),   # 3
    (  2,   5, 0.0472, 0.1983, 0.0209,   65*1.5),   # 4
    (  2,   6, 0.0581, 0.1763, 0.0187,   65*1.5),   # 5
    (  4,   6, 0.0119, 0.0414, 0.0045,   90*1.5),   # 6
    (  5,   7, 0.0460, 0.1160, 0.0102,   70*1.5),   # 7
    (  6,   7, 0.0267, 0.0820, 0.0085,   90*1.5),   # 8
    (  6,   8, 0.0120, 0.0420, 0.0045,   70*1.5),   # 9
    (  6,   9, 0.0000, 0.2080, 0.0000,   65*1.5),   # 10 transformer
    (  6,  10, 0.0000, 0.5560, 0.0000,   32*1.5),   # 11 transformer
    (  9,  11, 0.0000, 0.2080, 0.0000,   65*1.5),   # 12 transformer
    (  9,  10, 0.0000, 0.1100, 0.0000,   65*1.5),   # 13
    (  4,  12, 0.0000, 0.2560, 0.0000,   65*1.5),   # 14 transformer
    ( 12,  13, 0.0000, 0.1400, 0.0000,   65*1.5),   # 15 transformer
    ( 12,  14, 0.1231, 0.2559, 0.0000,   32*1.5),   # 16
    ( 12,  15, 0.0662, 0.1304, 0.0000,   32*1.5),   # 17
    ( 12,  16, 0.0945, 0.1987, 0.0000,   32*1.5),   # 18
    ( 14,  15, 0.2210, 0.1997, 0.0000,   16*1.5),   # 19
    ( 16,  17, 0.0824, 0.1932, 0.0000,   16*1.5),   # 20
    ( 15,  18, 0.1070, 0.2185, 0.0000,   16*1.5),   # 21
    ( 18,  19, 0.0639, 0.1292, 0.0000,   16*1.5),   # 22
    ( 19,  20, 0.0340, 0.0680, 0.0000,   32*1.5),   # 23
    ( 10,  20, 0.0936, 0.2090, 0.0000,   32*1.5),   # 24
    ( 10,  17, 0.0324, 0.0845, 0.0000,   32*1.5),   # 25
    ( 10,  21, 0.0348, 0.0749, 0.0000,   32*1.5),   # 26
    ( 10,  22, 0.0727, 0.1499, 0.0000,   32*1.5),   # 27
    ( 21,  22, 0.0116, 0.0236, 0.0000,   32*1.5),   # 28
    ( 15,  23, 0.1000, 0.2020, 0.0000,   16*1.5),   # 29
    ( 22,  24, 0.1150, 0.1790, 0.0000,   16*1.5),   # 30
    ( 23,  24, 0.1320, 0.2700, 0.0000,   16*1.5),   # 31
    ( 24,  25, 0.1885, 0.3292, 0.0000,   16*1.5),   # 32
    ( 25,  26, 0.2544, 0.3800, 0.0000,   16*1.5),   # 33
    ( 25,  27, 0.1093, 0.2087, 0.0000,   16*1.5),   # 34
    ( 28,  27, 0.0000, 0.3960, 0.0000,   65*1.5),   # 35 transformer
    ( 27,  29, 0.2198, 0.4153, 0.0000,   16*1.5),   # 36
    ( 27,  30, 0.3202, 0.6027, 0.0000,   16*1.5),   # 37
    ( 29,  30, 0.2399, 0.4533, 0.0000,   16*1.5),   # 38
    (  8,  28, 0.0636, 0.2000, 0.0214,   32*1.5),   # 39
    (  6,  28, 0.0169, 0.0599, 0.0065,   32*1.5),   # 40
]

N_LINE = len(LINES)

# ─────────────────────────────────────────────
# Quick sanity checks
# ─────────────────────────────────────────────
def _verify():
    total_cap = sum(g['Pmax'] for g in GENERATORS)
    assert total_cap > TOTAL_PEAK_LOAD, "Insufficient generation capacity"
    reserve = (total_cap - TOTAL_PEAK_LOAD) / TOTAL_PEAK_LOAD
    print(f"[ieee30] Peak load: {TOTAL_PEAK_LOAD:.1f} MW | "
          f"Total cap: {total_cap:.1f} MW | "
          f"Reserve margin: {reserve:.1%}")
    print(f"[ieee30] Buses: {N_BUS} | Lines: {N_LINE} | Gens: {N_GEN}")
    print(f"[ieee30] N-1 contingency set size: {N_LINE} (one per line outage)")

if __name__ == '__main__':
    _verify()