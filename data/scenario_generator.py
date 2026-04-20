import numpy as np
from typing import Optional
from data.ieee30_system import (
    GENERATORS, BUS_LOAD_MW, N_BUS, T_HORIZON, N_GEN, N_SEG
)


# ─────────────────────────────────────────────
# Canonical 24-hour load shape (normalised to peak = 1.0)
# Two-Gaussian model:
#   morning peak centred at t=9, width=2.5
#   evening peak centred at t=19, width=2.0
# Overnight trough ≈ 0.45 of peak
# ─────────────────────────────────────────────
def _daily_shape() -> np.ndarray:
    t = np.arange(T_HORIZON, dtype=float)
    morning = np.exp(-0.5 * ((t - 9.0) / 2.5) ** 2)
    evening = 0.90 * np.exp(-0.5 * ((t - 19.0) / 2.0) ** 2)
    base    = 0.45
    shape   = base + (1.0 - base) * np.maximum(morning, evening)
    return shape / shape.max()          # normalise to [0.45, 1.0]


# ─────────────────────────────────────────────
# Simple solar PV profile (pu of installed capacity)
# Sunrise ~t=6, sunset ~t=20, peak at t=12
# ─────────────────────────────────────────────
def _solar_profile() -> np.ndarray:
    t = np.arange(T_HORIZON, dtype=float)
    irr = np.exp(-0.5 * ((t - 12.0) / 3.0) ** 2)
    irr[irr < 0.05] = 0.0              # dark hours → zero generation
    return irr


DAILY_SHAPE   = _daily_shape()
SOLAR_PROFILE = _solar_profile()


class ScenarioGenerator:
    """
    Generates random (load, cost) scenarios for MILP-UC training.

    Parameters
    ----------
    n_scenarios      : total number of scenarios to generate
    load_scale_mean  : mean of per-scenario load scale factor
    load_scale_std   : std  of per-scenario load scale factor
    load_noise_std   : per-bus per-hour spatial noise std (fraction)
    cost_noise_std   : fractional noise on fuel cost coefficients
    pv_capacity_mw   : installed PV capacity at bus 5 (MW), 0 = no PV
    seed             : random seed for reproducibility
    """

    def __init__(
        self,
        n_scenarios     : int   = 210,
        load_scale_mean : float = 1.1,
        load_scale_std  : float = 0.15,
        load_noise_std  : float = 0.03,
        cost_noise_std  : float = 0.10,
        pv_capacity_mw  : float = 30.0,   # 30 MW PV at bus 5
        seed            : int   = 42,
    ):
        self.n_scenarios      = n_scenarios
        self.load_scale_mean  = load_scale_mean
        self.load_scale_std   = load_scale_std
        self.load_noise_std   = load_noise_std
        self.cost_noise_std   = cost_noise_std
        self.pv_capacity_mw   = pv_capacity_mw
        self.rng              = np.random.default_rng(seed)

        # Pre-compute base bus load vector (MW)
        self.base_load = np.array(
            [BUS_LOAD_MW.get(n + 1, 0.0) for n in range(N_BUS)],
            dtype=float,
        )   # shape (N_BUS,)

        # Nominal fuel cost coefficients from system data
        self.nominal_costs = np.array(
            [GENERATORS[g]['c_p'] for g in range(N_GEN)],
            dtype=float,
        )   # shape (N_GEN, N_SEG)

    def _pv_generation(self) -> np.ndarray:
        """
        Solar PV generation at bus 5 for a given day (MW).
        We add ±10 % irradiance variability per scenario.
        Returns shape (T,).
        """
        irr_scale = self.rng.uniform(0.8, 1.1)
        return self.pv_capacity_mw * SOLAR_PROFILE * irr_scale

    def generate_one(self) -> dict:
        """
        Generate one training scenario Δs.

        Returns
        -------
        dict with:
          'net_load_bus'  : (N_BUS, T) net load after PV subtraction (MW)
          'gross_load_bus': (N_BUS, T) gross load before PV (MW)
          'pv_gen'        : (T,) PV generation (MW, added at bus 5)
          'costs'         : (N_GEN, N_SEG) fuel cost coefficients ($/MWh)
          'load_scale'    : per-scenario scale factor (scalar)
        """
        # ── Load scale factor ───────────────────────────────────────────────
        scale = float(np.clip(
            self.rng.normal(self.load_scale_mean, self.load_scale_std),
            0.60, 1.20
        ))

        # ── Hourly bus loads (N_BUS, T) ─────────────────────────────────────
        # Outer product: base_load[n] × shape[t] × scale × noise[n,t]
        noise = self.rng.normal(
            1.0, self.load_noise_std, size=(N_BUS, T_HORIZON)
        )
        noise = np.clip(noise, 0.5, 1.5)

        gross = np.outer(self.base_load, DAILY_SHAPE) * scale * noise
        gross = np.maximum(gross, 0.0)   # no negative load

        # ── PV generation — subtract from bus 5 load (eq. 28) ───────────────
        pv = self._pv_generation()                           # shape (T,)
        net_load = gross.copy()
        bus5_idx = 5 - 1                                     # 0-indexed
        net_load[bus5_idx, :] = np.maximum(
            gross[bus5_idx, :] - pv, 0.0
        )

        # ── Generation cost coefficients (eq. 29) ───────────────────────────
        cost_noise = self.rng.normal(
            1.0, self.cost_noise_std, size=(N_GEN, N_SEG)
        )
        cost_noise = np.clip(cost_noise, 0.5, 1.8)
        costs = self.nominal_costs * cost_noise
        costs = np.maximum(costs, 0.10)                      # no negative costs

        # Ensure cost monotonicity: segment 2 must cost more than segment 1
        # (piecewise-linear convex cost curve requirement)
        for g in range(N_GEN):
            if costs[g, 1] < costs[g, 0]:
                costs[g, 1] = costs[g, 0] * 1.1

        return {
            'net_load_bus'  : net_load,
            'gross_load_bus': gross,
            'pv_gen'        : pv,
            'costs'         : costs,
            'load_scale'    : scale,
        }

    def generate_all(self) -> list[dict]:
        """Generate all n_scenarios scenarios."""
        return [self.generate_one() for _ in range(self.n_scenarios)]

    def get_train_val_test_split(
        self,
        scenarios: list[dict],
        train_frac: float = 0.70,
        val_frac:   float = 0.15,
    ) -> tuple[list, list, list]:
        """
        Split scenarios into train / validation / test sets.
        70 / 15 / 15 split matching Venkatesh et al.
        Shuffle before splitting to avoid temporal ordering bias.
        """
        idx = self.rng.permutation(len(scenarios))
        n = len(idx)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)

        train_idx = idx[:n_train]
        val_idx   = idx[n_train:n_train + n_val]
        test_idx  = idx[n_train + n_val:]

        train = [scenarios[i] for i in train_idx]
        val   = [scenarios[i] for i in val_idx]
        test  = [scenarios[i] for i in test_idx]

        return train, val, test