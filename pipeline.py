import argparse
import json
import logging
import sys
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


def validate_system():
    log.info("=" * 60)
    log.info("STAGE 1: SYSTEM VALIDATION")
    log.info("=" * 60)

    from data.ieee30_system import (
        GENERATORS, LINES, N_BUS, N_LINE, N_GEN, T_HORIZON,
        TOTAL_PEAK_LOAD, GAMMA
    )
    log.info(f"System: {N_BUS} buses, {N_LINE} lines, {N_GEN} generators")
    log.info(f"Horizon: {T_HORIZON}h | Peak load: {TOTAL_PEAK_LOAD:.1f} MW")
    total_cap = sum(g['Pmax'] for g in GENERATORS)
    log.info(f"Total capacity: {total_cap:.1f} MW | "
             f"Reserve margin: {(total_cap/TOTAL_PEAK_LOAD - 1):.1%}")

    # Check that all generator buses exist in the 30-bus system
    gen_buses = set(g['bus'] for g in GENERATORS)
    all_buses  = set(range(1, N_BUS + 1))
    assert gen_buses.issubset(all_buses), "Generator on non-existent bus!"
    log.info(f"Generator buses: {sorted(gen_buses)} ✓")

    # ── Check DC power flow ───────────────────────────────────────
    from utils.dc_powerflow import build_susceptance_matrix, compute_angles_and_flows
    B = build_susceptance_matrix(N_BUS, LINES, removed_line_idx=None)
    assert B.shape == (N_BUS, N_BUS), "B matrix wrong shape"
    assert np.allclose(B, B.T, atol=1e-10), "B matrix not symmetric"
    # Diagonal must be positive (sum of susceptances)
    assert (np.diag(B) >= 0).all(), "Negative diagonal in B"
    log.info(f"Susceptance matrix B ({N_BUS}×{N_BUS}): symmetric, positive diagonal ✓")

    # Test power flow with uniform generation
    test_inj = np.zeros(N_BUS)
    from data.ieee30_system import BUS_LOAD_MW, S_BASE
    for n in range(N_BUS):
        test_inj[n] = -BUS_LOAD_MW.get(n+1, 0.0)
    # Put all generation at bus 1 (slack)
    test_inj[0] = sum(BUS_LOAD_MW.values())
    try:
        theta, flows = compute_angles_and_flows(test_inj, LINES, N_BUS)
        log.info(f"DC power flow test: max |flow| = {np.abs(flows).max():.1f} MW ✓")
    except ValueError as e:
        log.error(f"DC power flow failed: {e}")
        return False

    # ── Check Gurobi ─────────────────────────────────────────────
    try:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        m = gp.Model(env=env)
        x = m.addVar()
        m.setObjective(x, gp.GRB.MINIMIZE)
        m.addConstr(x >= 1.0)
        m.optimize()
        assert abs(x.X - 1.0) < 1e-6
        m.dispose(); env.dispose()
        log.info("Gurobi: licence valid, test solve passed ✓")
    except Exception as e:
        log.error(f"Gurobi check failed: {e}")
        log.error("Ensure gurobipy is installed and a valid licence is active.")
        return False

    # ── Check scenario generator ──────────────────────────────────
    from data.scenario_generator import ScenarioGenerator
    gen = ScenarioGenerator(n_scenarios=5, seed=0)
    scenarios = gen.generate_all()
    assert len(scenarios) == 5
    for s in scenarios:
        assert s['net_load_bus'].shape == (N_BUS, T_HORIZON)
        assert (s['net_load_bus'] >= 0).all(), "Negative net load"
        # Net load should not exceed total capacity
        max_load = s['net_load_bus'].sum(axis=0).max()
        assert max_load <= total_cap * 1.1, f"Load {max_load:.1f} MW exceeds capacity"
    log.info(f"Scenario generator: 5 test scenarios OK ✓")
    log.info("  Load range: "
             f"{min(s['net_load_bus'].sum(axis=0).max() for s in scenarios):.1f}–"
             f"{max(s['net_load_bus'].sum(axis=0).max() for s in scenarios):.1f} MW")

    log.info("\nAll validation checks passed ✓\n")
    return True


def dry_run():
    log.info("=" * 60)
    log.info("DRY RUN: 3 scenarios × 6 topologies = 18 UC solves")
    log.info("=" * 60)

    from data.ieee30_system import N_BUS, N_GEN, T_HORIZON, LINES
    from data.scenario_generator import ScenarioGenerator
    from solver.milp_uc import solve_milp_uc
    from dataset.graph_builder import build_sample, build_adjacency
    import time

    gen = ScenarioGenerator(n_scenarios=3, seed=99)
    scenarios = gen.generate_all()

    contingencies = [-1, 0, 5, 10, 20, 35]   # base + 5 specific lines
    results = []

    for s_idx, scenario in enumerate(scenarios):
        for c_idx in contingencies:
            removed = None if c_idx == -1 else c_idx
            desc    = "base" if removed is None else f"line {removed} out"
            t0 = time.time()

            res = solve_milp_uc(
                net_load_bus     = scenario['net_load_bus'],
                costs = scenario['costs'],
                removed_line_idx = removed,
                mip_gap          = 0.01,     # looser gap for speed in dry run
                time_limit       = 30.0,
                verbose          = False,
            )
            dt = time.time() - t0

            status = res.get('status', 'error')
            cost   = res.get('obj_cost')
            gap    = res.get('mip_gap')

            log.info(
                f"  s={s_idx} c={c_idx:>3} ({desc:15s}) | "
                f"status={status:10s} | "
                f"cost=${cost:,.0f}" if cost else
                f"  s={s_idx} c={c_idx:>3} ({desc:15s}) | status={status}"
                + f"  [{dt:.2f}s]"
            )

            if res.get('z') is not None:
                # Report commitment statistics
                z = res['z']   # (N_GEN, T)
                on_fraction = z.mean()
                log.info(f"    Commitment rate: {on_fraction:.2%} "
                         f"| Generators ON/OFF pattern:"
                         f" {[int(z[g].sum()) for g in range(N_GEN)]} h/day")

                # Build the graph sample
                sample = build_sample(
                    net_load_bus     = scenario['net_load_bus'],
                    costs            = scenario['costs'],
                    z_optimal        = z,
                    removed_line_idx = removed,
                    scenario_idx     = s_idx,
                    contingency_idx  = c_idx,
                )
                results.append(sample)

                # Verify sample shape integrity
                assert sample['X'].shape         == (N_BUS, T_HORIZON, 4)
                assert sample['A'].shape         == (N_BUS, N_BUS)
                assert sample['z'].shape         == (N_GEN, T_HORIZON)
                assert sample['M'].shape         == (N_BUS, N_GEN)
                assert sample['edge_index'].shape[0] == 2

    log.info(f"\nDry run complete: {len(results)}/{len(scenarios)*len(contingencies)} "
             f"samples built successfully.")
    log.info(f"Sample shapes:")
    if results:
        s = results[0]
        for k, v in s.items():
            if isinstance(v, np.ndarray):
                log.info(f"  {k:12s}: {v.shape}  dtype={v.dtype}")
    return results


def full_run(n_scenarios: int = 210, output_dir: str = 'dataset_output',
             max_workers: int = 4):
    """Run the complete dataset generation pipeline."""
    log.info("=" * 60)
    log.info(f"FULL RUN: {n_scenarios} scenarios × {42} topologies")
    log.info(f"Output: {output_dir}/")
    log.info("=" * 60)

    from dataset.augmented_dataset import AugmentedDatasetBuilder
    builder = AugmentedDatasetBuilder(
        n_scenarios = n_scenarios,
        output_dir  = output_dir,
        max_workers = max_workers,
        mip_gap     = 0.001,
        time_limit  = 120.0,
        seed        = 42,
    )
    summary = builder.build()
    return summary


def report_dataset_stats(output_dir: str = 'dataset_output'):
    """Print statistics about a generated dataset."""
    import os
    from data.ieee30_system import N_GEN, N_LINE, GENERATORS, LINES

    for split in ('train', 'val', 'test'):
        path = os.path.join(output_dir, f'{split}.npz')
        if not os.path.exists(path):
            continue
        data = dict(np.load(path, allow_pickle=True))
        z    = data['z']    # (S, N_GEN, T)
        S    = z.shape[0]

        log.info(f"\n── {split.upper()} split: {S} samples ──")
        log.info("Generator commitment rates (h/day averaged over scenarios):")
        for g in range(N_GEN):
            rate   = z[:, g, :].mean()
            avg_on = z[:, g, :].sum(axis=1).mean()
            log.info(f"  Gen {g} (bus {GENERATORS[g]['bus']}): "
                     f"{rate:.1%} ON rate, {avg_on:.1f} h/day avg")

        is_base   = data['is_base_case']
        n_base    = is_base.sum()
        n_conting = S - n_base
        log.info(f"Base-case samples: {n_base} | Contingency samples: {n_conting}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IEEE 30-bus UC Pipeline')
    parser.add_argument(
        '--mode', choices=['validate', 'dry_run', 'full', 'stats'],
        default='dry_run',
        help='Pipeline mode'
    )
    parser.add_argument('--n_scenarios', type=int, default=210)
    parser.add_argument('--output_dir',  type=str, default='dataset_output')
    parser.add_argument('--max_workers', type=int, default=4)
    args = parser.parse_args()

    if args.mode == 'validate':
        ok = validate_system()
        sys.exit(0 if ok else 1)

    elif args.mode == 'dry_run':
        ok = validate_system()
        if ok:
            dry_run()

    elif args.mode == 'full':
        ok = validate_system()
        if ok:
            summary = full_run(args.n_scenarios, args.output_dir, args.max_workers)
            print(json.dumps(summary, indent=2))

    elif args.mode == 'stats':
        report_dataset_stats(args.output_dir)