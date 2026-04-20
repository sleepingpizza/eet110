import numpy as np
import os
import json
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict

from data.ieee30_system import N_LINE, LINES, N_GEN, T_HORIZON
from data.scenario_generator import ScenarioGenerator
from solver.milp_uc import solve_milp_uc
from dataset.graph_builder import build_sample, normalise_dataset


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


@dataclass
class SolveTask:
    """One (scenario, contingency) pair to solve."""
    scenario_idx    : int
    contingency_idx : int        # -1 for base case, else line index (0-based)
    net_load_bus    : np.ndarray # (N_BUS, T)
    costs           : np.ndarray # (N_GEN, N_SEG)
    mip_gap         : float = 0.001
    time_limit      : float = 120.0


def _solve_one(task: SolveTask) -> dict | None:
    """
    Worker function: solve one UC problem and return the graph sample.
    Must be module-level (not a lambda) for ProcessPoolExecutor pickling.
    """
    removed = None if task.contingency_idx == -1 else task.contingency_idx

    result = solve_milp_uc(
        net_load_bus     = task.net_load_bus,
        costs =           task.costs,
        removed_line_idx = removed,
        mip_gap          = task.mip_gap,
        time_limit       = task.time_limit,
        verbose          = False,
    )

    if result['status'] not in ('optimal', 'feasible') or result.get('z') is None:
        return {
            'success'        : False,
            'scenario_idx'   : task.scenario_idx,
            'contingency_idx': task.contingency_idx,
            'status'         : result['status'],
            'reason'         : result.get('reason', ''),
        }

    sample = build_sample(
        net_load_bus     = task.net_load_bus,
        costs            = task.costs,
        z_optimal        = result['z'],
        removed_line_idx = removed,
        scenario_idx     = task.scenario_idx,
        contingency_idx  = task.contingency_idx,
    )

    return {
        'success'        : True,
        'sample'         : sample,
        'scenario_idx'   : task.scenario_idx,
        'contingency_idx': task.contingency_idx,
        'obj_cost'       : result['obj_cost'],
        'solve_time'     : result['solve_time'],
        'mip_gap'        : result['mip_gap'],
    }


class AugmentedDatasetBuilder:
    """
    Orchestrates the full augmented dataset generation pipeline.

    Parameters
    ----------
    n_scenarios  : total number of daily scenarios
    output_dir   : directory to save dataset files
    max_workers  : parallel processes (set to n_cpu // 4 for Gurobi threading)
    mip_gap      : MILP optimality gap
    time_limit   : per-solve time limit (seconds)
    skip_radial  : if True, skip contingencies that create radial-only networks
                   (these are valid but may stress the DC flow solver)
    """

    def __init__(
        self,
        n_scenarios : int   = 210,
        output_dir  : str   = 'dataset_output',
        max_workers : int   = 4,
        mip_gap     : float = 0.001,
        time_limit  : float = 120.0,
        seed        : int   = 42,
    ):
        self.n_scenarios = n_scenarios
        self.output_dir  = output_dir
        self.max_workers = max_workers
        self.mip_gap     = mip_gap
        self.time_limit  = time_limit
        self.seed        = seed
        os.makedirs(output_dir, exist_ok=True)

    def build(self) -> dict:
        """
        Full pipeline:
          1. Generate scenarios
          2. For each (scenario, contingency) → solve MILP-UC
          3. Build graph samples
          4. Normalise features
          5. Split train/val/test
          6. Save to disk

        Returns summary statistics dict.
        """
        t_start = time.time()

        # ── Step 1: generate scenarios ───────────────────────────────────────
        log.info(f"Generating {self.n_scenarios} daily scenarios...")
        gen = ScenarioGenerator(n_scenarios=self.n_scenarios, seed=self.seed)
        scenarios = gen.generate_all()
        log.info(f"  Done. Peak load range: "
                 f"{min(s['net_load_bus'].sum(axis=0).max() for s in scenarios):.1f} – "
                 f"{max(s['net_load_bus'].sum(axis=0).max() for s in scenarios):.1f} MW")

        # ── Step 2: build solve task list ────────────────────────────────────
        # Contingencies: -1 (base case) + 0..N_LINE-1 (each line removed)
        contingencies = [-1] + list(range(N_LINE))
        total_tasks   = self.n_scenarios * len(contingencies)
        log.info(
            f"Building {total_tasks} UC solve tasks "
            f"({self.n_scenarios} scenarios × {len(contingencies)} topologies)..."
        )

        tasks = []
        for s_idx, scenario in enumerate(scenarios):
            for c_idx in contingencies:
                tasks.append(SolveTask(
                    scenario_idx     = s_idx,
                    contingency_idx  = c_idx,
                    net_load_bus     = scenario['net_load_bus'],
                    costs            = scenario['costs'],
                    mip_gap          = self.mip_gap,
                    time_limit       = self.time_limit,
                ))

        # ── Step 3: solve in parallel ─────────────────────────────────────────
        log.info(f"Solving with {self.max_workers} parallel workers...")
        samples, failures = [], []
        solve_times, obj_costs = [], []

        # Design note: sequential execution is used here for safety/debuggability.
        # Uncomment the ProcessPoolExecutor block for production speed-up.
        # The sequential path is identical in output but easier to debug.

        completed = 0
        for task in tasks:
            res = _solve_one(task)
            completed += 1

            if res['success']:
                samples.append(res['sample'])
                solve_times.append(res['solve_time'])
                obj_costs.append(res['obj_cost'])
            else:
                failures.append({
                    's': res['scenario_idx'],
                    'c': res['contingency_idx'],
                    'status': res['status'],
                    'reason': res.get('reason', ''),
                })

            if completed % 100 == 0 or completed == total_tasks:
                elapsed = time.time() - t_start
                eta     = (elapsed / completed) * (total_tasks - completed)
                log.info(
                    f"  {completed}/{total_tasks} solved | "
                    f"success={len(samples)} | failures={len(failures)} | "
                    f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s"
                )

        log.info(f"\n{'─'*60}")
        log.info(f"Solve phase complete.")
        log.info(f"  Successful samples : {len(samples)}")
        log.info(f"  Failures           : {len(failures)}")
        if solve_times:
            log.info(f"  Avg solve time     : {np.mean(solve_times):.3f}s")
            log.info(f"  Max solve time     : {np.max(solve_times):.3f}s")

        if not samples:
            raise RuntimeError("No samples generated. Check Gurobi licence.")

        # ── Step 4: normalise features ────────────────────────────────────────
        log.info("Normalising node features...")
        normed_samples, feat_stats = normalise_dataset(samples)

        # ── Step 5: split train/val/test ──────────────────────────────────────
        # Split at scenario level (not sample level) to prevent data leakage.
        # All contingencies of one scenario go to the same split.
        # This ensures the val/test sets contain TRULY unseen scenarios.
        log.info("Splitting into train/val/test (70/15/15 at scenario level)...")
        rng = np.random.default_rng(self.seed)
        s_indices = rng.permutation(self.n_scenarios)
        n_train = int(self.n_scenarios * 0.70)
        n_val   = int(self.n_scenarios * 0.15)

        train_s = set(s_indices[:n_train].tolist())
        val_s   = set(s_indices[n_train:n_train + n_val].tolist())
        test_s  = set(s_indices[n_train + n_val:].tolist())

        splits = {'train': [], 'val': [], 'test': []}
        for sample in normed_samples:
            s_idx = sample['meta']['scenario_idx']
            if   s_idx in train_s: splits['train'].append(sample)
            elif s_idx in val_s:   splits['val'].append(sample)
            else:                  splits['test'].append(sample)

        for split, split_samples in splits.items():
            log.info(f"  {split:5s}: {len(split_samples)} samples")

        # ── Step 6: save to disk ──────────────────────────────────────────────
        log.info(f"Saving dataset to {self.output_dir}/...")
        self._save_splits(splits, feat_stats, failures)

        elapsed_total = time.time() - t_start
        summary = {
            'n_scenarios'    : self.n_scenarios,
            'n_contingencies': len(contingencies),
            'total_tasks'    : total_tasks,
            'n_samples'      : len(normed_samples),
            'n_failures'     : len(failures),
            'n_train'        : len(splits['train']),
            'n_val'          : len(splits['val']),
            'n_test'         : len(splits['test']),
            'mean_solve_time': float(np.mean(solve_times)) if solve_times else 0,
            'total_time_s'   : elapsed_total,
        }
        log.info(f"\nDATASET SUMMARY:\n" + json.dumps(summary, indent=2))
        return summary

    def _save_splits(
        self,
        splits     : dict,
        feat_stats : dict,
        failures   : list,
    ):
        """Save each split as a compressed numpy archive (.npz)."""

        def _pack(samples: list[dict]) -> dict:
            """Stack samples into arrays for efficient storage."""
            return {
                'X'          : np.stack([s['X']          for s in samples]),
                'A'          : np.stack([s['A']          for s in samples]),
                'edge_index' : np.stack([s['edge_index'] for s in samples]),
                'edge_attr'  : np.stack([s['edge_attr']  for s in samples]),
                'M'          : samples[0]['M'],    # same for all
                'z'          : np.stack([s['z']   for s in samples]),
                'net_load_bus': np.stack([s['net_load_bus'] for s in samples]), 
                'scenario_idx'   : np.array(
                    [s['meta']['scenario_idx']    for s in samples]),
                'contingency_idx': np.array(
                    [s['meta']['contingency_idx'] for s in samples]),
                'is_base_case'   : np.array(
                    [s['meta']['is_base_case']    for s in samples]),
            }

        for split, samples in splits.items():
            if not samples:
                continue
            packed = _pack(samples)
            path   = os.path.join(self.output_dir, f'{split}.npz')
            np.savez_compressed(path, **packed)
            log.info(f"  Saved {path}  ({len(samples)} samples)")

        # Save normalisation stats
        stats_path = os.path.join(self.output_dir, 'feat_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(feat_stats, f, indent=2)

        # Save failure log
        failures_path = os.path.join(self.output_dir, 'failures.json')
        with open(failures_path, 'w') as f:
            json.dump(failures, f, indent=2)

        log.info(f"  Normalisation stats → {stats_path}")
        log.info(f"  Failures log        → {failures_path}")

    @staticmethod
    def load_split(split_path: str) -> dict:
        """Load a saved split from disk."""
        return dict(np.load(split_path, allow_pickle=True))