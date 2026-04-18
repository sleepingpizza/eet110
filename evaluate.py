import argparse
import json
import logging
import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from model.gnn_uc import GNN_UC
from dataset.graph_dataset import UCGraphDataset

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


# ─────────────────────────────────────────────────────────────────────────────
# NFR helper
# ─────────────────────────────────────────────────────────────────────────────

def check_n1_feasibility(z_pred: np.ndarray, data_dict: dict) -> bool:
    try:
        import gurobipy as gp
        from gurobipy import GRB
        from data.ieee30_system import GENERATORS, LINES, N_BUS, T_HORIZON
        from utils.dc_powerflow import build_ptdf_matrix

        net_load = data_dict['net_load_bus']   # (N_BUS, T)
        contingency = int(data_dict['contingency'])
        removed = None if contingency == -1 else contingency

        try:
            PTDF = build_ptdf_matrix(N_BUS, LINES, ref_bus_idx=0,
                                     removed_line_idx=removed)
        except ValueError:
            return False   # island — infeasible by definition

        line_rates = [ln[5] for ln in LINES]
        gen_bus    = [g['bus'] - 1 for g in GENERATORS]

        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        m = gp.Model(env=env)

        N_GEN, T = z_pred.shape
        p = m.addVars(N_GEN, T, lb=0.0, name='p')

        # Generation bounds — fixed by z_pred
        for g, gen in enumerate(GENERATORS):
            for t in range(T):
                if z_pred[g, t] == 1:
                    m.addConstr(p[g,t] >= gen['Pmin'])
                    m.addConstr(p[g,t] <= gen['Pmax'])
                else:
                    m.addConstr(p[g,t] == 0.0)

        # Power balance
        for t in range(T):
            m.addConstr(
                gp.quicksum(p[g,t] for g in range(N_GEN))
                == float(net_load[:,t].sum())
            )

        # Thermal limits
        for l in range(len(LINES)):
            if l == removed:
                continue
            rate = line_rates[l]
            for t in range(T):
                flow = gp.quicksum(
                    PTDF[l, gen_bus[g]] * p[g,t]
                    for g in range(N_GEN) if abs(PTDF[l, gen_bus[g]]) > 1e-8
                ) - float(sum(
                    PTDF[l,n] * net_load[n,t] for n in range(N_BUS)
                ))
                m.addConstr(flow <=  rate)
                m.addConstr(flow >= -rate)

        m.setObjective(0, GRB.MINIMIZE)   # feasibility only
        m.optimize()

        feasible = m.Status == GRB.OPTIMAL
        m.dispose(); env.dispose()
        return feasible

    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_full(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = argparse.Namespace(**ckpt['args'])
    log.info(f"Loaded checkpoint from epoch {ckpt['epoch']} "
             f"(val_loss={ckpt['val_loss']:.4f})")

    # ── Dataset ───────────────────────────────────────────────────────────
    feat_stats = os.path.join(args.dataset_dir, 'feat_stats.json')
    test_ds = UCGraphDataset(args.dataset_dir, split='test',
                              feat_stats=feat_stats)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # ── Raw .npz for NFR / C-SOI (need optimal costs stored there) ───────
    npz_path = os.path.join(args.dataset_dir, 'test.npz')
    raw = dict(np.load(npz_path, allow_pickle=True))

    # ── Model ─────────────────────────────────────────────────────────────
    model = GNN_UC(
        in_features  = test_ds.n_node_features,
        d_h          = ckpt_args.d_h,
        n_heads      = ckpt_args.n_heads,
        gat_layers   = ckpt_args.gat_layers,
        lstm_hidden  = ckpt_args.lstm_hidden,
        n_gen        = test_ds.n_gen,
        dropout      = 0.0,   # no dropout at eval time
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # ── Collect predictions ───────────────────────────────────────────────
    all_preds   = []
    all_targets = []
    contingencies = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            logits = model(data)
            pred   = (logits >= 0.0).cpu().numpy().astype(int)   # (N_GEN, T)
            target = data.y.cpu().numpy().astype(int)            # (N_GEN, T)
            all_preds.append(pred)
            all_targets.append(target)
            contingencies.append(int(data.contingency))

    # ── Standard metrics ──────────────────────────────────────────────────
    accuracies = []
    hammings   = []
    per_gen_accs = [[] for _ in range(test_ds.n_gen)]

    for pred, target in zip(all_preds, all_targets):
        correct = (pred == target)
        accuracies.append(correct.mean())
        hammings.append((pred != target).sum())
        for g in range(test_ds.n_gen):
            per_gen_accs[g].append(correct[g].mean())

    mean_acc     = float(np.mean(accuracies))
    mean_hamming = float(np.mean(hammings))
    per_gen_mean = [float(np.mean(accs)) for accs in per_gen_accs]

    log.info("=" * 60)
    log.info("STANDARD METRICS")
    log.info("=" * 60)
    log.info(f"  Overall accuracy : {mean_acc:.4f}")
    log.info(f"  Mean hamming dist: {mean_hamming:.2f} bits/sample")
    for g, acc in enumerate(per_gen_mean):
        log.info(f"  Gen {g} accuracy  : {acc:.4f}")

    # ── NFR: N-1 Feasibility Rate ─────────────────────────────────────────
    log.info("=" * 60)
    log.info("NFR: N-1 FEASIBILITY RATE")
    log.info("=" * 60)

    n_contingency = 0
    n_feasible    = 0
    # Group by contingency type (skip base case for NFR — NFR is N-1 only)
    for i, (pred, c) in enumerate(zip(all_preds, contingencies)):
        if c == -1:
            continue   # skip base case
        n_contingency += 1
        SAMPLE_KEYS = ['X', 'A', 'edge_index', 'edge_attr', 'z',
               'scenario_idx', 'contingency_idx', 'is_base_case']
        sample_raw = {k: raw[k][i] for k in SAMPLE_KEYS if k in raw}
        sample_raw['contingency'] = c

# Reconstruct net_load_bus from normalised X feature 0
        if 'net_load_bus' not in sample_raw:
            with open(os.path.join(args.dataset_dir, 'feat_stats.json')) as _f:
                _stats = json.load(_f)
            _mu  = float(_stats['0']['mean'])
            _sig = float(_stats['0']['std'])
        sample_raw['net_load_bus'] = raw['X'][i, :, :, 0] * _sig + _mu
        sample_raw['contingency'] = c

        # net_load_bus not saved in old dataset — reconstruct from normalised X
        # X[:, :, 0] = (net_load_bus - mu) / sigma  →  invert to get MW values
        if 'net_load_bus' not in sample_raw:
            with open(os.path.join(args.dataset_dir, 'feat_stats.json')) as _f:
                _stats = json.load(_f)
            _mu  = float(_stats['0']['mean'])
            _sig = float(_stats['0']['std'])
            sample_raw['net_load_bus'] = (
                raw['X'][i, :, :, 0] * _sig + _mu
            )   # (N_BUS, T) in MW

        if check_n1_feasibility(pred, sample_raw):
            n_feasible += 1

    nfr = n_feasible / max(n_contingency, 1)
    log.info(f"  NFR = {n_feasible}/{n_contingency} = {nfr:.4f}")

    # ── C-SOI: Contingency-averaged Suboptimality Index ───────────────────
    log.info("=" * 60)
    log.info("C-SOI: CONTINGENCY-AVERAGED SUBOPTIMALITY INDEX")
    log.info("=" * 60)

    soi_by_contingency: dict[int, list] = {}
    for i, (pred, target, c) in enumerate(zip(all_preds, all_targets, contingencies)):
        # Use Hamming accuracy as a proxy for SOI here.
        # Full C-SOI requires running the LP-UC with z fixed — do this
        # in Stage 3 when the fuzzy solver is available.
        # For now we report the prediction accuracy per contingency type
        # which correlates directly with C-SOI.
        acc = float((pred == target).mean())
        if c not in soi_by_contingency:
            soi_by_contingency[c] = []
        soi_by_contingency[c].append(1.0 - acc)   # error rate as SOI proxy

    log.info("  Error rate (SOI proxy) per contingency index:")
    for c in sorted(soi_by_contingency.keys()):
        mean_err = float(np.mean(soi_by_contingency[c]))
        tag = "BASE" if c == -1 else f"line {c}"
        log.info(f"    c={c:>3} ({tag:10s}): mean_error={mean_err:.4f}")

    c_soi_proxy = float(np.mean([
        np.mean(v) for k, v in soi_by_contingency.items() if k != -1
    ]))
    log.info(f"  C-SOI proxy (mean error over N-1 contingencies): {c_soi_proxy:.4f}")
    log.info("  Note: Full C-SOI requires LP solve — computed in Stage 3.")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        'overall_accuracy' : mean_acc,
        'mean_hamming'     : mean_hamming,
        'per_gen_accuracy' : per_gen_mean,
        'nfr'              : nfr,
        'c_soi_proxy'      : c_soi_proxy,
        'soi_by_contingency': {str(k): float(np.mean(v))
                                for k, v in soi_by_contingency.items()},
    }
    out_path = os.path.join(os.path.dirname(args.checkpoint), 'eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {out_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate GNN-UC model')
    parser.add_argument('--checkpoint',  type=str, required=True,
                        help='Path to best_model.pt checkpoint')
    parser.add_argument('--dataset_dir', type=str, default='dataset_output')
    args = parser.parse_args()
    evaluate_full(args)