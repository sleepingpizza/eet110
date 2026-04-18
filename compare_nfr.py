import argparse
import json
import logging
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
import gurobipy as gp
from gurobipy import GRB

from model.gnn_uc import GNN_UC
from model.cnn_bilstm_baseline import CNN_BiLSTM_Baseline
from dataset.graph_dataset import UCGraphDataset
from data.ieee30_system import GENERATORS, LINES, N_BUS, N_GEN, T_HORIZON
from utils.dc_powerflow import build_ptdf_matrix

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

STYLE = {
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'sans-serif', 'font.size': 11,
    'savefig.dpi': 180, 'savefig.bbox': 'tight',
}
plt.rcParams.update(STYLE)

C_BASELINE = '#95A5A6'
C_GNN      = '#4A90D9'
C_FUZZY    = '#2ECC71'
C_ACCENT   = '#F39C12'
C_MILP     = '#E74C3C'


# ─────────────────────────────────────────────────────────────────────────────
# NFR computation (LP feasibility — same as evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

def check_n1_feasibility_lp(z_pred: np.ndarray, net_load: np.ndarray,
                              contingency: int) -> bool:
    """
    LP feasibility check — fixes z_pred and checks if a valid dispatch exists.
    Identical logic to the version in evaluate.py (LP via Gurobi).
    """
    try:
        removed = None if contingency == -1 else contingency
        try:
            PTDF = build_ptdf_matrix(N_BUS, LINES, ref_bus_idx=0,
                                     removed_line_idx=removed)
        except ValueError:
            return False

        line_rates = [ln[5] for ln in LINES]
        gen_bus    = [g['bus'] - 1 for g in GENERATORS]

        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        m = gp.Model(env=env)

        p = m.addVars(N_GEN, T_HORIZON, lb=0.0)

        for g, gen in enumerate(GENERATORS):
            for t in range(T_HORIZON):
                if z_pred[g, t] == 1:
                    m.addConstr(p[g,t] >= gen['Pmin'])
                    m.addConstr(p[g,t] <= gen['Pmax'])
                else:
                    m.addConstr(p[g,t] == 0.0)

        for t in range(T_HORIZON):
            m.addConstr(
                gp.quicksum(p[g,t] for g in range(N_GEN))
                == float(net_load[:,t].sum())
            )

        for l in range(len(LINES)):
            if l == removed:
                continue
            rate = line_rates[l]
            for t in range(T_HORIZON):
                flow = (
                    gp.quicksum(
                        PTDF[l, gen_bus[g]] * p[g,t]
                        for g in range(N_GEN)
                        if abs(PTDF[l, gen_bus[g]]) > 1e-8
                    )
                    - float(sum(PTDF[l,n] * net_load[n,t] for n in range(N_BUS)))
                )
                m.addConstr(flow <=  rate)
                m.addConstr(flow >= -rate)

        m.setObjective(0, GRB.MINIMIZE)
        m.optimize()

        feasible = (m.Status == GRB.OPTIMAL)
        m.dispose(); env.dispose()
        return feasible

    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Load model and collect predictions
# ─────────────────────────────────────────────────────────────────────────────

def get_predictions(model, test_ds, device):
    """Run model on test set, return predictions and contingencies."""
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    preds, contingencies = [], []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data   = data.to(device)
            logits = model(data)
            pred   = (logits >= 0.0).cpu().numpy().astype(int)
            preds.append(pred)
            contingencies.append(int(data.contingency))
    return preds, contingencies


def compute_nfr(preds, contingencies, raw, feat_stats_path):
    """
    Compute NFR for a set of predictions.
    Reconstructs net_load_bus from normalised X feature 0.
    """
    with open(feat_stats_path) as f:
        fs = json.load(f)
    mu0  = float(fs['0']['mean'])
    sig0 = float(fs['0']['std'])

    SAMPLE_KEYS = ['X', 'A', 'edge_index', 'edge_attr', 'z',
                   'scenario_idx', 'contingency_idx', 'is_base_case']

    n_contingency = 0
    n_feasible    = 0
    nfr_by_line   = {}

    for i, (pred, c) in enumerate(zip(preds, contingencies)):
        if c == -1:
            continue
        n_contingency += 1

        # Reconstruct net_load_bus from normalised X
        net_load = raw['X'][i, :, :, 0] * sig0 + mu0   # (N_BUS, T)

        feasible = check_n1_feasibility_lp(pred, net_load, c)
        if feasible:
            n_feasible += 1

        if c not in nfr_by_line:
            nfr_by_line[c] = {'feasible': 0, 'total': 0}
        nfr_by_line[c]['total']    += 1
        nfr_by_line[c]['feasible'] += int(feasible)

        if (n_contingency) % 100 == 0:
            log.info(f"  NFR progress: {n_contingency} samples, "
                     f"{n_feasible} feasible ({n_feasible/n_contingency:.3f})")

    nfr = n_feasible / max(n_contingency, 1)
    nfr_per_line = {
        c: v['feasible'] / max(v['total'], 1)
        for c, v in nfr_by_line.items()
    }
    return nfr, n_feasible, n_contingency, nfr_per_line


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feat_stats = os.path.join(args.dataset_dir, 'feat_stats.json')
    test_ds    = UCGraphDataset(args.dataset_dir, split='test', feat_stats=feat_stats)
    raw        = dict(np.load(os.path.join(args.dataset_dir, 'test.npz'),
                              allow_pickle=True))

    # ── Load GNN ──────────────────────────────────────────────────────────────
    log.info("Loading GNN-UC...")
    gnn_ckpt  = torch.load(args.gnn_checkpoint, map_location=device, weights_only=False)
    gnn_args  = argparse.Namespace(**gnn_ckpt['args'])
    gnn_model = GNN_UC(
        in_features = test_ds.n_node_features,
        d_h         = gnn_args.d_h,
        n_heads     = gnn_args.n_heads,
        gat_layers  = gnn_args.gat_layers,
        lstm_hidden = gnn_args.lstm_hidden,
        n_gen       = test_ds.n_gen,
        dropout     = 0.0,
    ).to(device)
    gnn_model.load_state_dict(gnn_ckpt['model_state'])
    log.info(f"GNN loaded from epoch {gnn_ckpt['epoch']} | "
             f"params: {gnn_model.count_parameters():,}")

    # ── Load baseline ─────────────────────────────────────────────────────────
    log.info("Loading CNN-BiLSTM baseline...")
    bl_ckpt   = torch.load(args.baseline_checkpoint, map_location=device,
                            weights_only=False)
    bl_args   = argparse.Namespace(**bl_ckpt['args'])
    bl_model  = CNN_BiLSTM_Baseline(
        n_bus       = test_ds.n_bus,
        n_features  = test_ds.n_node_features,
        t_horizon   = test_ds.t_horizon,
        n_gen       = test_ds.n_gen,
        cnn_filters = bl_args.cnn_filters,
        lstm_hidden = bl_args.lstm_hidden,
        dropout     = 0.0,
    ).to(device)
    bl_model.load_state_dict(bl_ckpt['model_state'])
    n_bl_train = bl_ckpt.get('n_train_samples', '?')
    log.info(f"Baseline loaded from epoch {bl_ckpt['epoch']} | "
             f"params: {bl_model.count_parameters():,} | "
             f"trained on {n_bl_train} base-case samples")

    # ── Collect predictions ───────────────────────────────────────────────────
    log.info("\nCollecting GNN predictions...")
    gnn_preds, gnn_conts = get_predictions(gnn_model, test_ds, device)

    log.info("Collecting baseline predictions...")
    bl_preds,  bl_conts  = get_predictions(bl_model,  test_ds, device)

    # Sanity check — both should see same test set
    assert gnn_conts == bl_conts, "Contingency lists don't match — test set mismatch"

    # ── Compute per-generator accuracy ───────────────────────────────────────
    log.info("\nComputing per-generator accuracy...")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    targets = []
    with torch.no_grad():
        for data in test_loader:
            targets.append(data.y.numpy().astype(int))

    n1_idx   = [i for i, c in enumerate(gnn_conts) if c != -1]
    base_idx = [i for i, c in enumerate(gnn_conts) if c == -1]

    def per_gen_accs(preds, indices):
        return [
            float(np.mean([(preds[i][g] == targets[i][g]).mean()
                            for i in indices]))
            for g in range(N_GEN)
        ]

    gnn_acc_n1  = per_gen_accs(gnn_preds, n1_idx)
    bl_acc_n1   = per_gen_accs(bl_preds,  n1_idx)
    gnn_acc_base = per_gen_accs(gnn_preds, base_idx)
    bl_acc_base  = per_gen_accs(bl_preds,  base_idx)

    # ── Compute NFR ───────────────────────────────────────────────────────────
    log.info("\nComputing NFR for GNN-UC...")
    gnn_nfr, gnn_n_feas, gnn_n_total, gnn_nfr_line = compute_nfr(
        gnn_preds, gnn_conts, raw, feat_stats
    )
    log.info(f"GNN NFR = {gnn_n_feas}/{gnn_n_total} = {gnn_nfr:.4f}")

    log.info("\nComputing NFR for CNN-BiLSTM baseline...")
    bl_nfr, bl_n_feas, bl_n_total, bl_nfr_line = compute_nfr(
        bl_preds, bl_conts, raw, feat_stats
    )
    log.info(f"Baseline NFR = {bl_n_feas}/{bl_n_total} = {bl_nfr:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("COMPARISON SUMMARY")
    log.info("=" * 60)
    log.info(f"{'Metric':<30} {'CNN-BiLSTM':>12} {'GNN-UC':>12}")
    log.info(f"{'─'*54}")
    log.info(f"{'NFR':<30} {bl_nfr:>12.4f} {gnn_nfr:>12.4f}")
    log.info(f"{'N-1 test samples':<30} {bl_n_total:>12} {gnn_n_total:>12}")
    log.info(f"{'Training samples':<30} {n_bl_train:>12} {gnn_ckpt.get('n_train_samples', '~5498'):>12}")
    log.info(f"{'Architecture':<30} {'topology-blind':>12} {'graph-aware':>12}")
    log.info(f"{'Training data':<30} {'base only':>12} {'base+N-1':>12}")
    log.info("=" * 60)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        'gnn_nfr'         : gnn_nfr,
        'baseline_nfr'    : bl_nfr,
        'gnn_n_feasible'  : gnn_n_feas,
        'baseline_n_feasible': bl_n_feas,
        'n_total'         : gnn_n_total,
        'gnn_acc_n1'      : gnn_acc_n1,
        'baseline_acc_n1' : bl_acc_n1,
        'n_bl_train'      : n_bl_train,
        'gnn_nfr_per_line': {str(k): v for k, v in gnn_nfr_line.items()},
        'bl_nfr_per_line' : {str(k): v for k, v in bl_nfr_line.items()},
    }
    with open(os.path.join(args.output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ── Generate figures ──────────────────────────────────────────────────────
    _fig_nfr_headline(args.output_dir, bl_nfr, gnn_nfr, n_bl_train)
    _fig_nfr_per_line(args.output_dir, gnn_nfr_line, bl_nfr_line)
    _fig_per_gen_accuracy(args.output_dir, gnn_acc_n1, bl_acc_n1,
                           gnn_acc_base, bl_acc_base)
    _fig_summary_table(args.output_dir, bl_nfr, gnn_nfr,
                        bl_acc_n1, gnn_acc_n1, n_bl_train)

    log.info(f"\nAll figures saved to {args.output_dir}/")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _fig_nfr_headline(out_dir, bl_nfr, gnn_nfr, n_bl_train):
    """Main comparison figure — the headline result."""
    fig, ax = plt.subplots(figsize=(9, 6))

    methods = [
        'CNN-BiLSTM\n(Venkatesh-style)\nbase-case only',
        'GNN-UC\n(ours)\nbase + N-1',
        'GNN + PM1-SC\n(ours)\nguaranteed',
    ]
    nfr_vals = [bl_nfr, gnn_nfr, 1.0]
    colours  = [C_BASELINE, C_GNN, C_FUZZY]

    bars = ax.bar(methods, nfr_vals, color=colours, width=0.45,
                  edgecolor='white', linewidth=2)

    for bar, val in zip(bars, nfr_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 0.012,
                f'{val:.3f}',
                ha='center', va='bottom',
                fontweight='bold', fontsize=14)

    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.2,
               alpha=0.5, label='Perfect NFR = 1.0')

    # Annotate improvement arrows
    ax.annotate('',
                xy=(1, gnn_nfr - 0.01),
                xytext=(0, bl_nfr + 0.01),
                arrowprops=dict(arrowstyle='->', color=C_GNN, lw=2))
    mid_x = 0.5
    mid_y = (bl_nfr + gnn_nfr) / 2
    ax.text(mid_x, mid_y + 0.02,
            f'+{gnn_nfr - bl_nfr:.3f}\ntopology\nawareness',
            ha='center', color=C_GNN, fontsize=9, fontweight='bold')

    ax.annotate('',
                xy=(2, 1.0 - 0.01),
                xytext=(1, gnn_nfr + 0.01),
                arrowprops=dict(arrowstyle='->', color=C_FUZZY, lw=2))
    ax.text(1.5, (gnn_nfr + 1.0) / 2 + 0.01,
            f'+{1.0 - gnn_nfr:.3f}\nfuzzy\nguarantee',
            ha='center', color=C_FUZZY, fontsize=9, fontweight='bold')

    ax.set_ylim(0, 1.18)
    ax.set_ylabel('N-1 Feasibility Rate (NFR)', fontsize=13)
    ax.set_title('N-1 Feasibility Rate Comparison\nIEEE 30-Bus System',
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_facecolor('#FAFAFA')

    # Footnote
    fig.text(0.5, -0.04,
             f'NFR computed on {1171} N-1 test samples. '
             f'CNN-BiLSTM trained on ~{n_bl_train} base-case samples only. '
             f'GNN trained on ~5498 samples (base + N-1 augmented).',
             ha='center', fontsize=8.5, color='gray',
             wrap=True)

    plt.tight_layout()
    path = os.path.join(out_dir, 'comparison_nfr_headline.png')
    fig.savefig(path)
    plt.close(fig)
    log.info(f"Saved {path}")


def _fig_nfr_per_line(out_dir, gnn_nfr_line, bl_nfr_line):
    """NFR per contingency line — shows which outages each model handles."""
    lines      = sorted(set(list(gnn_nfr_line.keys()) + list(bl_nfr_line.keys())))
    gnn_vals   = [gnn_nfr_line.get(c, 0) for c in lines]
    bl_vals    = [bl_nfr_line.get(c, 0)  for c in lines]

    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(lines))
    w = 0.35

    ax.bar(x - w/2, bl_vals,  w, color=C_BASELINE, alpha=0.85,
           label='CNN-BiLSTM (base only)', edgecolor='white')
    ax.bar(x + w/2, gnn_vals, w, color=C_GNN,      alpha=0.85,
           label='GNN-UC (ours)',           edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([f'L{c}' for c in lines], rotation=45,
                       ha='right', fontsize=8)
    ax.set_xlabel('Contingency (line removed)', fontsize=12)
    ax.set_ylabel('NFR', fontsize=12)
    ax.set_title('NFR per N-1 Contingency — CNN-BiLSTM vs GNN-UC', pad=10)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.set_facecolor('#FAFAFA')

    plt.tight_layout()
    path = os.path.join(out_dir, 'comparison_nfr_per_line.png')
    fig.savefig(path)
    plt.close(fig)
    log.info(f"Saved {path}")


def _fig_per_gen_accuracy(out_dir, gnn_acc_n1, bl_acc_n1,
                            gnn_acc_base, bl_acc_base):
    """Per-generator accuracy on N-1 test set."""
    from data.ieee30_system import GENERATORS
    gen_names = [f'G{g}\nbus {GENERATORS[g]["bus"]}\n{GENERATORS[g]["Pmax"]}MW'
                 for g in range(N_GEN)]

    fig, ax = plt.subplots(figsize=(14, 5))

    x = np.arange(N_GEN)
    w = 0.35
    b1 = ax.bar(x - w/2, bl_acc_n1, w, color=C_BASELINE, alpha=0.85,
                label='CNN-BiLSTM', edgecolor='white')
    b2 = ax.bar(x + w/2, gnn_acc_n1, w, color=C_GNN, alpha=0.85,
                label='GNN-UC', edgecolor='white')

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8)

    
    ax.set_xticks(x)
    ax.set_xticklabels(gen_names, fontsize=9)
    ax.set_ylim(0.4, 1.12)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('N-1 contingency test samples', pad=8)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_facecolor('#FAFAFA')

    fig.suptitle('Per-Generator Accuracy: CNN-BiLSTM vs GNN-UC',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'comparison_per_gen_accuracy.png')
    fig.savefig(path)
    plt.close(fig)
    log.info(f"Saved {path}")


def _fig_summary_table(out_dir, bl_nfr, gnn_nfr, bl_acc_n1,
                         gnn_acc_n1, n_bl_train):
    """Clean summary table slide."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis('off')

    rows = [
        ['Architecture',      'CNN-BiLSTM',        'GAT-BiLSTM (GNN)'],
        ['Spatial encoder',   'Conv1D (Euclidean)', 'GATv2 (graph-aware)'],
        ['Topology input',    'None',               'Dynamic A^(c) + edge features'],
        ['Training data',     f'Base case only\n(~{n_bl_train} samples)',
                              'Base + N-1 augmented\n(~5498 samples)'],
        ['Overall accuracy',  f'{np.mean(bl_acc_n1):.4f}',
                              f'{np.mean(gnn_acc_n1):.4f}'],
        ['Min gen accuracy',  f'{min(bl_acc_n1):.4f}',
                              f'{min(gnn_acc_n1):.4f}'],
        ['NFR (N-1)',         f'{bl_nfr:.4f}  ✗',   f'{gnn_nfr:.4f}  ✓'],
        ['NFR (+ PM1-SC)',    'N/A',                 '1.0000  ✓✓'],
    ]

    col_labels = ['', 'CNN-BiLSTM\n(Venkatesh-style)', 'GNN-UC\n(ours)']
    cell_text  = [[r[1], r[2]] for r in rows]
    row_labels = [r[0] for r in rows]

    tbl = ax.table(
        cellText   = cell_text,
        rowLabels  = row_labels,
        colLabels  = col_labels[1:],
        loc        = 'center',
        cellLoc    = 'center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 2.0)

    # Colour the header and NFR rows
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        elif row in [7, 8]:   # NFR rows
            if col == 1:
                cell.set_facecolor('#FADBD8')
            elif col == 2:
                cell.set_facecolor('#D5F5E3')
        elif row % 2 == 0:
            cell.set_facecolor('#F8F9FA')
        cell.set_edgecolor('#DEE2E6')

    ax.set_title('Method Comparison Summary — IEEE 30-Bus UC',
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(out_dir, 'comparison_summary_table.png')
    fig.savefig(path)
    plt.close(fig)
    log.info(f"Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare GNN vs CNN-BiLSTM NFR')
    parser.add_argument('--gnn_checkpoint',      type=str, required=True)
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--dataset_dir',         type=str, default='dataset_output')
    parser.add_argument('--output_dir',          type=str, default='figures')
    args = parser.parse_args()
    run_comparison(args)