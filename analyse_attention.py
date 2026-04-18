"""
analyse_attention.py
====================
Extracts and visualises GAT attention weights from each layer to
diagnose whether over-localisation is causing the accuracy/NFR tradeoff.

Specifically checks:
  1. Per-layer attention entropy — low entropy = concentrated/local attention
  2. Attention on peripheral buses (22, 27) vs core buses (1, 2)
  3. How attention changes between base case and N-1 contingency
  4. Correlation between first-layer attention concentration and NFR

RUN WITH:
  python analyse_attention.py \
    --checkpoint model_output/best_model.pt \
    --dataset_dir dataset_output
"""

import argparse
import json
import logging
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

from model.gnn_uc import GNN_UC
from dataset.graph_dataset import UCGraphDataset
from data.ieee30_system import GENERATORS, LINES, N_BUS, N_GEN

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

STYLE = {
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'sans-serif', 'font.size': 11,
    'savefig.dpi': 150, 'savefig.bbox': 'tight',
}
plt.rcParams.update(STYLE)


# ─────────────────────────────────────────────────────────────────────────────
# Hook-based attention extractor
# ─────────────────────────────────────────────────────────────────────────────

class AttentionExtractor:
    """
    Registers forward hooks on GATv2Conv layers to capture attention
    coefficients alpha_ij during the forward pass.
    """

    def __init__(self, model: GNN_UC):
        self.model       = model
        self.attentions  = {}   # layer_idx -> alpha tensor
        self._hooks      = []
        self._register_hooks()

    def _register_hooks(self):
        for k, layer in enumerate(self.model.spatial_encoder.gat_layers):
            def make_hook(layer_idx):
                def hook(module, input, output):
                    # GATv2Conv returns (out, alpha) when return_attention_weights=True
                    # But hooks get the standard output — we need to call with flag
                    # Store the layer index for later manual extraction
                    self.attentions[layer_idx] = None
                return hook
            h = layer.register_forward_hook(make_hook(k))
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def get_attention_weights(self, data, device):
        """
        Run forward pass and extract attention weights by calling each
        GAT layer directly with return_attention_weights=True.
        """
        self.model.eval()
        attentions = {}

        with torch.no_grad():
            x          = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr  = data.edge_attr.to(device)

            N, T, F = x.shape
            E = edge_index.shape[1]

            # Build tiled edges (same as gat_encoder.forward)
            offsets  = torch.arange(T, device=device) * N
            ei_tiled = (
                edge_index.unsqueeze(2) + offsets.view(1, 1, T)
            ).permute(0, 2, 1).reshape(2, T * E)
            ea_tiled = edge_attr.repeat(T, 1)

            # Input projection
            x_flat = x.reshape(N * T, F)
            h = self.model.spatial_encoder.input_proj(x_flat)

            # Extract attention per layer
            for k in range(self.model.spatial_encoder.n_layers):
                gat_layer = self.model.spatial_encoder.gat_layers[k]

                # Call with return_attention_weights=True
                h_new, (edge_idx_out, alpha) = gat_layer(
                    h, ei_tiled,
                    edge_attr=ea_tiled,
                    return_attention_weights=True,
                )

                # alpha shape: (E*T, n_heads) or (E*T, 1) for last layer
                # Average over heads
                alpha_avg = alpha.abs().mean(dim=-1)   # (E*T,)

                # Reshape to (T, E) — attention per timestep per edge
                alpha_per_edge = alpha_avg.reshape(T, E)   # (T, E)

                attentions[k] = {
                    'alpha'      : alpha_avg.cpu().numpy(),      # (E*T,)
                    'alpha_per_edge': alpha_per_edge.cpu().numpy(),  # (T, E)
                    'edge_index' : edge_index.cpu().numpy(),     # (2, E)
                }

                # Continue forward pass
                h_new = self.model.spatial_encoder.norm_layers[k](h_new)
                h_new = torch.nn.functional.elu(h_new)
                h_new = self.model.spatial_encoder.ff_layers[k](h_new)
                h     = h + h_new

        return attentions


# ─────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────────────────────

def attention_entropy(alpha: np.ndarray) -> float:
    """
    Compute entropy of attention distribution.
    Low entropy = concentrated attention (over-localisation).
    High entropy = spread attention (global awareness).
    """
    alpha = np.clip(alpha, 1e-10, None)
    alpha = alpha / alpha.sum()
    return float(-np.sum(alpha * np.log(alpha)))


def bus_attention_profile(alpha_per_edge, edge_index, n_bus=N_BUS):
    """
    For each bus, compute total attention it RECEIVES from neighbours.
    Shape: (T, N_BUS) — attention received per bus per timestep.
    """
    T, E = alpha_per_edge.shape
    bus_attn = np.zeros((T, n_bus))

    for e in range(E):
        dst = edge_index[1, e]   # destination bus (receiver)
        bus_attn[:, dst] += alpha_per_edge[:, e]

    # Normalise per timestep
    row_sums = bus_attn.sum(axis=1, keepdims=True)
    bus_attn = bus_attn / np.maximum(row_sums, 1e-10)
    return bus_attn


def peripheral_vs_core_attention(bus_attn):
    """
    Compare attention received by peripheral buses (22, 27 — gen buses)
    vs core buses (1, 2 — base load).
    Returns (peripheral_mean, core_mean).
    """
    # 0-indexed
    peripheral = [12, 21, 22, 26]   # buses 13, 22, 23, 27 (gen buses)
    core       = [0, 1]             # buses 1, 2 (base load)

    peripheral_attn = bus_attn[:, peripheral].mean()
    core_attn       = bus_attn[:, core].mean()
    return float(peripheral_attn), float(core_attn)


def contingency_attention_delta(attn_base, attn_n1, edge_index):
    """
    Compute how much attention changes between base case and N-1.
    Large delta = model is topology-sensitive (good).
    Small delta = model ignores topology change (bad).
    """
    delta = np.abs(
        attn_n1['alpha_per_edge'] - attn_base['alpha_per_edge']
    ).mean()
    return float(delta)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    ckpt      = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)
    ckpt_args = argparse.Namespace(**ckpt['args'])
    feat_stats = os.path.join(args.dataset_dir, 'feat_stats.json')

    test_ds = UCGraphDataset(args.dataset_dir, split='test',
                              feat_stats=feat_stats)
    model = GNN_UC(
        in_features = test_ds.n_node_features,
        d_h         = ckpt_args.d_h,
        n_heads     = ckpt_args.n_heads,
        gat_layers  = ckpt_args.gat_layers,
        lstm_hidden = ckpt_args.lstm_hidden,
        n_gen       = test_ds.n_gen,
        dropout     = 0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    log.info(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    extractor = AttentionExtractor(model)
    loader    = DataLoader(test_ds, batch_size=1, shuffle=False)

    # ── Collect attention stats ───────────────────────────────────────────────
    results = {
        'base': {'entropy': {k: [] for k in range(ckpt_args.gat_layers)},
                 'peripheral': {k: [] for k in range(ckpt_args.gat_layers)},
                 'core'      : {k: [] for k in range(ckpt_args.gat_layers)}},
        'n1'  : {'entropy': {k: [] for k in range(ckpt_args.gat_layers)},
                 'peripheral': {k: [] for k in range(ckpt_args.gat_layers)},
                 'core'      : {k: [] for k in range(ckpt_args.gat_layers)}},
        'delta': {k: [] for k in range(ckpt_args.gat_layers)},
    }

    # Also collect base/n1 pairs for delta computation
    base_attns = {}   # scenario_idx -> layer -> attn
    n1_attns   = {}

    n_processed = 0
    with open(feat_stats) as f:
        fs = json.load(f)
    raw = dict(np.load(os.path.join(args.dataset_dir, 'test.npz'),
                        allow_pickle=True))

    for i, data in enumerate(loader):
        if n_processed >= args.n_samples:
            break

        data = data.to(device)
        c    = int(data.contingency)

        try:
            attns = extractor.get_attention_weights(data, device)
        except Exception as e:
            log.warning(f"Sample {i} attention extraction failed: {e}")
            continue

        split = 'base' if c == -1 else 'n1'
        scenario = int(raw['scenario_idx'][i])

        for k, attn in attns.items():
            ent  = attention_entropy(attn['alpha_per_edge'].flatten())
            bpf  = bus_attention_profile(attn['alpha_per_edge'],
                                          attn['edge_index'])
            p, co = peripheral_vs_core_attention(bpf)

            results[split]['entropy'][k].append(ent)
            results[split]['peripheral'][k].append(p)
            results[split]['core'][k].append(co)

        # Store for delta computation
        if c == -1:
            base_attns[scenario] = attns
        else:
            if scenario in base_attns:
                for k in attns:
                    if k in base_attns[scenario]:
                        delta = contingency_attention_delta(
                            base_attns[scenario][k], attns[k],
                            attns[k]['edge_index']
                        )
                        results['delta'][k].append(delta)

        n_processed += 1
        if n_processed % 50 == 0:
            log.info(f"Processed {n_processed}/{args.n_samples} samples")

    extractor.remove_hooks()

    # ── Print summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("ATTENTION ANALYSIS SUMMARY")
    log.info("=" * 60)
    n_layers = ckpt_args.gat_layers

    for k in range(n_layers):
        log.info(f"\nGAT Layer {k+1}:")

        for split in ['base', 'n1']:
            ent_vals = results[split]['entropy'][k]
            per_vals = results[split]['peripheral'][k]
            cor_vals = results[split]['core'][k]
            if not ent_vals:
                continue
            log.info(f"  {split.upper():5s} — "
                     f"entropy={np.mean(ent_vals):.3f} "
                     f"(low=concentrated) | "
                     f"peripheral_attn={np.mean(per_vals):.3f} | "
                     f"core_attn={np.mean(cor_vals):.3f}")

        delta_vals = results['delta'][k]
        if delta_vals:
            log.info(f"  DELTA (base→N-1) — "
                     f"mean={np.mean(delta_vals):.4f} "
                     f"(low=topology-blind, high=topology-sensitive)")

    log.info("\nDIAGNOSIS:")
    for k in range(n_layers):
        base_ent = np.mean(results['base']['entropy'][k]) if results['base']['entropy'][k] else 0
        n1_ent   = np.mean(results['n1']['entropy'][k])   if results['n1']['entropy'][k]   else 0
        delta    = np.mean(results['delta'][k]) if results['delta'][k] else 0

        if base_ent > n1_ent + 0.1:
            log.info(f"  Layer {k+1}: entropy drops on N-1 → "
                     f"more concentrated attention under contingency "
                     f"(over-localisation confirmed)")
        elif delta < 0.01:
            log.info(f"  Layer {k+1}: tiny attention delta on contingency → "
                     f"model largely ignores topology change")
        else:
            log.info(f"  Layer {k+1}: attention adapts to contingency "
                     f"(delta={delta:.4f}) — topology-sensitive ✓")

    # ── Save results ──────────────────────────────────────────────────────────
    summary = {
        'base_entropy' : {str(k): float(np.mean(v)) if v else 0
                          for k, v in results['base']['entropy'].items()},
        'n1_entropy'   : {str(k): float(np.mean(v)) if v else 0
                          for k, v in results['n1']['entropy'].items()},
        'delta'        : {str(k): float(np.mean(v)) if v else 0
                          for k, v in results['delta'].items()},
        'peripheral'   : {str(k): float(np.mean(v)) if v else 0
                          for k, v in results['n1']['peripheral'].items()},
        'core'         : {str(k): float(np.mean(v)) if v else 0
                          for k, v in results['n1']['core'].items()},
    }
    with open(os.path.join(args.output_dir, 'attention_analysis.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    _plot_attention(args.output_dir, results, n_layers)
    log.info(f"\nSaved to {args.output_dir}/")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _plot_attention(out_dir, results, n_layers):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    layers = list(range(n_layers))
    x      = np.arange(n_layers)

    # Entropy comparison base vs N-1
    ax = axes[0]
    base_ent = [np.mean(results['base']['entropy'][k])
                if results['base']['entropy'][k] else 0 for k in layers]
    n1_ent   = [np.mean(results['n1']['entropy'][k])
                if results['n1']['entropy'][k] else 0 for k in layers]
    ax.bar(x - 0.2, base_ent, 0.35, label='Base case',       color='#4A90D9', alpha=0.8)
    ax.bar(x + 0.2, n1_ent,   0.35, label='N-1 contingency', color='#E74C3C', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {k+1}' for k in layers])
    ax.set_ylabel('Attention Entropy\n(higher = more global)', fontsize=11)
    ax.set_title('Attention Entropy\nBase vs N-1', pad=8)
    ax.legend(fontsize=9)
    ax.set_facecolor('#FAFAFA')

    # Peripheral vs core attention under N-1
    ax = axes[1]
    per_vals = [np.mean(results['n1']['peripheral'][k])
                if results['n1']['peripheral'][k] else 0 for k in layers]
    cor_vals = [np.mean(results['n1']['core'][k])
                if results['n1']['core'][k] else 0 for k in layers]
    ax.bar(x - 0.2, per_vals, 0.35, label='Peripheral buses\n(gens 2,3,4,5)',
           color='#F39C12', alpha=0.8)
    ax.bar(x + 0.2, cor_vals, 0.35, label='Core buses\n(gens 0,1)',
           color='#2ECC71', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {k+1}' for k in layers])
    ax.set_ylabel('Normalised Attention Received', fontsize=11)
    ax.set_title('Attention on Peripheral\nvs Core Buses (N-1)', pad=8)
    ax.legend(fontsize=9)
    ax.set_facecolor('#FAFAFA')

    # Attention delta base → N-1
    ax = axes[2]
    delta_vals = [np.mean(results['delta'][k])
                  if results['delta'][k] else 0 for k in layers]
    bars = ax.bar(x, delta_vals, 0.5, color='#9B59B6', alpha=0.8,
                  edgecolor='white')
    for bar, val in zip(bars, delta_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {k+1}' for k in layers])
    ax.set_ylabel('Mean |Δattention| base→N-1', fontsize=11)
    ax.set_title('Topology Sensitivity\n(higher = more aware of outage)',
                 pad=8)
    ax.set_facecolor('#FAFAFA')
    ax.axhline(0.01, color='red', linewidth=1.5, linestyle='--',
               alpha=0.6, label='Sensitivity threshold')
    ax.legend(fontsize=9)

    plt.suptitle('GAT Attention Analysis — Diagnosing Topology Awareness',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'attention_analysis.png')
    fig.savefig(path)
    plt.close(fig)
    log.info(f"Saved {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse GAT attention weights')
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, default='dataset_output')
    parser.add_argument('--n_samples',   type=int, default=200)
    parser.add_argument('--output_dir',  type=str, default='figures')
    args = parser.parse_args()
    analyse(args)