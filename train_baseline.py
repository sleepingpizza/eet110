#ew baseline
import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from model.cnn_bilstm_baseline import CNN_BiLSTM_Baseline
from dataset.graph_dataset import UCGraphDataset

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

PMAX = [80, 80, 40, 50, 30, 50]


@torch.no_grad()
def evaluate_baseline(model, cache, criterion):
    model.eval()
    total_loss = 0.0; total_acc = 0.0; total_hamming = 0.0
    n_samples  = 0
    gen_correct = None; gen_total = None

    for data in cache:
        logits = model(data)
        y      = data.y

        loss           = criterion(logits, y)
        total_loss    += loss.item()
        pred           = (logits >= 0.0).float()
        total_acc     += (pred == y).float().mean().item()
        total_hamming += (pred != y).float().sum().item()

        if gen_correct is None:
            n_gen       = y.shape[0]
            gen_correct = torch.zeros(n_gen, device=y.device)
            gen_total   = torch.zeros(n_gen, device=y.device)

        gen_correct += (pred == y).float().sum(dim=1)
        gen_total   += float(y.shape[1])
        n_samples   += 1

    per_gen_acc = (gen_correct / gen_total).cpu().tolist() if gen_correct is not None else []
    return {
        'loss'       : total_loss    / max(n_samples, 1),
        'accuracy'   : total_acc     / max(n_samples, 1),
        'hamming'    : total_hamming / max(n_samples, 1),
        'per_gen_acc': per_gen_acc,
    }


def train_baseline(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # ── Load full datasets ────────────────────────────────────────────────────
    feat_stats = os.path.join(args.dataset_dir, 'feat_stats.json')
    train_ds   = UCGraphDataset(args.dataset_dir, split='train', feat_stats=feat_stats)
    val_ds     = UCGraphDataset(args.dataset_dir, split='val',   feat_stats=feat_stats)

    # ── Pre-cache on GPU ──────────────────────────────────────────────────────
    log.info("Pre-caching dataset on GPU...")
    all_train = [d.to(device) for d in train_ds]
    all_val   = [d.to(device) for d in val_ds]

    # ── CRITICAL: filter to base-case only for training ───────────────────────
    # This is what makes the baseline topology-blind in the Venkatesh sense —
    # it never sees a contingency topology during training.
    train_base = [d for d in all_train if int(d.contingency) == -1]
    val_base   = [d for d in all_val   if int(d.contingency) == -1]

    log.info(f"Full train: {len(all_train)} | Base-case only: {len(train_base)}")
    log.info(f"Full val:   {len(all_val)}   | Base-case only: {len(val_base)}")
    log.info(f"Sample ratio GNN/baseline: {len(all_train)/max(len(train_base),1):.1f}x")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CNN_BiLSTM_Baseline(
        n_bus       = train_ds.n_bus,
        n_features  = train_ds.n_node_features,
        t_horizon   = train_ds.t_horizon,
        n_gen       = train_ds.n_gen,
        cnn_filters = args.cnn_filters,
        lstm_hidden = args.lstm_hidden,
        dropout     = args.dropout,
    ).to(device)
    log.info(f"CNN-BiLSTM baseline parameters: {model.count_parameters():,}")

  
    pos_weight = torch.tensor(
        [1, 1, 5, 5, 5, 15], dtype=torch.float32
    ).unsqueeze(-1).to(device)
    criterion = nn.BCE5WithLogitsLoss(pos_weight=pos_weight)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=10
    )

    # ── Training ──────────────────────────────────────────────────────────────
    best_min_hard_acc = 0.0
    patience_counter  = 0
    history           = []
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, 'best_baseline.pt')

    log.info("=" * 60)
    log.info("BASELINE TRAINING: CNN-BiLSTM on base-case only")
    log.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        random.shuffle(train_base)
        for batch in train_base:
            optimiser.zero_grad()
            logits = model(batch)
            loss   = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / max(len(train_base), 1)
        elapsed        = time.time() - t0

        # Evaluate on base-case val only (model was trained on base-case)
        val_metrics = evaluate_baseline(model, val_base, criterion)

        hard_gen_indices = [2, 3, 4, 5]
        hard_gen_accs    = [val_metrics['per_gen_acc'][g] for g in hard_gen_indices]
        min_hard_acc     = min(hard_gen_accs)

        old_lr = optimiser.param_groups[0]['lr']
        scheduler.step(-min_hard_acc)
        new_lr = optimiser.param_groups[0]['lr']
        lr_tag = f" [lr {old_lr:.1e}→{new_lr:.1e}]" if new_lr != old_lr else ""

        log.info(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"train={avg_train_loss:.4f} | "
            f"val={val_metrics['loss']:.4f} | "
            f"acc={val_metrics['accuracy']:.4f} | "
            f"min_hard={min_hard_acc:.4f} | "
            f"{elapsed:.1f}s{lr_tag}"
        )
        per_gen_str = ' '.join(
            f"g{g}:{v:.3f}" for g, v in enumerate(val_metrics['per_gen_acc'])
        )
        log.info(f"  per-gen: {per_gen_str}")

        history.append({
            'epoch'       : epoch,
            'train_loss'  : avg_train_loss,
            'val_loss'    : val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'min_hard_acc': min_hard_acc,
            'per_gen_acc' : val_metrics['per_gen_acc'],
        })

        if min_hard_acc > best_min_hard_acc:
            best_min_hard_acc = min_hard_acc
            patience_counter  = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_loss'   : val_metrics['loss'],
                'val_metrics': val_metrics,
                'min_hard_acc': min_hard_acc,
                'args'       : vars(args),
                'n_train_samples': len(train_base),
            }, checkpoint_path)
            log.info(f"  ✓ checkpoint saved (min_hard_acc={min_hard_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    with open(os.path.join(args.output_dir, 'baseline_train_metrics.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # ── Test evaluation on base-case test set ─────────────────────────────────
    log.info("\nEvaluating on test set (base-case only)...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])

    test_ds   = UCGraphDataset(args.dataset_dir, split='test', feat_stats=feat_stats)
    test_all  = [d.to(device) for d in test_ds]
    test_base = [d for d in test_all if int(d.contingency) == -1]

    test_metrics = evaluate_baseline(model, test_base, criterion)

    log.info("=" * 60)
    log.info("BASELINE TEST RESULTS (base-case only)")
    log.info("=" * 60)
    log.info(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    log.info(f"  Hamming  : {test_metrics['hamming']:.1f}")
    for g, acc in enumerate(test_metrics['per_gen_acc']):
        log.info(f"  Gen {g}    : {acc:.4f}")

    with open(os.path.join(args.output_dir, 'baseline_test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    log.info(f"\nDone. Run compare_nfr.py to compute NFR and generate figures.")
    return test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN-BiLSTM baseline')
    parser.add_argument('--dataset_dir',  type=str,   default='dataset_output')
    parser.add_argument('--output_dir',   type=str,   default='baseline_output')
    parser.add_argument('--cnn_filters',  type=int,   default=32)
    parser.add_argument('--lstm_hidden',  type=int,   default=64)
    parser.add_argument('--dropout',      type=float, default=0.10)
    parser.add_argument('--epochs',       type=int,   default=300)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience',     type=int,   default=40)
    args = parser.parse_args()
    train_baseline(args)