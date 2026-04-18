#ablations at some point
import argparse
import json
import logging
import os
import random
import time

import torch
import torch.nn as nn

from model.gnn_uc import GNN_UC
from dataset.graph_dataset import UCGraphDataset
from torch_geometric.loader import DataLoader

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

PMAX = [80, 80, 40, 50, 30, 150]   # from ieee30_system.py — for capacity-weighted acc


@torch.no_grad()
def evaluate(model, cache, criterion):
    """Run model on pre-cached GPU list, return aggregated metrics."""
    model.eval()
    total_loss    = 0.0
    total_acc     = 0.0
    total_hamming = 0.0
    n_samples     = 0
    gen_correct   = None
    gen_total     = None

    for data in cache:
        logits = model(data)    # (N_GEN, T) — data already on device
        y      = data.y         # (N_GEN, T)

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


def train(args):
    if args.batch_size != 1:
        raise ValueError("batch_size > 1 not supported. Run with --batch_size 1.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    feat_stats = os.path.join(args.dataset_dir, 'feat_stats.json')
    train_ds   = UCGraphDataset(args.dataset_dir, split='train', feat_stats=feat_stats)
    val_ds     = UCGraphDataset(args.dataset_dir, split='val',   feat_stats=feat_stats)

    log.info(
        f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples | "
        f"n_bus={train_ds.n_bus}, n_gen={train_ds.n_gen}, "
        f"T={train_ds.t_horizon}, F={train_ds.n_node_features}"
    )

    # ── BUG 14 FIX: pre-cache entire dataset on GPU ───────────────────────────
    log.info("Pre-caching dataset on GPU...")
    t_cache = time.time()
    train_cache = [d.to(device) for d in train_ds]
    val_cache   = [d.to(device) for d in val_ds]
    log.info(f"Cached {len(train_cache)} train + {len(val_cache)} val samples "
             f"in {time.time() - t_cache:.1f}s")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GNN_UC(
        in_features = train_ds.n_node_features,
        d_h         = args.d_h,
        n_heads     = args.n_heads,
        gat_layers  = args.gat_layers,
        lstm_hidden = args.lstm_hidden,
        n_gen       = train_ds.n_gen,
        dropout     = args.dropout,
    ).to(device)
    log.info(f"Model parameters: {model.count_parameters():,}")

    # Verify all parameters are on the correct device
    for name, param in model.named_parameters():
        assert param.device.type == device.type, \
            f"Parameter {name} on {param.device}, expected {device}"
    log.info("All model parameters confirmed on correct device ✓")

    # ── Loss ──────────────────────────────────────────────────────────────────
    # Manual pos_weights: gen 0/1 are always-ON (trivial), boost hard generators
    pos_weight = torch.tensor(
        [1.0,   # gen 0: always ON — neutral weight
         1.0,   # gen 1: always ON — neutral weight
         5.0,   # gen 2: boosted
         5.0,   # gen 3: boosted — hard generator
         5.0,   # gen 4: reduced from 7.4 — was destabilising
         15.0],  # gen 5: boosted — hardest generator (bus 27, Pmax=150)
        dtype=torch.float32,
    ).unsqueeze(-1).to(device)   # (N_GEN, 1)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Track -min_hard_acc (scheduler minimises, we want to maximise accuracy)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=10
    )

    # ── Training state ────────────────────────────────────────────────────────
    best_min_hard_acc = 0.0
    patience_counter  = 0
    history           = []
    checkpoint_path   = os.path.join(args.output_dir, 'best_model.pt')
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("STAGE 2: GNN-UC TRAINING")
    log.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        # BUG 14 FIX: iterate over pre-cached GPU tensors, no .to(device) needed
        random.shuffle(train_cache)
        for batch in train_cache:
            optimiser.zero_grad()
            logits = model(batch)
            loss   = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / max(len(train_cache), 1)
        elapsed        = time.time() - t0

        val_metrics = evaluate(model, val_cache, criterion)

        # ── BUG 12 FIX: hard-gen tracking INSIDE epoch loop ──────────────────
        hard_gen_indices = [2, 3, 4, 5]
        hard_gen_accs    = [val_metrics['per_gen_acc'][g] for g in hard_gen_indices]
        min_hard_acc     = min(hard_gen_accs)

        # ── BUG 13 FIX: single scheduler.step() per epoch ────────────────────
        old_lr = optimiser.param_groups[0]['lr']
        scheduler.step(-min_hard_acc)   # minimise negative = maximise accuracy
        new_lr = optimiser.param_groups[0]['lr']
        lr_tag = f" [lr {old_lr:.1e}→{new_lr:.1e}]" if new_lr != old_lr else ""

        # Capacity-weighted accuracy
        total_pmax     = sum(PMAX)
        cap_weight_acc = sum(
            val_metrics['per_gen_acc'][g] * PMAX[g] / total_pmax
            for g in range(train_ds.n_gen)
        )

        log.info(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"train={avg_train_loss:.4f} | "
            f"val={val_metrics['loss']:.4f} | "
            f"acc={val_metrics['accuracy']:.4f} | "
            f"cap_w={cap_weight_acc:.4f} | "
            f"min_hard={min_hard_acc:.4f} | "
            f"hamming={val_metrics['hamming']:.1f} | "
            f"{elapsed:.1f}s{lr_tag}"
        )
        per_gen_str = ' '.join(
            f"g{g}:{v:.3f}" for g, v in enumerate(val_metrics['per_gen_acc'])
        )
        log.info(f"  per-gen: {per_gen_str}")

        history.append({
            'epoch'            : epoch,
            'train_loss'       : avg_train_loss,
            'val_loss'         : val_metrics['loss'],
            'val_accuracy'     : val_metrics['accuracy'],
            'val_hamming'      : val_metrics['hamming'],
            'per_gen_acc'      : val_metrics['per_gen_acc'],
            'min_hard_acc'     : min_hard_acc,
            'cap_weighted_acc' : cap_weight_acc,
        })

        # ── BUG 15 FIX: actually save the checkpoint ──────────────────────────
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
            }, checkpoint_path)
            log.info(f"  ✓ checkpoint saved (min_hard_acc={min_hard_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {args.patience} epochs, "
                    f"best min_hard_acc={best_min_hard_acc:.4f})"
                )
                break

    # ── Save history ───────────────────────────────────────────────────────────
    with open(os.path.join(args.output_dir, 'train_metrics.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # ── Final test evaluation ──────────────────────────────────────────────────
    log.info("\nLoading best checkpoint for test evaluation...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    log.info(f"Loaded checkpoint from epoch {ckpt['epoch']} "
             f"(min_hard_acc={ckpt['min_hard_acc']:.4f})")

    test_ds    = UCGraphDataset(args.dataset_dir, split='test', feat_stats=feat_stats)
    test_cache = [d.to(device) for d in test_ds]
    test_metrics = evaluate(model, test_cache, criterion)

    log.info("=" * 60)
    log.info("TEST SET RESULTS")
    log.info("=" * 60)
    log.info(f"  Loss     : {test_metrics['loss']:.4f}")
    log.info(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    log.info(f"  Hamming  : {test_metrics['hamming']:.1f} bits/sample")
    for g, acc in enumerate(test_metrics['per_gen_acc']):
        log.info(f"  Gen {g}    : {acc:.4f}")

    total_pmax    = sum(PMAX)
    cap_w_acc     = sum(
        test_metrics['per_gen_acc'][g] * PMAX[g] / total_pmax
        for g in range(len(PMAX))
    )
    log.info(f"  Capacity-weighted acc: {cap_w_acc:.4f}")

    with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    mean_acc    = test_metrics['accuracy']
    min_gen_acc = min(test_metrics['per_gen_acc'])

    log.info("=" * 60)
    if mean_acc >= 0.88 and min_gen_acc >= 0.78:
        log.info(f"✓ STAGE 2 PASSED — proceed to Stage 3 (fuzzy MILP)")
        log.info(f"  mean={mean_acc:.4f} >= 0.88  |  min={min_gen_acc:.4f} >= 0.78")
        log.info(f"  cap_weighted={cap_w_acc:.4f}")
    else:
        log.info(f"✗ STAGE 2 NOT YET PASSED")
        log.info(f"  mean={mean_acc:.4f} (need 0.88)  |  min={min_gen_acc:.4f} (need 0.78)")
    log.info("=" * 60)

    return test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GNN-UC (Stage 2)')
    parser.add_argument('--dataset_dir',  type=str,   default='dataset_output')
    parser.add_argument('--output_dir',   type=str,   default='model_output')
    parser.add_argument('--d_h',          type=int,   default=64)
    parser.add_argument('--n_heads',      type=int,   default=4)
    parser.add_argument('--gat_layers',   type=int,   default=2)
    parser.add_argument('--lstm_hidden',  type=int,   default=64)
    parser.add_argument('--dropout',      type=float, default=0.10)
    parser.add_argument('--epochs',       type=int,   default=300)
    parser.add_argument('--batch_size',   type=int,   default=1)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience',     type=int,   default=40)
    args = parser.parse_args()
    train(args)