"""
EXP-007: Fine-tune evaluation — within-dataset and cross-dataset.

Compares MAE, JEPA, and Hybrid (α=0.5) with full fine-tuning (not just
linear probe). This typically boosts all methods and may change rankings.

Runs:
  1. Within-dataset on UCI-HAR (3 seeds) — fast sanity check
  2. Cross-dataset leave-one-out (5 splits) — main result

Usage:
    python -m scripts.finetune_eval
"""

import json
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

from src.models.mae import build_mae
from src.models.jepa import build_jepa
from src.models.hybrid import build_hybrid
from src.data.dataset import get_har_dataloaders
from src.analysis.evaluator import linear_probe
from scripts.train_mae import train_mae
from scripts.train_jepa import train_jepa
from scripts.train_hybrid import train_hybrid


DATASETS = ["uci_har", "pamap2", "dsads", "mhealth", "usc_had"]

# Pretraining config
SEED = 42
PT_EPOCHS = 100
BATCH_SIZE = 128
LR = 1.5e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 10
MASK_RATIO = 0.75
PATCH_LEN = 16
USE_AMP = True
ALPHA = 0.5

# Fine-tune config
FT_EPOCHS = 50
FT_LR = 5e-5
FT_WD = 0.01


def finetune_and_eval(
    encoder: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_classes: int,
    epochs: int = 50,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """Fine-tune encoder + linear head, return test metrics."""
    torch.manual_seed(seed)

    # Deep copy encoder so we don't mutate the original
    enc = copy.deepcopy(encoder).to(device)
    head = nn.Linear(enc.embed_dim, n_classes).to(device)
    enc.train()
    head.train()

    optimizer = torch.optim.AdamW(
        list(enc.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    for epoch in range(epochs):
        enc.train()
        head.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda"):
                emb = enc.get_cls_token(x)
                logits = head(emb)
                loss = criterion(logits, y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

    # Evaluate
    enc.eval()
    head.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            with torch.amp.autocast("cuda"):
                emb = enc.get_cls_token(x)
                logits = head(emb)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(y.numpy())

    pred_y = np.concatenate(all_preds)
    test_y = np.concatenate(all_labels)

    return {
        "accuracy": float(accuracy_score(test_y, pred_y)),
        "macro_f1": float(f1_score(test_y, pred_y, average="macro")),
        "weighted_f1": float(f1_score(test_y, pred_y, average="weighted")),
    }


def pretrain_method(method, pretrain_loader, device):
    """Pretrain and return the encoder."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    n_channels, seq_length = 6, 128

    if method == "mae":
        model = build_mae(n_channels=n_channels, seq_length=seq_length,
                          patch_len=PATCH_LEN, mask_ratio=MASK_RATIO)
        encoder, _ = train_mae(model, pretrain_loader, epochs=PT_EPOCHS, lr=LR,
                               weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS,
                               device=device, checkpoint_dir=None, log_every=50,
                               use_amp=USE_AMP, seed=SEED)
    elif method == "jepa":
        model = build_jepa(n_channels=n_channels, seq_length=seq_length,
                           patch_len=PATCH_LEN, target_mode="sigreg",
                           mask_ratio=MASK_RATIO)
        encoder, _ = train_jepa(model, pretrain_loader, epochs=PT_EPOCHS, lr=LR,
                                weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS,
                                device=device, checkpoint_dir=None, log_every=50,
                                use_amp=USE_AMP, seed=SEED)
    elif method == "hybrid":
        model = build_hybrid(n_channels=n_channels, seq_length=seq_length,
                             patch_len=PATCH_LEN, alpha=ALPHA, mask_ratio=MASK_RATIO)
        encoder, _ = train_hybrid(model, pretrain_loader, epochs=PT_EPOCHS, lr=LR,
                                  weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS,
                                  device=device, checkpoint_dir=None, log_every=50,
                                  use_amp=USE_AMP, seed=SEED)
    return encoder


def load_all_datasets(processed_dir):
    datasets = {}
    for name in DATASETS:
        ds_dir = processed_dir / name
        if not (ds_dir / "X.npy").exists():
            continue
        datasets[name] = {
            "X": np.load(ds_dir / "X.npy"),
            "y": np.load(ds_dir / "y.npy"),
            "subjects": np.load(ds_dir / "subjects.npy"),
            "split": np.load(ds_dir / "split.npy"),
        }
    return datasets


def make_pretrain_loader(datasets, exclude, batch_size):
    all_X = [ds["X"] for name, ds in datasets.items() if name != exclude]
    X_pool = np.concatenate(all_X, axis=0)
    X_tensor = torch.from_numpy(X_pool.transpose(0, 2, 1))
    y_dummy = torch.zeros(len(X_tensor), dtype=torch.long)
    return DataLoader(TensorDataset(X_tensor, y_dummy), batch_size=batch_size,
                      shuffle=True, num_workers=0, pin_memory=True, drop_last=True)


def make_eval_loaders(ds_dict, batch_size):
    X, y, split = ds_dict["X"], ds_dict["y"], ds_dict["split"]
    X_tr = torch.from_numpy(X[split == 0].transpose(0, 2, 1))
    y_tr = torch.from_numpy(y[split == 0])
    X_te = torch.from_numpy(X[split == 1].transpose(0, 2, 1))
    y_te = torch.from_numpy(y[split == 1])
    train_l = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size,
                          shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    test_l = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size,
                         shuffle=False, num_workers=0, pin_memory=True)
    return train_l, test_l, int(y.max() + 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    exp_dir = project_root / "experiments" / "exp_007_finetune"
    exp_dir.mkdir(parents=True, exist_ok=True)

    methods = ["mae", "jepa", "hybrid"]

    print("=" * 70)
    print("EXP-007: Fine-Tune Evaluation")
    print("=" * 70)
    print(f"Methods: {methods}")
    print(f"FT config: {FT_EPOCHS} epochs, lr={FT_LR}, wd={FT_WD}")
    print()

    all_datasets = load_all_datasets(processed_dir)
    available = list(all_datasets.keys())

    # ============================================================
    # Part 1: Within-dataset UCI-HAR (3 seeds)
    # ============================================================
    print("\n" + "=" * 70)
    print("Part 1: Within-Dataset UCI-HAR (Fine-Tune, 3 Seeds)")
    print("=" * 70)

    uci_data_dir = processed_dir / "uci_har"
    uci_train, uci_test = get_har_dataloaders(uci_data_dir, batch_size=BATCH_SIZE)
    uci_n_classes = uci_train.dataset.n_classes

    within_results = {m: [] for m in methods}
    within_seeds = [42, 123, 456]

    for method in methods:
        print(f"\n--- {method.upper()} ---")
        for seed in within_seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            encoder = pretrain_method(method, uci_train, device)
            ft_res = finetune_and_eval(
                encoder, uci_train, uci_test, uci_n_classes,
                epochs=FT_EPOCHS, lr=FT_LR, weight_decay=FT_WD,
                device=device, seed=seed,
            )
            lp_res = linear_probe(encoder.to(device), uci_train, uci_test, device=device)

            within_results[method].append({
                "seed": seed,
                "ft_macro_f1": ft_res["macro_f1"],
                "ft_accuracy": ft_res["accuracy"],
                "lp_macro_f1": lp_res["macro_f1"],
                "lp_accuracy": lp_res["accuracy"],
            })
            print(f"  seed={seed}: FT f1={ft_res['macro_f1']:.4f}, "
                  f"LP f1={lp_res['macro_f1']:.4f}")

    # Within-dataset summary
    print(f"\n{'='*70}")
    print("Within-Dataset UCI-HAR Summary (mean ± std)")
    print(f"{'='*70}")
    print(f"{'Method':<10} {'FT Macro-F1':>18} {'LP Macro-F1':>18} {'FT-LP Δ':>10}")
    print("-" * 60)
    for m in methods:
        ft_vals = [r["ft_macro_f1"] for r in within_results[m]]
        lp_vals = [r["lp_macro_f1"] for r in within_results[m]]
        ft_mean, ft_std = np.mean(ft_vals), np.std(ft_vals)
        lp_mean, lp_std = np.mean(lp_vals), np.std(lp_vals)
        print(f"{m.upper():<10} {ft_mean:.4f} ± {ft_std:.4f}   "
              f"{lp_mean:.4f} ± {lp_std:.4f}   {ft_mean - lp_mean:+.4f}")

    # ============================================================
    # Part 2: Cross-dataset (single seed, fine-tune vs linear probe)
    # ============================================================
    print(f"\n\n{'='*70}")
    print("Part 2: Cross-Dataset Leave-One-Out (Fine-Tune vs Linear Probe)")
    print(f"{'='*70}")

    cross_results = {m: {} for m in methods}

    for held_out in available:
        print(f"\n--- Held-out: {held_out.upper()} ---")
        pretrain_loader = make_pretrain_loader(all_datasets, held_out, BATCH_SIZE)
        eval_train, eval_test, n_classes = make_eval_loaders(
            all_datasets[held_out], BATCH_SIZE
        )

        for method in methods:
            print(f"  {method.upper()}: ", end="", flush=True)
            encoder = pretrain_method(method, pretrain_loader, device)

            # Fine-tune
            ft_res = finetune_and_eval(
                encoder, eval_train, eval_test, n_classes,
                epochs=FT_EPOCHS, lr=FT_LR, weight_decay=FT_WD,
                device=device, seed=SEED,
            )
            # Linear probe
            lp_res = linear_probe(encoder.to(device), eval_train, eval_test, device=device)

            cross_results[method][held_out] = {
                "ft_macro_f1": ft_res["macro_f1"],
                "ft_accuracy": ft_res["accuracy"],
                "lp_macro_f1": lp_res["macro_f1"],
                "lp_accuracy": lp_res["accuracy"],
            }
            print(f"FT={ft_res['macro_f1']:.4f}, LP={lp_res['macro_f1']:.4f}")

    # Cross-dataset summary
    print(f"\n\n{'='*70}")
    print("CROSS-DATASET FINE-TUNE RESULTS (Macro-F1)")
    print(f"{'='*70}")
    header = f"{'Held-out':<12}"
    for m in methods:
        header += f" {m.upper()+'-FT':>10}"
    for m in methods:
        header += f" {m.upper()+'-LP':>10}"
    print(header)
    print("-" * 84)

    ft_avgs = {m: [] for m in methods}
    lp_avgs = {m: [] for m in methods}

    for ds in available:
        line = f"{ds:<12}"
        for m in methods:
            ft_f1 = cross_results[m][ds]["ft_macro_f1"]
            ft_avgs[m].append(ft_f1)
            line += f" {ft_f1:>10.4f}"
        for m in methods:
            lp_f1 = cross_results[m][ds]["lp_macro_f1"]
            lp_avgs[m].append(lp_f1)
            line += f" {lp_f1:>10.4f}"
        print(line)

    print("-" * 84)
    line = f"{'AVERAGE':<12}"
    for m in methods:
        line += f" {np.mean(ft_avgs[m]):>10.4f}"
    for m in methods:
        line += f" {np.mean(lp_avgs[m]):>10.4f}"
    print(line)

    # Best method
    ft_means = {m: float(np.mean(ft_avgs[m])) for m in methods}
    best_ft = max(ft_means, key=ft_means.get)
    print(f"\nBest cross-dataset FT: {best_ft.upper()} ({ft_means[best_ft]:.4f})")
    print(f"Hybrid FT vs MAE FT: {ft_means['hybrid'] - ft_means['mae']:+.4f}")

    # --- Save ---
    output = {
        "experiment": "EXP-007",
        "timestamp": datetime.now().isoformat(),
        "within_dataset": {
            m: {
                "per_seed": within_results[m],
                "ft_mean": float(np.mean([r["ft_macro_f1"] for r in within_results[m]])),
                "ft_std": float(np.std([r["ft_macro_f1"] for r in within_results[m]])),
                "lp_mean": float(np.mean([r["lp_macro_f1"] for r in within_results[m]])),
                "lp_std": float(np.std([r["lp_macro_f1"] for r in within_results[m]])),
            }
            for m in methods
        },
        "cross_dataset": {
            m: {ds: cross_results[m][ds] for ds in available}
            for m in methods
        },
        "cross_dataset_ft_avg": ft_means,
    }
    with open(exp_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {exp_dir / 'results.json'}")


if __name__ == "__main__":
    main()
