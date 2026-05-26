"""
EXP-005: Cross-dataset leave-one-out with Hybrid JEPA+MAE.

Compares MAE, JEPA (SIGReg), and Hybrid (JEPA + alpha*MAE) across
5 HAR benchmarks in the leave-one-out protocol.

Usage:
    python -m scripts.cross_dataset_hybrid
"""

import json
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from datetime import datetime

from src.models.mae import build_mae
from src.models.jepa import build_jepa
from src.models.hybrid import build_hybrid
from src.analysis.evaluator import linear_probe
from scripts.train_mae import train_mae
from scripts.train_jepa import train_jepa
from scripts.train_hybrid import train_hybrid


DATASETS = ["uci_har", "pamap2", "dsads", "mhealth", "usc_had"]

SEED = 42
EPOCHS = 100
BATCH_SIZE = 128
LR = 1.5e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 10
MASK_RATIO = 0.75
PATCH_LEN = 16
USE_AMP = True
ALPHA = 0.25  # hybrid mixing weight


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
    ds = TensorDataset(X_tensor, y_dummy)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=0, pin_memory=True, drop_last=True), len(X_tensor)


def make_eval_loaders(ds_dict, batch_size):
    X, y, split = ds_dict["X"], ds_dict["y"], ds_dict["split"]
    X_train = torch.from_numpy(X[split == 0].transpose(0, 2, 1))
    y_train = torch.from_numpy(y[split == 0])
    X_test = torch.from_numpy(X[split == 1].transpose(0, 2, 1))
    y_test = torch.from_numpy(y[split == 1])
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size,
                               shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


def run_split(method, pretrain_loader, eval_train, eval_test, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    n_channels, seq_length = 6, 128

    if method == "mae":
        model = build_mae(n_channels=n_channels, seq_length=seq_length,
                          patch_len=PATCH_LEN, mask_ratio=MASK_RATIO)
        encoder, _ = train_mae(model, pretrain_loader, epochs=EPOCHS, lr=LR,
                               weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS,
                               device=device, checkpoint_dir=None, log_every=50,
                               use_amp=USE_AMP, seed=SEED)

    elif method == "jepa":
        model = build_jepa(n_channels=n_channels, seq_length=seq_length,
                           patch_len=PATCH_LEN, target_mode="sigreg",
                           mask_ratio=MASK_RATIO)
        encoder, _ = train_jepa(model, pretrain_loader, epochs=EPOCHS, lr=LR,
                                weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS,
                                device=device, checkpoint_dir=None, log_every=50,
                                use_amp=USE_AMP, seed=SEED)

    elif method == "hybrid":
        model = build_hybrid(n_channels=n_channels, seq_length=seq_length,
                             patch_len=PATCH_LEN, alpha=ALPHA,
                             mask_ratio=MASK_RATIO)
        encoder, _ = train_hybrid(model, pretrain_loader, epochs=EPOCHS, lr=LR,
                                  weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS,
                                  device=device, checkpoint_dir=None, log_every=50,
                                  use_amp=USE_AMP, seed=SEED)
    else:
        raise ValueError(method)

    encoder = encoder.to(device)
    return linear_probe(encoder, eval_train, eval_test, device=device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    exp_dir = project_root / "experiments" / "exp_005_hybrid_cross_dataset"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXP-005: Cross-Dataset Eval — MAE vs JEPA vs Hybrid (JEPA+MAE)")
    print("=" * 70)
    print(f"Alpha (hybrid MAE weight): {ALPHA}")
    print()

    all_datasets = load_all_datasets(processed_dir)
    available = list(all_datasets.keys())
    print(f"Datasets: {available}\n")

    methods = ["mae", "jepa", "hybrid"]
    results = {m: {} for m in methods}

    t_start = time.time()

    for held_out in available:
        print(f"\n{'='*70}")
        print(f"  Held-out: {held_out.upper()}")
        print(f"{'='*70}")

        pretrain_loader, n_pt = make_pretrain_loader(all_datasets, held_out, BATCH_SIZE)
        eval_train, eval_test = make_eval_loaders(all_datasets[held_out], BATCH_SIZE)
        print(f"  Pretrain: {n_pt} windows | Eval: {len(eval_train.dataset)}/"
              f"{len(eval_test.dataset)} train/test")

        for method in methods:
            print(f"\n  --- {method.upper()} ---")
            t0 = time.time()
            res = run_split(method, pretrain_loader, eval_train, eval_test, device)
            elapsed = time.time() - t0
            res["time_s"] = round(elapsed, 1)
            results[method][held_out] = res
            print(f"  {method.upper():>6s} => acc={res['accuracy']:.4f}, "
                  f"f1={res['macro_f1']:.4f} ({elapsed:.0f}s)")

    # --- Summary ---
    total_time = time.time() - t_start
    print(f"\n\n{'='*70}")
    print("CROSS-DATASET RESULTS (Linear Probe, Macro-F1)")
    print(f"{'='*70}")
    header = f"{'Held-out':<12}"
    for m in methods:
        header += f" {m.upper():>10}"
    header += f" {'Best':>10}"
    print(header)
    print("-" * 70)

    avgs = {m: [] for m in methods}
    wins = {m: 0 for m in methods}

    for ds in available:
        line = f"{ds:<12}"
        f1s = {}
        for m in methods:
            f1 = results[m][ds]["macro_f1"]
            f1s[m] = f1
            avgs[m].append(f1)
            line += f" {f1:>10.4f}"

        best_m = max(f1s, key=f1s.get)
        wins[best_m] += 1
        line += f" {best_m.upper():>10}"
        print(line)

    print("-" * 70)
    line = f"{'AVERAGE':<12}"
    avg_vals = {}
    for m in methods:
        avg_vals[m] = float(np.mean(avgs[m]))
        line += f" {avg_vals[m]:>10.4f}"
    best_avg = max(avg_vals, key=avg_vals.get)
    line += f" {best_avg.upper():>10}"
    print(line)

    print(f"\nWins: " + ", ".join(f"{m.upper()}={wins[m]}" for m in methods))
    print(f"Hybrid vs MAE:  {avg_vals['hybrid'] - avg_vals['mae']:+.4f}")
    print(f"Hybrid vs JEPA: {avg_vals['hybrid'] - avg_vals['jepa']:+.4f}")
    print(f"Total time: {total_time/60:.1f} minutes")

    # --- Save ---
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        "experiment": "EXP-005",
        "name": "Cross-dataset: MAE vs JEPA vs Hybrid",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LR, "mask_ratio": MASK_RATIO, "patch_len": PATCH_LEN,
            "alpha": ALPHA,
        },
        "results_macro_f1": {
            m: {ds: float(results[m][ds]["macro_f1"]) for ds in available}
            for m in methods
        },
        "results_accuracy": {
            m: {ds: float(results[m][ds]["accuracy"]) for ds in available}
            for m in methods
        },
        "summary": {
            "avg_macro_f1": {m: float(v) for m, v in avg_vals.items()},
            "wins": {m: int(v) for m, v in wins.items()},
            "total_time_min": round(total_time / 60, 1),
        },
    }
    with open(exp_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {exp_dir / 'results.json'}")


if __name__ == "__main__":
    main()
