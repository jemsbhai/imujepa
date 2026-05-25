"""
EXP-004: Cross-dataset leave-one-out evaluation.

For each of 5 datasets, pool the other 4 for pretraining (self-supervised,
no labels), then evaluate on the held-out dataset via linear probe.

This is the standard protocol used by MoPFormer (NeurIPS 2025) and is the
key experiment for demonstrating cross-domain generalization.

Usage:
    python -m scripts.cross_dataset_eval
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
from src.data.dataset import HARDataset, get_har_dataloaders
from src.analysis.evaluator import linear_probe
from scripts.train_mae import train_mae
from scripts.train_jepa import train_jepa


DATASETS = ["uci_har", "pamap2", "dsads", "mhealth", "usc_had"]

# --- Config ---
SEED = 42
EPOCHS = 100
BATCH_SIZE = 128
LR = 1.5e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 10
MASK_RATIO = 0.75
PATCH_LEN = 16
USE_AMP = True


def load_all_datasets(processed_dir: Path):
    """Load all datasets into a dict."""
    datasets = {}
    for name in DATASETS:
        ds_dir = processed_dir / name
        if not (ds_dir / "X.npy").exists():
            print(f"  WARNING: {name} not available, skipping.")
            continue
        X = np.load(ds_dir / "X.npy")  # (N, T, C)
        y = np.load(ds_dir / "y.npy")
        subjects = np.load(ds_dir / "subjects.npy")
        split = np.load(ds_dir / "split.npy")
        meta = dict(np.load(ds_dir / "meta.npz", allow_pickle=True))
        datasets[name] = {
            "X": X, "y": y, "subjects": subjects,
            "split": split, "meta": meta,
        }
        print(f"  {name:12s}: {X.shape[0]:6d} windows, "
              f"{len(np.unique(y)):2d} classes")
    return datasets


def make_pretrain_loader(datasets: dict, exclude: str, batch_size: int):
    """Pool all datasets except `exclude` into a single pretraining DataLoader.
    No labels needed — just the raw windows for self-supervised learning."""
    all_X = []
    for name, ds in datasets.items():
        if name == exclude:
            continue
        all_X.append(ds["X"])

    X_pool = np.concatenate(all_X, axis=0)  # (N_total, T, C)
    # Transpose to channels-first: (N, T, C) -> (N, C, T)
    X_tensor = torch.from_numpy(X_pool.transpose(0, 2, 1))
    # Dummy labels (not used in SSL, but DataLoader expects them)
    y_dummy = torch.zeros(len(X_tensor), dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_dummy)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    return loader, len(X_tensor)


def make_eval_loaders(ds_dict: dict, batch_size: int):
    """Create train/test DataLoaders for the held-out evaluation dataset."""
    X = ds_dict["X"]  # (N, T, C)
    y = ds_dict["y"]
    split = ds_dict["split"]

    X_train = torch.from_numpy(X[split == 0].transpose(0, 2, 1))
    y_train = torch.from_numpy(y[split == 0])
    X_test = torch.from_numpy(X[split == 1].transpose(0, 2, 1))
    y_test = torch.from_numpy(y[split == 1])

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    # Attach n_classes for compatibility
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    return train_loader, test_loader


def run_single_split(
    method: str,
    pretrain_loader,
    eval_train_loader,
    eval_test_loader,
    n_channels: int,
    seq_length: int,
    device: str,
):
    """Pretrain on pooled data, evaluate on held-out dataset."""
    if method == "mae":
        model = build_mae(
            n_channels=n_channels, seq_length=seq_length,
            patch_len=PATCH_LEN, mask_ratio=MASK_RATIO,
        )
        encoder, history = train_mae(
            mae=model, train_loader=pretrain_loader,
            epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
            warmup_epochs=WARMUP_EPOCHS, device=device,
            checkpoint_dir=None, log_every=50,
            use_amp=USE_AMP, seed=SEED,
        )
    elif method == "jepa":
        model = build_jepa(
            n_channels=n_channels, seq_length=seq_length,
            patch_len=PATCH_LEN, target_mode="sigreg",
            mask_ratio=MASK_RATIO,
        )
        encoder, history = train_jepa(
            jepa=model, train_loader=pretrain_loader,
            epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
            warmup_epochs=WARMUP_EPOCHS, device=device,
            checkpoint_dir=None, log_every=50,
            use_amp=USE_AMP, seed=SEED,
        )
    else:
        raise ValueError(method)

    # Linear probe on held-out dataset
    encoder = encoder.to(device)

    # We need the eval loaders to return data with get_cls_token working
    # The encoder expects (B, C, T) which our loaders already provide
    results = linear_probe(encoder, eval_train_loader, eval_test_loader, device=device)
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    exp_dir = project_root / "experiments" / "exp_004_cross_dataset"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXP-004: Cross-Dataset Leave-One-Out Evaluation")
    print("=" * 60)
    print(f"Datasets: {DATASETS}")
    print(f"Methods:  MAE vs JEPA (SIGReg)")
    print(f"Protocol: Pretrain on 4, evaluate on held-out 1")
    print()

    # Load all datasets
    print("Loading datasets...")
    all_datasets = load_all_datasets(processed_dir)
    available = list(all_datasets.keys())
    print(f"Available: {available}\n")

    n_channels = 6
    seq_length = 128

    # Results storage
    results_table = {method: {} for method in ["mae", "jepa"]}

    t_total_start = time.time()

    for held_out in available:
        print(f"\n{'='*60}")
        print(f"  Held-out dataset: {held_out.upper()}")
        print(f"{'='*60}")

        # Create pretrain loader (pool everything except held-out)
        pretrain_loader, n_pretrain = make_pretrain_loader(
            all_datasets, exclude=held_out, batch_size=BATCH_SIZE,
        )
        print(f"  Pretraining on {n_pretrain} windows from "
              f"{[d for d in available if d != held_out]}")

        # Create eval loaders for held-out dataset
        eval_train, eval_test = make_eval_loaders(
            all_datasets[held_out], batch_size=BATCH_SIZE,
        )
        n_eval_train = len(eval_train.dataset)
        n_eval_test = len(eval_test.dataset)
        n_classes = int(all_datasets[held_out]["y"].max() + 1)
        print(f"  Evaluating on {held_out}: {n_eval_train} train, "
              f"{n_eval_test} test, {n_classes} classes")

        for method in ["mae", "jepa"]:
            print(f"\n  --- {method.upper()} ---")
            t0 = time.time()

            result = run_single_split(
                method=method,
                pretrain_loader=pretrain_loader,
                eval_train_loader=eval_train,
                eval_test_loader=eval_test,
                n_channels=n_channels,
                seq_length=seq_length,
                device=device,
            )

            elapsed = time.time() - t0
            result["time_s"] = round(elapsed, 1)
            results_table[method][held_out] = result

            print(f"  {method.upper()} on {held_out}: "
                  f"acc={result['accuracy']:.4f}, "
                  f"macro_f1={result['macro_f1']:.4f} "
                  f"({elapsed:.0f}s)")

    # --- Summary ---
    total_time = time.time() - t_total_start
    print(f"\n\n{'='*60}")
    print("CROSS-DATASET RESULTS (Linear Probe, Macro-F1)")
    print(f"{'='*60}")
    print(f"{'Held-out':<12} {'MAE':>10} {'JEPA':>10} {'Delta':>10}")
    print("-" * 42)

    mae_f1s, jepa_f1s = [], []
    for ds in available:
        mae_f1 = results_table["mae"][ds]["macro_f1"]
        jepa_f1 = results_table["jepa"][ds]["macro_f1"]
        delta = jepa_f1 - mae_f1
        mae_f1s.append(mae_f1)
        jepa_f1s.append(jepa_f1)
        print(f"{ds:<12} {mae_f1:>10.4f} {jepa_f1:>10.4f} {delta:>+10.4f}")

    avg_mae = np.mean(mae_f1s)
    avg_jepa = np.mean(jepa_f1s)
    avg_delta = avg_jepa - avg_mae
    print("-" * 42)
    print(f"{'AVERAGE':<12} {avg_mae:>10.4f} {avg_jepa:>10.4f} {avg_delta:>+10.4f}")

    wins = sum(j > m for j, m in zip(jepa_f1s, mae_f1s))
    print(f"\nJEPA wins on {wins}/{len(available)} datasets")
    print(f"Total time: {total_time/60:.1f} minutes")

    # --- Save ---
    output = {
        "experiment": "EXP-004",
        "name": "Cross-dataset leave-one-out: MAE vs JEPA",
        "timestamp": datetime.now().isoformat(),
        "datasets": available,
        "config": {
            "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LR, "mask_ratio": MASK_RATIO, "patch_len": PATCH_LEN,
        },
        "results": {
            method: {
                ds: {k: v for k, v in res.items()}
                for ds, res in method_results.items()
            }
            for method, method_results in results_table.items()
        },
        "summary": {
            "mae_avg_macro_f1": float(avg_mae),
            "jepa_avg_macro_f1": float(avg_jepa),
            "delta": float(avg_delta),
            "jepa_wins": wins,
            "total_time_min": round(total_time / 60, 1),
        },
    }
    results_path = exp_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
