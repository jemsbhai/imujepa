"""
EXP-006: Alpha ablation for Hybrid JEPA+MAE.

Tests alpha in {0.1, 0.5} cross-dataset (alpha=0.25 already done in EXP-005).
Determines the optimal mixing weight for the reconstruction auxiliary loss.

Usage:
    python -m scripts.alpha_ablation
"""

import json
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from datetime import datetime

from src.models.hybrid import build_hybrid
from src.analysis.evaluator import linear_probe
from scripts.train_hybrid import train_hybrid


DATASETS = ["uci_har", "pamap2", "dsads", "mhealth", "usc_had"]
ALPHAS = [0.1, 0.5]  # 0.25 already in EXP-005

SEED = 42
EPOCHS = 100
BATCH_SIZE = 128
LR = 1.5e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 10
MASK_RATIO = 0.75
PATCH_LEN = 16
USE_AMP = True


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
                      num_workers=0, pin_memory=True, drop_last=True)


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    exp_dir = project_root / "experiments" / "exp_006_alpha_ablation"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXP-006: Alpha Ablation for Hybrid JEPA+MAE")
    print("=" * 70)
    print(f"Alphas to test: {ALPHAS} (0.25 from EXP-005)")
    print()

    all_datasets = load_all_datasets(processed_dir)
    available = list(all_datasets.keys())

    # Load EXP-005 results for alpha=0.25
    exp005_path = project_root / "experiments" / "exp_005_hybrid_cross_dataset" / "results.json"
    with open(exp005_path) as f:
        exp005 = json.load(f)
    alpha_025_f1 = exp005["results_macro_f1"]["hybrid"]

    results = {}  # alpha -> {dataset: macro_f1}

    # Add existing alpha=0.25 results
    results[0.25] = {ds: float(alpha_025_f1[ds]) for ds in available}

    t_start = time.time()

    for alpha in ALPHAS:
        print(f"\n{'='*70}")
        print(f"  Alpha = {alpha}")
        print(f"{'='*70}")

        results[alpha] = {}

        for held_out in available:
            print(f"\n  --- Held-out: {held_out} ---")
            torch.manual_seed(SEED)
            np.random.seed(SEED)

            pretrain_loader = make_pretrain_loader(all_datasets, held_out, BATCH_SIZE)
            eval_train, eval_test = make_eval_loaders(all_datasets[held_out], BATCH_SIZE)

            model = build_hybrid(
                n_channels=6, seq_length=128, patch_len=PATCH_LEN,
                alpha=alpha, mask_ratio=MASK_RATIO,
            )
            encoder, _ = train_hybrid(
                model, pretrain_loader, epochs=EPOCHS, lr=LR,
                weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS,
                device=device, checkpoint_dir=None, log_every=50,
                use_amp=USE_AMP, seed=SEED,
            )

            encoder = encoder.to(device)
            res = linear_probe(encoder, eval_train, eval_test, device=device)
            results[alpha][held_out] = float(res["macro_f1"])

            print(f"    alpha={alpha}, {held_out}: f1={res['macro_f1']:.4f}")

    total_time = time.time() - t_start

    # --- Summary ---
    print(f"\n\n{'='*70}")
    print("ALPHA ABLATION RESULTS (Cross-Dataset, Macro-F1)")
    print(f"{'='*70}")

    all_alphas = sorted(results.keys())
    header = f"{'Held-out':<12}" + "".join(f" {'α='+str(a):>10}" for a in all_alphas)
    print(header)
    print("-" * 70)

    avgs = {a: [] for a in all_alphas}
    for ds in available:
        line = f"{ds:<12}"
        for a in all_alphas:
            f1 = results[a][ds]
            avgs[a].append(f1)
            line += f" {f1:>10.4f}"
        print(line)

    print("-" * 70)
    line = f"{'AVERAGE':<12}"
    avg_vals = {}
    for a in all_alphas:
        avg_vals[a] = float(np.mean(avgs[a]))
        line += f" {avg_vals[a]:>10.4f}"
    print(line)

    best_alpha = max(avg_vals, key=avg_vals.get)
    print(f"\nBest alpha: {best_alpha} (avg macro-F1 = {avg_vals[best_alpha]:.4f})")
    print(f"Total time: {total_time/60:.1f} minutes")

    # Also show MAE and pure JEPA from EXP-005 for context
    mae_f1 = exp005["results_macro_f1"]["mae"]
    jepa_f1 = exp005["results_macro_f1"]["jepa"]
    mae_avg = float(np.mean([mae_f1[ds] for ds in available]))
    jepa_avg = float(np.mean([jepa_f1[ds] for ds in available]))
    print(f"\nContext: MAE avg={mae_avg:.4f}, JEPA avg={jepa_avg:.4f}")
    print(f"Best hybrid vs MAE: {avg_vals[best_alpha] - mae_avg:+.4f}")
    print(f"Best hybrid vs JEPA: {avg_vals[best_alpha] - jepa_avg:+.4f}")

    # --- Save ---
    output = {
        "experiment": "EXP-006",
        "name": "Alpha ablation for Hybrid JEPA+MAE",
        "timestamp": datetime.now().isoformat(),
        "alphas_tested": all_alphas,
        "results_macro_f1": {str(a): r for a, r in results.items()},
        "averages": {str(a): float(v) for a, v in avg_vals.items()},
        "best_alpha": float(best_alpha),
        "context": {"mae_avg": mae_avg, "jepa_avg": jepa_avg},
        "total_time_min": round(total_time / 60, 1),
    }
    with open(exp_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {exp_dir / 'results.json'}")


if __name__ == "__main__":
    main()
