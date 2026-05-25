"""
EXP-003: Multi-seed evaluation of MAE vs JEPA on UCI-HAR.

Runs both methods across 3 seeds to confirm the performance gap is
statistically stable, not a lucky seed artifact.

Usage:
    python -m scripts.multiseed_eval
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from src.models.mae import build_mae
from src.models.jepa import build_jepa
from src.data.dataset import get_har_dataloaders
from src.analysis.evaluator import linear_probe, format_results


def pretrain_and_eval(
    method: str,
    train_loader,
    test_loader,
    n_channels: int,
    seq_length: int,
    seed: int,
    epochs: int = 100,
    device: str = "cuda",
):
    """Pretrain a model and evaluate via linear probe. Returns results dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    patch_len = 16
    mask_ratio = 0.75
    lr = 1.5e-4
    weight_decay = 0.05
    warmup_epochs = 10

    if method == "mae":
        from src.models.mae import build_mae
        from scripts.train_mae import train_mae

        model = build_mae(
            n_channels=n_channels, seq_length=seq_length,
            patch_len=patch_len, mask_ratio=mask_ratio,
        )
        encoder, history = train_mae(
            mae=model, train_loader=train_loader,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            warmup_epochs=warmup_epochs, device=device,
            checkpoint_dir=None, log_every=50,
            use_amp=True, seed=seed,
        )

    elif method == "jepa":
        from src.models.jepa import build_jepa
        from scripts.train_jepa import train_jepa

        model = build_jepa(
            n_channels=n_channels, seq_length=seq_length,
            patch_len=patch_len, target_mode="sigreg",
            mask_ratio=mask_ratio,
        )
        encoder, history = train_jepa(
            jepa=model, train_loader=train_loader,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            warmup_epochs=warmup_epochs, device=device,
            checkpoint_dir=None, log_every=50,
            use_amp=True, seed=seed,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Linear probe
    encoder = encoder.to(device)
    results = linear_probe(encoder, train_loader, test_loader, device=device)
    results["seed"] = seed
    results["final_train_loss"] = history[-1].get("loss", history[-1].get("total_loss"))
    return results


def main():
    SEEDS = [42, 123, 456]
    EPOCHS = 100
    BATCH_SIZE = 128
    METHODS = ["mae", "jepa"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed" / "uci_har"
    exp_dir = project_root / "experiments" / "exp_003_multiseed"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXP-003: Multi-Seed Evaluation — MAE vs JEPA on UCI-HAR")
    print("=" * 60)
    print(f"Seeds: {SEEDS}")
    print(f"Epochs: {EPOCHS}")
    print(f"Methods: {METHODS}")
    print()

    # Load data
    train_loader, test_loader = get_har_dataloaders(
        data_dir, batch_size=BATCH_SIZE, num_workers=0,
    )
    train_ds = train_loader.dataset
    n_channels = train_ds.n_channels
    seq_length = train_ds.seq_length

    all_results = {}

    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"  Method: {method.upper()}")
        print(f"{'='*60}")

        method_results = []
        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")
            result = pretrain_and_eval(
                method=method,
                train_loader=train_loader,
                test_loader=test_loader,
                n_channels=n_channels,
                seq_length=seq_length,
                seed=seed,
                epochs=EPOCHS,
                device=device,
            )
            method_results.append(result)
            print(f"  acc={result['accuracy']:.4f}, "
                  f"macro_f1={result['macro_f1']:.4f}, "
                  f"weighted_f1={result['weighted_f1']:.4f}")

        # Aggregate
        accs = [r["accuracy"] for r in method_results]
        f1s = [r["macro_f1"] for r in method_results]
        wf1s = [r["weighted_f1"] for r in method_results]

        all_results[method] = {
            "per_seed": method_results,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
            "weighted_f1_mean": float(np.mean(wf1s)),
            "weighted_f1_std": float(np.std(wf1s)),
        }

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY: Multi-Seed Results (mean ± std)")
    print("=" * 60)
    print(f"{'Method':<12} {'Accuracy':>16} {'Macro-F1':>16} {'Weighted-F1':>16}")
    print("-" * 60)
    for method in METHODS:
        r = all_results[method]
        print(
            f"{method.upper():<12} "
            f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f} "
            f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} "
            f"{r['weighted_f1_mean']:.4f} ± {r['weighted_f1_std']:.4f}"
        )

    # Delta
    delta_f1 = all_results["jepa"]["macro_f1_mean"] - all_results["mae"]["macro_f1_mean"]
    print(f"\nJEPA vs MAE delta (macro-F1): {delta_f1:+.4f}")

    # Check if JEPA wins on every seed
    mae_f1s = [r["macro_f1"] for r in all_results["mae"]["per_seed"]]
    jepa_f1s = [r["macro_f1"] for r in all_results["jepa"]["per_seed"]]
    wins = sum(j > m for j, m in zip(jepa_f1s, mae_f1s))
    print(f"JEPA wins on {wins}/{len(SEEDS)} seeds")

    # --- Save ---
    output = {
        "experiment": "EXP-003",
        "name": "Multi-seed MAE vs JEPA on UCI-HAR",
        "timestamp": datetime.now().isoformat(),
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "results": all_results,
    }
    results_path = exp_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
