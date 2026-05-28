"""
EXP-008: Few-shot evaluation and EXP-009: Noise robustness.

Two quick experiments that test distinctive properties of each SSL objective:
  1. Few-shot: How well do representations transfer with limited labels?
     (1%, 5%, 10%, 25%, 100% of training labels)
  2. Noise robustness: How well do frozen representations hold up under
     injected Gaussian noise at test time? (SNR = 20, 10, 5, 0 dB)

Both use cross-dataset pretrained encoders (pretrain on 4, eval on UCI-HAR)
since that's the most realistic scenario.

Usage:
    python -m scripts.fewshot_noise_eval
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

from src.models.mae import build_mae
from src.models.jepa import build_jepa
from src.models.hybrid import build_hybrid
from src.analysis.evaluator import extract_embeddings
from scripts.train_mae import train_mae
from scripts.train_jepa import train_jepa
from scripts.train_hybrid import train_hybrid


DATASETS = ["uci_har", "pamap2", "dsads", "mhealth", "usc_had"]

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


def pretrain_encoder(method, pretrain_loader, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    nc, sl = 6, 128
    if method == "mae":
        model = build_mae(n_channels=nc, seq_length=sl, patch_len=PATCH_LEN, mask_ratio=MASK_RATIO)
        enc, _ = train_mae(model, pretrain_loader, epochs=PT_EPOCHS, lr=LR,
                           weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS,
                           device=device, checkpoint_dir=None, log_every=50,
                           use_amp=USE_AMP, seed=SEED)
    elif method == "jepa":
        model = build_jepa(n_channels=nc, seq_length=sl, patch_len=PATCH_LEN,
                           target_mode="sigreg", mask_ratio=MASK_RATIO)
        enc, _ = train_jepa(model, pretrain_loader, epochs=PT_EPOCHS, lr=LR,
                            weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS,
                            device=device, checkpoint_dir=None, log_every=50,
                            use_amp=USE_AMP, seed=SEED)
    elif method == "hybrid":
        model = build_hybrid(n_channels=nc, seq_length=sl, patch_len=PATCH_LEN,
                             alpha=ALPHA, mask_ratio=MASK_RATIO)
        enc, _ = train_hybrid(model, pretrain_loader, epochs=PT_EPOCHS, lr=LR,
                              weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS,
                              device=device, checkpoint_dir=None, log_every=50,
                              use_amp=USE_AMP, seed=SEED)
    return enc


def linear_probe_from_embeddings(train_emb, train_y, test_emb, test_y):
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, n_jobs=-1)
    clf.fit(train_emb, train_y)
    pred = clf.predict(test_emb)
    return {
        "accuracy": float(accuracy_score(test_y, pred)),
        "macro_f1": float(f1_score(test_y, pred, average="macro")),
    }


# ================================================================
# Few-shot: subsample training labels
# ================================================================
def run_fewshot(encoder, train_loader, test_loader, device, fractions):
    """Extract embeddings once, then subsample train labels at various fractions."""
    encoder = encoder.to(device)
    train_emb, train_y = extract_embeddings(encoder, train_loader, device)
    test_emb, test_y = extract_embeddings(encoder, test_loader, device)

    results = {}
    for frac in fractions:
        n_total = len(train_y)
        n_sample = max(1, int(n_total * frac))

        # Stratified subsample
        rng = np.random.RandomState(SEED)
        classes = np.unique(train_y)
        indices = []
        for c in classes:
            c_idx = np.where(train_y == c)[0]
            n_c = max(1, int(len(c_idx) * frac))
            indices.extend(rng.choice(c_idx, size=n_c, replace=False).tolist())

        sub_emb = train_emb[indices]
        sub_y = train_y[indices]

        res = linear_probe_from_embeddings(sub_emb, sub_y, test_emb, test_y)
        res["n_train_samples"] = len(indices)
        results[frac] = res

    return results


# ================================================================
# Noise robustness: inject Gaussian noise at test time
# ================================================================
def add_noise_snr(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add Gaussian noise to achieve target SNR in dB."""
    signal_power = x.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(x) * noise_power.sqrt()
    return x + noise


def run_noise_robustness(encoder, train_loader, test_loader, device, snr_levels):
    """Extract clean train embeddings, then test at various noise levels."""
    encoder = encoder.to(device)
    encoder.eval()

    # Clean train embeddings (always clean — noise only at test time)
    train_emb, train_y = extract_embeddings(encoder, train_loader, device)

    results = {}
    for snr in snr_levels:
        # Extract test embeddings with noise
        all_emb, all_y = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                if snr is not None:
                    x = add_noise_snr(x, snr)
                emb = encoder.get_cls_token(x)
                all_emb.append(emb.cpu().numpy())
                all_y.append(y.numpy())

        test_emb = np.concatenate(all_emb)
        test_y = np.concatenate(all_y)

        res = linear_probe_from_embeddings(train_emb, train_y, test_emb, test_y)
        results[snr] = res

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    exp_dir_fs = project_root / "experiments" / "exp_008_fewshot"
    exp_dir_noise = project_root / "experiments" / "exp_009_noise"
    exp_dir_fs.mkdir(parents=True, exist_ok=True)
    exp_dir_noise.mkdir(parents=True, exist_ok=True)

    methods = ["mae", "jepa", "hybrid"]
    all_datasets = load_all_datasets(processed_dir)

    # We pretrain on everything EXCEPT UCI-HAR, then evaluate on UCI-HAR
    held_out = "uci_har"
    print("=" * 70)
    print("EXP-008 & EXP-009: Few-Shot + Noise Robustness")
    print("=" * 70)
    print(f"Pretrain on: {[d for d in all_datasets if d != held_out]}")
    print(f"Evaluate on: {held_out}")
    print()

    pretrain_loader = make_pretrain_loader(all_datasets, held_out, BATCH_SIZE)

    # Eval loaders for UCI-HAR
    ds = all_datasets[held_out]
    X_tr = torch.from_numpy(ds["X"][ds["split"] == 0].transpose(0, 2, 1))
    y_tr = torch.from_numpy(ds["y"][ds["split"] == 0])
    X_te = torch.from_numpy(ds["X"][ds["split"] == 1].transpose(0, 2, 1))
    y_te = torch.from_numpy(ds["y"][ds["split"] == 1])
    eval_train = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=True)
    eval_test = DataLoader(TensorDataset(X_te, y_te), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)

    # Pretrain all encoders
    encoders = {}
    for method in methods:
        print(f"\nPretraining {method.upper()}...")
        encoders[method] = pretrain_encoder(method, pretrain_loader, device)

    # ============================================================
    # EXP-008: Few-shot
    # ============================================================
    fractions = [0.01, 0.05, 0.10, 0.25, 1.0]
    print(f"\n\n{'='*70}")
    print(f"EXP-008: Few-Shot Evaluation (label fractions: {fractions})")
    print(f"{'='*70}")

    fewshot_results = {}
    for method in methods:
        fewshot_results[method] = run_fewshot(
            encoders[method], eval_train, eval_test, device, fractions
        )

    # Summary table
    print(f"\n{'Fraction':<10}", end="")
    for m in methods:
        print(f" {m.upper():>10}", end="")
    print()
    print("-" * 44)

    for frac in fractions:
        line = f"{frac:<10.0%}"
        for m in methods:
            f1 = fewshot_results[m][frac]["macro_f1"]
            line += f" {f1:>10.4f}"
        print(line)

    # Save EXP-008
    output_fs = {
        "experiment": "EXP-008",
        "timestamp": datetime.now().isoformat(),
        "held_out": held_out,
        "fractions": fractions,
        "results": {
            m: {str(f): fewshot_results[m][f] for f in fractions}
            for m in methods
        },
    }
    with open(exp_dir_fs / "results.json", "w") as f:
        json.dump(output_fs, f, indent=2)

    # ============================================================
    # EXP-009: Noise robustness
    # ============================================================
    snr_levels = [None, 20, 10, 5, 0]  # None = clean
    print(f"\n\n{'='*70}")
    print(f"EXP-009: Noise Robustness (SNR dB: {snr_levels})")
    print(f"{'='*70}")

    noise_results = {}
    for method in methods:
        noise_results[method] = run_noise_robustness(
            encoders[method], eval_train, eval_test, device, snr_levels
        )

    # Summary table
    print(f"\n{'SNR (dB)':<12}", end="")
    for m in methods:
        print(f" {m.upper():>10}", end="")
    print()
    print("-" * 44)

    for snr in snr_levels:
        label = "Clean" if snr is None else f"{snr} dB"
        line = f"{label:<12}"
        for m in methods:
            f1 = noise_results[m][snr]["macro_f1"]
            line += f" {f1:>10.4f}"
        print(line)

    # Compute degradation (clean - worst noise)
    print(f"\n{'Degradation (Clean → 0dB):'}")
    for m in methods:
        clean = noise_results[m][None]["macro_f1"]
        noisy = noise_results[m][0]["macro_f1"]
        deg = clean - noisy
        print(f"  {m.upper()}: {clean:.4f} → {noisy:.4f} (Δ={deg:+.4f})")

    # Save EXP-009
    output_noise = {
        "experiment": "EXP-009",
        "timestamp": datetime.now().isoformat(),
        "held_out": held_out,
        "snr_levels_db": [s if s is not None else "clean" for s in snr_levels],
        "results": {
            m: {
                str(s) if s is not None else "clean": noise_results[m][s]
                for s in snr_levels
            }
            for m in methods
        },
    }
    with open(exp_dir_noise / "results.json", "w") as f:
        json.dump(output_noise, f, indent=2)

    print(f"\nSaved EXP-008 to {exp_dir_fs / 'results.json'}")
    print(f"Saved EXP-009 to {exp_dir_noise / 'results.json'}")


if __name__ == "__main__":
    main()
