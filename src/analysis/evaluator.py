"""
Linear probe and fine-tune evaluation for pretrained encoders.

Standard SSL evaluation protocol:
  1. Freeze encoder, extract CLS embeddings for all train/test samples
  2. Train a linear classifier (LogisticRegression or nn.Linear) on embeddings
  3. Report accuracy, macro-F1, weighted-F1

Also supports full fine-tuning where encoder weights are updated.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from typing import Optional


@torch.no_grad()
def extract_embeddings(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract CLS embeddings from a frozen encoder.

    Returns:
        embeddings: (N, D) numpy array
        labels: (N,) numpy array
    """
    encoder.eval()
    all_embs = []
    all_labels = []

    for x, y in dataloader:
        x = x.to(device)
        emb = encoder.get_cls_token(x)  # (B, D)
        all_embs.append(emb.cpu().numpy())
        all_labels.append(y.numpy())

    embeddings = np.concatenate(all_embs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return embeddings, labels


def linear_probe(
    encoder: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cuda",
    max_iter: int = 1000,
) -> dict:
    """
    Evaluate encoder with a sklearn LogisticRegression linear probe.

    This is fast, deterministic (given embeddings), and the standard
    protocol used by I-JEPA, TS-JEPA, Laya, and most SSL papers.

    Returns:
        dict with accuracy, macro_f1, weighted_f1, and timing info.
    """
    t0 = time.time()

    # Extract embeddings
    train_emb, train_y = extract_embeddings(encoder, train_loader, device)
    test_emb, test_y = extract_embeddings(encoder, test_loader, device)

    t_extract = time.time() - t0

    # Train linear classifier
    clf = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        multi_class="multinomial",
        C=1.0,
        n_jobs=-1,
    )
    clf.fit(train_emb, train_y)

    # Predict
    pred_y = clf.predict(test_emb)

    # Metrics
    acc = accuracy_score(test_y, pred_y)
    macro_f1 = f1_score(test_y, pred_y, average="macro")
    weighted_f1 = f1_score(test_y, pred_y, average="weighted")

    t_total = time.time() - t0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_train": len(train_y),
        "n_test": len(test_y),
        "embed_dim": train_emb.shape[1],
        "extract_time_s": round(t_extract, 2),
        "total_time_s": round(t_total, 2),
    }


def finetune_eval(
    encoder: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_classes: int,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """
    Fine-tune encoder + linear head end-to-end.

    Returns:
        dict with accuracy, macro_f1, weighted_f1, and training history.
    """
    torch.manual_seed(seed)

    embed_dim = encoder.embed_dim
    head = nn.Linear(embed_dim, n_classes).to(device)
    encoder = encoder.to(device)
    encoder.train()

    # Optimizer over encoder + head
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = []
    t0 = time.time()

    for epoch in range(epochs):
        encoder.train()
        head.train()
        epoch_loss = 0.0
        n_correct = 0
        n_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            emb = encoder.get_cls_token(x)  # (B, D)
            logits = head(emb)              # (B, n_classes)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            n_correct += (logits.argmax(dim=1) == y).sum().item()
            n_total += x.size(0)

        scheduler.step()
        train_acc = n_correct / n_total
        avg_loss = epoch_loss / n_total
        history.append({"epoch": epoch + 1, "loss": avg_loss, "train_acc": train_acc})

    # Evaluate
    encoder.eval()
    head.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            emb = encoder.get_cls_token(x)
            logits = head(emb)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(y.numpy())

    pred_y = np.concatenate(all_preds)
    test_y = np.concatenate(all_labels)

    acc = accuracy_score(test_y, pred_y)
    macro_f1 = f1_score(test_y, pred_y, average="macro")
    weighted_f1 = f1_score(test_y, pred_y, average="weighted")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "epochs": epochs,
        "final_train_loss": history[-1]["loss"],
        "final_train_acc": history[-1]["train_acc"],
        "total_time_s": round(time.time() - t0, 2),
        "seed": seed,
    }


def multi_seed_eval(
    eval_fn,
    seeds: list[int] = [42, 123, 456],
    **kwargs,
) -> dict:
    """
    Run evaluation across multiple seeds, report mean +/- std.

    Args:
        eval_fn: callable that accepts seed=... and returns a dict with metrics.
        seeds: list of random seeds.
        **kwargs: passed to eval_fn.

    Returns:
        dict with per-seed results and aggregated mean/std.
    """
    results_per_seed = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        result = eval_fn(seed=seed, **kwargs)
        result["seed"] = seed
        results_per_seed.append(result)
        print(f"  Seed {seed}: acc={result['accuracy']:.4f}, "
              f"macro_f1={result['macro_f1']:.4f}")

    # Aggregate
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    agg = {}
    for m in metrics:
        vals = [r[m] for r in results_per_seed]
        agg[f"{m}_mean"] = np.mean(vals)
        agg[f"{m}_std"] = np.std(vals)

    return {
        "per_seed": results_per_seed,
        "aggregate": agg,
        "seeds": seeds,
    }


def format_results(results: dict, name: str = "") -> str:
    """Pretty-print evaluation results."""
    lines = []
    if name:
        lines.append(f"=== {name} ===")

    if "aggregate" in results:
        agg = results["aggregate"]
        lines.append(
            f"Accuracy:    {agg['accuracy_mean']:.4f} +/- {agg['accuracy_std']:.4f}"
        )
        lines.append(
            f"Macro-F1:    {agg['macro_f1_mean']:.4f} +/- {agg['macro_f1_std']:.4f}"
        )
        lines.append(
            f"Weighted-F1: {agg['weighted_f1_mean']:.4f} +/- {agg['weighted_f1_std']:.4f}"
        )
    else:
        lines.append(f"Accuracy:    {results['accuracy']:.4f}")
        lines.append(f"Macro-F1:    {results['macro_f1']:.4f}")
        lines.append(f"Weighted-F1: {results['weighted_f1']:.4f}")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from src.data.dataset import HARDataset, get_har_dataloaders
    from src.models.vit import vit_small

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "uci_har"

    # Load data
    train_loader, test_loader = get_har_dataloaders(data_dir, batch_size=64)
    train_ds = train_loader.dataset
    print(f"Train: {len(train_ds)}, Test: {len(test_loader.dataset)}")
    print(f"Classes: {train_ds.n_classes}, Channels: {train_ds.n_channels}, "
          f"Seq len: {train_ds.seq_length}")

    # Random encoder (no pretraining -- baseline sanity check)
    encoder = vit_small(
        n_channels=train_ds.n_channels,
        seq_length=train_ds.seq_length,
        patch_len=16,
    ).to(device)
    print(f"\nEncoder: {encoder.num_parameters:,} params")

    # Linear probe on random encoder
    print("\n--- Linear Probe (random encoder) ---")
    results = linear_probe(encoder, train_loader, test_loader, device=device)
    print(format_results(results, "Random Encoder Linear Probe"))
    print(f"Time: {results['total_time_s']:.1f}s")
