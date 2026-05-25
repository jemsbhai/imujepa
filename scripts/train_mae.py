"""
MAE-IMU pretraining script.

Trains a Masked Autoencoder on IMU data (self-supervised, no labels needed).
After pretraining, evaluates the frozen encoder via linear probe.

Usage:
    python -m scripts.train_mae
"""

import json
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime

from src.models.mae import build_mae
from src.data.dataset import HARDataset, get_har_dataloaders
from src.analysis.evaluator import linear_probe, format_results


def train_mae(
    mae: nn.Module,
    train_loader,
    epochs: int = 100,
    lr: float = 1.5e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 10,
    device: str = "cuda",
    checkpoint_dir: Path = None,
    log_every: int = 10,
    use_amp: bool = True,
    seed: int = 42,
):
    """
    Pretrain MAE with reconstruction objective.

    Returns:
        encoder: the pretrained encoder (frozen-ready)
        history: list of per-epoch dicts
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    mae = mae.to(device)
    mae.train()

    # Optimizer
    optimizer = torch.optim.AdamW(
        mae.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    # LR schedule: linear warmup + cosine decay
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        mae.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, _ in train_loader:  # labels ignored (self-supervised)
            x = x.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, info = mae(x)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(mae.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        lr_current = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_start

        entry = {
            "epoch": epoch,
            "loss": round(avg_loss, 6),
            "lr": round(lr_current, 8),
            "elapsed_s": round(elapsed, 1),
        }
        history.append(entry)

        # Logging
        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"loss={avg_loss:.6f} | "
                f"lr={lr_current:.2e} | "
                f"time={elapsed:.0f}s"
            )

        # Checkpoint best model
        if avg_loss < best_loss and checkpoint_dir is not None:
            best_loss = avg_loss
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": mae.encoder.state_dict(),
                "mae_state_dict": mae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": {
                    "lr": lr, "weight_decay": weight_decay,
                    "warmup_epochs": warmup_epochs, "epochs": epochs,
                    "seed": seed, "use_amp": use_amp,
                },
            }, checkpoint_dir / "best.pt")

    # Save final checkpoint
    if checkpoint_dir is not None:
        torch.save({
            "epoch": epochs,
            "encoder_state_dict": mae.encoder.state_dict(),
            "mae_state_dict": mae.state_dict(),
            "loss": avg_loss,
        }, checkpoint_dir / "final.pt")

    print(f"\nPretraining complete. Best loss: {best_loss:.6f}")
    return mae.encoder, history


def main():
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed" / "uci_har"
    exp_dir = project_root / "experiments" / "exp_001_mae_ucihar"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MAE-IMU Pretraining — EXP-001")
    print("=" * 60)
    print(f"Device:     {device}")
    print(f"Dataset:    UCI-HAR")
    print(f"Epochs:     {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Mask ratio: {MASK_RATIO}")
    print(f"Patch len:  {PATCH_LEN}")
    print(f"LR:         {LR}")
    print(f"Seed:       {SEED}")
    print()

    # --- Data ---
    train_loader, test_loader = get_har_dataloaders(
        data_dir, batch_size=BATCH_SIZE, num_workers=0,
    )
    train_ds = train_loader.dataset
    print(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"Test:  {len(test_loader.dataset)} samples")
    print(f"Shape: ({train_ds.n_channels}, {train_ds.seq_length})")
    print()

    # --- Model ---
    mae = build_mae(
        n_channels=train_ds.n_channels,
        seq_length=train_ds.seq_length,
        patch_len=PATCH_LEN,
        mask_ratio=MASK_RATIO,
    )
    print(f"MAE params:     {mae.num_parameters:,}")
    print(f"  Encoder:      {mae.encoder_parameters:,}")
    print(f"  Decoder:      {mae.decoder_parameters:,}")
    print()

    # --- Pretrain ---
    encoder, history = train_mae(
        mae=mae,
        train_loader=train_loader,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        device=device,
        checkpoint_dir=exp_dir / "checkpoints",
        log_every=10,
        use_amp=USE_AMP,
        seed=SEED,
    )

    # --- Evaluate ---
    print("\n" + "=" * 60)
    print("Linear Probe Evaluation (frozen encoder)")
    print("=" * 60)
    encoder = encoder.to(device)
    results = linear_probe(encoder, train_loader, test_loader, device=device)
    print(format_results(results, "MAE-IMU Pretrained"))
    print(f"Time: {results['total_time_s']:.1f}s")

    # --- Save results ---
    output = {
        "experiment": "EXP-001",
        "name": "MAE-IMU pretraining on UCI-HAR",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LR, "weight_decay": WEIGHT_DECAY,
            "warmup_epochs": WARMUP_EPOCHS, "mask_ratio": MASK_RATIO,
            "patch_len": PATCH_LEN, "use_amp": USE_AMP,
        },
        "linear_probe": results,
        "training_history": history,
    }
    results_path = exp_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
