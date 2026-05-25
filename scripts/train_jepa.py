"""
IMU-JEPA pretraining script.

Trains IMU-JEPA on IMU data (self-supervised, no labels needed).
After pretraining, evaluates the frozen context encoder via linear probe.

Usage:
    python -m scripts.train_jepa
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

from src.models.jepa import build_jepa
from src.data.dataset import HARDataset, get_har_dataloaders
from src.analysis.evaluator import linear_probe, format_results


def train_jepa(
    jepa: nn.Module,
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
    ema_momentum_schedule: tuple = (0.996, 1.0),
):
    """
    Pretrain IMU-JEPA with latent prediction objective.

    Returns:
        encoder: the pretrained context encoder (frozen-ready)
        history: list of per-epoch dicts
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    jepa = jepa.to(device)
    jepa.train()

    # Only optimize context encoder + predictor (target encoder is frozen)
    trainable_params = [
        p for p in jepa.parameters() if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
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
        jepa.train()
        epoch_pred_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0

        # Update EMA momentum (cosine schedule)
        if jepa.target_mode == "ema":
            progress = (epoch - 1) / max(epochs - 1, 1)
            m_start, m_end = ema_momentum_schedule
            momentum = m_end - (m_end - m_start) * (1 + np.cos(np.pi * progress)) / 2
            jepa.set_ema_momentum(momentum)

        for x, _ in train_loader:
            x = x.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, info = jepa(x)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_pred_loss += info["pred_loss"]
            epoch_reg_loss += info["reg_loss"]
            epoch_total_loss += info["total_loss"]
            n_batches += 1

        scheduler.step()

        avg_pred = epoch_pred_loss / n_batches
        avg_reg = epoch_reg_loss / n_batches
        avg_total = epoch_total_loss / n_batches
        lr_current = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_start

        entry = {
            "epoch": epoch,
            "pred_loss": round(avg_pred, 6),
            "reg_loss": round(avg_reg, 6),
            "total_loss": round(avg_total, 6),
            "lr": round(lr_current, 8),
            "elapsed_s": round(elapsed, 1),
        }
        if jepa.target_mode == "ema":
            entry["ema_momentum"] = round(momentum, 6)
        history.append(entry)

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            extra = f"ema_m={momentum:.4f} | " if jepa.target_mode == "ema" else ""
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"pred={avg_pred:.6f} | "
                f"reg={avg_reg:.6f} | "
                f"{extra}"
                f"lr={lr_current:.2e} | "
                f"time={elapsed:.0f}s"
            )

        # Checkpoint best model
        if avg_total < best_loss and checkpoint_dir is not None:
            best_loss = avg_total
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": jepa.context_encoder.state_dict(),
                "predictor_state_dict": jepa.predictor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_total,
            }, checkpoint_dir / "best.pt")

    # Save final checkpoint
    if checkpoint_dir is not None:
        torch.save({
            "epoch": epochs,
            "encoder_state_dict": jepa.context_encoder.state_dict(),
            "predictor_state_dict": jepa.predictor.state_dict(),
            "loss": avg_total,
        }, checkpoint_dir / "final.pt")

    print(f"\nPretraining complete. Best loss: {best_loss:.6f}")
    return jepa.context_encoder, history


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
    TARGET_MODE = "sigreg"
    USE_AMP = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed" / "uci_har"
    exp_dir = project_root / "experiments" / "exp_002_jepa_ucihar"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("IMU-JEPA Pretraining — EXP-002")
    print("=" * 60)
    print(f"Device:      {device}")
    print(f"Dataset:     UCI-HAR")
    print(f"Epochs:      {EPOCHS}")
    print(f"Batch size:  {BATCH_SIZE}")
    print(f"Mask ratio:  {MASK_RATIO}")
    print(f"Patch len:   {PATCH_LEN}")
    print(f"Target mode: {TARGET_MODE}")
    print(f"LR:          {LR}")
    print(f"Seed:        {SEED}")
    print()

    # --- Data ---
    train_loader, test_loader = get_har_dataloaders(
        data_dir, batch_size=BATCH_SIZE, num_workers=0,
    )
    train_ds = train_loader.dataset
    print(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"Test:  {len(test_loader.dataset)} samples")
    print()

    # --- Model ---
    jepa = build_jepa(
        n_channels=train_ds.n_channels,
        seq_length=train_ds.seq_length,
        patch_len=PATCH_LEN,
        target_mode=TARGET_MODE,
        mask_ratio=MASK_RATIO,
    )
    print(f"Trainable params: {jepa.num_parameters:,}")
    print(f"  Encoder:        {jepa.encoder_parameters:,}")
    print(f"  Predictor:      {jepa.predictor_parameters:,}")
    print()

    # --- Pretrain ---
    encoder, history = train_jepa(
        jepa=jepa,
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
    print(format_results(results, "IMU-JEPA Pretrained"))
    print(f"Time: {results['total_time_s']:.1f}s")

    # --- Compare to baselines ---
    print("\n--- Comparison ---")
    print(f"Random encoder:  acc=0.7411, macro_f1=0.7461")
    print(f"MAE pretrained:  acc=0.8012, macro_f1=0.8083  (EXP-001)")
    print(f"JEPA pretrained: acc={results['accuracy']:.4f}, "
          f"macro_f1={results['macro_f1']:.4f}  (EXP-002)")
    delta_mae = results["macro_f1"] - 0.8083
    print(f"JEPA vs MAE delta: {delta_mae:+.4f} macro_f1")

    # --- Save results ---
    output = {
        "experiment": "EXP-002",
        "name": f"IMU-JEPA ({TARGET_MODE}) pretraining on UCI-HAR",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LR, "weight_decay": WEIGHT_DECAY,
            "warmup_epochs": WARMUP_EPOCHS, "mask_ratio": MASK_RATIO,
            "patch_len": PATCH_LEN, "target_mode": TARGET_MODE,
            "use_amp": USE_AMP,
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
