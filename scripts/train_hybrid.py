"""
Hybrid JEPA+MAE training function.

Usage as module:
    from scripts.train_hybrid import train_hybrid
"""

import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


def train_hybrid(
    model: nn.Module,
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
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95),
    )

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_jepa = 0.0
        epoch_mae = 0.0
        epoch_reg = 0.0
        epoch_total = 0.0
        n_batches = 0

        for x, _ in train_loader:
            x = x.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, info = model(x)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_jepa += info["jepa_loss"]
            epoch_mae += info["mae_loss"]
            epoch_reg += info["reg_loss"]
            epoch_total += info["total_loss"]
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - t_start

        entry = {
            "epoch": epoch,
            "jepa_loss": round(epoch_jepa / n_batches, 6),
            "mae_loss": round(epoch_mae / n_batches, 6),
            "reg_loss": round(epoch_reg / n_batches, 6),
            "total_loss": round(epoch_total / n_batches, 6),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "elapsed_s": round(elapsed, 1),
        }
        history.append(entry)

        avg_total = epoch_total / n_batches
        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"jepa={entry['jepa_loss']:.4f} mae={entry['mae_loss']:.4f} "
                f"reg={entry['reg_loss']:.4f} | "
                f"lr={entry['lr']:.2e} | {elapsed:.0f}s"
            )

        if avg_total < best_loss and checkpoint_dir is not None:
            best_loss = avg_total
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": model.context_encoder.state_dict(),
                "loss": avg_total,
            }, checkpoint_dir / "best.pt")

    if checkpoint_dir is not None:
        torch.save({
            "epoch": epochs,
            "encoder_state_dict": model.context_encoder.state_dict(),
            "loss": avg_total,
        }, checkpoint_dir / "final.pt")

    print(f"\nPretraining complete. Best loss: {best_loss:.6f}")
    return model.context_encoder, history
