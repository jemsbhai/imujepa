"""
IMU-JEPA: Joint-Embedding Predictive Architecture for IMU data.

Core contribution: predicts masked patch representations in latent space
instead of reconstructing raw signals. Same ViT encoder as MAE baseline,
different objective.

Architecture:
  1. Randomly mask patches
  2. Context encoder processes VISIBLE patches only
  3. Target encoder (EMA or stop-grad+SIGReg) processes ALL patches
  4. Predictor predicts target embeddings at MASKED positions from context
  5. Loss = Smooth L1 between predicted and target embeddings (latent space)

Supports two target modes:
  - "ema": standard JEPA with exponential moving average target encoder
  - "sigreg": LeJEPA with stop-gradient + SIGReg variance/covariance regularization
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vit import ViTEncoder, TransformerBlock


class Predictor(nn.Module):
    """
    Lightweight Transformer predictor for JEPA.

    Takes context encoder outputs (visible patches) and learnable mask tokens
    (at masked positions), all with positional embeddings, and predicts the
    target encoder's representations at the masked positions.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        n_patches: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_patches = n_patches

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positional embeddings for all patch positions
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        context_emb: torch.Tensor,
        visible_idx: torch.Tensor,
        masked_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_emb: (B, n_visible, D) from context encoder (no CLS)
            visible_idx: (n_visible,) patch indices that are visible
            masked_idx:  (n_masked,) patch indices that are masked

        Returns:
            pred: (B, n_masked, D) predicted embeddings at masked positions
        """
        B, n_vis, D = context_emb.shape
        n_masked = len(masked_idx)

        # Build full sequence: visible embeddings + mask tokens
        full_seq = self.mask_token.to(context_emb.dtype).expand(B, self.n_patches, -1).clone()
        full_seq[:, visible_idx, :] = context_emb

        # Add positional embeddings
        full_seq = full_seq + self.pos_embed.to(context_emb.dtype)

        # Process through predictor Transformer
        for block in self.blocks:
            full_seq = block(full_seq)
        full_seq = self.norm(full_seq)

        # Extract predictions at masked positions only
        pred = full_seq[:, masked_idx, :]  # (B, n_masked, D)
        return pred


class IMU_JEPA(nn.Module):
    """
    Joint-Embedding Predictive Architecture for IMU data.

    Args:
        encoder: ViTEncoder instance (context encoder)
        predictor: Predictor instance
        target_mode: "ema" or "sigreg"
        ema_momentum: initial EMA momentum (only for target_mode="ema")
        lambda_var: SIGReg variance regularization weight
        lambda_cov: SIGReg covariance regularization weight
        mask_ratio: fraction of patches to mask
    """

    def __init__(
        self,
        encoder: ViTEncoder,
        predictor: Predictor,
        target_mode: str = "sigreg",
        ema_momentum: float = 0.996,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.context_encoder = encoder
        self.predictor = predictor
        self.target_mode = target_mode
        self.mask_ratio = mask_ratio
        self.n_patches = encoder.n_patches

        # Target encoder: deep copy of context encoder
        self.target_encoder = copy.deepcopy(encoder)

        if target_mode == "ema":
            # Freeze target encoder — updated via EMA only
            for p in self.target_encoder.parameters():
                p.requires_grad = False
            self.ema_momentum = ema_momentum
        elif target_mode == "sigreg":
            # Target encoder shares architecture but uses stop-gradient
            # No EMA updates needed; regularization prevents collapse
            for p in self.target_encoder.parameters():
                p.requires_grad = False
            self.lambda_var = lambda_var
            self.lambda_cov = lambda_cov
        else:
            raise ValueError(f"Unknown target_mode: {target_mode}")

    def _generate_mask(self, device: torch.device):
        """Generate random patch mask."""
        n_masked = int(self.n_patches * self.mask_ratio)
        n_visible = self.n_patches - n_masked

        perm = torch.randperm(self.n_patches, device=device)
        visible_idx = perm[:n_visible].sort().values
        masked_idx = perm[n_visible:].sort().values

        # Bool mask for encoder: True = VISIBLE
        mask = torch.zeros(self.n_patches, dtype=torch.bool, device=device)
        mask[visible_idx] = True

        return visible_idx, masked_idx, mask

    @torch.no_grad()
    def _update_target_ema(self):
        """Update target encoder via exponential moving average."""
        for param_ctx, param_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            param_tgt.data.mul_(self.ema_momentum).add_(
                param_ctx.data, alpha=1.0 - self.ema_momentum
            )

    def _sigreg_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        SIGReg (Sketched Isotropic Gaussian Regularization) from LeJEPA.

        Two terms:
          1. Variance: encourage std of each dimension >= 1
          2. Covariance: penalize off-diagonal covariance (decorrelate dims)

        Args:
            embeddings: (B, N, D) or (B*N, D) context encoder output
        """
        if embeddings.dim() == 3:
            # Flatten batch and patches: (B, N, D) -> (B*N, D)
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])

        # Center
        emb = embeddings - embeddings.mean(dim=0, keepdim=True)
        N, D = emb.shape

        # Variance term: encourage std >= 1 along each dimension
        std = emb.std(dim=0)  # (D,)
        var_loss = F.relu(1.0 - std).mean()

        # Covariance term: penalize off-diagonal entries
        cov = (emb.T @ emb) / (N - 1)  # (D, D)
        # Zero out diagonal
        off_diag = cov - torch.diag(cov.diagonal())
        cov_loss = (off_diag ** 2).sum() / D

        return self.lambda_var * var_loss + self.lambda_cov * cov_loss

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, T) raw IMU input

        Returns:
            loss: scalar JEPA loss (+ SIGReg if enabled)
            info: dict with loss components and metadata
        """
        B = x.shape[0]
        device = x.device

        # Generate mask
        visible_idx, masked_idx, mask = self._generate_mask(device)

        # --- Context encoder: visible patches only ---
        context_out = self.context_encoder(x, mask=mask)  # (B, n_vis+1, D)
        if self.context_encoder.use_cls:
            context_patches = context_out[:, 1:, :]  # (B, n_vis, D) — no CLS
        else:
            context_patches = context_out

        # --- Target encoder: ALL patches, no gradient ---
        with torch.no_grad():
            if self.target_mode == "sigreg":
                # Copy current context encoder weights to target (stop-grad)
                for param_ctx, param_tgt in zip(
                    self.context_encoder.parameters(),
                    self.target_encoder.parameters(),
                ):
                    param_tgt.data.copy_(param_ctx.data)

            target_out = self.target_encoder(x)  # (B, N+1, D) all patches
            if self.target_encoder.use_cls:
                target_patches = target_out[:, 1:, :]  # (B, N, D)
            else:
                target_patches = target_out

            # Extract targets at masked positions
            targets = target_patches[:, masked_idx, :]  # (B, n_masked, D)

        # --- Predictor: predict masked embeddings from context ---
        predictions = self.predictor(
            context_patches, visible_idx, masked_idx
        )  # (B, n_masked, D)

        # --- Prediction loss: Smooth L1 in latent space ---
        pred_loss = F.smooth_l1_loss(predictions, targets)

        # --- SIGReg regularization on context encoder output ---
        reg_loss = torch.tensor(0.0, device=device)
        if self.target_mode == "sigreg":
            reg_loss = self._sigreg_loss(context_out)

        # --- Total loss ---
        total_loss = pred_loss + reg_loss

        # --- EMA update (if applicable) ---
        if self.target_mode == "ema":
            self._update_target_ema()

        info = {
            "pred_loss": pred_loss.item(),
            "reg_loss": reg_loss.item(),
            "total_loss": total_loss.item(),
            "n_visible": len(visible_idx),
            "n_masked": len(masked_idx),
        }

        return total_loss, info

    def set_ema_momentum(self, momentum: float):
        """Update EMA momentum (for cosine schedule during training)."""
        self.ema_momentum = momentum

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters (context encoder + predictor)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def encoder_parameters(self) -> int:
        return sum(p.numel() for p in self.context_encoder.parameters())

    @property
    def predictor_parameters(self) -> int:
        return sum(p.numel() for p in self.predictor.parameters())


def build_jepa(
    n_channels: int = 6,
    seq_length: int = 128,
    patch_len: int = 16,
    encoder_dim: int = 384,
    encoder_layers: int = 6,
    encoder_heads: int = 6,
    predictor_layers: int = 2,
    predictor_heads: int = 4,
    target_mode: str = "sigreg",
    mask_ratio: float = 0.75,
    **kwargs,
) -> IMU_JEPA:
    """Build IMU-JEPA with specified config."""
    n_patches = seq_length // patch_len

    encoder = ViTEncoder(
        n_channels=n_channels,
        seq_length=seq_length,
        patch_len=patch_len,
        embed_dim=encoder_dim,
        num_layers=encoder_layers,
        num_heads=encoder_heads,
        use_cls=True,
    )

    predictor = Predictor(
        embed_dim=encoder_dim,
        num_layers=predictor_layers,
        num_heads=predictor_heads,
        n_patches=n_patches,
    )

    return IMU_JEPA(
        encoder=encoder,
        predictor=predictor,
        target_mode=target_mode,
        mask_ratio=mask_ratio,
        **kwargs,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for mode in ["sigreg", "ema"]:
        print(f"--- IMU-JEPA ({mode}) ---")
        jepa = build_jepa(
            n_channels=6, seq_length=128, patch_len=16,
            target_mode=mode, mask_ratio=0.75,
        ).to(device)

        print(f"  Trainable params:  {jepa.num_parameters:,}")
        print(f"  Encoder params:    {jepa.encoder_parameters:,}")
        print(f"  Predictor params:  {jepa.predictor_parameters:,}")

        # Forward with AMP
        x = torch.randn(16, 6, 128, device=device)
        with torch.amp.autocast("cuda"):
            loss, info = jepa(x)

        print(f"  Forward OK:")
        print(f"    pred_loss={info['pred_loss']:.6f}")
        print(f"    reg_loss={info['reg_loss']:.6f}")
        print(f"    total_loss={info['total_loss']:.6f}")
        print(f"    visible={info['n_visible']}, masked={info['n_masked']}")

        loss.backward()
        enc_grad = sum(
            p.grad.abs().sum().item()
            for p in jepa.context_encoder.parameters()
            if p.grad is not None
        )
        pred_grad = sum(
            p.grad.abs().sum().item()
            for p in jepa.predictor.parameters()
            if p.grad is not None
        )
        print(f"    Encoder grad norm:   {enc_grad:.4f}")
        print(f"    Predictor grad norm: {pred_grad:.4f}")

        # Verify target encoder has no gradients
        tgt_grad = sum(
            1 for p in jepa.target_encoder.parameters()
            if p.grad is not None
        )
        print(f"    Target encoder grads: {tgt_grad} (should be 0)")
        print()
