"""
Masked Autoencoder (MAE) for IMU time series.

Controlled baseline: same ViT encoder as IMU-JEPA, but with a
signal-space reconstruction objective instead of latent prediction.

Architecture:
  1. Randomly mask patches (default 75%)
  2. Encode only visible patches through the ViT encoder
  3. Insert learnable mask tokens at masked positions
  4. Decode all tokens through a lightweight Transformer decoder
  5. Reconstruct original patch values at masked positions (MSE loss)

Reference: He et al., "Masked Autoencoders Are Scalable Vision Learners"
"""

import torch
import torch.nn as nn
from einops import rearrange

from src.models.vit import ViTEncoder, TransformerBlock


class MAE_IMU(nn.Module):
    """
    Masked Autoencoder for multi-channel IMU data.

    Args:
        encoder: ViTEncoder instance (shared architecture with JEPA)
        decoder_dim: Decoder embedding dimension
        decoder_layers: Number of decoder Transformer blocks
        decoder_heads: Number of decoder attention heads
        mask_ratio: Fraction of patches to mask
    """

    def __init__(
        self,
        encoder: ViTEncoder,
        decoder_dim: int = 192,
        decoder_layers: int = 2,
        decoder_heads: int = 3,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.n_patches = encoder.n_patches
        self.patch_dim = encoder.patch_embed.input_dim  # patch_len * n_channels

        # Encoder-to-decoder projection
        self.enc_to_dec = nn.Linear(encoder.embed_dim, decoder_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Decoder positional embeddings (for all patches, no CLS)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, decoder_dim)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # Decoder Transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, decoder_heads, mlp_ratio=4.0)
            for _ in range(decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # Reconstruction head: project back to patch pixel space
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_dim)

    def _generate_mask(self, batch_size: int, device: torch.device):
        """
        Generate random mask. Same mask for all samples in batch
        (standard in MAE/JEPA implementations for efficiency).

        Returns:
            visible_idx: (n_visible,) indices of visible patches
            masked_idx:  (n_masked,) indices of masked patches
            mask: (N,) bool tensor, True = VISIBLE
        """
        n_masked = int(self.n_patches * self.mask_ratio)
        n_visible = self.n_patches - n_masked

        # Random permutation
        perm = torch.randperm(self.n_patches, device=device)
        visible_idx = perm[:n_visible].sort().values
        masked_idx = perm[n_visible:].sort().values

        # Bool mask: True = visible
        mask = torch.zeros(self.n_patches, dtype=torch.bool, device=device)
        mask[visible_idx] = True

        return visible_idx, masked_idx, mask

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert raw input to patch targets: (B, C, T) -> (B, N, patch_dim)."""
        return rearrange(
            x, "b c (n p) -> b n (p c)", p=self.encoder.patch_len
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, T) raw IMU input

        Returns:
            loss: scalar reconstruction loss (MSE on masked patches only)
            info: dict with additional info for logging
        """
        B, C, T = x.shape
        device = x.device

        # Generate mask
        visible_idx, masked_idx, mask = self._generate_mask(B, device)
        n_visible = len(visible_idx)
        n_masked = len(masked_idx)

        # Encode only visible patches
        encoded = self.encoder(x, mask=mask)  # (B, n_visible+1, enc_dim) with CLS

        # Remove CLS token for decoder (we only reconstruct patches)
        if self.encoder.use_cls:
            encoded = encoded[:, 1:, :]  # (B, n_visible, enc_dim)

        # Project to decoder dimension
        decoded = self.enc_to_dec(encoded)  # (B, n_visible, dec_dim)

        # Build full sequence with mask tokens
        # Match dtype of decoded (critical for mixed-precision / AMP)
        full_seq = self.mask_token.to(decoded.dtype).expand(B, self.n_patches, -1).clone()

        # Insert visible patch embeddings at their original positions
        full_seq[:, visible_idx, :] = decoded

        # Add positional embeddings
        full_seq = full_seq + self.decoder_pos_embed.to(decoded.dtype)

        # Decode
        for block in self.decoder_blocks:
            full_seq = block(full_seq)
        full_seq = self.decoder_norm(full_seq)

        # Reconstruct: predict patch pixel values
        pred = self.decoder_pred(full_seq)  # (B, N, patch_dim)

        # Target: original patches
        target = self._patchify(x)  # (B, N, patch_dim)

        # Loss: MSE only on masked patches
        pred_masked = pred[:, masked_idx, :]    # (B, n_masked, patch_dim)
        target_masked = target[:, masked_idx, :]  # (B, n_masked, patch_dim)
        loss = nn.functional.mse_loss(pred_masked, target_masked)

        info = {
            "n_visible": n_visible,
            "n_masked": n_masked,
            "mask_ratio": n_masked / self.n_patches,
            "recon_loss": loss.item(),
        }

        return loss, info

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def encoder_parameters(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters())

    @property
    def decoder_parameters(self) -> int:
        return self.num_parameters - self.encoder_parameters


def build_mae(
    n_channels: int = 6,
    seq_length: int = 128,
    patch_len: int = 16,
    encoder_dim: int = 384,
    encoder_layers: int = 6,
    encoder_heads: int = 6,
    decoder_dim: int = 192,
    decoder_layers: int = 2,
    decoder_heads: int = 3,
    mask_ratio: float = 0.75,
) -> MAE_IMU:
    """Build MAE with specified config."""
    encoder = ViTEncoder(
        n_channels=n_channels,
        seq_length=seq_length,
        patch_len=patch_len,
        embed_dim=encoder_dim,
        num_layers=encoder_layers,
        num_heads=encoder_heads,
        use_cls=True,
    )
    return MAE_IMU(
        encoder=encoder,
        decoder_dim=decoder_dim,
        decoder_layers=decoder_layers,
        decoder_heads=decoder_heads,
        mask_ratio=mask_ratio,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build MAE with UCI-HAR dimensions
    mae = build_mae(
        n_channels=6,
        seq_length=128,
        patch_len=16,
        mask_ratio=0.75,
    ).to(device)

    print(f"MAE total params:   {mae.num_parameters:,}")
    print(f"  Encoder params:   {mae.encoder_parameters:,}")
    print(f"  Decoder params:   {mae.decoder_parameters:,}")
    print(f"  Patches per input: {mae.n_patches}")
    print(f"  Patch dim:         {mae.patch_dim}")
    print(f"  Mask ratio:        {mae.mask_ratio}")

    # Forward pass with AMP (tests the dtype fix)
    x = torch.randn(16, 6, 128, device=device)
    with torch.amp.autocast("cuda"):
        loss, info = mae(x)
    print(f"\nForward pass OK (AMP):")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Visible: {info['n_visible']}, Masked: {info['n_masked']}")

    loss.backward()
    print(f"  Backward pass OK")
