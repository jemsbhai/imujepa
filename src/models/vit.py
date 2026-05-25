"""
Vision Transformer (ViT) encoder for 1D multi-channel time series.

Shared backbone for both MAE baseline and IMU-JEPA. The architecture:
  1. Patch tokenization along time axis
  2. Linear projection to embedding dimension
  3. Learnable positional embeddings + optional CLS token
  4. Stack of Transformer encoder blocks
  5. LayerNorm output

Input:  (B, C, T)  -- batch, channels, timesteps
Output: (B, N+1, D) if use_cls else (B, N, D) -- patch embeddings
"""

import math
import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbedding(nn.Module):
    """Tokenize multi-channel time series into patch embeddings.

    Input:  (B, C, T) -- e.g., (32, 6, 128)
    Output: (B, N, D) -- e.g., (32, 8, 384) with patch_len=16
    """

    def __init__(self, n_channels: int, patch_len: int, embed_dim: int):
        super().__init__()
        self.patch_len = patch_len
        self.n_channels = n_channels
        self.input_dim = patch_len * n_channels
        self.proj = nn.Linear(self.input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        assert T % self.patch_len == 0, (
            f"Sequence length {T} not divisible by patch_len {self.patch_len}"
        )
        n_patches = T // self.patch_len

        # (B, C, T) -> (B, N, patch_len, C) -> (B, N, patch_len * C)
        x = rearrange(x, "b c (n p) -> b n (p c)", p=self.patch_len)
        x = self.proj(x)  # (B, N, D)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention (uses PyTorch 2.x efficient attention)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward MLP block."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    ViT encoder for 1D multi-channel time series.

    Args:
        n_channels: Number of input channels (e.g., 6 for acc+gyro)
        seq_length: Length of input sequence in samples (e.g., 128)
        patch_len: Samples per patch (e.g., 16)
        embed_dim: Transformer embedding dimension (e.g., 384)
        num_layers: Number of Transformer blocks (e.g., 6)
        num_heads: Number of attention heads (e.g., 6)
        mlp_ratio: MLP hidden dim multiplier (e.g., 4.0)
        dropout: Dropout rate (e.g., 0.0)
        use_cls: Whether to prepend a [CLS] token
    """

    def __init__(
        self,
        n_channels: int = 6,
        seq_length: int = 128,
        patch_len: int = 16,
        embed_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_cls: bool = True,
    ):
        super().__init__()
        assert seq_length % patch_len == 0
        self.n_patches = seq_length // patch_len
        self.embed_dim = embed_dim
        self.use_cls = use_cls
        self.patch_len = patch_len

        # Patch embedding
        self.patch_embed = PatchEmbedding(n_channels, patch_len, embed_dim)

        # CLS token
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embedding (includes CLS position if used)
        n_positions = self.n_patches + (1 if use_cls else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_positions, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) input tensor
            mask: (B, N) bool tensor. True = KEEP this patch, False = DROP it.
                  Used by JEPA to pass only visible patches through encoder.
                  If None, all patches are processed.

        Returns:
            (B, N', D) where N' = num_kept_patches + (1 if use_cls else 0)
        """
        # Tokenize
        x = self.patch_embed(x)  # (B, N, D)
        B, N, D = x.shape

        if mask is not None:
            # Select only visible patches and their positional embeddings
            # mask shape: (B, N), True = keep
            # We need to handle variable-length sequences per batch element
            # For simplicity and efficiency, assume same mask across batch
            # (standard in I-JEPA / TS-JEPA implementations)
            assert mask.shape == (B, N) or mask.shape == (N,)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(B, -1)

            # Use first sample's mask (assumed uniform across batch)
            keep_idx = mask[0].nonzero(as_tuple=True)[0]  # (n_visible,)

            x = x[:, keep_idx, :]  # (B, n_visible, D)
            patch_pos = self.pos_embed[:, 1:, :] if self.use_cls else self.pos_embed
            patch_pos = patch_pos[:, keep_idx, :]  # (1, n_visible, D)

            if self.use_cls:
                cls = self.cls_token.expand(B, -1, -1)
                cls_pos = self.pos_embed[:, :1, :]
                x = torch.cat([cls, x], dim=1)
                pos = torch.cat([cls_pos, patch_pos], dim=1)
            else:
                pos = patch_pos

            x = x + pos
        else:
            # Standard full forward pass
            if self.use_cls:
                cls = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls, x], dim=1)  # (B, N+1, D)
            x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x

    def get_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: return only the CLS embedding for classification."""
        assert self.use_cls, "CLS token not enabled"
        out = self.forward(x)  # (B, N+1, D)
        return out[:, 0, :]    # (B, D)

    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: return only patch embeddings (no CLS)."""
        out = self.forward(x)
        if self.use_cls:
            return out[:, 1:, :]  # (B, N, D)
        return out

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --- Preset configurations ---

def vit_tiny(n_channels=6, seq_length=128, patch_len=16, **kwargs):
    return ViTEncoder(
        n_channels=n_channels, seq_length=seq_length, patch_len=patch_len,
        embed_dim=192, num_layers=4, num_heads=3, **kwargs,
    )

def vit_small(n_channels=6, seq_length=128, patch_len=16, **kwargs):
    return ViTEncoder(
        n_channels=n_channels, seq_length=seq_length, patch_len=patch_len,
        embed_dim=384, num_layers=6, num_heads=6, **kwargs,
    )

def vit_base(n_channels=6, seq_length=128, patch_len=16, **kwargs):
    return ViTEncoder(
        n_channels=n_channels, seq_length=seq_length, patch_len=patch_len,
        embed_dim=768, num_layers=12, num_heads=12, **kwargs,
    )


if __name__ == "__main__":
    # Smoke test with UCI-HAR dimensions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    for name, factory in [("ViT-Tiny", vit_tiny), ("ViT-Small", vit_small), ("ViT-Base", vit_base)]:
        model = factory(n_channels=6, seq_length=128, patch_len=16).to(device)
        x = torch.randn(8, 6, 128, device=device)

        # Full forward
        out = model(x)
        cls_emb = model.get_cls_token(x)
        patch_emb = model.get_patch_embeddings(x)

        print(f"{name}: {model.num_parameters:,} params")
        print(f"  Full output:     {list(out.shape)}")
        print(f"  CLS embedding:   {list(cls_emb.shape)}")
        print(f"  Patch embeddings:{list(patch_emb.shape)}")

        # Masked forward (keep 3 of 8 patches)
        mask = torch.zeros(8, dtype=torch.bool, device=device)
        mask[[0, 3, 7]] = True
        out_masked = model(x, mask=mask)
        print(f"  Masked (3/8 vis):{list(out_masked.shape)}")
        print()
