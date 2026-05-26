"""
Hybrid IMU-JEPA: combines JEPA latent prediction + lightweight MAE reconstruction.

Motivation: LuMamba (2026) found that JEPA alone produces "diffuse embeddings"
while MAE alone yields "structured but less generalizable representations."
Combining both achieves the most robust cross-domain performance.

Architecture:
  Same ViT encoder processes visible patches. Two heads:
  1. JEPA predictor: predicts target encoder embeddings at masked positions
  2. MAE decoder: reconstructs raw patch values at masked positions

Loss: L_total = L_jepa + alpha * L_mae

The MAE decoder is lightweight (1-2 layers) and acts as a regularizer that
anchors the latent space to signal structure, preventing the "diffuse embeddings"
failure mode of pure JEPA.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.vit import ViTEncoder, TransformerBlock
from src.models.jepa import Predictor


class HybridJEPA(nn.Module):
    """
    Hybrid JEPA + MAE for IMU data.

    Args:
        encoder: ViTEncoder (context encoder)
        predictor: JEPA Predictor
        decoder_dim: MAE decoder embedding dimension
        decoder_layers: MAE decoder Transformer layers
        decoder_heads: MAE decoder attention heads
        alpha: weight for MAE reconstruction loss
        lambda_var: SIGReg variance regularization
        lambda_cov: SIGReg covariance regularization
        mask_ratio: fraction of patches to mask
    """

    def __init__(
        self,
        encoder: ViTEncoder,
        predictor: Predictor,
        decoder_dim: int = 192,
        decoder_layers: int = 1,
        decoder_heads: int = 3,
        alpha: float = 0.25,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.context_encoder = encoder
        self.predictor = predictor
        self.alpha = alpha
        self.mask_ratio = mask_ratio
        self.n_patches = encoder.n_patches
        self.patch_dim = encoder.patch_embed.input_dim
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

        # Target encoder (stop-gradient copy for JEPA)
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # --- MAE decoder (lightweight) ---
        self.enc_to_dec = nn.Linear(encoder.embed_dim, decoder_dim)
        self.mae_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mae_mask_token, std=0.02)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, decoder_dim)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, decoder_heads, mlp_ratio=4.0)
            for _ in range(decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_dim)

    def _generate_mask(self, device):
        n_masked = int(self.n_patches * self.mask_ratio)
        n_visible = self.n_patches - n_masked
        perm = torch.randperm(self.n_patches, device=device)
        visible_idx = perm[:n_visible].sort().values
        masked_idx = perm[n_visible:].sort().values
        mask = torch.zeros(self.n_patches, dtype=torch.bool, device=device)
        mask[visible_idx] = True
        return visible_idx, masked_idx, mask

    def _patchify(self, x):
        return rearrange(x, "b c (n p) -> b n (p c)", p=self.context_encoder.patch_len)

    def _sigreg_loss(self, embeddings):
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        emb = embeddings - embeddings.mean(dim=0, keepdim=True)
        N, D = emb.shape
        std = emb.std(dim=0)
        var_loss = F.relu(1.0 - std).mean()
        cov = (emb.T @ emb) / (N - 1)
        off_diag = cov - torch.diag(cov.diagonal())
        cov_loss = (off_diag ** 2).sum() / D
        return self.lambda_var * var_loss + self.lambda_cov * cov_loss

    def forward(self, x):
        B, C, T = x.shape
        device = x.device

        visible_idx, masked_idx, mask = self._generate_mask(device)

        # --- Context encoder: visible patches only ---
        context_out = self.context_encoder(x, mask=mask)
        if self.context_encoder.use_cls:
            context_patches = context_out[:, 1:, :]
        else:
            context_patches = context_out

        # === JEPA HEAD: predict target embeddings ===
        with torch.no_grad():
            for p_ctx, p_tgt in zip(
                self.context_encoder.parameters(),
                self.target_encoder.parameters(),
            ):
                p_tgt.data.copy_(p_ctx.data)

            target_out = self.target_encoder(x)
            if self.target_encoder.use_cls:
                target_patches = target_out[:, 1:, :]
            else:
                target_patches = target_out
            targets = target_patches[:, masked_idx, :]

        predictions = self.predictor(context_patches, visible_idx, masked_idx)
        jepa_loss = F.smooth_l1_loss(predictions, targets)

        # === MAE HEAD: reconstruct raw patches ===
        decoded = self.enc_to_dec(context_patches)
        full_seq = self.mae_mask_token.to(decoded.dtype).expand(B, self.n_patches, -1).clone()
        full_seq[:, visible_idx, :] = decoded
        full_seq = full_seq + self.decoder_pos_embed.to(decoded.dtype)

        for block in self.decoder_blocks:
            full_seq = block(full_seq)
        full_seq = self.decoder_norm(full_seq)
        recon = self.decoder_pred(full_seq)

        target_patches_raw = self._patchify(x)
        mae_loss = F.mse_loss(
            recon[:, masked_idx, :],
            target_patches_raw[:, masked_idx, :],
        )

        # === SIGReg ===
        reg_loss = self._sigreg_loss(context_out)

        # === Total ===
        total_loss = jepa_loss + self.alpha * mae_loss + reg_loss

        info = {
            "jepa_loss": jepa_loss.item(),
            "mae_loss": mae_loss.item(),
            "reg_loss": reg_loss.item(),
            "total_loss": total_loss.item(),
            "n_visible": len(visible_idx),
            "n_masked": len(masked_idx),
        }
        return total_loss, info

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def encoder_parameters(self):
        return sum(p.numel() for p in self.context_encoder.parameters())


def build_hybrid(
    n_channels=6, seq_length=128, patch_len=16,
    encoder_dim=384, encoder_layers=6, encoder_heads=6,
    predictor_layers=2, predictor_heads=4,
    decoder_dim=192, decoder_layers=1, decoder_heads=3,
    alpha=0.25, mask_ratio=0.75, **kwargs,
):
    n_patches = seq_length // patch_len
    encoder = ViTEncoder(
        n_channels=n_channels, seq_length=seq_length, patch_len=patch_len,
        embed_dim=encoder_dim, num_layers=encoder_layers,
        num_heads=encoder_heads, use_cls=True,
    )
    predictor = Predictor(
        embed_dim=encoder_dim, num_layers=predictor_layers,
        num_heads=predictor_heads, n_patches=n_patches,
    )
    return HybridJEPA(
        encoder=encoder, predictor=predictor,
        decoder_dim=decoder_dim, decoder_layers=decoder_layers,
        decoder_heads=decoder_heads, alpha=alpha,
        mask_ratio=mask_ratio, **kwargs,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for alpha in [0.1, 0.25, 0.5]:
        model = build_hybrid(alpha=alpha).to(device)
        x = torch.randn(16, 6, 128, device=device)

        with torch.amp.autocast("cuda"):
            loss, info = model(x)
        loss.backward()

        print(f"alpha={alpha}: total={info['total_loss']:.4f}, "
              f"jepa={info['jepa_loss']:.4f}, mae={info['mae_loss']:.4f}, "
              f"reg={info['reg_loss']:.4f}, params={model.num_parameters:,}")
