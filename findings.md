# Findings

## Curated Summary

**IMU-JEPA with SIGReg consistently outperforms MAE by +3.8% macro-F1 on UCI-HAR
across 3 seeds, with 3x lower variance.**

Under controlled multi-seed evaluation (seeds 42, 123, 456), JEPA achieves 83.6%
± 0.10% macro-F1 on UCI-HAR linear probe, compared to MAE's 79.9% ± 0.31%
(EXP-003). JEPA wins on all 3 seeds. The consistent +3.8% gap and lower variance
validate two claims: (1) latent prediction is superior to signal-space reconstruction
for IMU representation learning, and (2) SIGReg provides more stable training
dynamics than MAE's reconstruction loss.

Both findings are currently limited to a single dataset (UCI-HAR, 30 subjects,
6 activities, waist-mounted). Cross-dataset generalization experiments are required
to confirm these findings hold across sensor placements, devices, and activity sets.

---

## Raw Findings Log

### 2026-05-25 — EXP-003: Multi-Seed MAE vs JEPA on UCI-HAR

**Key result:** JEPA beats MAE by +3.8% macro-F1 consistently across 3 seeds
(83.6% ± 0.10% vs 79.9% ± 0.31%), winning on all 3 seeds.

- MAE:  acc=0.7905±0.0031, macro_f1=0.7985±0.0031
- JEPA: acc=0.8301±0.0009, macro_f1=0.8360±0.0010
- JEPA variance 3x lower — SIGReg stabilizes training
- Config: ViT-Small, patch_len=16, mask_ratio=0.75, 100 epochs, 3 seeds

### 2026-05-25 — EXP-002: IMU-JEPA (SIGReg) Pretraining on UCI-HAR

**Key result:** JEPA achieves 84.3% macro-F1 vs MAE's 80.8% (+3.5%) on UCI-HAR
linear probe, validating latent prediction > signal reconstruction for IMU data.

### 2026-05-25 — EXP-001: MAE-IMU Pretraining on UCI-HAR

**Key result:** MAE pretrained encoder achieves 80.8% macro-F1 on UCI-HAR linear
probe, +6.2% over random encoder (74.6%).
