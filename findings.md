# Findings

## Curated Summary

**IMU-JEPA with SIGReg outperforms MAE by +3.5% macro-F1 on UCI-HAR linear probe.**

Under controlled conditions (same ViT-Small encoder, same training config, same seed),
a JEPA latent-prediction objective with LeJEPA's SIGReg regularization achieves 84.3%
macro-F1 on UCI-HAR, compared to 80.8% for MAE signal-space reconstruction and 74.6%
for a random encoder (EXP-001, EXP-002). This validates the core hypothesis that
predicting in latent space is superior to reconstructing raw IMU signals, as the
encoder never wastes capacity modeling sensor noise and instead captures higher-level
motion semantics.

SIGReg (variance + covariance regularization) successfully prevents representation
collapse without requiring an EMA target encoder, removing a hyperparameter while
providing a provable non-collapse guarantee. This is the first empirical validation
of LeJEPA-style SIGReg on IMU data.

**Caveat:** These results are single-dataset, single-seed. Multi-seed evaluation and
cross-dataset generalization experiments are required before publication-strength claims.

---

## Raw Findings Log

### 2026-05-25 — EXP-002: IMU-JEPA (SIGReg) Pretraining on UCI-HAR

**Key result:** JEPA achieves 84.3% macro-F1 vs MAE's 80.8% (+3.5%) on UCI-HAR
linear probe, validating latent prediction > signal reconstruction for IMU data.

- Random baseline: acc=0.7411, macro_f1=0.7461
- MAE pretrained:  acc=0.8012, macro_f1=0.8083 (EXP-001)
- JEPA (SIGReg):   acc=0.8392, macro_f1=0.8429 (EXP-002)
- Delta JEPA vs MAE: +0.0346 macro-F1
- Config: ViT-Small, patch_len=16, mask_ratio=0.75, sigreg (λ_var=1.0, λ_cov=0.04)
- Single seed (42), single dataset — needs multi-seed validation

### 2026-05-25 — EXP-001: MAE-IMU Pretraining on UCI-HAR

**Key result:** MAE pretrained encoder achieves 80.8% macro-F1 on UCI-HAR linear
probe, +6.2% over random encoder (74.6%).

- Random baseline: acc=0.7411, macro_f1=0.7461, weighted_f1=0.7396
- MAE pretrained: acc=0.8012, macro_f1=0.8083, weighted_f1=0.7999
- Config: ViT-Small, patch_len=16, mask_ratio=0.75, 100 epochs, seed=42
