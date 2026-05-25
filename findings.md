# Findings

## Curated Summary

**MAE pretraining establishes a viable baseline for IMU self-supervised learning.**
A ViT-Small encoder (10.7M params) pretrained with a masked autoencoder objective
(75% masking, MSE reconstruction) on UCI-HAR for 100 epochs achieves 80.1% accuracy
/ 80.8% macro-F1 under frozen linear probe — a +6% improvement over the random
encoder baseline (74.1% / 74.6%). This confirms that the ViT patch-tokenization
approach learns useful IMU representations from self-supervised pretraining and
establishes the performance floor that IMU-JEPA must exceed (EXP-001).

---

## Raw Findings Log

### 2026-05-25 — EXP-001: MAE-IMU Pretraining on UCI-HAR

**Key result:** MAE pretrained encoder achieves 80.8% macro-F1 on UCI-HAR linear
probe, +6.2% over random encoder (74.6%).

- Random baseline: acc=0.7411, macro_f1=0.7461, weighted_f1=0.7396
- MAE pretrained: acc=0.8012, macro_f1=0.8083, weighted_f1=0.7999
- Config: ViT-Small, patch_len=16, mask_ratio=0.75, 100 epochs, seed=42
- Training time: ~2-3 min on RTX 4090 Laptop GPU
