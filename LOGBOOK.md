# Experimental Logbook — IMU-JEPA

**Paper working title:** IMU-JEPA: Joint-Embedding Predictive Architecture for Wearable Sensor Activity Recognition  
**Target venues:** NeurIPS / ICLR / IMWUT  
**Start date:** 2026-05-25  
**Researchers:** Muntaser Syed

---

## EXP-000: Environment Validation

**Date:** 2026-05-25 07:40 EST
**Type:** computational
**Status:** completed

### Hypothesis
Verify that the development environment (PyTorch, CUDA, mixed-precision) is correctly configured and can train a small ViT on synthetic IMU-shaped data.

### Independent Variables
- None (validation only)

### Dependent Variables / Metrics
- Environment installs without error
- GPU detected and CUDA operational
- Forward + backward pass completes on synthetic (500, 6) tensor
- Mixed-precision (torch.cuda.amp) functional

### Control Conditions
- N/A

### Protocol
1. Run scripts/validate_env.py
2. Verify all checks pass
3. Inspect environment.json snapshot

### Environment
- **Hardware:** NVIDIA GeForce RTX 4090 Laptop GPU (17.2 GB VRAM), 64GB RAM, AMD64
- **Software:** Windows 11 (10.0.26200), Python 3.12.2, PyTorch 2.6.0+cu124, CUDA 12.4, cuDNN 90100
- **Git commit:** initial commit
- **Seeds:** 42

### Results
- All 12 dependencies present and importable
- CUDA available, BF16 supported
- Forward/backward pass on synthetic (8, 500, 6) IMU tensor: OK
- Mixed-precision (FP16) autocast: OK
- No missing packages

### Observations
- GPU is the Laptop variant (17.2GB VRAM vs 24GB desktop). Still ample for
  our model sizes (5-10M params). Batch size may need to be 128 instead of
  256 if VRAM pressure appears during full JEPA training, but unlikely.
- pynvml deprecation warning present (cosmetic, no functional impact).
- All dependency versions are recent (numpy 2.4, scipy 1.17, PyTorch 2.6).

### Interpretation
Environment is fully operational. Ready to proceed to Phase 1 (baseline reproduction).

### Artifacts
- Environment snapshot: experiments/exp_000_env_validation/environment.json


---

## EXP-001: MAE-IMU Pretraining on UCI-HAR

**Date:** 2026-05-25
**Type:** computational
**Status:** completed

### Hypothesis
A masked autoencoder (signal-space reconstruction objective) pretrained on UCI-HAR
will produce encoder representations that outperform a random encoder on linear probe
HAR classification. This establishes the baseline that IMU-JEPA must beat.

### Independent Variables
- Pretraining objective: MAE (MSE reconstruction of masked patches)
- Mask ratio: 0.75
- Epochs: 100

### Dependent Variables / Metrics
- Linear probe accuracy, macro-F1, weighted-F1 on UCI-HAR test set

### Control Conditions
- Baseline: random (untrained) ViT-Small encoder
- Same encoder architecture (ViT-Small, 10.7M params)
- Same linear probe protocol (sklearn LogisticRegression, C=1.0)

### Protocol
1. Build MAE with ViT-Small encoder + lightweight decoder (984K params)
2. Pretrain 100 epochs on UCI-HAR train split (7,352 windows)
3. Freeze encoder, run linear probe on train/test split

### Environment
- **Hardware:** RTX 4090 Laptop GPU (17.2 GB), 64GB RAM
- **Software:** Python 3.12.2, PyTorch 2.6.0+cu124, CUDA 12.4
- **Config:** batch_size=128, lr=1.5e-4, weight_decay=0.05, warmup=10 epochs, AMP=true
- **Seed:** 42

### Results
| Method           | Accuracy | Macro-F1 | Weighted-F1 |
|------------------|----------|----------|-------------|
| Random encoder   | 0.7411   | 0.7461   | 0.7396      |
| MAE pretrained   | 0.8012   | 0.8083   | 0.7999      |
| **Delta**        | **+6.0%**| **+6.2%**| **+6.0%**   |

### Observations
- Pretraining completed in ~2-3 minutes on RTX 4090.
- +6% improvement confirms that self-supervised pretraining on IMU data works
  with our ViT encoder architecture.
- 80.1% accuracy is reasonable for UCI-HAR with a linear probe (no fine-tuning).
  Published baselines with fine-tuning typically reach 90%+.

### Interpretation
MAE pretraining successfully learns useful IMU representations. The 80.8% macro-F1
is the number IMU-JEPA must exceed to demonstrate that latent prediction is superior
to signal-space reconstruction for IMU data (Gate 2 threshold).

### Artifacts
- Checkpoint: experiments/exp_001_mae_ucihar/checkpoints/best.pt
- Results: experiments/exp_001_mae_ucihar/results.json
