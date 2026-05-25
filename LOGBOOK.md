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
