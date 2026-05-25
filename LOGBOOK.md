# Experimental Logbook — IMU-JEPA

**Paper working title:** IMU-JEPA: Joint-Embedding Predictive Architecture for Wearable Sensor Activity Recognition  
**Target venues:** NeurIPS / ICLR / IMWUT  
**Start date:** 2026-05-25  
**Researchers:** Muntaser Syed

---

## EXP-000: Environment Validation

**Date:** 2026-05-25
**Type:** computational
**Status:** planned

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
1. Create and activate Python venv
2. Install PyTorch with CUDA support
3. Install project dependencies
4. Run environment validation script (scripts/validate_env.py)
5. Confirm GPU name, VRAM, CUDA version, PyTorch version in output

### Environment
- **Hardware:** RTX 4090 (24GB), 64GB RAM
- **Software:** Windows, Python 3.11+, PyTorch 2.x, CUDA 12.x
- **Git commit:** (initial commit)
- **Seeds:** 42

### Results
(pending)

### Observations
(pending)

### Interpretation
(pending)

### Artifacts
- Logs: experiments/exp_000_env_validation/logs/
