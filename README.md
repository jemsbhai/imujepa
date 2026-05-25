# IMU-JEPA

**Joint-Embedding Predictive Architecture for Inertial Measurement Unit Data**

Self-supervised representation learning for wearable IMU sensors using latent prediction instead of signal-space reconstruction. Targeting top-tier ML/sensing venues (NeurIPS, ICLR, IMWUT).

## Status

🔬 **Phase 0: Infrastructure Setup** — scaffolding, environment, data pipeline

## Key Idea

Apply JEPA's latent-prediction objective to multi-channel IMU data (accelerometer + gyroscope) for Human Activity Recognition. Unlike masked autoencoders that waste capacity reconstructing sensor noise, JEPA predicts in a learned latent space, discarding noise and focusing on semantic motion dynamics.

## Architecture

- **Encoder:** ViT-Small (6 layers, 6 heads, D=384, ~7M params)
- **Predictor:** Lightweight Transformer (2 layers, 4 heads, ~1.5M params)
- **Regularization:** LeJEPA SIGReg (provable collapse prevention, no EMA tuning)
- **Tokenization:** Temporal patch tokenization along time axis

## Datasets

### Pretraining
- NHANES 2011-2014 (~28k hours, ~11k participants, wrist accelerometer, public)

### Evaluation (HAR Benchmarks)
- UCI-HAR, PAMAP2, DSADS, MHealth, USC-HAD, Realworld
- Cross-dataset leave-one-out protocol (matches MoPFormer NeurIPS 2025)

## Hardware Requirements

- NVIDIA GPU with >= 12GB VRAM (developed on RTX 4090)
- 32GB+ system RAM
- ~200GB disk for datasets + checkpoints

## Setup

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Reproduction

See `LOGBOOK.md` for full experimental protocol. Each experiment is identified by `EXP-XXX` and has a frozen config in `experiments/exp_XXX/config.yaml`.

## License

MIT

## Citation

TBD upon publication.
