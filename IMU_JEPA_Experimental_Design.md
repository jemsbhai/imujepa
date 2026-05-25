# IMU-JEPA Experimental Design
## Hardware: Windows, 64GB RAM, RTX 4090 (24GB VRAM)

---

## 1. Hardware Feasibility Assessment

Your RTX 4090 is overkill for this project in the best possible way.

**Why:** The models in this domain are tiny. Sonata is 3.77M parameters. CHARM is 7M.
MoPFormer is a small ViT (~5-10M). Compare this to LLMs at billions of parameters.
A 10M-parameter ViT in FP32 with AdamW optimizer states uses ~120MB of VRAM.
Even with batch size 512 and activations, you'll use 2-4GB total. You have 24GB.

**Estimated compute budget for full project:**

| Phase | Data | Model | Est. Time (4090) |
|-------|------|-------|-------------------|
| Pretrain on NHANES (28k hrs, 100 epochs) | ~16M windows | 5-10M params | 48-96 hours |
| Pretrain on HAR pooled (cross-dataset) | ~500k windows | 5-10M params | 2-4 hours |
| Linear probe eval (6 datasets) | tiny | frozen encoder | < 30 min total |
| Fine-tune eval (6 datasets, 3 seeds) | tiny | full model | 2-3 hours total |
| Ablations (masking, regularization, etc.) | varies | varies | 20-40 hours |
| **Total estimated wall-clock time** | | | **~100-200 hours** |

This is 4-8 days of continuous GPU time. Entirely feasible on a single 4090.

**RAM:** 64GB is sufficient. NHANES raw data is ~50-80GB on disk but you'll load
windows on-the-fly via a PyTorch DataLoader with memory-mapped files. No issue.

**Disk:** Budget ~200GB for datasets + checkpoints + logs.

**Software stack:** PyTorch 2.x, WSL2 or native Windows. Either works.
Mixed-precision (FP16/BF16 via torch.cuda.amp) will roughly halve training time
and VRAM usage; use it by default.

---

## 2. Model Architecture: IMU-JEPA

### 2.1 Overview

The architecture follows the standard JEPA template with IMU-specific adaptations:

```
Raw IMU window (T x C)      e.g., 500 samples x 6 channels
        |
   [Patch Tokenizer]         non-overlapping patches along time axis
        |
   [Context Encoder]         ViT-Small or ViT-Base (shared weights)
        |                        |
  visible patches          masked patch positions
        |                        |
   [Predictor]              [Target Encoder] (EMA or SIGReg-frozen)
        |                        |
   predicted latents        target latents
        |________________________|
                  |
            Huber Loss in latent space
```

### 2.2 Tokenization

Input: a window of shape (T, C) where T=500 (5s at 100Hz) and C=6 (3-axis acc + 3-axis gyro).

Patch strategy: divide the time axis into P non-overlapping patches of length L.
Each patch is a (L, C) matrix, flattened to a vector of dimension L*C.

Recommended starting point: L=25 (0.25s at 100Hz), giving P=20 patches per window.
Each patch token has raw dimension 25*6 = 150, projected to D=256 via a linear layer.

Ablate: L in {10, 16, 25, 50} to find the sweet spot.

For 3-channel-only pretraining (NHANES has no gyro): L*3 = 75 input dim per patch.
Transfer to 6-channel downstream: re-initialize the input projection layer's extra
columns with small random weights, keeping the encoder frozen initially.

### 2.3 Encoder

ViT-Small: 6 layers, 6 heads, D=384, MLP ratio 4. ~7M params.
This matches CHARM (7M) and is 2x Sonata (3.77M), well within 4090 budget.

Positional encoding: learnable 1D positional embeddings (P positions).
Optionally add a channel-type embedding (acc vs gyro vs mag) if doing multimodal.

Prepend a [CLS] token for downstream classification.

### 2.4 Predictor

Lightweight Transformer: 2 layers, 4 heads, D=384.
Input: context encoder outputs for visible patches + learnable mask tokens at
masked positions, all with positional embeddings.
Output: predicted latent representations at masked positions.

~1.5M additional params. Total model: ~8.5M params.

### 2.5 Target Encoder

Option A (baseline): EMA of context encoder. Momentum schedule: 0.996 -> 1.0
(cosine schedule over training, same as I-JEPA).

Option B (proposed): LeJEPA with SIGReg. Replace EMA with a second forward pass
through the same encoder but with stop-gradient, plus SIGReg regularization
(variance + covariance terms on the embeddings). This removes the EMA momentum
hyperparameter entirely and provides a provable non-collapse guarantee.

Ablate: A vs B. If B matches or beats A, it's a contribution (no prior work
has tested SIGReg on IMU data).

### 2.6 Masking Strategy

Three strategies to ablate:

**Block masking (I-JEPA style):** mask 1-2 contiguous blocks of 4-8 patches each.
Context sees the rest. Targets are the masked blocks. Total masking ratio ~40-60%.

**Random patch masking (MAE/Audio-JEPA style):** randomly mask individual patches.
Masking ratio 60-80% (TS-JEPA found >70% optimal for time series).

**Cross-channel masking (novel for IMU-JEPA):** mask entire sensor axes.
E.g., mask all gyroscope channels for some patches, forcing the encoder to predict
gyro latents from accelerometer context. This tests cross-modal predictability
and is directly relevant to partial-sensor deployment scenarios.

### 2.7 Loss

Smooth L1 (Huber) loss between predicted and target latent representations,
averaged over masked positions. Same as I-JEPA and TS-JEPA.

### 2.8 Regularization (if using SIGReg / LeJEPA)

Add variance and covariance regularization terms on encoder output embeddings:
- Variance term: encourage each embedding dimension to have variance >= 1
- Covariance term: penalize off-diagonal covariance to decorrelate dimensions

Lambda_var = 1.0, lambda_cov = 0.04 (LeJEPA defaults). Ablate.

---

## 3. Experimental Protocol

### Phase 0: Infrastructure Setup (Week 1)

- Set up PyTorch environment (conda/venv, CUDA, mixed-precision)
- Download and preprocess all datasets (see Section 4)
- Implement base DataLoader for IMU windows
- Implement evaluation harness (linear probe + fine-tune)

### Phase 1: Reproduce Baselines (Weeks 2-3)

**Goal:** Establish the performance bar on your exact hardware and data splits.

**1a. LIMU-BERT reproduction**
- Code is public: github.com/zhoupf/LIMU-BERT-Public
- Datasets: HHAR, UCI-HAR, MotionSense, USC-HAD
- This gives you a fast sanity check and calibrates your evaluation pipeline

**1b. MoPFormer reproduction (if code available) or report their numbers**
- Check if NeurIPS 2025 code is released
- If not, use their reported numbers as comparison targets
- Their six datasets: PAMAP2, DSADS, MHealth, Realworld, UCI-HAR, USC-HAD

**1c. Vanilla MAE-IMU baseline**
- Implement a standard masked autoencoder on IMU patches (same tokenization
  as IMU-JEPA but with pixel-space reconstruction loss)
- This is your controlled ablation: same architecture, different objective
  (reconstruction vs latent prediction)
- Train and eval on same datasets

### Phase 2: IMU-JEPA Pretraining (Weeks 4-6)

**2a. Small-scale validation (cross-dataset pretraining)**
- Pool the unlabeled data from 5 of 6 HAR benchmarks
- Pretrain IMU-JEPA on the pooled data
- Evaluate on the held-out 6th dataset via linear probe
- Repeat for all 6 leave-one-out splits
- This matches MoPFormer's evaluation protocol exactly
- Estimated time: 2-4 hours per split, ~24 hours total

**2b. Large-scale pretraining on NHANES**
- Download NHANES 2011-2014 raw 80Hz wrist accelerometer data
- Preprocess: resample to 100Hz, segment into 5s windows, normalize
- Note: NHANES is accelerometer-only (3 channels), no gyroscope
- Pretrain IMU-JEPA on ~16M windows for 100-200 epochs
- Estimated time: 48-96 hours (2-4 days)
- Evaluate: freeze encoder, attach linear probe on each of 6 HAR benchmarks
- Compare to LIMU-BERT, MoPFormer, MAE-IMU baseline

### Phase 3: Ablation Studies (Weeks 7-9)

Each ablation is a full pretrain + eval cycle. Budget ~4-8 hours each on the
cross-dataset (small-scale) setup. Run on NHANES only for the winning config.

**3a. Masking strategy ablation**
- Block masking (I-JEPA style) at ratios 40%, 60%, 75%
- Random patch masking at ratios 60%, 70%, 80%
- Cross-channel masking (mask entire acc or gyro axes)
- Best masking strategy proceeds

**3b. Regularization ablation**
- EMA-only (standard JEPA)
- SIGReg-only (LeJEPA)
- EMA + SIGReg combined
- SIGReg + lightweight reconstruction auxiliary loss (hybrid, per LuMamba finding)

**3c. Patch length ablation**
- L in {10, 16, 25, 50} samples
- Trade-off: shorter patches = more tokens = finer temporal resolution but
  higher compute; longer patches = coarser but more context per token

**3d. Model scale ablation** (optional, for scaling law argument)
- ViT-Tiny (3M), ViT-Small (7M), ViT-Base (15M)
- All fit comfortably on 4090
- Plot downstream accuracy vs model size at fixed pretraining compute

### Phase 4: Cross-Domain Generalization (Weeks 10-11)

This is where a strong paper differentiates from a workshop paper.

**4a. Cross-device transfer**
- Pretrain on NHANES (Actigraph GT3X+ wrist)
- Evaluate on datasets using different devices and body locations:
  - UCI-HAR (Samsung Galaxy S II, waist)
  - PAMAP2 (Colibri, wrist + chest + ankle)
  - USC-HAD (MotionNode, hip)
- Report: how much does each method degrade when device/location changes?

**4b. Few-shot evaluation**
- For each downstream dataset, evaluate with 1%, 5%, 10%, 25%, 100% of labels
- This tests sample efficiency, where JEPA should shine
  (EchoJEPA showed 79% accuracy at 1% labels vs 42% for MAE)
- Plot: label efficiency curves per method

**4c. Noise robustness**
- Inject Gaussian noise at SNR = {20, 10, 5, 0} dB into test data
- Evaluate frozen encoder performance at each noise level
- Hypothesis: JEPA degrades less than MAE because it never learned to
  reconstruct noise in the first place

### Phase 5: Analysis and Visualization (Week 12)

**5a. Representation quality**
- t-SNE/UMAP of learned embeddings colored by activity class
- Compare JEPA vs MAE vs LIMU-BERT embedding structure
- Compute embedding covariance rank (effective dimensionality)
  to verify non-collapse

**5b. Koopman connection (optional, for theory contribution)**
- On a controlled synthetic IMU dataset (simulated pendulum or gait model),
  verify that JEPA encoder learns Koopman eigenfunctions
- Compare JEPA vs MAE latent dynamics linearity

**5c. Latent prediction visualization**
- For a test window, show: original signal, masked positions, and
  what the predictor predicts in latent space
- Compare with MAE's pixel-space reconstruction

---

## 4. Dataset Preparation

### 4.1 Pretraining Corpus

**NHANES 2011-2014** (primary pretraining)
- Source: https://wwwn.cdc.gov/Nchs/Nhanes/search/datapage.aspx?Component=Examination
- Format: per-participant tar.bz2 archives containing hourly CSV files
  with 80Hz tri-axial acceleration (x, y, z)
- Preprocessing pipeline:
  1. Extract all participant archives
  2. Concatenate hourly CSVs per participant
  3. Resample 80Hz -> 100Hz (linear interpolation)
  4. Remove non-wear periods (zero-variance detection)
  5. Segment into 5-second windows (500 samples) with no overlap
  6. Z-score normalize per participant
  7. Save as memory-mapped numpy arrays or HDF5 for fast DataLoader access
- Expected yield: ~10-16M windows from ~11k participants

### 4.2 Evaluation Benchmarks

All freely downloadable. Standardize to 100Hz, 5-second windows, 6-channel
(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z) where available.

| Dataset | Download | Channels | Resample | Notes |
|---------|----------|----------|----------|-------|
| UCI-HAR | UCI ML Repository | 6 (acc+gyro) | 50->100Hz | Pre-segmented, use raw inertial |
| PAMAP2 | UCI ML Repository | 36+ (multi-loc) | 100Hz native | Use wrist IMU (6ch) for fair compare |
| DSADS | UCI ML Repository | 45 (multi-loc) | 25->100Hz | 5 body positions |
| MHealth | UCI ML Repository | 23 | 50->100Hz | Chest+wrist+ankle |
| USC-HAD | USC | 6 (acc+gyro) | 100Hz native | Hip-mounted |
| Realworld | TU Darmstadt | 63 (multi-loc) | 50->100Hz | 7 positions |
| HHAR | archive.ics.uci.edu | 6 (acc+gyro) | varies | Multi-device, multi-user |
| MotionSense | GitHub | 6 (acc+gyro) | 50->100Hz | iPhone in pocket |

### 4.3 Data Split Protocol

Follow MoPFormer's protocol for direct comparability:
- Cross-dataset: pretrain on 5 datasets, evaluate on 6th (leave-one-out)
- Within-dataset: subject-disjoint train/val/test splits (never leak subjects)
- Report: mean and std over 3 random seeds for all results

---

## 5. Evaluation Metrics

| Metric | Purpose | When |
|--------|---------|------|
| **Macro-F1** | Primary HAR metric (class-balanced) | All experiments |
| **Weighted-F1** | Secondary (accounts for class imbalance) | All experiments |
| **Top-1 Accuracy** | For comparability with older papers | All experiments |
| **Linear probe accuracy** | Tests representation quality (frozen encoder) | Phases 2-4 |
| **Fine-tune accuracy** | Tests full-model performance | Phase 2-3 |
| **Effective rank** | Embedding dimensionality (collapse detection) | Phase 5 |
| **Label efficiency AUC** | Area under few-shot curve | Phase 4b |
| **Noise degradation slope** | Robustness to noise injection | Phase 4c |

---

## 6. Expected Results Table (Template)

### Main Results: Cross-Dataset HAR (Leave-One-Out, Linear Probe, Macro-F1)

| Method | PAMAP2 | DSADS | MHealth | Realworld | UCI-HAR | USC-HAD | **Avg** |
|--------|--------|-------|---------|-----------|---------|---------|---------|
| Supervised (no pretrain) | -- | -- | -- | -- | -- | -- | -- |
| LIMU-BERT | -- | -- | -- | -- | -- | -- | -- |
| MoPFormer (reported) | -- | -- | -- | -- | -- | -- | -- |
| MAE-IMU (ours) | -- | -- | -- | -- | -- | -- | -- |
| **IMU-JEPA (ours)** | -- | -- | -- | -- | -- | -- | -- |
| IMU-JEPA + SIGReg | -- | -- | -- | -- | -- | -- | -- |
| IMU-JEPA + cross-ch mask | -- | -- | -- | -- | -- | -- | -- |

### NHANES Pretrained -> HAR Transfer (Linear Probe, Macro-F1)

| Method | UCI-HAR | PAMAP2 | USC-HAD | HHAR | MotionSense | **Avg** |
|--------|---------|--------|---------|------|-------------|---------|
| Yuan-SSL (reported) | -- | -- | -- | -- | -- | -- |
| MAE-IMU (ours) | -- | -- | -- | -- | -- | -- |
| **IMU-JEPA (ours)** | -- | -- | -- | -- | -- | -- |

---

## 7. Decision Thresholds and Go/No-Go Gates

**Gate 1 (end of Phase 1):** Can you reproduce LIMU-BERT numbers within 2%?
If no: debug evaluation pipeline before proceeding. Do not pretrain IMU-JEPA
on a broken eval setup.

**Gate 2 (end of Phase 2a):** Does IMU-JEPA beat MAE-IMU (your own controlled
baseline) by at least 1% avg macro-F1 on cross-dataset eval?
- If yes: the latent prediction objective is working. Proceed to NHANES pretraining.
- If no: investigate masking strategy and regularization first (mini-ablation).

**Gate 3 (end of Phase 2b):** Does NHANES-pretrained IMU-JEPA beat or match
MoPFormer's reported numbers?
- If beats by 3%+ avg: target NeurIPS/ICLR main track with Angle B.
- If matches within 1-2%: target IMWUT or SenSys with Angle A + thorough ablation.
- If loses by >2%: pivot to Angle C (edge deployment) or Angle D (world model /
  activity anticipation) where the comparison is less direct.

**Gate 4 (end of Phase 4):** Does IMU-JEPA show clear advantages on cross-device
transfer AND/OR few-shot AND/OR noise robustness?
- If yes on >= 2 of 3: strong paper with clear story.
- If yes on 1: still publishable but needs additional contribution.
- If no on all: the "JEPA for IMU" thesis may not hold empirically.
  Consider hybrid objective (JEPA + light reconstruction, per LuMamba finding).

---

## 8. Timeline Summary

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Setup | Environment, datasets downloaded, DataLoader working |
| 2-3 | Baselines | LIMU-BERT reproduced, MAE-IMU baseline trained, Gate 1 cleared |
| 4-5 | IMU-JEPA v1 | Cross-dataset pretrain + eval, Gate 2 cleared |
| 6 | NHANES pretrain | Large-scale pretrain (runs ~4 days), Gate 3 assessed |
| 7-9 | Ablations | Masking, regularization, patch-length, scale experiments |
| 10-11 | Generalization | Cross-device, few-shot, noise robustness, Gate 4 assessed |
| 12 | Analysis | Visualizations, embedding analysis, paper outline |
| 13-16 | Writing | Draft, internal review, revision, submission |

**Total: ~4 months to submission-ready paper.**

---

## 9. Risks and Mitigations

**Risk: NHANES is accelerometer-only, HAR benchmarks need 6-axis.**
Mitigation: (a) Pretrain on 3ch, add zero-init projection for gyro channels
at transfer time; (b) alternatively, use cross-dataset pretraining (which has
all 6 channels) as the primary protocol and NHANES as a scaling experiment.

**Risk: JEPA collapses on stationary IMU segments (sitting/sleeping).**
Mitigation: SIGReg regularization; filter out low-variance windows during
pretraining; use the hybrid JEPA + reconstruction objective if needed.

**Risk: MoPFormer or another JEPA-for-HAR paper appears during your work.**
Mitigation: (a) Differentiate via SIGReg (no prior IMU work uses it);
(b) differentiate via cross-channel masking; (c) differentiate via the
NHANES large-scale pretraining angle. Also: move fast.

**Risk: Single-GPU pretraining limits batch size for contrastive baselines.**
Mitigation: JEPA does not need large batches (no negative pairs). This is
actually an advantage of the architecture you can highlight in the paper.
For contrastive baselines (SimCLR-HAR), use gradient accumulation.

**Risk: Resampling artifacts from 50/80Hz -> 100Hz.**
Mitigation: Use proper anti-aliasing (scipy.signal.resample_poly, not
linear interpolation). Report original and resampled rates. Alternatively,
use 50Hz as the standard rate (some papers do this).
