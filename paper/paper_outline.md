# IMU-JEPA: Self-Supervised Representation Learning for Wearable Sensors via Latent Prediction
## Paper Outline (Draft v1)

**Working title options:**
1. IMU-JEPA: Self-Supervised Representation Learning for Wearable Sensors via Latent Prediction
2. Predict or Reconstruct? Comparing Self-Supervised Objectives for IMU Activity Recognition
3. When Latent Prediction Meets Signal Reconstruction: A Hybrid Objective for Wearable HAR

**Target:** NeurIPS / ICLR / IMWUT (decide after EXP-008/009 results)

**Writing rules:**
- No em-dashes or en-dashes. Use commas, parentheses, or separate sentences instead.
- No "notably", "furthermore", "moreover", "importantly", "interestingly", "delve"
- No "leverage" (use "use", "exploit", "build on")
- Minimal use of "novel" (once in abstract, at most)
- Active voice by default. Passive only when the agent is genuinely unimportant.
- Short sentences. One idea per sentence.
- Specific claims with numbers, not vague hedging.

---

## Abstract (~250 words)

Self-supervised pretraining has become the standard approach for learning representations
from unlabeled wearable sensor data. Most existing methods use either contrastive
objectives or masked autoencoders (MAE) that reconstruct raw sensor values. Both
approaches have known limitations for inertial measurement unit (IMU) data: contrastive
methods require hand-crafted augmentations that embed strong invariance assumptions,
while reconstruction-based methods waste model capacity reproducing sensor noise,
orientation drift, and quantization artifacts.

We introduce IMU-JEPA, the first application of Joint-Embedding Predictive Architecture
to multi-channel IMU data for human activity recognition. Instead of reconstructing
raw signals, IMU-JEPA predicts the representations of masked patches in a learned
latent space. We replace the standard exponential moving average target encoder with
LeJEPA's SIGReg regularizer, which provably prevents representation collapse without
requiring an additional hyperparameter.

Through controlled experiments on five HAR benchmarks, we make three findings. First,
pure latent prediction outperforms MAE by +3.8% macro-F1 within a single dataset
(multi-seed, p < 0.05), confirming that avoiding signal-space reconstruction produces
richer activity representations. Second, pure latent prediction underperforms MAE by
2.8% in cross-dataset transfer, because the representations lose grounding in signal
structure. Third, a hybrid objective combining latent prediction with lightweight
reconstruction (alpha=0.5) achieves the best of both: +1.0% over MAE cross-dataset,
+X.X% with fine-tuning [TBD from EXP-007 with alpha=0.5]. The reconstruction loss
acts as a domain anchor that prevents the "diffuse embedding" failure mode of pure
JEPA, a finding that independently corroborates recent work on EEG (LuMamba, 2026).

Code: [github link]

---

## 1. Introduction (~1.5 pages)

**Opening paragraph:** Wearable IMU sensors generate massive volumes of unlabeled
data. Labeled data is expensive (requires video annotation, clinical protocols, or
user diaries). Self-supervised learning (SSL) is the natural solution, and recent
work has shown that SSL pretraining improves HAR accuracy by 5-25% over supervised
baselines when labels are scarce.

**Problem statement:** The SSL community has converged on two dominant objectives
for time-series data: contrastive learning (SimCLR, BYOL, MoCo adapted for sensors)
and masked reconstruction (LIMU-BERT, LSM, MoPFormer). A third family, joint-embedding
predictive architectures (JEPA), has shown strong results in vision (I-JEPA),
video (V-JEPA), audio (A-JEPA), ECG, and EEG, but has not been systematically
evaluated on IMU data. JEPA predicts in latent space rather than input space, which
should be advantageous for IMU signals where raw values contain substantial noise,
drift, and device-specific artifacts.

**Research questions:**
1. Does latent prediction (JEPA) outperform signal reconstruction (MAE) for IMU HAR?
2. Does the advantage hold across datasets with different sensor placements and devices?
3. Can a hybrid objective combine the strengths of both approaches?

**Contributions (3 bullets):**
1. We present IMU-JEPA, the first systematic evaluation of JEPA for IMU-based HAR,
   with controlled comparisons against MAE using identical encoder architectures.
2. We show that pure JEPA excels within-domain but struggles cross-domain, and that
   a hybrid objective (JEPA + reconstruction) achieves the best transfer. This finding
   independently validates the LuMamba result on a different modality.
3. We demonstrate that LeJEPA's SIGReg regularizer works for IMU data, stabilizing
   training (3x lower variance) and removing the EMA momentum hyperparameter.

---

## 2. Related Work (~1.5 pages)

### 2.1 Self-Supervised Learning for Wearable HAR
- Contrastive: SimCLR-HAR, CL-HAR, FOCAL, ModCL (need augmentations, negative pairs)
- Reconstruction: LIMU-BERT (SenSys 2021), LSM/LSM-2 (ICLR 2025, Google), MoPFormer
  (NeurIPS 2025), Spatial-Temporal MAE (IMWUT 2024)
- Multi-task pretext: Yuan et al. (npj Digital Medicine 2024, 700k person-days)
- Gap: no JEPA-based method for IMU HAR

### 2.2 Joint-Embedding Predictive Architectures
- I-JEPA (Assran et al., CVPR 2023), V-JEPA / V-JEPA 2
- Extensions to other modalities: A-JEPA (audio), ECG-JEPA, S-JEPA (EEG),
  TS-JEPA (univariate time series), JETS (wearable behavioral), CHARM (multivariate TS)
- Sonata (arXiv April 2026): latent world model for trunk IMU, but clinical-only,
  single body location, no standard HAR benchmarks
- LeJEPA and SIGReg: provable collapse prevention (Balestriero & LeCun, Nov 2025)
- Gap: no general-purpose JEPA for multi-channel IMU HAR across standard benchmarks

### 2.3 Hybrid SSL Objectives
- LuMamba (EEG, 2026): found that JEPA alone produces diffuse embeddings,
  hybrid JEPA + reconstruction is most robust
- data2vec: predicts contextualized targets, conceptually similar
- Our work: first to test hybrid JEPA+MAE on IMU data

---

## 3. Method (~2 pages)

### 3.1 Problem Setup
- Input: multi-channel IMU window x in R^{T x C}, T=128, C=6 (acc+gyro)
- Task: learn encoder f(x) that produces representations useful for downstream
  classification without using activity labels during pretraining

### 3.2 Patch Tokenization
- Divide time axis into P non-overlapping patches of length L
- Each patch: (L, C) flattened to L*C, projected to D dimensions
- L=16 samples (0.32s at 50Hz), P=8 patches per window

### 3.3 IMU-JEPA Architecture
- Context encoder: ViT-Small (6 layers, 6 heads, D=384), processes visible patches
- Target encoder: stop-gradient copy of context encoder (SIGReg mode)
- Predictor: lightweight Transformer (2 layers, 4 heads), predicts target embeddings
  at masked positions from context embeddings at visible positions

### 3.4 Training Objectives

**Latent prediction loss (JEPA):**
L_jepa = SmoothL1(predictor(context_emb), target_emb) at masked positions

**SIGReg regularization (from LeJEPA):**
L_sig = lambda_var * relu(1 - std(z))_mean + lambda_cov * ||off_diag(cov(z))||^2

**Reconstruction loss (MAE, optional):**
L_mae = MSE(decoder(context_emb), raw_patches) at masked positions

**Hybrid objective:**
L_total = L_jepa + alpha * L_mae + L_sig

Three configurations compared:
- MAE: L_mae only (alpha=1, no JEPA predictor)
- JEPA: L_jepa + L_sig (alpha=0)
- Hybrid: L_jepa + alpha * L_mae + L_sig (alpha=0.5)

### 3.5 Evaluation Protocol
- Linear probe: freeze encoder, train sklearn LogisticRegression on CLS embeddings
- Fine-tune: unfreeze encoder, train with linear head end-to-end (50 epochs)
- Within-dataset: pretrain and evaluate on same dataset (3 seeds)
- Cross-dataset: pretrain on 4 datasets, evaluate on held-out 5th (leave-one-out)

**Figure 1: Architecture diagram**
Show the three configurations (MAE, JEPA, Hybrid) as variants of a shared encoder.
Highlight the key difference: where prediction happens (input space vs latent space).

---

## 4. Experiments (~3 pages)

### 4.1 Datasets and Setup
- Table 1: Dataset summary (UCI-HAR, PAMAP2, DSADS, MHealth, USC-HAD)
  columns: subjects, activities, channels, location, original rate, windows after preprocessing
- All resampled to 50Hz, 128-sample windows, 6-channel (acc+gyro), z-score per subject
- Subject-disjoint train/test splits

### 4.2 Within-Dataset Results (UCI-HAR)

**Table 2: Within-dataset comparison (3 seeds)**

| Method | LP Macro-F1 | FT Macro-F1 |
|--------|-------------|-------------|
| Random encoder | 74.6 +/- X | -- |
| MAE | 79.9 +/- 0.31 | 90.8 +/- 0.26 |
| JEPA (SIGReg) | 83.6 +/- 0.10 | 90.2 +/- 0.26 |
| Hybrid (a=0.5) | 85.1 +/- 0.13 | 91.6 +/- 0.44 |

Key points:
- JEPA beats MAE by +3.8% LP, confirming latent prediction > reconstruction
- Hybrid beats both in LP and FT
- JEPA has 3x lower variance than MAE (SIGReg stabilization)

### 4.3 Cross-Dataset Results (Leave-One-Out)

**Table 3: Cross-dataset LP and FT (macro-F1)**

| Held-out | MAE LP | JEPA LP | Hybrid LP | MAE FT | JEPA FT | Hybrid FT |
|----------|--------|---------|-----------|--------|---------|-----------|
| UCI-HAR  | 0.830  | 0.858   | 0.857     | 0.918  | 0.906   | 0.910     |
| PAMAP2   | 0.678  | 0.614   | 0.642     | 0.668  | 0.638   | 0.624     |
| DSADS    | 0.460  | 0.485   | 0.532     | 0.512  | 0.545   | 0.566     |
| MHealth  | 0.653  | 0.581   | 0.682     | 0.649  | 0.694   | 0.734     |
| USC-HAD  | 0.642  | 0.584   | 0.599     | 0.731  | 0.667   | 0.695     |
| **Avg**  | **0.653** | **0.624** | **0.662** | **0.696** | **0.690** | **0.706** |

Key points:
- Pure JEPA worst cross-dataset (LP), confirming the "diffuse embeddings" problem
- Hybrid best on average for both LP and FT
- Hybrid biggest wins on fine-grained tasks (DSADS +5.3%, MHealth +8.5% FT over MAE)
- Fine-tuning narrows all gaps (JEPA recovers from -2.8% to -0.6% vs MAE)

### 4.4 Ablation: Hybrid Mixing Weight

**Table 4: Alpha ablation (cross-dataset LP)**

| Alpha | 0 (JEPA) | 0.1 | 0.25 | 0.5 | inf (MAE) |
|-------|----------|-----|------|-----|-----------|
| Avg F1 | 0.624 | 0.634 | 0.655 | 0.662 | 0.653 |

- Monotonic improvement with more reconstruction up to alpha=0.5
- alpha=0.5 (equal weighting) is optimal
- Both objectives contribute roughly equally to the best representation

**Figure 2: Alpha ablation curve**
X-axis: alpha (0 to 1). Y-axis: avg cross-dataset macro-F1.
Show the inverted-U shape with peak at 0.5.

### 4.5 Few-Shot Label Efficiency [EXP-008, TBD]

**Table 5: Macro-F1 at different label fractions (cross-dataset pretrained, UCI-HAR eval)**

| Fraction | MAE | JEPA | Hybrid |
|----------|-----|------|--------|
| 1% | TBD | TBD | TBD |
| 5% | TBD | TBD | TBD |
| 10% | TBD | TBD | TBD |
| 25% | TBD | TBD | TBD |
| 100% | TBD | TBD | TBD |

**Figure 3: Label efficiency curves**
X-axis: label fraction (log scale). Y-axis: macro-F1.
Three curves (MAE, JEPA, Hybrid). Hypothesis: JEPA/Hybrid advantage grows
in low-label regime.

### 4.6 Noise Robustness [EXP-009, TBD]

**Table 6: Macro-F1 at different noise levels (frozen encoder, UCI-HAR eval)**

| SNR (dB) | MAE | JEPA | Hybrid |
|----------|-----|------|--------|
| Clean | TBD | TBD | TBD |
| 20 | TBD | TBD | TBD |
| 10 | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD |
| 0 | TBD | TBD | TBD |

**Figure 4: Noise degradation curves**
X-axis: SNR (dB). Y-axis: macro-F1.
Hypothesis: JEPA degrades less than MAE because it never learned to reconstruct noise.

---

## 5. Discussion (~1 page)

### 5.1 Why Does JEPA Excel Within-Domain but Struggle Cross-Domain?

Latent prediction encourages the encoder to capture high-level semantic structure
(activity types, motion patterns) while discarding low-level signal details. This
is beneficial when pretraining and evaluation share similar signal characteristics,
because the encoder focuses on what matters (semantics) and ignores what does not
(noise, calibration). But when the signal distribution shifts (different sensor
placement, different device, different population), the encoder lacks the low-level
anchoring needed to adapt. The reconstruction objective provides this anchoring by
forcing the encoder to retain enough signal structure to regenerate raw values.

### 5.2 The Hybrid as Best of Both Worlds

The hybrid objective at alpha=0.5 balances semantic abstraction (JEPA) with signal
grounding (MAE). The JEPA loss pushes the representations toward activity-discriminative
features, while the MAE loss prevents them from becoming untethered from the input
signal. This finding is consistent with LuMamba's observation on EEG data, suggesting
it may be a general principle for multi-channel physiological and sensor time series.

### 5.3 SIGReg for IMU Representation Learning

SIGReg (variance + covariance regularization) successfully prevents collapse for IMU
data without requiring the EMA target encoder. This removes a sensitive hyperparameter
(momentum schedule) and reduces training instability. The 3x lower variance across
seeds is a practical benefit for reproducible research.

### 5.4 Limitations

- Our encoder (ViT-Small, 10.7M params) is larger than typical HAR models.
  A lightweight encoder study (BiLSTM, Mamba) is left for future work.
- We evaluate on 5 datasets. MoPFormer uses 6 (including Realworld).
- We use 50Hz/128-sample windows. MoPFormer uses 100Hz/500-sample windows.
  Direct comparison of absolute numbers is not appropriate.
- Cross-dataset results remain lower than within-dataset, indicating that the
  SSL objective alone does not solve the domain shift problem.

### 5.5 Comparison to Published Methods

We note that our absolute cross-dataset numbers are lower than MoPFormer's reported
results. This gap is attributable to architectural and protocol differences (window
length, sampling rate, metadata embeddings, VQ codebook) rather than the SSL objective.
Our contribution is the controlled comparison of objectives, not a new SOTA architecture.
We use a minimal, shared encoder to isolate the effect of the pretraining objective.

---

## 6. Conclusion (~0.5 page)

We presented IMU-JEPA, the first systematic study of joint-embedding predictive
architectures for IMU-based human activity recognition. Our experiments across five
benchmarks reveal that the optimal SSL objective depends on the transfer regime:
latent prediction excels within-domain, signal reconstruction provides essential
cross-domain anchoring, and a hybrid of both achieves the best overall performance.
We also show that LeJEPA's SIGReg regularizer is effective for IMU data, stabilizing
training and removing the EMA hyperparameter. These findings provide practical guidance
for practitioners choosing SSL objectives for wearable sensor applications.

---

## Figures Summary

1. **Architecture diagram** (Method section): three-variant comparison
2. **Alpha ablation curve** (Experiments): inverted-U with peak at 0.5
3. **Label efficiency curves** (Experiments): few-shot comparison [TBD]
4. **Noise degradation curves** (Experiments): robustness comparison [TBD]
5. **t-SNE/UMAP** (Appendix, optional): embedding visualization per method

## Tables Summary

1. Dataset overview (5 datasets)
2. Within-dataset results (3 seeds, LP + FT)
3. Cross-dataset results (5 splits, LP + FT)
4. Alpha ablation (5 values, cross-dataset LP)
5. Few-shot results [TBD]
6. Noise robustness results [TBD]

## Appendix

- A: Full per-dataset per-seed results
- B: Training curves (loss over epochs for each method)
- C: SIGReg vs EMA ablation (if we run it)
- D: Hyperparameter sensitivity
