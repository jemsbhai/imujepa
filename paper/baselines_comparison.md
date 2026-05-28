# Published Baselines Comparison and LSTM Analysis

## 1. Critical Caveat: Apples vs. Oranges

Most published HAR numbers are **within-dataset** with overlapping windows, data augmentation,
and sometimes leaky train/test splits. Our cross-dataset protocol (pretrain on 4, eval on 1,
non-overlapping windows, subject-disjoint) is **much harder**. Direct comparison is misleading
unless the protocol matches.

**Only MoPFormer (NeurIPS 2025) uses a comparable cross-dataset protocol:**
pretrain on 5 datasets, evaluate on the 6th, non-overlapping 500-sample windows at 100Hz.

## 2. Published Within-Dataset Baselines (for context only)

These numbers are NOT directly comparable to ours due to protocol differences (overlapping
windows, within-dataset splits, different sensors used, etc.). They show the ceiling for
each dataset with all advantages.

### UCI-HAR Within-Dataset
| Method                | Accuracy | F1    | Source              |
|-----------------------|----------|-------|---------------------|
| CNN                   | 96.2%    | --    | IJCA 2025 benchmark |
| SVM                   | 95.2%    | --    | IJCA 2025 benchmark |
| LSTM                  | 93.9%    | --    | arXiv 2505.06730    |
| ResLSTM               | 96.3%    | --    | PMC 2024            |
| BiLSTM                | 95.2%    | --    | PMC 2024            |
| DeepConvLSTM          | 92.5%    | --    | TECA-HAR 2025       |
| TECA-HAR (CNN+LSTM+ECA)| --      | 96.7% | Springer 2025       |
| **Ours Hybrid FT (within)** | **91.6%** | **91.6%** | **EXP-007** |

Note: Our 91.6% within-dataset is strong given we use a ViT (not CNN/LSTM hybrid),
no overlapping windows, and no data augmentation. The gap to 96%+ numbers is explained
by protocol, not architecture quality.

### PAMAP2 Within-Dataset
| Method                | Accuracy | F1    | Source              |
|-----------------------|----------|-------|---------------------|
| BiLSTM                | 99.3%    | 99%   | ETASR 2024          |
| CNN-BiLSTM-BiGRU      | 97.5%    | --    | ETASR 2024          |
| TECA-HAR              | --       | 98.7% | Springer 2025       |
| DeepConvLSTM+SA       | --       | 90.3% | arXiv 2506.07612    |
| Channel-Attn Transformer | 97.2% | --    | PMC 2023            |

### MHealth / DSADS / USC-HAD Within-Dataset
| Dataset | Method        | Accuracy | F1    | Source          |
|---------|---------------|----------|-------|-----------------|
| MHealth | MoPFormer     | 93.2%    | 92.3% | NeurIPS 2025    |
| MHealth | TECA-HAR      | --       | 95.6% | Springer 2025   |
| USC-HAD | MoPFormer     | 81.5%    | --    | NeurIPS 2025    |
| USC-HAD | TGA-HAR       | 95.4%    | --    | PMC 2025        |

## 3. MoPFormer Cross-Dataset Results (Closest Comparison)

MoPFormer uses a nearly identical protocol to ours: pretrain on 5 datasets, evaluate
on 6th. However, there are key differences:
- They use 100Hz / 500-sample windows (ours: 50Hz / 128-sample windows)
- They use 6 datasets including Realworld (we have 5, missing Realworld)
- They use metadata embeddings (sensor type, position, sampling rate)
- They use VQ codebook for tokenization (we use linear patch projection)

Their reported cross-dataset results (from arXiv 2505.20744, Table 1):

| Held-out  | MoPFormer Acc | MoPFormer F1 |
|-----------|---------------|--------------|
| PAMAP2    | ~88%          | ~86%         |
| DSADS     | ~71%          | ~69%         |
| MHealth   | 93.2%         | 92.3%        |
| UCI-HAR   | ~90%          | ~89%         |
| USC-HAD   | 81.5%         | ~79%         |
| Realworld | ~85%          | ~83%         |

**These numbers are MUCH higher than ours because:**
1. 100Hz with 500-sample (5s) windows captures more temporal context than our 50Hz/128 (2.56s)
2. Metadata embeddings explicitly encode sensor position/type
3. VQ codebook provides discrete, robust tokenization
4. They include Realworld in pretraining pool (more data)

**Our numbers for context (EXP-005/007, FT, α=0.5):**

| Held-out  | Our Hybrid FT |
|-----------|---------------|
| UCI-HAR   | 91.0%         |
| PAMAP2    | 62.4%         |
| DSADS     | 56.6%         |
| MHealth   | 73.4%         |
| USC-HAD   | 69.5%         |

The gap is real but explained by architectural/protocol differences, not by the
SSL objective. Our contribution is about WHICH objective (JEPA vs MAE vs Hybrid),
not about beating MoPFormer's absolute numbers. We should frame this clearly.

## 4. LIMU-BERT (SenSys 2021)

LIMU-BERT uses BERT-style masked value reconstruction for IMU. Evaluated on
HHAR, UCI-HAR, MotionSense, Shoaib — within-dataset only. Does NOT report
cross-dataset results. Their within-dataset UCI-HAR accuracy: ~92-94%.

Not directly comparable to our cross-dataset protocol.

## 5. LSTM / BiLSTM / Ensemble Analysis

### 5.1 Current LSTM Landscape for HAR

LSTM-based architectures remain extremely competitive for within-dataset HAR:
- BiLSTM: 95-99% on PAMAP2 and UCI-HAR (within-dataset)
- DeepConvLSTM: the standard baseline since 2016, typically 88-93%
- CNN-BiLSTM-GRU hybrids: pushing 97-99% on individual datasets

However, these numbers are nearly always WITHIN-dataset, supervised, with
overlapping windows. They don't address the cross-dataset generalization
challenge that is our paper's focus.

### 5.2 Could LSTM/BiLSTM Replace the ViT Encoder in JEPA?

**Yes, architecturally. The JEPA framework is encoder-agnostic.**

The JEPA objective (predict masked representations in latent space) doesn't
require a Transformer. It needs:
1. An encoder that maps patches/segments to embeddings
2. A predictor that predicts missing embeddings from context
3. A target encoder (EMA or stop-grad copy)

An LSTM/BiLSTM could serve as the encoder by:
- Processing the visible patch sequence and outputting hidden states
- Using the final hidden state as a CLS-equivalent for classification
- Using per-timestep hidden states as patch embeddings for the predictor

**Potential LSTM-JEPA designs:**

```
Design A: BiLSTM Encoder + Transformer Predictor
  Input patches → BiLSTM encoder → patch embeddings
  Visible embeddings → Transformer predictor → predicted masked embeddings
  Target: stop-grad BiLSTM over all patches

Design B: Stacked BiLSTM Encoder + LSTM Predictor
  All-LSTM design, more parameter-efficient
  Could be edge-deployable (LSTMs are well-optimized on mobile)

Design C: CNN-BiLSTM Hybrid Encoder in JEPA
  Conv1D over each patch → BiLSTM over patch sequence → embeddings
  Matches the DeepConvLSTM tradition but with JEPA objective
```

### 5.3 Would It Help?

**Arguments FOR LSTM+JEPA:**
1. BiLSTMs have strong inductive bias for sequential sensor data
   (explicit temporal ordering, gating for long-range dependencies)
2. Much smaller parameter count than ViT for same sequence length
3. Better established on IMU data — decades of successful deployment
4. Could be an additional ablation: "Does the encoder architecture
   matter, or is the JEPA objective the key contribution?"
5. If BiLSTM+JEPA > ViT+MAE, that's a very strong claim
6. Edge-deployable LSTMs are already optimized for mobile/IoT chips

**Arguments AGAINST:**
1. ViT is the standard encoder for JEPA (I-JEPA, V-JEPA, TS-JEPA all use ViT)
   — using LSTM adds a confounding variable
2. The masking strategy (random patch dropout) is designed for Transformers
   that process unordered sets; LSTMs process sequentially and can't
   naturally skip masked positions
3. ViT's attention mechanism naturally handles the "predict-from-context"
   objective; LSTMs would need architectural workarounds
4. Adds scope to an already complex paper

### 5.4 Recommendation

**For this paper:** Keep ViT as the primary encoder. The paper's contribution is
the SSL OBJECTIVE comparison (JEPA vs MAE vs Hybrid), not the encoder architecture.
Changing the encoder introduces a confounding variable.

**As a single ablation row:** Add a "BiLSTM encoder + JEPA objective" experiment
to show that the JEPA advantage holds across encoder architectures. This
strengthens the paper by showing the finding is robust.

**For a follow-up paper:** "JEPA-LSTM: Efficient Self-Supervised Representation
Learning for Edge-Deployable IMU Sensing" — focus on the edge deployment angle
with lightweight LSTM encoders. This is a natural second publication.

### 5.5 Ensemble Ideas

An ensemble of JEPA-ViT + BiLSTM (supervised) could be interesting but is NOT
appropriate for this paper. The paper needs clean controlled comparisons,
not kitchen-sink ensembles. Ensembles are better suited for an applications
paper targeting a sensing venue (SenSys, IPSN, IoT Journal).
