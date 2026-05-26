# Findings

## Curated Summary

**The optimal SSL objective for IMU HAR depends on the transfer regime.**

Across 5 HAR benchmarks in a cross-dataset leave-one-out protocol, we find:

- **Within-domain:** JEPA with SIGReg beats MAE by +3.8% macro-F1 (83.6% vs 79.9%,
  3 seeds, all wins). Latent prediction captures richer activity semantics than
  signal reconstruction.

- **Cross-domain:** Pure JEPA is worst (-2.8% vs MAE avg). Without signal-level
  anchoring, latent representations become "diffuse" and fail to transfer across
  sensor placements, devices, and activity vocabularies.

- **Hybrid (JEPA + 0.25×MAE):** Best average cross-domain transfer (65.5% vs MAE's
  65.3% vs JEPA's 62.4%). Recovers MAE's robustness while retaining JEPA's
  discrimination on compatible domains (UCI-HAR +3.7%, DSADS +4.7%).

This independently validates the LuMamba (2026) finding on EEG — combining latent
prediction with lightweight reconstruction yields the most robust representations
across heterogeneous sensor domains.

**Paper-ready claims (with caveats):**
1. First systematic comparison of JEPA vs MAE vs Hybrid objectives on IMU HAR
2. First application of LeJEPA's SIGReg to IMU data (provable collapse prevention)
3. Within-domain: JEPA > MAE (+3.8%, multi-seed confirmed)
4. Cross-domain: Hybrid >= MAE > JEPA (hybrid wins on avg, JEPA worst)
5. The reconstruction loss acts as a domain anchor for latent-prediction models

**Remaining work:**
- Multi-seed cross-dataset evaluation (currently single-seed)
- Alpha ablation (test 0.1, 0.25, 0.5 for hybrid mixing weight)
- Few-shot and noise robustness evaluation
- Comparison to published MoPFormer / LIMU-BERT numbers
- Fine-tune evaluation (current results are linear probe only)

---

## Raw Findings Log

### 2026-05-25 — EXP-005: Cross-Dataset — MAE vs JEPA vs Hybrid

| Held-out | MAE    | JEPA   | Hybrid | Best   |
|----------|--------|--------|--------|--------|
| UCI-HAR  | 0.8296 | 0.8575 | 0.8669 | HYBRID |
| PAMAP2   | 0.6781 | 0.6136 | 0.6447 | MAE    |
| DSADS    | 0.4600 | 0.4845 | 0.5072 | HYBRID |
| MHealth  | 0.6528 | 0.5814 | 0.6242 | MAE    |
| USC-HAD  | 0.6419 | 0.5842 | 0.6321 | MAE    |
| **AVG**  | **0.6525** | **0.6242** | **0.6550** | **HYBRID** |

### 2026-05-25 — EXP-004: Cross-Dataset — MAE vs JEPA (no hybrid)

JEPA wins 3/5 individual datasets but loses on average (-0.9% vs MAE).
Motivated the hybrid objective investigation.

### 2026-05-25 — EXP-003: Multi-Seed MAE vs JEPA on UCI-HAR

JEPA: 83.6% ± 0.10% vs MAE: 79.9% ± 0.31% macro-F1. JEPA wins 3/3 seeds.
JEPA has 3x lower variance — SIGReg stabilizes training.

### 2026-05-25 — EXP-002: IMU-JEPA (SIGReg) on UCI-HAR

JEPA 84.3% vs MAE 80.8% macro-F1 (+3.5%). Gate 2 cleared.

### 2026-05-25 — EXP-001: MAE-IMU on UCI-HAR

MAE 80.8% vs random 74.6% macro-F1. Pretraining works.
