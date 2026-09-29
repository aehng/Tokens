# Predictor V2 Architecture Bake-Off Results

> **HISTORICAL / SUPERSEDED FOR CURRENT SELECTION:** This earlier small-corpus bakeoff and its legacy candidate labels are preserved as research history. Its 36 TRAIN / 12 DEV / 12 consumed diagnostic TEST results are not current architecture-selection evidence.

> [!WARNING]
> **METHODOLOGY CORRECTION NOTICE (2026-09-28)**  
> **Single-Seed vs Multi-Seed:** The table below reflects the initial single-seed run (seed 42). To control for neural weight initialization variance, multi-seed results across 3 random seeds (42, 43, 44) are provided in `docs/predictor_v2_multiseed_bakeoff_results.md`.  
> **Holdout Discipline:** Architecture selection decisions are based **exclusively on the DEV split**. The test split shown below was examined in the pilot and is treated as diagnostic data.

---

## Executive Summary

Empirical head-to-head evaluation of five lightweight candidate architectures trained strictly on canonical Vanilla Phi continuation labels on the deterministic TRAIN split (36 prompts) and evaluated on DEV (12 prompts) and FROZEN TEST (12 prompts).

---

## 1. DEV Split Head-to-Head Comparison (Architecture Selection)

| Architecture | Parameters | Model Size | Dev DP Steps (K=32) | % Candidate Pool | % Global Ceiling | Precision@32 | Dead Slot % | Occurrence AUPRC | Count MAE |
|---|---|---|---|---|---|---|---|---|---|
| **Ridge** | 44 | 0.51 KB | **221** | **31.8%** | 12.7% | 28.1% | 71.9% | 0.4108 | 0.96 |
| **PooledMLP** | 3,194,503 | 12483.37 KB | **346** | **49.7%** | 19.9% | 44.0% | 56.0% | 0.3988 | 0.57 |
| **CNNRanker** | 3,225,367 | 12606.03 KB | **319** | **45.8%** | 18.3% | 40.6% | 59.4% | 0.3698 | 0.50 |
| **GRURanker** | 3,275,143 | 12799.42 KB | **340** | **48.9%** | 19.5% | 40.4% | 59.6% | 0.4390 | 0.51 |
| **TransformerRanker** | 4,465,671 | 17453.01 KB | **266** | **38.2%** | 15.3% | 33.9% | 66.1% | 0.3798 | 0.56 |

---

## 2. FROZEN TEST Split Comparison (Diagnostic Holdout)

| Architecture | Parameters | Test DP Steps (K=32) | % Candidate Pool | % Global Ceiling | Precision@32 | Dead Slot % | Occurrence AUPRC | Count MAE |
|---|---|---|---|---|---|---|---|
| **Ridge** | 44 | **323** | **42.2%** | 18.5% | 37.2% | 62.8% | 0.4462 | 1.02 |
| **PooledMLP** | 3,194,503 | **316** | **41.3%** | 18.1% | 43.0% | 57.0% | 0.4150 | 0.62 |
| **CNNRanker** | 3,225,367 | **338** | **44.2%** | 19.4% | 44.8% | 55.2% | 0.4070 | 0.55 |
| **GRURanker** | 3,275,143 | **374** | **48.9%** | 21.4% | 44.0% | 56.0% | 0.4074 | 0.58 |
| **TransformerRanker** | 4,465,671 | **247** | **32.3%** | 14.2% | 32.3% | 67.7% | 0.3280 | 0.61 |

---

## 3. Scaling with Budget K (DEV Split)

| Architecture | K=8 DP Steps (% Cap) | K=16 DP Steps (% Cap) | K=32 DP Steps (% Cap) |
|---|---|---|---|
| **Ridge** | 120 (23.9%) | 155 (24.1%) | 221 (31.8%) |
| **PooledMLP** | 101 (20.1%) | 213 (33.1%) | 346 (49.7%) |
| **CNNRanker** | 84 (16.7%) | 165 (25.6%) | 319 (45.8%) |
| **GRURanker** | 131 (26.1%) | 218 (33.9%) | 340 (48.9%) |
| **TransformerRanker** | 138 (27.5%) | 180 (27.9%) | 266 (38.2%) |
