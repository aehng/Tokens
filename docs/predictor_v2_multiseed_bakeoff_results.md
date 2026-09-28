# Predictor V2 Multi-Seed Architecture Bake-Off Results

**Date:** 2026-09-28  
**Branch:** `codex/predictor-v2-methodology-correction`  
**Supervision:** Canonical Microsoft Phi-3.5-mini-instruct greedy continuation labels (36 TRAIN prompts)  
**Selection Split:** Strictly DEV split (12 prompts)  
**Seeds Evaluated:** 42, 43, 44 (for all neural models); Deterministic analytical fit for Ridge  

---

## Executive Summary

To eliminate single-seed initialization noise and establish definitive Pareto selection, all candidate architectures were evaluated across **3 random seeds (42, 43, 44)** on the strictly partitioned DEV split (12 prompts):

1. **Ridge Baseline:** Deterministic closed-form regression (44 parameters, 0.06 ms CPU latency). Achieves **31.35% Candidate-Pool capture** (221 DP steps, 28.1% precision).
2. **CNNRanker (Top Recommended Challenger):**
   - Achieves **44.78% $\pm$ 3.94% Candidate-Pool capture** (315.7 $\pm$ 27.8 steps, 42.0% $\pm$ 2.0% precision).
   - **Delta over Ridge:** **+13.43 percentage points**. Exceeds the +5.0 pp quality gate on **all 3 random seeds** (39.29%, 46.67%, 48.37%).
   - **CPU Inference Latency (L=512):** **3.21 ms**, operating safely inside the $\le 5.0\text{ ms}$ latency budget.
   - **Verdict:** **PASSED_RECOMMENDED** (Primary neural architecture recommendation).
3. **PooledMLP (High Initialization Variance):**
   - Achieves **34.42% $\pm$ 6.44% Candidate-Pool capture** (242.7 $\pm$ 45.4 steps).
   - Suffers severe seed sensitivity: while seeds 42 and 43 scored 37.87% and 40.00%, **seed 44 degraded to 25.39%**, falling below the deterministic Ridge baseline (31.35%).
   - Mean gain over Ridge is only +3.07 pp, failing the pre-registered Pareto gate.
   - **Verdict:** **FAILED_QUALITY_IMPROVEMENT_GATE**.
4. **GRURanker (Latency Disqualified):**
   - Achieves the highest raw capture (**49.51% $\pm$ 2.81%**, 349.0 $\pm$ 19.8 steps).
   - However, sequential unrolling results in **14.73 ms CPU latency at L=512** (and 28.61 ms at L=1024), exceeding the 5.0 ms TTFT budget by nearly 3x.
   - **Verdict:** **FAILED_LATENCY_BUDGET**.
5. **TransformerRanker (Dominated):**
   - Achieves **41.51% $\pm$ 2.49% capture** at **4.58 ms CPU latency** with 4.47M parameters.
   - Strictly dominated by CNNRanker across parameters (4.47M vs 3.23M), latency (4.58 ms vs 3.21 ms), and capture (41.51% vs 44.78%).

---

## 1. DEV Split Multi-Seed Evaluation (Mean $\pm$ Std Dev)

| Architecture | Parameters | CPU Latency (L=512) | Dev DP Steps (K=32) | % Candidate Pool Captured | Precision@32 | Occurrence AUPRC | Pareto Gate Verdict |
|---|---|---|---|---|---|---|---|
| **Ridge** (Baseline) | 44 | **0.06 ms** | 221.0 $\pm$ 0.0 | **31.35% $\pm$ 0.0%** | 28.1% $\pm$ 0.0% | 0.4108 $\pm$ 0.0000 | **BASELINE CONTROL** |
| **PooledMLP** | 3,194,503 | 1.39 ms | 242.7 $\pm$ 45.4 | 34.42% $\pm$ 6.44% | 29.2% $\pm$ 8.4% | 0.3603 $\pm$ 0.0367 | **FAILED QUALITY GATE** |
| **CNNRanker** | 3,225,367 | **3.21 ms** | **315.7 $\pm$ 27.8** | **44.78% $\pm$ 3.94%** | **42.0% $\pm$ 2.0%** | 0.3883 $\pm$ 0.0125 | **★ PASSED RECOMMENDED** |
| **GRURanker** | 3,275,143 | 14.73 ms | 349.0 $\pm$ 19.8 | 49.51% $\pm$ 2.81% | 43.5% $\pm$ 3.1% | 0.4427 $\pm$ 0.0273 | **FAILED LATENCY BUDGET** |
| **TransformerRanker** | 4,465,671 | 4.58 ms | 292.7 $\pm$ 17.6 | 41.51% $\pm$ 2.49% | 38.5% $\pm$ 3.2% | 0.3866 $\pm$ 0.0072 | **DOMINATED BY CNN** |

---

## 2. Seed Breakdown per Architecture

| Architecture | Seed | Dev DP Steps (K=32) | % Candidate Pool | Precision@32 | Occurrence AUPRC | Train Time |
|---|---|---|---|---|---|---|
| **Ridge** | 42 | 221 | 31.35% | 28.1% | 0.4108 | 0.0s |
| **PooledMLP** | 42 | 267 | 37.87% | 31.0% | 0.3636 | 5.9s |
| **PooledMLP** | 43 | 282 | 40.00% | 38.5% | 0.4035 | 5.3s |
| **PooledMLP** | 44 | 179 | 25.39% | 18.2% | 0.3139 | 5.4s |
| **CNNRanker** | 42 | 277 | 39.29% | 39.6% | 0.3733 | 11.2s |
| **CNNRanker** | 43 | 329 | 46.67% | 44.5% | 0.4039 | 10.9s |
| **CNNRanker** | 44 | 341 | 48.37% | 41.9% | 0.3877 | 10.8s |
| **GRURanker** | 42 | 377 | 53.48% | 39.6% | 0.4780 | 18.1s |
| **GRURanker** | 43 | 334 | 47.38% | 47.1% | 0.4114 | 17.6s |
| **GRURanker** | 44 | 336 | 47.66% | 43.8% | 0.4386 | 17.8s |
| **TransformerRanker** | 42 | 298 | 42.27% | 37.0% | 0.3813 | 16.5s |
| **TransformerRanker** | 43 | 311 | 44.11% | 43.0% | 0.3968 | 15.9s |
| **TransformerRanker** | 44 | 269 | 38.16% | 35.4% | 0.3818 | 16.3s |

---

## 3. Strict Pareto Decision Gate Analysis

A neural candidate qualifies over the Ridge baseline **if and only if**:
$$\text{Dev Capture}(\text{Neural}) \ge \text{Dev Capture}(\text{Ridge}) + 5.0\text{ pp} \quad \forall \text{ seeds}$$
$$\text{CPU Latency (L=512, p50)} \le 5.0\text{ ms}$$

### Evaluation Against Criteria:
- **CNNRanker:**
  - $\Delta$ over Ridge across seeds: $+7.94\text{ pp}$ (Seed 42), $+15.32\text{ pp}$ (Seed 43), $+17.02\text{ pp}$ (Seed 44).
  - All seeds strictly surpass the $+5.0\text{ pp}$ threshold.
  - Latency: $3.21\text{ ms} \le 5.0\text{ ms}$.
  - **Decision:** **SELECTED AS PRIMARY NEURAL RECOMMENDATION.**
- **PooledMLP:**
  - Seed 44 capture ($25.39\%$) is $-5.96\text{ pp}$ worse than Ridge.
  - Fails the multi-seed stability requirement.
- **GRURanker:**
  - Latency is $14.73\text{ ms} > 5.0\text{ ms}$.
  - Fails the inference latency budget.
