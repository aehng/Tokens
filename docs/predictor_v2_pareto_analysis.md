# Predictor V2 Pareto Analysis & End-to-End Loss Funnel

> **HISTORICAL / SUPERSEDED FOR CURRENT SELECTION:** The architecture and bottleneck conclusions below come from earlier small-corpus/legacy candidate-label experiments. They are retained as history and do not replace broader live Phi failure attribution on the corrected DEV data.

> [!WARNING]
> **METHODOLOGY CORRECTION NOTICE (2026-09-28)**  
> **Epistemic Classification Standard Applied:** Metrics are strictly partitioned into `EXACT` (mathematically certified 0-1 ILP solver), `MEASURED` (verified executed code), `ESTIMATED` (multi-seed sample distribution), and `PROJECTED` (unmeasured conjectures).  
> **Funnel Correction:** Former stages 4 ("Continuation Safety Adjusted"), 5 ("Live Hypertoken Emission"), and 6 ("Quality-Preserved Savings") were calculated using static multiplier approximations ($\times 0.45, \times 0.70, \times 0.85$) rather than executed experiments. They have been removed from the empirical measured funnel and marked `NOT YET MEASURED`.

---

## Executive Summary

- **Architecture Selection (DEV Split):** **PooledMLP** achieves the best quality-latency trade-off (1.39 ms CPU latency, beating Ridge baseline by $>15\text{ pp}$ across random seeds). **Ridge** provides a deterministic ultra-low-latency control (0.06 ms).
- **Primary Bottleneck Identified:** Candidate generation is the dominant measured loss stage (-57.6% of the Global Occurrence ceiling is lost before any ranking occurs).

---

## 1. Pareto Frontier & Compute Tradeoff (Prompt L=512)

| Architecture | Params | Model Size | CPU Latency p50 | CPU Latency p90 | Cold Latency | Dev % Cand Pool | Dev % Global | Pareto Status |
|---|---|---|---|---|---|---|---|---|
| **Ridge** | 44 | 0.51 KB | **0.06 ms** | 0.07 ms | 0.16 ms | **31.8%** | 13.5% | **Pareto Frontier** (Baseline Control) |
| **PooledMLP** | 3,194,503 | 12483.37 KB | **1.39 ms** | 1.85 ms | 1.89 ms | **49.7%** | 21.1% | **Pareto Frontier** (★ Selected) |
| **CNNRanker** | 3,225,367 | 12606.03 KB | **3.21 ms** | 4.48 ms | 6.10 ms | **45.8%** | 19.4% | Dominated (by PooledMLP) |
| **TransformerRanker** | 4,465,671 | 17453.01 KB | **4.58 ms** | 5.26 ms | 5.03 ms | **38.2%** | 16.2% | Dominated (by PooledMLP) |
| **GRURanker** | 3,275,143 | 12799.42 KB | **14.73 ms** | 19.56 ms | 15.33 ms | **48.9%** | 20.7% | Dominated / Latency Exceeded |

---

## 2. Latency Across Prompt Lengths (CPU)

| Architecture | L=128 p50 | L=512 p50 | L=1024 p50 | L=1024 p90 |
|---|---|---|---|---|
| **Ridge** | 0.06 ms | 0.06 ms | 0.07 ms | 0.08 ms |
| **PooledMLP** | 1.23 ms | 1.39 ms | 1.02 ms | 1.82 ms |
| **CNNRanker** | 2.14 ms | 3.21 ms | 4.93 ms | 5.24 ms |
| **GRURanker** | 4.54 ms | 14.73 ms | 26.63 ms | 34.81 ms |
| **TransformerRanker** | 2.57 ms | 4.58 ms | 7.19 ms | 7.69 ms |

---

## 3. Empirical Measured Opportunity Loss Funnel (K=32, 60 Prompts)

The empirical funnel encompasses **strictly measured** stages with verified code execution:

| Stage | Name | Epistemic Status | Decode Steps Saved | % of Global Ceiling | Incremental Loss | Primary Mechanism |
|---|---|---|---|---|---|---|
| **1** | **Global Occurrence Oracle** | **EXACT** | 8,806 steps | 100.0% | 0 | Theoretical physical ceiling of base model continuation |
| **2** | **Fixed Candidate Pool Oracle** | **EXACT** | 3,737 steps | 42.44% | **-5,069 steps (-57.56%)** | **Prompt candidate generator recall deficit** |
| **3** | **Best Offline Ranker (PooledMLP)** | **MEASURED** | 1,858 steps | 21.10% | -1,879 steps (-21.34%) | Scorer ranking & top-K slot allocation errors |

> [!IMPORTANT]
> **Dominant Bottleneck Finding:** Candidate Generation Loss (-57.6%) is by far the largest single drop in the entire pipeline. Expanding prompt candidate recall (e.g. through learned prefix retrieval or association expansion) will yield far more net compression than further scaling the ranker parameter count.

---

## 4. Hypothetical Downstream Funnel (Unmeasured Projections)

> [!WARNING]
> **The following downstream stages are NOT YET MEASURED.** They are unverified projections derived from heuristic scaling and must not be cited as empirical findings until executed via live model decoding.

| Projected Stage | Status | Heuristic Multiplier | Projected Steps | Projected % Ceiling | Verification Requirement |
|---|---|---|---|---|---|
| **Continuation Safety Filter** | **PROJECTED (NOT YET MEASURED)** | $\sim 0.45\times$ | ~836 steps | ~9.5% | Contextual continuation KL divergence measurement |
| **Live Hypertoken Emission** | **PROJECTED (NOT YET MEASURED)** | $\sim 0.70\times$ | ~585 steps | ~6.6% | Realized token emission on live vLLM serving engine |
| **Quality-Preserved Net Savings** | **PROJECTED (NOT YET MEASURED)** | $\sim 0.85\times$ | ~497 steps | ~5.6% | End-to-end task accuracy benchmark (MBPP / GSM8K) |
