# Predictor V2 Pareto Analysis & End-to-End Loss Funnel

## Executive Summary

- Selected **TOP TWO ARCHITECTURES** for live attribution testing: **Ridge** and **PooledMLP**.
- **Primary Bottleneck Identified:** Candidate generation is the dominant loss stage (57.6% of global ceiling lost before ranking).

## 1. Pareto Frontier & Compute Tradeoff (Prompt L=512)

| Architecture | Params | Model Size | CPU Latency p50 | CPU Latency p90 | Cold Latency | Dev % Cand Pool | Dev % Global | Status |
|---|---|---|---|---|---|---|---|---|
| **Ridge** | 44 | 0.51 KB | **0.06 ms** | 0.07 ms | 0.16 ms | **31.8%** | 13.5% | **Pareto Frontier** (★ Selected) |
| **PooledMLP** | 3,194,503 | 12483.37 KB | **1.39 ms** | 1.85 ms | 1.89 ms | **49.7%** | 21.1% | **Pareto Frontier** (★ Selected) |
| **CNNRanker** | 3,225,367 | 12606.03 KB | **3.21 ms** | 4.48 ms | 6.10 ms | **45.8%** | 19.4% | Dominated (by PooledMLP) |
| **TransformerRanker** | 4,465,671 | 17453.01 KB | **4.58 ms** | 5.26 ms | 5.03 ms | **38.2%** | 16.2% | Dominated (by PooledMLP) |
| **GRURanker** | 3,275,143 | 12799.42 KB | **14.73 ms** | 19.56 ms | 15.33 ms | **48.9%** | 20.7% | Dominated (by PooledMLP) |

## 2. Latency Across Prompt Lengths (CPU)

| Architecture | L=128 p50 | L=512 p50 | L=1024 p50 | L=1024 p90 |
|---|---|---|---|---|
| **Ridge** | 0.06 ms | 0.06 ms | 0.07 ms | 0.08 ms |
| **PooledMLP** | 1.23 ms | 1.39 ms | 1.02 ms | 1.82 ms |
| **CNNRanker** | 2.14 ms | 3.21 ms | 4.93 ms | 5.24 ms |
| **GRURanker** | 4.54 ms | 14.73 ms | 26.63 ms | 34.81 ms |
| **TransformerRanker** | 2.57 ms | 4.58 ms | 7.19 ms | 7.69 ms |

## 3. End-to-End Opportunity Loss Funnel (K=32, 60 Prompts)

| Stage | Realized / Available Steps | % of Global Ceiling | Incremental Loss (Steps) | Loss % | Primary Mechanism |
|---|---|---|---|---|---|
| **1. Global Occurrence Oracle** | 8806 steps | 100.0% | 0 | 0.0% | Theoretical physical maximum |
| **2. Fixed Candidate Pool** | 3737 steps | 42.4% | -5069 | -57.6% | Candidate generator recall deficit |
| **3. Best Offline Ranker** | 1186 steps | 13.5% | -2551 | -29.0% | Scorer ranking & slot budgeting errors |
| **4. Continuation Safety Adjusted** | 534 steps | 6.1% | -652 | -7.4% | Destabilizing phrases filtered out |
| **5. Live Hypertoken Emission** | 374 steps | 4.2% | -160 | -1.8% | Model fails to emit valid seeded token |
| **6. Quality-Preserved Savings** | 318 steps | 3.6% | -56 | -0.6% | Continuation divergence / truncation |

> [!IMPORTANT]
> **Key Funnel Finding:** Candidate Generation Loss is by far the largest single drop in the entire pipeline (-57.6%). Expanding prompt candidate recall (e.g. through learned prefix retrieval or association expansion) will yield far more net compression than further scaling the ranker parameter count.
