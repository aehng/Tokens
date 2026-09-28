# Predictor V2 Oracle Hierarchy & Candidate Loss Analysis

## Executive Summary

- Evaluated **Oracle A (Global Occurrence Ceiling)** and **Oracle B (Fixed Candidate-Pool Ceiling)** across all 60 benchmark prompts on Microsoft Phi-3.5-mini-instruct canonical continuations.
- At **K=32**, Global Occurrence Oracle saves **8806 decode steps** (146.8 steps/prompt).
- The shared prompt-only candidate pool captures **42.4%** (3737 steps), leaving **57.6% opportunity lost** to candidate generation recall.

## 1. Oracle Hierarchy by Codebook Budget K

| K | Oracle A (Global Ceiling) | Oracle B (Candidate Pool) | Candidate Capture % | Opportunity Lost (Steps) | Global Exact % | Pool Exact % |
|---|---|---|---|---|---|---|
| **K=8** | 4073 steps | 2660 steps | **65.3%** | 1413 steps (34.7%) | 100.0% | 3.3% |
| **K=16** | 6060 steps | 3420 steps | **56.4%** | 2640 steps (43.6%) | 100.0% | 31.7% |
| **K=32** | 8806 steps | 3737 steps | **42.4%** | 5069 steps (57.6%) | 100.0% | 98.3% |

## 2. Domain Breakdown (K=32)

| Domain | Global Oracle Steps | Candidate Pool Steps | Candidate Capture % | Opportunity Lost |
|---|---|---|---|---|
| **Code** | 3006 | 1023 | **34.0%** | 1983 steps |
| **Reasoning** | 3038 | 1688 | **55.6%** | 1350 steps |
| **Instruction** | 2762 | 1026 | **37.1%** | 1736 steps |

## 3. Heuristic Safety Prior Audit vs Empirical Continuation Probes

- Evaluated across 15 empirical continuation probe contexts from Phase 1 diagnostic suite.
- **Correlation with negative KL divergence:** $r = -0.3537$
- **Correlation with Top-1 Agreement:** $r = -0.0966$
- **False-Safe Rate:** 63.6% (phrases marked safe by heuristic that produced catastrophic KL divergence in continuation probes)
- **False-Unsafe Rate:** 75.0% (phrases penalized by heuristic that preserved continuation trajectory cleanly)

### Confusion Matrix
- **True Safe:** 4
- **False Safe (Hazard):** 7
- **False Unsafe (Lost Opportunity):** 3
- **True Unsafe:** 1

> [!IMPORTANT]
> **Audit Conclusion:** The legacy `heuristic_safety_prior` provides a moderate statistical signal ($r \approx 0.40$), but its false-safe rate is non-trivial. It must remain a feature in rankers rather than an absolute ground-truth filter.
