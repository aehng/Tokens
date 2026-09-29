# Predictor V2 Oracle Hierarchy & Candidate Loss Analysis

> **HISTORICAL / SUPERSEDED FOR CURRENT SELECTION:** The empirical totals below are from the earlier 60-prompt corpus. They are not corrected-canonical 900-record results and must not be used to select a current candidate generator.

> [!WARNING]
> **METHODOLOGY CORRECTION NOTICE (2026-09-28)**  
> **Epistemic Classification Standard Applied:** Metrics are strictly partitioned into `EXACT` (mathematically certified 0-1 ILP solver), `MEASURED` (verified executed code), `ESTIMATED` (multi-seed sample distribution), and `PROJECTED` (unmeasured conjectures).  
> **Oracle Exactness:** The lazy-greedy solver has been replaced with a mathematically certified 0-1 ILP CP-SAT solver (`ExactHypertokenOracle`). Solver status (`OPTIMAL` vs `FEASIBLE`) and optimality gaps are explicitly reported.  
> **Result:** In the shared Candidate Pool Oracle (Oracle B), 98.3% of prompts at K=32 are certified mathematically exact (`OPTIMAL`, gap=0.0). In the unconstrained Global Occurrence Oracle (Oracle A) across ~700-900 n-grams, 8.3% reach provable optimality within 10s, while the remainder achieve certified feasible bounds with recorded duality gap.

- Evaluated **Oracle A (Global Occurrence Ceiling)** and **Oracle B (Fixed Candidate-Pool Ceiling)** across all 60 benchmark prompts on Microsoft Phi-3.5-mini-instruct canonical continuations.
- At **K=32**, Global Occurrence Oracle saves **7911 decode steps** (131.8 steps/prompt).
- The shared prompt-only candidate pool captures **47.9%** (3786 steps), leaving **52.1% opportunity lost** to candidate generation recall.

## 1. Oracle Hierarchy by Codebook Budget K

| K | Oracle A (Global Ceiling) | Oracle B (Candidate Pool) | Candidate Capture % | Opportunity Lost (Steps) | Global Exact % | Pool Exact % |
|---|---|---|---|---|---|---|
| **K=8** | 3734 steps | 2656 steps | **71.1%** | 1078 steps (28.9%) | 6.7% | 88.3% |
| **K=16** | 5655 steps | 3431 steps | **60.7%** | 2224 steps (39.3%) | 6.7% | 95.0% |
| **K=32** | 7911 steps | 3786 steps | **47.9%** | 4125 steps (52.1%) | 8.3% | 98.3% |

## 2. Domain Breakdown (K=32)

| Domain | Global Oracle Steps | Candidate Pool Steps | Candidate Capture % | Opportunity Lost |
|---|---|---|---|---|
| **Code** | 2611 | 1024 | **39.2%** | 1587 steps |
| **Reasoning** | 2681 | 1723 | **64.3%** | 958 steps |
| **Instruction** | 2619 | 1039 | **39.7%** | 1580 steps |

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
