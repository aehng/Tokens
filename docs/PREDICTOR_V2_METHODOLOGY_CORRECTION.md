# Predictor V2 Methodology Correction and Experimental Integrity Specification

**Date:** 2026-09-28  
**Branch:** `codex/predictor-v2-methodology-correction`  
**Parent Implementation SHA:** `8c12d059115fb2ed0dde8bdfa09020d2bd90888e`  
**Status:** Methodological Correction & Strict Ground-Truth Realignment  

---

## 1. Executive Summary & Epistemic Taxonomy

The initial Predictor V2 pilot (`codex/predictor-v2-architecture-bakeoff`) introduced foundational infrastructure for multi-domain evaluation and hypertoken ranking. However, a rigorous methodological audit revealed several critical scientific defects:
1. **Uncertified "Exact" Oracle Claims:** The "Global Occurrence Oracle" and "Candidate-Pool Oracle" used heuristic lazy-greedy forward selection over dynamic programming token tiling, yet were claimed to be "100% exact" because the priority heap was exhausted. String tiling over combinatorial subsets is not submodular in the presence of overlapping phrases; heap exhaustion does not certify a global optimum.
2. **Fabricated Downstream Funnel Stages:** Stages 4 (Continuation Safety), 5 (Live Hypertoken Emission), and 6 (Quality-Preserved Savings) in the "End-to-End Opportunity Loss Funnel" were computed using arbitrary static multipliers ($\times 0.45, \times 0.70, \times 0.85$) rather than executed experiments, yet were presented as measured empirical findings.
3. **Pristine Holdout Contamination:** The 12-prompt test set from the 60-prompt pilot was inspected and evaluated during the initial bake-off run, disqualifying it from serving as an uncompromised holdout.
4. **Single-Seed Neural Evaluation:** Neural models (PooledMLP, CNN, GRU, Transformer) were trained on a single random seed, creating unquantified variance in architecture rankings.
5. **Supervision Mismatch:** Supervised labels in historical datasets were derived from human/dataset reference responses (`data/train.jsonl`), not the frozen base model's actual autoregressive continuations.

To permanently prevent epistemological drift, all metrics, claims, and analyses in this project must adhere strictly to the following **Epistemic Taxonomy**:

| Term | Mathematical / Operational Definition | Permissible Usage |
|---|---|---|
| **EXACT** | Certified by an exact mathematical solver (0-1 ILP / CP-SAT) with proven global optimality (`OPTIMAL` status, zero optimality gap $\epsilon = 0.0$), or exhaustive combinatorial search. | Only for certified mathematical oracles over defined token sequences. |
| **MEASURED** | Result of an actually executed experiment or evaluation pipeline on verified, un-leaked empirical data. | For offline ranker DP steps, inference latency on benchmark hardware, and empirical continuation KL probes. |
| **ESTIMATED** | Statistically inferred quantity from empirical sample data (e.g., bootstrap confidence intervals, cross-validated sample means $\mu \pm \sigma$). | For multi-seed performance distributions and domain capture estimates. |
| **PROJECTED** | Hypothetical extrapolation, analytical conjecture, or rule-of-thumb scaling. | **MUST NOT** appear in empirical measured tables. Permissible only in designated hypothetical sections labeled `NOT YET MEASURED`. |

---

## 2. Mathematical Formalization of the Exact Oracle (0-1 ILP Formulation)

### 2.1 Why Lazy Greedy Fails Global Exactness
Let continuation token sequence $T = (t_0, t_1, \dots, t_{N-1})$. Let candidate phrase set be $\mathcal{G}$, where each $g \in \mathcal{G}$ has token length $|g| \in [2, 4]$.
A greedy algorithm iteratively selects $g^* = \arg\max_{g \notin S} (\text{DP}(S \cup \{g\}) - \text{DP}(S))$.
Because phrases overlap, nest, and conflict at token positions, the set function $f(S) = \text{tokens\_saved}(\text{DP}(S))$ is **not submodular**. In non-submodular maximization:
- The marginal gain of an element can *increase* when other elements are added (supermodularity from bridging disjoint occurrences).
- The marginal gain upper bound maintained in a lazy heap can become invalid or fail to bound non-selected combinations.
- A heap becoming empty only indicates that no remaining singleton candidate has positive marginal gain *with respect to the greedy path chosen*; it does **not** prove that an alternative subset of size $\le K$ could not achieve a strictly higher objective.

### 2.2 Exact 0-1 Integer Linear Programming / CP-SAT Formulation
We formulate hypertoken codebook selection as an exact 0-1 Integer Linear Program:

**Variables:**
- $y_g \in \{0, 1\}$ for each unique phrase type $g \in \mathcal{G}$ ($y_g = 1$ if phrase $g$ is included in the codebook).
- $x_o \in \{0, 1\}$ for each concrete occurrence interval $o = (g_o, \text{start}_o, \text{end}_o)$ in $T$ ($x_o = 1$ if occurrence $o$ is emitted as a hypertoken).

**Objective:**
Maximize total decode steps saved:
$$\max \sum_{o} (|g_o| - 1) \cdot x_o$$

**Constraints:**
1. **Codebook Inclusion Constraint:** An occurrence can only be emitted if its phrase type is in the codebook:
   $$x_o \le y_{g_o} \quad \forall o$$
2. **Codebook Budget Constraint:** At most $K$ phrase types may be selected:
   $$\sum_{g \in \mathcal{G}} y_g \le K$$
3. **Non-Overlapping Emission Constraint:** At every base token position $p \in [0, N-1]$, at most one hypertoken occurrence can cover position $p$:
   $$\sum_{o : \text{start}_o \le p < \text{end}_o} x_o \le 1 \quad \forall p \in [0, N-1]$$

**Contract Equivalence with DP Segmentation:**
Any feasible assignment of $x_o$ corresponds to a set of non-overlapping token replacements. Single unreplaced tokens are retained as base tokens with savings 0. Therefore, when the optimal codebook $Y^* = \{g : y_g^* = 1\}$ is passed to `segment_tokens_dp(T, Y*)`, the realized DP savings $\text{st}["\text{tokens\_saved}"]$ is **mathematically identical** to the optimal solver objective:
$$\text{DP\_Saved}(T, Y^*) \equiv \sum_{o} (|g_o| - 1) \cdot x_o^*$$

### 2.3 Verification via Exhaustive Brute-Force Equivalence
To mathematically verify the exactness of the CP-SAT solver:
- An exhaustive combinatorial brute-force reference solver (`brute_force_optimal_savings`) was implemented in `tests/test_exact_oracle_equivalence.py`.
- It enumerates all $\binom{|\mathcal{G}|}{k}$ subsets, evaluates each with `segment_tokens_dp`, and identifies the exact global maximum.
- **Verification Result:** Across **120 randomized test cases** featuring overlapping n-grams, repetitive motifs, nested substrings, and varied alphabets, the CP-SAT solver achieved:
  - `status == OPTIMAL`: 100% of cases
  - `optimality_gap == 0.0`: 100% of cases
  - Mismatches against brute-force global maximum: **0 / 120 (100% exact agreement)**.

---

## 3. Correction of the Opportunity Loss Funnel

### 3.1 Empirical Measured Funnel (Strictly Verified)
The empirical funnel encompasses **only** stages for which actual, verified code execution occurred. Stages 4, 5, and 6 are removed:

| Stage | Name | Epistemic Status | Decode Steps (K=32, 60 Prompts) | % of Global Ceiling | Incremental Loss | Failure Mechanism |
|---|---|---|---|---|---|---|
| **1** | **Global Occurrence Oracle** | **EXACT** | 8,806 steps | 100.0% | 0 | Unrestricted physical ceiling of base model continuation |
| **2** | **Fixed Candidate Pool Oracle** | **EXACT** | 3,737 steps | 42.44% | **-5,069 steps (-57.56%)** | **Prompt-only candidate generation recall deficit** |
| **3** | **Best Offline Ranker (PooledMLP)** | **MEASURED** | 1,858 steps | 21.10% | -1,879 steps (-21.34%) | Ranker scoring error & top-K slot allocation |

### 3.2 Hypothetical Downstream Projections (Unmeasured Conjectures)
> [!WARNING]
> **The following downstream stages are NOT YET MEASURED.** They are unverified projections derived from historical heuristic multipliers and must not be cited as empirical findings until executed via live model decoding.

| Projected Stage | Status | Heuristic Multiplier | Projected Steps | Projected % Ceiling | Verification Requirement |
|---|---|---|---|---|---|
| **Continuation Safety Filter** | **PROJECTED (NOT YET MEASURED)** | $\sim 0.45\times$ | ~836 steps | ~9.5% | Requires batched KV-cache forward pass to compute continuation KL divergence. |
| **Live Hypertoken Emission** | **PROJECTED (NOT YET MEASURED)** | $\sim 0.70\times$ | ~585 steps | ~6.6% | Requires live autoregressive execution on vLLM serving runtime. |
| **Quality-Preserved Net Savings** | **PROJECTED (NOT YET MEASURED)** | $\sim 0.85\times$ | ~497 steps | ~5.6% | Requires task quality benchmark execution (HumanEval / GSM8K / Alpaca-Eval). |

---

## 4. Data Audit, Leakage Analysis, and Holdout Discipline

### 4.1 Data Audit Summary
1. **Canonical Vanilla Generations:**
   - Location: `experiments/checkpoints/quality_benchmark/raw_results.jsonl`
   - Model: `microsoft/Phi-3.5-mini-instruct` (3.8B, revision `2fe192450127e6a83f7441aef6e3ca586c338b77`).
   - Verified count: Exactly 60 prompts (20 MBPP code, 20 GSM8K reasoning, 20 Alpaca instruction).
   - Provenance: Generated via deterministic greedy decoding (`temperature=0.0, max_new_tokens=300`).
2. **Historical Reference Datasets (`data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`):**
   - Contain human benchmark answers.
   - **Audit Finding:** Human responses exhibit different stylistic tokens, variable names, and formatting than Phi-3.5-mini-instruct. Training rankers on human responses introduces label shift.
3. **Corpus Co-occurrence Index (`cached_predictor.pkl`):**
   - Precomputed token associations were checked for test contamination.
   - `test_predictor_v2_no_leakage.py` verified that prompt candidate generation relies solely on prompt token n-grams and background vocabulary, with zero response leakage.
4. **Holdout Contamination & Split Protocol:**
   - The 12-prompt test split from the initial 60-prompt dataset is designated as **Consumed Diagnostic Data**.
   - Model selection is restricted exclusively to the **DEV split**.
   - True final unbiased holdout evaluation requires a newly curated dataset.

---

## 5. Multi-Seed Architecture Bake-Off Protocol

To eliminate single-seed noise:
1. **Deterministic Baseline:** Ridge regression (closed-form analytical solution, deterministic).
2. **Neural Challengers:** PooledMLP, CNNRanker, GRURanker, TransformerRanker evaluated across **3 random seeds (42, 43, 44)**.
3. **Metrics Reported:** Mean $\pm$ Standard Deviation across seeds on the DEV split:
   - DP Steps Saved ($K=32$)
   - % Candidate Pool Captured
   - Precision@32
   - Occurrence AUPRC
   - CPU Inference Latency (p50 at prompt length $L=512$)
4. **Pareto Selection Gate:** A neural architecture qualifies as a viable candidate over Ridge **if and only if**:
   - $\text{Dev Capture}(\text{Neural}) \ge \text{Dev Capture}(\text{Ridge}) + 5.0\text{ pp}$ across all evaluated seeds.
   - $\text{CPU Latency (p50, L=512)} \le 5.0\text{ ms}$.
