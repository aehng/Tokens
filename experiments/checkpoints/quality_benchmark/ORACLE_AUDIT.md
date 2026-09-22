# Phase 0: Scientific Audit of the Current "Oracle" Codebook

## Executive Summary

The historical benchmark and offline simulation code used a function called `compute_oracle_codebook(tokens, k=32)` to represent the "Oracle Upper Bound" or "Optimal Oracle" compression ceiling.

This audit definitively proves that **the current oracle is NOT globally optimal**. It is an **Answer-Aware Greedy Frequency Heuristic with Exact DP Segmentation**. 

Across the 60 validation responses in `data/cached_pure_pred_val_60.json`:
- Out of 1,787 allocated codebook slots ($K=32$), **928 slots (51.9%) were completely dead** (never emitted once during exact DP segmentation).
- Over half of the allocated capacity in the "oracle" is wasted on overlapping subphrase duplicates (e.g. including `(1, 2)` and `(2, 3)` alongside `(1, 2, 3)`).
- On controlled synthetic and real samples, codebooks with fewer, non-redundant phrases achieve strictly higher compression than the greedy "oracle".

Calling this heuristic a "true upper bound" or "globally optimal oracle" is scientifically inaccurate. Going forward, it is strictly designated as the **Answer-Aware Greedy Oracle**.

---

## 1. Mathematical Formulation & What the Greedy Oracle Optimizes

The current implementation in `src/evaluation/offline_segmenter.py` performs the following steps:
1. **Rolling N-Gram Extraction:** Slides a window of lengths $l \in [2, 3]$ over the true response sequence $T = (t_1, t_2, \dots, t_N)$ to compute raw frequency counts $C(g)$.
2. **Independent Greedy Scoring:** Assigns each n-gram $g$ an unconditioned score:
   $$S(g) = C(g) \times (|g| - 1)$$
   This score assumes that every occurrence of $g$ in $T$ will be tiled and will save $(|g| - 1)$ tokens, completely independent of all other candidates selected.
3. **Truncated Top-K Selection:** Sorts all candidates by $S(g)$ descending and takes the top $K$ items:
   $$\mathcal{C}_{\text{greedy}} = \{g_1, g_2, \dots, g_K\}$$
4. **Post-Hoc Optimal Tiling:** Passes $\mathcal{C}_{\text{greedy}}$ to `segment_tokens_dp(T, \mathcal{C}_{\text{greedy}})` to find the minimal token tiling via dynamic programming.

**Core Flaw:**
While the *tiling* (Step 4) is exact and optimal for a *fixed* set of phrases, the *selection* of the phrase set (Step 2 and 3) is myopic and greedy. It does not account for phrase mutual exclusivity, subphrase containment, or diminishing marginal returns.

---

## 2. Five Points of Failure & Redundancy

### A. Subphrase Shadowing (The Substring Redundancy Problem)
If a 3-token phrase $g_3 = (A, B, C)$ occurs frequently, its constituent subphrases $g_2^{(1)} = (A, B)$ and $g_2^{(2)} = (B, C)$ are guaranteed to occur at least as frequently.
- $S(A, B, C) = C \times 2$
- $S(A, B) \ge C \times 1$
- $S(B, C) \ge C \times 1$

When $S(A, B, C)$ is large, both $(A, B)$ and $(B, C)$ appear immediately below it in the ranked list. If all three are selected into $\mathcal{C}_{\text{greedy}}$, the DP segmenter will prefer the 3-token step (saving 2 tokens instead of 1). Consequently:
- $(A, B)$ and $(B, C)$ are **shadowed** at all joint occurrences.
- They consume $2$ of the $K$ codebook slots while providing **zero marginal savings**.

### B. Minimal Synthetic Counterexample (Proof of Suboptimality)
Consider sequence:
$$T = [1, 2, 3, 99, 1, 2, 3, 88, 1, 2, 3, 77, 4, 5, 66, 4, 5]$$
Base tokens: 17.

- **Greedy Oracle ($K=3$):**
  - Evaluates counts: `(1, 2, 3)` occurs 3x (score 6); `(1, 2)` occurs 3x (score 3); `(2, 3)` occurs 3x (score 3); `(4, 5)` occurs 2x (score 2).
  - Selects top 3: `{(1, 2, 3), (1, 2), (2, 3)}`.
  - Tiling result: Uses `(1, 2, 3)` 3 times. Slots `(1, 2)` and `(2, 3)` are dead.
  - **Tokens saved: 6** (Compressed length: 11). Unique hypertokens used: **1 / 3**.

- **Better Non-Greedy Codebook ($K=2$):**
  - Selects `{(1, 2, 3), (4, 5)}`.
  - Tiling result: Uses `(1, 2, 3)` 3 times and `(4, 5)` 2 times.
  - **Tokens saved: 8** (Compressed length: 9). Unique hypertokens used: **2 / 2**.

$$\Delta = +33.3\% \text{ more tokens saved using } 33\% \text{ fewer slots!}$$

### C. Self-Overlap & Window Contention
For repetitive tokens such as `[A, A, A, A]`, a rolling window counts three occurrences of `(A, A)`. In reality, non-overlapping tiling can only pick two occurrences. The greedy score overestimates potential savings by $50\%$.

### D. Empirical Validation Across 60 Real Held-Out Validation Prompts
We executed the greedy oracle across all 60 validation responses:
- **Total candidate slots allocated:** 1,787 slots
- **Slots actually tiled by DP:** 859 slots
- **Dead slots:** **928 slots (51.9% dead capacity!)**
- Realized compression: 40.10%

More than half of the oracle's allocated vocabulary is completely wasted on dead subphrases that never get emitted.

### E. Prompt vs. Response Separation Audit
Audit of `src/evaluation/comprehensive_suite.py`, `experiments/run_corrected_offline_benchmark.py`, and `offline_segmenter.py`:
- `compute_oracle_codebook` receives response tokens `r_ids` only.
- In `run_corrected_offline_benchmark.py`, the response codebook was also evaluated on the prompt (`segment_tokens_dp(p_ids, oracle_cb)`) for auxiliary diagnostic metrics, but the codebook itself was derived purely from the response.
- There is no unexpected contamination of prompt tokens into the response codebook calculation.

---

## 3. Terminology & Documentation Updates

1. **Renaming:** In all future publications, documentation, and code comments, `compute_oracle_codebook` must be referred to as the **Answer-Aware Greedy Oracle**.
2. **Ceiling Disclaimer:** The greedy oracle is a *lower bound on the true oracle ceiling* (i.e. a stronger search algorithm will find a codebook with higher compression).
3. **Target for Phase 1:** Build a Stronger Compression Oracle (exact combinatorial search for small instances; marginal-gain beam/swap search for full instances) to establish the true mathematical ceiling.
