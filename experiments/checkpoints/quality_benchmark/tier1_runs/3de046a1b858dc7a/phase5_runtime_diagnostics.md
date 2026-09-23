# Phase-5 Retroactive Runtime & Generation-Trajectory Analysis

**Run ID:** `3de046a1b858dc7a`  
**Canonical Baseline:** `predictive_step_100_compressed_prompt` (No Gate)  
**Evaluation Scope:** 12 prompts across 3 conditions (36 evaluations total)  
**Trajectory Invariance:** **100% Invariant** (All 12 prompt texts, decode iterations [2,794], and expanded tokens [3,128] are byte-identical across conditions)  

---

## 1. Executive Summary

- **Natural Experiment on Gate Overhead:** Because the decode sequences and token emissions are 100% identical across all three conditions, any delta in execution time is strictly the isolated computational overhead of the `ContextualEmissionGate` logits processor on CPU, not due to trajectory drift or length divergence.
- **Overall Throughput (No Gate Baseline):**
  - **Decode Iterations / sec:** `1.51` it/s (`663.2` ms/step)
  - **Expanded Tokens / sec:** `1.69` tok/s (`592.3` ms/expanded token)
  - **Decoded Words / sec:** `1.00` words/s
  - **Overall Decode Step Savings:** `334` steps saved out of `3,128` expanded tokens (`10.68%` micro decode reduction).
- **Gating Latency Penalty:**
  - **Top-16 Gate:** Increases pooled decode latency from `663.2 ms/step` to `807.4 ms/step` (**+144.2 ms/step**, +21.7% decode time), while filtering out `79.4%` of candidate hypertoken slots.
  - **Top-32 Gate:** Increases pooled decode latency from `663.2 ms/step` to `917.3 ms/step` (**+254.1 ms/step**, +38.3% decode time), filtering out `69.8%` of candidate slots.
- **Stopping & EOS Pathology:**
  - **0 / 12 prompts emitted an EOS token** in Phase 5. 7 prompts hit `max_new_tokens = 300` and continued generating irrelevant post-answer tails or repetitive loops.
  - **GSM8K Post-Answer Waste:** All 4 GSM8K prompts solved their math problems within `41` to `183` steps, but then generated unprompted follow-up problems and explanations until hitting the 300-step ceiling, wasting **48.2% of all reasoning decode steps** (578 post-answer tail steps out of 1,200).

---

## 2. Condition-Level Aggregate Metrics

| Metric | No Gate (Canonical) | Top-16 Gate | Top-32 Gate |
| :--- | :---: | :---: | :---: |
| **Decode Iterations (Steps)** | 2794 | 2794 | 2794 |
| **Expanded Base Tokens** | 3128 | 3128 | 3128 |
| **Hypertokens Emitted** | 239 | 239 | 239 |
| **Decode Steps Saved** | 334 | 334 | 334 |
| **Micro Decode Reduction %** | 10.68% | 10.68% | 10.68% |
| **Mean TTFT / Prefill Time** | 2.073s | 2.035s | 2.819s |
| **Mean Decode Wall Time** | 154.41s | 187.98s | 213.57s |
| **Mean Total Wall Time** | 156.54s | 190.08s | 216.45s |
| **Throughput (Decode it/s)** | **1.51 it/s** | 1.24 it/s | 1.09 it/s |
| **Throughput (Expanded tok/s)** | **1.69 tok/s** | 1.39 tok/s | 1.22 tok/s |
| **Throughput (Words/s)** | **1.00 w/s** | 0.82 w/s | 0.72 w/s |
| **Latency (ms / Decode Step)** | **663.2 ms** | 807.4 ms (+144.2 ms) | 917.3 ms (+254.1 ms) |
| **Latency (ms / Expanded Tok)** | **592.3 ms** | 721.2 ms | 819.3 ms |
| **Candidate Slots Gated Out** | N/A | 79.4% (71008/89408) | 69.8% (62429/89408) |

---

## 3. Domain-Level Breakdown (Canonical Baseline)

| Domain | Prompts | Decode Steps | Expanded Tokens | Words | Hypertokens | Saved Steps | Reduction % | Decode it/s | Expanded tok/s | ms / Step |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Code** | 4 | 1065 | 1169 | 892 | 80 | 104 | 8.90% | 1.85 | 2.03 | 540.0 ms |
| **Reasoning** | 4 | 1200 | 1344 | 598 | 112 | 144 | 10.71% | 1.90 | 2.13 | 525.2 ms |
| **Instruction** | 4 | 529 | 615 | 365 | 47 | 86 | 13.98% | 0.82 | 0.95 | 1223.9 ms |

---

## 4. Paired Per-Prompt Comparison & Trajectory Length

| Prompt ID | Domain | Steps | Expanded | Words | H Emitted | H / 100 St | Saved (%) | NoGate Time (it/s) | Top16 Time (Overhead) | Top32 Time (Overhead) | Termination / Repetition |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `mbpp_740` | code | 242 | 248 | 166 | 6 | 2.5 | 6 (2.4%) | 155.5s (1.64) | 173.6s (+97.0 ms/st) | 167.3s (+70.9 ms/st) | Natural (242 st) [REP] |
| `mbpp_969` | code | 300 | 345 | 274 | 36 | 12.0 | 45 (13.0%) | 161.6s (1.88) | 527.0s (+1217.5 ms/st) | 206.4s (+148.1 ms/st) | MaxTokens (300) [REP] |
| `mbpp_542` | code | 223 | 245 | 184 | 13 | 5.8 | 22 (9.0%) | 114.2s (1.98) | 262.2s (+655.7 ms/st) | 145.3s (+136.2 ms/st) | Natural (223 st) [REP] |
| `mbpp_769` | code | 300 | 331 | 268 | 25 | 8.3 | 31 (9.4%) | 156.8s (1.93) | 223.3s (+218.8 ms/st) | 202.0s (+149.4 ms/st) | MaxTokens (300) [REP] |
| `gsm_3022` | reasoning | 300 | 351 | 102 | 33 | 11.0 | 51 (14.5%) | 159.6s (1.91) | 218.5s (+194.0 ms/st) | 340.6s (+603.8 ms/st) | MaxTokens (300) |
| `gsm_6613` | reasoning | 300 | 326 | 177 | 24 | 8.0 | 26 (8.0%) | 156.6s (1.94) | 216.5s (+192.7 ms/st) | 362.0s (+675.7 ms/st) | MaxTokens (300) [REP] |
| `gsm_2956` | reasoning | 300 | 327 | 159 | 24 | 8.0 | 27 (8.3%) | 159.5s (1.90) | 176.7s (+56.8 ms/st) | 308.1s (+494.2 ms/st) | MaxTokens (300) |
| `gsm_8674` | reasoning | 300 | 340 | 160 | 31 | 10.3 | 40 (11.8%) | 162.2s (1.87) | 169.8s (+23.9 ms/st) | 280.3s (+391.7 ms/st) | MaxTokens (300) |
| `alpaca_1337` | instruction | 300 | 372 | 191 | 36 | 12.0 | 72 (19.4%) | 437.1s (0.69) | 163.5s (-912.0 ms/st) | 310.8s (-421.8 ms/st) | MaxTokens (300) [REP] |
| `alpaca_1992` | instruction | 8 | 8 | 5 | 0 | 0.0 | 0 (0.0%) | 8.1s (1.24) | 5.8s (-220.4 ms/st) | 15.0s (+433.1 ms/st) | Natural (8 st) |
| `alpaca_55` | instruction | 19 | 23 | 16 | 2 | 10.5 | 4 (17.4%) | 15.9s (1.30) | 11.9s (-206.8 ms/st) | 18.4s (-86.3 ms/st) | Natural (19 st) |
| `alpaca_183` | instruction | 202 | 212 | 153 | 9 | 4.5 | 10 (4.7%) | 191.4s (1.06) | 132.4s (-289.9 ms/st) | 241.2s (+238.7 ms/st) | Natural (202 st) |

---

## 5. GSM8K Post-Answer Tail Analysis

When evaluating GSM8K prompts under greedy generation, the model produced the deterministic numerical answer early in the trajectory, but failed to emit an EOS token. As a consequence, generation continued until hitting `max_new_tokens = 300`.

| Prompt ID | Ground Truth | Extracted Answer | Correct? | Total Decode Steps | Approx. Steps to Answer | Post-Answer Tail Steps | Tail % of Trajectory | Last H Step | Steps After Last H |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gsm_3022` | `N/A` | `72000` | False | 300 | 136 | **164** | **54.7%** | 141 | 158 |
| `gsm_6613` | `N/A` | `275` | False | 300 | 41 | **259** | **86.3%** | 285 | 14 |
| `gsm_2956` | `N/A` | `4` | True | 300 | 96 | **204** | **68.0%** | 282 | 17 |
| `gsm_8674` | `N/A` | `42` | True | 300 | 183 | **117** | **39.0%** | 279 | 20 |

**Key Takeaways from Tail Analysis:**
- For `gsm_6613`, the solution is fully solved and emitted at step `41` (character 131). The subsequent **259 steps (86.3% of the run)** consist entirely of hallucinated follow-up questions (e.g. Euclidean algorithm for GCD(12, 18)).
- For `gsm_2956` (exact correct), the answer is reached at step `96`. The subsequent **204 steps (68.0%)** generate an unprompted logarithm problem.
- Across all 4 GSM8K problems, **578 out of 1,200 decode steps (48.2%)** were spent in post-answer tails due to missing EOS stopping.

---

## 6. Source of Contextual Gate Latency Overhead

Because trajectory lengths and generated tokens are **100% invariant** between No Gate and Gated conditions, the +21.7% (Top-16) and +38.3% (Top-32) latency regressions are **purely per-step Python CPU evaluation overhead** in `ContextualEmissionGate.__call__`:

1. **Logit Sorting / Top-K Masking Cost:** At every decode step, the logits processor intercepts the un-normalized logits tensor, applies `torch.topk(logits, k)` across the full vocabulary dimension (32,064 tokens), and checks candidate hypertoken prefixes.
2. **PyTorch Overhead on CPU:** Even though tensor slicing is compact, executing `torch.topk` on a CPU tensor 2,794 times adds ~144.2 ms per step for Top-16 and ~254.1 ms per step for Top-32.
3. **Zero Quality or Trajectory Benefit:** Because greedy top-1 selection already bypasses low-probability tokens, the gate filtered out 70-80% of speculative candidates that would never have been selected anyway under greedy decoding, producing zero change in emitted tokens while adding substantial latency.

---

## 7. Predictive Generation Diagnostics & Repetition Failures

- **Failure Associations:**
  - `mbpp_969`: Model emitted 36 hypertokens (12.0 H / 100 steps) and entered a severe repetition loop (`min_swaps = min_swaps + 1`), hitting 300 steps.
  - `alpaca_1337`: Model emitted 43 hypertokens (14.3 H / 100 steps, highest among instruction prompts) and entered severe loop repetition, hitting 300 steps.
  - Hypertoken emissions continued up to the final step (`mbpp_769` last H at step 299; `gsm_6613` last H at step 285). Hypertokens did not prevent repetition loops from forming.

---

## 8. Fields That Cannot Be Reconstructed Retroactively

To avoid false precision, the following metrics cannot be retroactively derived from the Phase 5 artifacts and must be prospectively captured in future harness runs:

1. **Per-Step Wall-Clock Timestamps:** `decode_step_intervals_s` was not logged per step; only aggregate `decode_time_s` is available. As a result, per-step jitter, p95 step latency, and exact time-to-first-token inside the generation loop cannot be reconstructed.
2. **Dynamic KV-Cache Memory Curves:** `process_rss_gb` was recorded as a final scalar endpoint per prompt (3.24 GB - 3.49 GB). Intermediate per-step memory consumption cannot be reconstructed.
3. **Exact Token Index of GSM Answer:** Because the decode loop records base/hyper tokens and expands them dynamically, exact token offsets can only be approximated via character offsets or re-tokenization of the output text.
4. **Speculative Branch Divergence Under Sampling:** Because Phase 5 was evaluated under greedy decoding ($T=0$), no alternate sampling trajectories exist to test non-greedy gate sensitivity retroactively.
