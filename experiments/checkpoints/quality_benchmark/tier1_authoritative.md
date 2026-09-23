# Tier-1 Authoritative Benchmark Report

**Tested Commit:** `e6b8e4250a7a22e58360556dd08d1f1fb3a8942c`  
**Run ID:** `81fc94a1eb26e970`  
**Protocol:** Fixed 12-Prompt Stratified Suite (4 MBPP Code, 4 GSM8K Reasoning, 4 Alpaca Instruction), `max_new_tokens=300`, greedy decoding (`do_sample=False`).  
**Evaluator Contract:** `phi_quality_evaluator_v2`, `mbpp_task_signature_v2`.  
**Model & Tokenizer:** `microsoft/Phi-3.5-mini-instruct` (revision `2fe192450127e6a83f7441aef6e3ca586c338b77`).  
**Zip2Zip Hub Revision:** `11c461733a79d2a5de6b814585c3361ca2aacbe7`.  
**Predictor:** `experiments/checkpoints/oracle_guided_predictor.pkl` (SHA256: `5ea21e53f119...`).  
**Joint Checkpoint:** `experiments/checkpoints/predictive_joint_pilot/checkpoint_step_100.pt` (SHA256: `2c3606c075ac...`).

---

## 1. Summary Matrix

| Metric | Vanilla Phi-3.5 | Official Zip2Zip (LZW) | Predictive Step 100 (Raw Prompt) | Predictive Step 100 (Compressed Prompt) |
| :--- | :---: | :---: | :---: | :---: |
| **MBPP Pass@1** | **1/4 (25.0%)** | 0/4 (0.0%) | 0/4 (0.0%) | 0/4 (0.0%) |
| **MBPP Syntax Validity** | **3/4 (75.0%)** | **3/4 (75.0%)** | 1/4 (25.0%) | 1/4 (25.0%) |
| **GSM8K Exact Accuracy** | **3/4 (75.0%)** | 2/4 (50.0%) | 2/4 (50.0%) | 2/4 (50.0%) |
| **Alpaca Mechanical Pass** | 2/4 (50.0%) | 2/4 (50.0%) | 2/4 (50.0%) | **3/4 (75.0%)** |
| **Net Decode Steps Saved** | 0 | **1,267** | 340 | 334 |
| **Micro Decode Reduction %** | 0.00% | **31.68%** | 9.25% | 10.68% |
| **Macro Decode Reduction %** | 0.00% | **25.81%** | 8.53% | 9.82% |
| **Quality-Preserved Saved** | 0 | **281 (7.03%)** | 75 (2.04%) | 81 (2.59%) |
| **Total Hypertokens Emitted** | 0 | 936 | 254 | 239 |
| **Mean Hypertokens / Output** | 0.00 | 78.00 | 21.17 | 19.92 |
| **Mean TTFT (Prefill Latency)** | **1.103s** | 18.584s | 4.510s | 1.624s |
| **Mean Wall Time / Prompt** | **50.59s** | 71.88s | 253.47s | 124.96s |
| **Effective Tok / Sec** | **5.80** | 4.64 | 1.21 | 2.09 |
| **EOS Reached** | 2/12 | 0/12 | 0/12 | 0/12 |
| **Truncations (Hit Cap)** | 10/12 | 7/12 | 10/12 | 7/12 |
| **Severe Trigram Repetitions** | 4/12 | 2/12 | 7/12 | 6/12 |

---

## 2. Phase 4 Evaluation: Raw vs. Compressed Prompt A/B

A strict, matched paired comparison was performed across all 12 prompts using the exact same codebooks and Step-100 model:

1. **Quality:** Compressed prompt equals or exceeds raw prompt on all domains:
   - Code: 0/4 Pass@1, 1/4 syntax valid on both.
   - Reasoning: 2/4 exact accuracy on both (`gsm_2956` and `gsm_8674` correct).
   - Instruction: **3/4 (75.0%)** mechanical pass for compressed prompt vs. 2/4 (50.0%) for raw prompt (cures formatting loop on `alpaca_1992`).
2. **Compression & Prefill:**
   - Prompt prefill length dropped from 49.58 to 42.67 tokens (**21.81% prompt compression** on average, reaching up to 60.87% on Alpaca).
   - Micro decode reduction increased from 9.25% to **10.68%**.
   - TTFT dropped from 4.510s to **1.624s** (64.0% reduction in prefill latency).
3. **Wall Clock:**
   - Average CPU wall time dropped by more than half: **124.96s vs 253.47s** (-50.7%).
4. **Canonical Decision:**
   > **Decision:** Canonical prompt representation is **`compressed_prompt`** (`predictive_codebook_dp_segmented`). It eliminates the train/inference mismatch, improves prefill latency, and reduces severe repetition.

---

## 3. Per-Prompt Task Breakdown

| Prompt ID | Domain | Vanilla Phi | Official Zip2Zip | Pred Step 100 (Raw) | Pred Step 100 (Compressed) | Notes / Behavior |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `mbpp_740` | Code | Fail (0 save) | Fail (50 save) | Fail (6 save) | Fail (6 save) | Correct syntax, assertion failure across all |
| `mbpp_969` | Code | Fail (0 save) | Fail (44 save) | Fail (45 save) | Fail (45 save) | Syntax error on zip2zip and predictive |
| `mbpp_542` | Code | **Pass** (0 save) | Fail (15 save) | Fail (26 save) | Fail (22 save) | Vanilla passes; predictive suffers function name drift |
| `mbpp_769` | Code | Fail (0 save) | Fail (27 save) | Fail (42 save) | Fail (31 save) | All fail assertion; predictive emits repetitive loops |
| `gsm_3022` | Reasoning | **Pass** (0 save) | Fail (92 save) | Fail (93 save) | Fail (51 save) | Arithmetic drift after ungrounded numeric hypers |
| `gsm_6613` | Reasoning | **Pass** (0 save) | **Pass** (116 save) | Fail (19 save) | Fail (26 save) | Zip2Zip answers 277; predictive drifts |
| `gsm_2956` | Reasoning | **Pass** (0 save) | **Pass** (81 save) | **Pass** (42 save) | **Pass** (27 save) | All correct; predictive saves 27-42 decode steps |
| `gsm_8674` | Reasoning | Fail (0 save) | Fail (116 save) | **Pass** (14 save) | **Pass** (40 save) | **Predictive win**: correct answer 42 (vanilla misses) |
| `alpaca_1337`| Instruction| **Pass** (0 save) | Fail (393 save) | Fail (16 save) | Fail (72 save) | Trigram repetition loop on few-shot formatting |
| `alpaca_1992`| Instruction| Fail (0 save) | Fail (249 save) | Fail (18 save) | **Pass** (0 save) | Compressed prompt stops cleanly |
| `alpaca_55`  | Instruction| Fail (0 save) | **Pass** (75 save) | **Pass** (14 save) | **Pass** (4 save) | Correct explanation; stops cleanly |
| `alpaca_183` | Instruction| **Pass** (0 save) | **Pass** (9 save) | **Pass** (5 save) | **Pass** (10 save) | Clean short biographical answer |
