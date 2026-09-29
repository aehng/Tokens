# Stage 1 Phi Quality and Compression: Offline Reanalysis

**Status labels:** MEASURED = directly computed from pinned source/records; INFERENCE = supported interpretation; HYPOTHESIS = unverified cause; NOT TESTED = no valid controlled result.

This report reads only the 45 DEV prompt records in `docs/stage1_attribution_records.jsonl`. It does not load the canonical 900-row dataset, run a model, or access FINAL.

## 1. Pinned provenance and prior C correction

- **MEASURED:** Stage 1 source commit `8af6589ba0711018637c8ce41f8584642258cc49`; 45 prompts and 180 records; Phi/tokenizer revision `2fe192450127e6a83f7441aef6e3ca586c338b77`; Step-100 checkpoint SHA-256 `2c3606c075ac96dff1f607043f58241251d837f2340ae950d3dc309e9820fd44`.
- **MEASURED:** C's exact supplied codebook is present in each record's `selected_h_slots`. The slots replay the original Vanilla-continuation phrase rule against A's observed sequence on all 45 prompts. C phrases were selected from canonical Vanilla-generated continuations, not the reference answer.
- **MEASURED:** C was **ORACLE CODEBOOK LIVE**: B/C/D share the wrapped Step-100 checkpoint; C received hindsight phrases but used ordinary greedy generation, with the model free to emit H or base tokens.
- **MEASURED:** prior 5.39% = `(16,712 C expanded tokens - 15,811 C decode calls) / 16,712 = 5.39%`. The 704 count is H emission events; they saved 901 base-token steps because H phrases span 2–4 tokens.
- **INFERENCE:** 5.39% is live realized compression on C's own output. It is not the ORACLE CEILING and does not measure all DP-compressible opportunity.

## 2. ORACLE CEILING: Oracle V2 on each realized sequence

Oracle V2 uses K=32, beam width 4, candidate limit 80, and the product constraints of valid base IDs 0–32010 excluding 0, 1, 2, and 32000–32010. DP segments the full realized sequence, including terminal IDs as ordinary uncompressed base tokens. Values are near-optimal under the repository's Oracle V2 beam search, not a proof of global optimality.

| Sequence | Prompts | Base tokens | DP steps | Tokens saved | Compression | Mean used phrases | Optimal H substitutions | Codebook lengths |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A, lengths 2–3 | 45 | 15167 | 9955 | 5212 | 34.36% | 28.67 | 3120 | `{'2': 405, '3': 885}` |
| A, lengths 2–4 | 45 | 15167 | 9647 | 5520 | 36.39% | 25.20 | 2490 | `{'2': 230, '3': 310, '4': 594}` |
| B, lengths 2–3 | 45 | 14902 | 10736 | 4166 | 27.96% | 24.62 | 2563 | `{'2': 365, '3': 743}` |
| B, lengths 2–4 | 45 | 14902 | 10464 | 4438 | 29.78% | 21.31 | 2048 | `{'2': 196, '3': 243, '4': 520}` |

## 3. A-derived C codebook across A/B/C outputs

This applies C's exact supplied phrases to A, B, and C sequences. Occurrences count overlapping n-gram matches; DP savings use a non-overlapping optimal tiling.

| Output sequence | Prompts | Base tokens | DP steps | Tokens saved | Compression | Phrase types occurring | Occurrences | Optimal substitutions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 45 | 15167 | 11393 | 3774 | 24.88% | 1440 | 5674 | 1746 |
| B | 45 | 14902 | 13734 | 1168 | 7.84% | 569 | 1443 | 656 |
| C | 45 | 16712 | 15159 | 1553 | 9.29% | 614 | 1991 | 898 |

- **MEASURED:** Against A's reference distribution, phrase-type occurrence change is B `-871` and C `-826`; occurrence-count change is B `-4231` and C `-3683`.
- **INFERENCE:** The B/C differences quantify how much A-derived candidate coverage moves with the checkpoint's realized wording; they do not explain why B changed wording.

## 4. C exact supplied codebook opportunity versus live realization

| Opportunity: ORACLE CODEBOOK LIVE: C supplied codebook on C output | Value |
|---|---:|
| Prompts | 45 |
| Base-equivalent tokens | 16712 |
| DP compressed steps | 15159 |
| Available tokens saved | 1553 |
| Available compression | 9.29% |
| Mean codebook size | 32.00 |
| Supplied phrase types that occur | 614 |
| Overlapping phrase occurrences | 1991 |
| Optimal H substitutions | 898 |
| Codebook phrase lengths | `{'2': 240, '3': 510, '4': 690}` |
| Live realization: C live H realization | Value |
|---|---:|
| Base-equivalent tokens | 16712 |
| Transformer decode calls | 15811 |
| H emissions | 704 |
| Realized tokens saved | 901 |
| Realized compression | 5.39% |
| Mean H span length | 2.280 |
| Emitted H phrase lengths | `{'2': 507, '3': 197}` |
- **MEASURED:** C opportunity-to-realization ratio = `0.5802 (58.02%)` (live savings / C-codebook DP opportunity).

## 5. D exact supplied codebook opportunity versus live realization

| Opportunity: D supplied codebook on D output | Value |
|---|---:|
| Prompts | 45 |
| Base-equivalent tokens | 13809 |
| DP compressed steps | 13096 |
| Available tokens saved | 713 |
| Available compression | 5.16% |
| Mean codebook size | 32.00 |
| Supplied phrase types that occur | 383 |
| Overlapping phrase occurrences | 588 |
| Optimal H substitutions | 337 |
| Codebook phrase lengths | `{'2': 149, '3': 359, '4': 932}` |
| Live realization: D live H realization | Value |
|---|---:|
| Base-equivalent tokens | 13809 |
| Transformer decode calls | 13536 |
| H emissions | 187 |
| Realized tokens saved | 273 |
| Realized compression | 1.98% |
| Mean H span length | 2.460 |
| Emitted H phrase lengths | `{'2': 101, '3': 86}` |
- **MEASURED:** D opportunity-to-realization ratio = `0.3829 (38.29%)`.

## 6. Quality evidence and causal limits

| Condition | Passes / 45 | Aggregate quality | Code | Reasoning | Instruction |
|---|---:|---:|---:|---:|---:|
| A_vanilla | 35 / 45 | 77.78% | 8/15 | 13/15 | 14/15 |
| B_h_disabled | 25 / 45 | 55.56% | 4/15 | 8/15 | 13/15 |
| C_oracle | 25 / 45 | 55.56% | 4/15 | 9/15 | 12/15 |
| D_real_predictor | 24 / 45 | 53.33% | 5/15 | 6/15 | 13/15 |

| Paired comparison | Pass→pass | Pass→fail | Fail→pass | Fail→fail | Quality rate delta |
|---|---:|---:|---:|---:|---:|
| A → B | 24 | 11 | 1 | 9 | -22.22 pp |
| B → C | 21 | 4 | 4 | 16 | +0.00 pp |
| C → D | 20 | 5 | 4 | 16 | -2.22 pp |

- **MEASURED:** A and B differ by 10 task passes on this sample, and B versus C has paired gains/losses shown above. The aggregate equality of B/C rates alone is not evidence that H mechanics are harmless.
- **INFERENCE:** A versus B cannot isolate LoRA because Stage 1 A is native Transformers while B also activates the Zip2Zip wrapper, embeddings/head, and position path. The Stage 1 comparison does not contain B0.
- **NOT TESTED:** A/B0 token equality; B0/B1 isolated weight effect; CF forced representation/state safety; direct Vanilla/B0/B1 base-logit fidelity; P-VANILLA versus P-LORA; CUDA timing comparability.
- **NOT TESTED:** LoRA-target predictor training labels do not exist in the Stage 1 DEV records. Scaling label generation is deferred until the B0/B1 ablation confirms whether Step-100 is a viable serving target.

## 7. Gates and next diagnostic

| Gate | Result | Evidence / blocker |
|---|---|---|
| 1 — ceiling, opportunity, realization are separate | **PASS** | Oracle V2 sequence ceiling, exact C/D supplied-codebook DP opportunity, and realized H savings are separately reported below and per prompt in JSON. |
| 2 — architecture fidelity A vs B0 | **NOT TESTED** | No B0 records exist. |
| 3 — LoRA fidelity B0 vs B1 | **NOT TESTED** | Current B combines wrapper and Step-100 weights. |
| 4 — H representation CF | **NOT TESTED** | No forced H continuation was run. |
| 5 — predictor target source | **NOT TESTED** | No verified B1 TRAIN continuations; do not scale while Step-100 viability is unresolved. |
| 6 — root cause | **OPEN** | Existing B quality loss is measured, but it cannot yet be causally assigned to wrapper versus checkpoint. |

- **INFERENCE — recommended next step:** run the smallest matched A/B0/B1/C/D live subset, starting with 4 Code, 4 Reasoning, and 4 Instruction prompts. Reuse existing A/B1/C/D records for these same prompt IDs; add B0, plus CF only after its CPU checks pass. Capture base-logit fidelity in the same run. Decide whether Step-100 remains a viable target before generating any LoRA-target TRAIN labels.
- **MEASURED:** The archived Stage 1 report states 76.8 minutes; its Kaggle kernel log records 4,076.95 seconds (67.95 minutes) for the benchmark body. The earlier experiment tracker reports approximately 1.65/5.0 cumulative hours; Kaggle's weekly quota is a separate counter. This offline reanalysis adds 0 GPU hours; FINAL remains untouched.
- **NOT TESTED:** No new GPU run or real training was launched in this reanalysis.

## Per-prompt machine-readable results

See `experiments/results/phi_quality_compression_reanalysis.json`; it contains the exact K=32 codebooks, all per-prompt sequence counts, DP tilings, occurrence counts, live savings, realization ratios, paired quality outcomes, and source constraints.

---

### Section summary

1. **Provenance:** C used Vanilla-derived hindsight phrases but was free to choose H; 5.39% was realized compression. 2. **Ceiling:** Oracle V2 computed K=32 near-optimal DP ceilings on A and B outputs. 3. **Coverage:** A-derived phrases were tested on A/B/C. 4–5. **Realization:** C/D exact stored codebooks were DP-segmented against their own output and compared with emitted H savings. 6–7. **Causality:** wrapper, LoRA, forced-H state, and predictor target-source causes remain open pending controlled comparisons.
