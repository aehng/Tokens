# Predictor V2 Dataset and Leakage Audit

## Executive Summary

- **Canonical Vanilla Phi Generations:** 60 prompts strictly verified from `experiments/checkpoints/quality_benchmark/raw_results.jsonl` under `microsoft/Phi-3.5-mini-instruct`.
- **Historical Datasets:** 9412 train, 1176 val, 1178 test samples examined.
- **Supervision Source Limitation:** Historical datasets contain human/benchmark reference answers. They **cannot** be used as surrogates for base model continuations without severe label mismatch.
- **Scaling Gate C Verdict:** `GPU_GENERATION_REQUIRED_FOR_600_CANONICAL_CONTINUATIONS`. Generating $\ge 600$ canonical base continuations requires a bounded GPU batch generation run.

## 1. Canonical Vanilla Dataset Audit

- **Total Prompts:** 60
- **Domain Breakdown:** Code=20, Reasoning=20, Instruction=20
- **Continuation Token Lengths:** Mean = 278.5, Min = 55, Max = 301
- **Model ID:** `microsoft/Phi-3.5-mini-instruct` (revision `2fe192450127e6a83f7441aef6e3ca586c338b77`)
- **Decoding Contract:** Greedy decoding (`do_sample=False, temperature=0.0`)

## 2. Historical Datasets vs Vanilla Continuations

| Split | Sample Count | Domain Breakdown | Mean Prompt Words | Mean Response Words | Response Provenance | Valid for Predictor V2? |
|---|---|---|---|---|---|---|
| **TRAIN** | 9,412 | reasoning:7033, instruction:1600, code:779 | 44.4 | 49.9 | Human / Reference | **NO (Label Mismatch)** |
| **VAL** | 1,176 | reasoning:879, instruction:200, code:97 | 44.9 | 50.1 | Human / Reference | **NO (Label Mismatch)** |
| **TEST** | 1,178 | reasoning:880, code:98, instruction:200 | 44.6 | 48.4 | Human / Reference | **NO (Label Mismatch)** |
| **Vanilla 60** | 60 | Code:20, Reas:20, Inst:20 | 58.4 | 142.1 | Frozen Phi-3.5 Continuation | **YES (Canonical)** |

## 3. Split Overlap & Leakage Analysis

- **Train vs Val ID Overlap:** 0
- **Train vs Test ID Overlap:** 0
- **Val vs Test ID Overlap:** 0
- **Vanilla 60 vs Historical Train ID Overlap:** 0 / 60

> [!IMPORTANT]
> The 60 benchmark prompts were historically drawn from validation/test slices, with 0 prompt IDs appearing in historical `train.jsonl`. Within the Predictor V2 pipeline, strict split discipline is maintained via `predictor_v2_split_manifest.json` (36 Train, 12 Dev, 12 Test).

## 4. Association Index & Candidate Generator Audit

- **Association Index Size:** 0 token entries.
- **Background Bank Size:** 0 phrases.
- **Leakage Check:** Verified via `test_predictor_v2_no_leakage.py`. Candidate pools are constructed strictly from prompt token IDs, co-occurrence associations, and background vocabulary. No future continuation tokens or response text are accessed during candidate generation.

## 5. Scaling Feasibility & Gate C Verdict

- **Available Raw Prompts:** 9,412 in train, 1,176 in val, 1,178 in test.
- **Available Canonical Phi-3.5 Continuations:** Exactly 60.
- **Gate C Assessment:** To scale the evaluation from 60 to $\ge 600$ prompts with scientific integrity, 540 additional prompts from `data/val.jsonl` must be decoded with frozen Microsoft Phi-3.5-mini-instruct on a GPU (e.g., Kaggle T4 batch job). We must NOT substitute human answers from `val.jsonl` as fake model continuations.
