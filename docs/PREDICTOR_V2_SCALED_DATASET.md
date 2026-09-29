# Predictor V2 Scaled Dataset Specification & Split Manifest

> **Status:** The ID and split inventory in this document remains the generation-pinned source of truth. Its original 300-token generation settings below are historical and superseded by the corrected canonical 1024-token Vanilla artifact described in Section 5.

**Date:** 2026-09-28  
**Branch:** `codex/predictor-v2-scale-and-candidate-recall`  
**Dataset Manifest SHA-256:** `bcd4ea237d65d4d449af923ec998e42d377cd3e9456010ed8f9615f2bbee3f98`  
**Deterministic Seed:** `42`  

---

## 1. Executive Summary

- **Target Dataset Scale:** **900 distinct prompt contexts** (reaching the preferred target beyond the 600 minimum).
- **Domain Balance:** Exactly **300 Code (MBPP)**, **300 Reasoning (GSM8K)**, and **300 Instruction (Alpaca)**.
- **Split Partitioning:**
  - **TRAIN (70%):** 630 prompts (210 Code, 210 Reasoning, 210 Instruction) drawn strictly from `data/train.jsonl`.
  - **DEV (15%):** 135 prompts (45 Code, 45 Reasoning, 45 Instruction) drawn strictly from unconsumed items in `data/val.jsonl`.
  - **FINAL (15%):** 135 prompts (45 Code, 45 Reasoning, 45 Instruction) drawn strictly from `data/test.jsonl`.
- **Untouched Holdout Integrity:** **FINAL IS GENUINELY UNTOUCHED.** 100% of FINAL prompts originate from `data/test.jsonl`, having zero overlap with training data, validation data, or the 60-prompt V1 pilot.
- **Zero Overlap:** Pairwise overlap between TRAIN, DEV, FINAL, and the 60 historically consumed pilot prompts is strictly **0**.

## 2. Split by Domain Matrix

| Split | Code (MBPP) | Reasoning (GSM8K) | Instruction (Alpaca) | Total Prompts | Split % | Source File | Eligible Operations |
|---|---|---|---|---|---|---|---|
| **TRAIN** | 210 | 210 | 210 | **630** | 70.0% | `data/train.jsonl` | Token-association index, candidate tuning, ranker training |
| **DEV** | 45 | 45 | 45 | **135** | 15.0% | `data/val.jsonl` | Candidate strategy selection, architecture selection |
| **FINAL** | 45 | 45 | 45 | **135** | 15.0% | `data/test.jsonl` | Held-out frozen evaluation ONLY (zero tuning) |
| **TOTAL** | **300** | **300** | **300** | **900** | **100.0%** | Multi-source | Full Scaled Dataset |

## 3. Historical Pilot Benchmark Prompts Status

- The 60 prompts from `data/cached_pure_pred_val_60.json` (20 code, 20 reasoning, 20 instruction) are designated as **`HISTORICALLY_CONSUMED_PILOT`**.
- They are explicitly **excluded** from TRAIN, DEV, and FINAL to prevent any diagnostic look-ahead bias.

## 4. Historical Initial Vanilla Generation Specification (Superseded)

- **Base Model:** `microsoft/Phi-3.5-mini-instruct` (3.8B)
- **Base Revision:** `2fe192450127e6a83f7441aef6e3ca586c338b77`
- **Tokenizer:** `microsoft/Phi-3.5-mini-instruct`
- **Generation Parameters:**
  - `do_sample`: `False` (deterministic greedy decoding)
  - `temperature`: `0.0`
  - `max_new_tokens`: `300` (historical; do not use for the corrected canonical artifact)
  - `pad_token_id`: `32000` (`eos_token_id`)
- **Prompt Template Code Version:** `mbpp_task_signature_v2` (assertions formatted into prompt for code; raw instruction prompt for GSM8K and Alpaca)
- **Label Contract:** Strictly derived from frozen base model token emissions. Human reference solutions are retained exclusively for task quality evaluation.

## 5. Corrected Canonical 900-Record Artifact

- **Records SHA-256:** `3a8f59791f6fd06479b2b2869d09d71b57bd1e05b897740b70f18b1941f22ee6`.
- **Generation contract:** Pinned Phi-3.5-mini-instruct revision `2fe192450127e6a83f7441aef6e3ca586c338b77`; native chat template; greedy decoding; float16; `max_new_tokens=1024`; EOS IDs `[32007, 32001, 32000]`; pad ID `32000`.
- **Termination:** 895 records ended at EOS; five hit the 1024-token ceiling. All five capped rows are in TRAIN: `mbpp_108`, `alpaca_104`, `alpaca_706`, `alpaca_1727`, and `alpaca_446`. They remain unchanged and included in the immutable source.
- **Split confirmation:** Generation selection and audit artifacts pin this document's manifest file SHA and manifest hash. The records match its assignment exactly: 630 TRAIN / 135 DEV / 135 FINAL, with 300 records in each domain. The later 540/180/180 expectation does not match the pinned inventory and was not applied.
- The actual JSONL and its independent inventory/provenance sidecars are recorded in `data/canonical_phi_continuations.manifest.json`, `data/predictor_v2_canonical_split_inventory.json`, and `data/predictor_v2_canonical_generation_provenance.json`.
