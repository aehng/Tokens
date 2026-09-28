"""Phase 1A & 1B: Source Manifest and Deterministic Split Builder for Scaled Dataset.

Audits available prompt sources:
- data/train.jsonl
- data/val.jsonl
- data/test.jsonl
- data/cached_pure_pred_val_60.json (historically consumed pilot)

Constructs:
- Source Manifest (classifying all candidate prompts by historical status and eligibility)
- Scaled Split Manifest (900 prompts: 300 Code, 300 Reasoning, 300 Instruction; 630 Train, 135 Dev, 135 Final)

Outputs:
- docs/predictor_v2_source_manifest.json
- docs/predictor_v2_scaled_dataset_manifest.json
- docs/PREDICTOR_V2_SCALED_DATASET.md
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set

PILOT_60_PATH = "data/cached_pure_pred_val_60.json"
DATASET_PATHS = {
    "train": "data/train.jsonl",
    "val": "data/val.jsonl",
    "test": "data/test.jsonl",
}

OUT_SOURCE_MANIFEST = "docs/predictor_v2_source_manifest.json"
OUT_SPLIT_MANIFEST = "docs/predictor_v2_scaled_dataset_manifest.json"
OUT_DOC_MD = "docs/PREDICTOR_V2_SCALED_DATASET.md"


def load_pilot_consumed_ids() -> Set[str]:
    consumed = set()
    if os.path.exists(PILOT_60_PATH):
        with open(PILOT_60_PATH, "r", encoding="utf-8") as f:
            for item in json.load(f):
                consumed.add(item["id"])
    return consumed


def build_manifests(seed: int = 42) -> Dict[str, Any]:
    print("=" * 80)
    print("BUILDING SCALED DATASET SOURCE & SPLIT MANIFESTS")
    print("=" * 80)

    pilot_consumed = load_pilot_consumed_ids()
    print(f"Loaded {len(pilot_consumed)} historically consumed pilot benchmark prompt IDs.")

    # 1. Audit all source items
    source_records = []
    by_source_and_domain = defaultdict(lambda: defaultdict(list))
    all_seen_ids = set()

    for src_name, path in DATASET_PATHS.items():
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                pid = item["id"]
                dom = item["domain"]
                prompt_text = item["prompt"]

                is_consumed = pid in pilot_consumed
                
                # Eligibility rules:
                # - train.jsonl prompts: eligible for TRAIN only (never DEV or FINAL)
                # - val.jsonl prompts (unconsumed): eligible for DEV (never TRAIN or FINAL)
                # - test.jsonl prompts: eligible for FINAL (never TRAIN or DEV)
                # - pilot_consumed: ineligible for all (consumed diagnostic data)
                if is_consumed:
                    eligible_train = False
                    eligible_dev = False
                    eligible_final = False
                    status = "HISTORICALLY_CONSUMED_PILOT"
                    reason = "Prompt was evaluated in 60-prompt V1 pilot; consumed diagnostic data."
                elif src_name == "train":
                    eligible_train = True
                    eligible_dev = False
                    eligible_final = False
                    status = "UNCONSUMED_TRAIN_POOL"
                    reason = "Standard training set partition; eligible for training only."
                elif src_name == "val":
                    eligible_train = False
                    eligible_dev = True
                    eligible_final = False
                    status = "UNCONSUMED_VAL_POOL"
                    reason = "Standard validation set partition; eligible for dev selection only."
                elif src_name == "test":
                    eligible_train = False
                    eligible_dev = False
                    eligible_final = True
                    status = "UNTOUCHED_TEST_POOL"
                    reason = "Standard test set partition; genuinely untouched holdout."
                else:
                    eligible_train = False
                    eligible_dev = False
                    eligible_final = False
                    status = "UNKNOWN"
                    reason = "Unrecognized source."

                rec = {
                    "prompt_id": pid,
                    "domain": dom,
                    "source_dataset": src_name,
                    "historical_usage_status": status,
                    "eligible_for_train": eligible_train,
                    "eligible_for_dev": eligible_dev,
                    "eligible_for_final": eligible_final,
                    "reason": reason,
                    "prompt_text": prompt_text,
                    "reference_response": item.get("response", ""),
                    "test_assert_statements": item.get("test_assert_statements", []),
                }
                source_records.append(rec)
                if not is_consumed:
                    by_source_and_domain[src_name][dom].append(rec)
                all_seen_ids.add(pid)

    print(f"Audited {len(source_records)} total source prompts.")
    for s in ["train", "val", "test"]:
        dom_counts = {d: len(by_source_and_domain[s][d]) for d in ["code", "reasoning", "instruction"]}
        print(f"  {s.upper()} eligible unconsumed: {dom_counts}")

    # 2. Select 900 prompts deterministically
    # Target: 300 Code, 300 Reasoning, 300 Instruction
    # Target Splits:
    #   TRAIN (70%): 630 prompts (210 Code, 210 Reasoning, 210 Instruction) from train.jsonl
    #   DEV   (15%): 135 prompts (45 Code, 45 Reasoning, 45 Instruction) from val.jsonl
    #   FINAL (15%): 135 prompts (45 Code, 45 Reasoning, 45 Instruction) from test.jsonl

    rng = random.Random(seed)

    train_selected = []
    dev_selected = []
    final_selected = []

    # Final split from test.jsonl
    for dom in ["code", "reasoning", "instruction"]:
        candidates = sorted(by_source_and_domain["test"][dom], key=lambda x: x["prompt_id"])
        rng.shuffle(candidates)
        final_selected.extend(candidates[:45])

    # Dev split from val.jsonl
    for dom in ["code", "reasoning", "instruction"]:
        candidates = sorted(by_source_and_domain["val"][dom], key=lambda x: x["prompt_id"])
        rng.shuffle(candidates)
        dev_selected.extend(candidates[:45])

    # Train split from train.jsonl
    for dom in ["code", "reasoning", "instruction"]:
        candidates = sorted(by_source_and_domain["train"][dom], key=lambda x: x["prompt_id"])
        rng.shuffle(candidates)
        train_selected.extend(candidates[:210])

    # Verify zero overlaps
    train_ids = set(x["prompt_id"] for x in train_selected)
    dev_ids = set(x["prompt_id"] for x in dev_selected)
    final_ids = set(x["prompt_id"] for x in final_selected)

    assert len(train_ids & dev_ids) == 0, "TRAIN and DEV overlap!"
    assert len(train_ids & final_ids) == 0, "TRAIN and FINAL overlap!"
    assert len(dev_ids & final_ids) == 0, "DEV and FINAL overlap!"
    assert len(train_ids & pilot_consumed) == 0, "TRAIN contains pilot consumed prompts!"
    assert len(dev_ids & pilot_consumed) == 0, "DEV contains pilot consumed prompts!"
    assert len(final_ids & pilot_consumed) == 0, "FINAL contains pilot consumed prompts!"

    assert len(train_ids) == 630, f"Expected 630 train prompts, got {len(train_ids)}"
    assert len(dev_ids) == 135, f"Expected 135 dev prompts, got {len(dev_ids)}"
    assert len(final_ids) == 135, f"Expected 135 final prompts, got {len(final_ids)}"

    total_scaled_count = len(train_ids) + len(dev_ids) + len(final_ids)
    print(f"\nConstructed Scaled 900-Prompt Split:")
    print(f"  TRAIN: {len(train_ids)} prompts (210 code, 210 reasoning, 210 instruction)")
    print(f"  DEV:   {len(dev_ids)} prompts (45 code, 45 reasoning, 45 instruction)")
    print(f"  FINAL: {len(final_ids)} prompts (45 code, 45 reasoning, 45 instruction)")
    print(f"  TOTAL: {total_scaled_count} distinct prompt contexts (300 code, 300 reasoning, 300 instruction)")
    print(f"  GENUINELY UNTOUCHED FINAL: YES (drawn 100% from unconsumed test.jsonl)")

    # 3. Create deterministic hash
    all_selected_records = sorted(
        train_selected + dev_selected + final_selected, key=lambda x: x["prompt_id"]
    )
    manifest_hasher = hashlib.sha256()
    for r in all_selected_records:
        line = f"{r['prompt_id']}|{r['domain']}|{r['source_dataset']}|{r['prompt_text']}"
        manifest_hasher.update(line.encode("utf-8"))
    manifest_hash = manifest_hasher.hexdigest()
    print(f"Dataset Manifest SHA-256: {manifest_hash}")

    # Build outputs
    split_manifest = {
        "schema": "predictor_v2_scaled_dataset_manifest_v1",
        "dataset_manifest_hash": manifest_hash,
        "seed": seed,
        "target_count": total_scaled_count,
        "counts_by_split": {
            "train": len(train_ids),
            "dev": len(dev_ids),
            "final": len(final_ids),
            "total": total_scaled_count,
        },
        "counts_by_domain": {
            "code": 300,
            "reasoning": 300,
            "instruction": 300,
        },
        "split_domain_matrix": {
            "train": {"code": 210, "reasoning": 210, "instruction": 210},
            "dev": {"code": 45, "reasoning": 45, "instruction": 45},
            "final": {"code": 45, "reasoning": 45, "instruction": 45},
        },
        "train_prompt_ids": sorted(list(train_ids)),
        "dev_prompt_ids": sorted(list(dev_ids)),
        "final_prompt_ids": sorted(list(final_ids)),
        "prompts": {
            r["prompt_id"]: {
                "prompt_id": r["prompt_id"],
                "domain": r["domain"],
                "source_dataset": r["source_dataset"],
                "split": "train" if r["prompt_id"] in train_ids else ("dev" if r["prompt_id"] in dev_ids else "final"),
                "prompt_text": r["prompt_text"],
                "reference_response": r["reference_response"],
                "test_assert_statements": r.get("test_assert_statements", []),
            }
            for r in all_selected_records
        },
    }

    # Save files
    os.makedirs("docs", exist_ok=True)
    with open(OUT_SPLIT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)
    print(f"Saved split manifest to {OUT_SPLIT_MANIFEST}")

    # Source manifest (lightweight summary of all candidate prompts)
    source_summary = {
        "schema": "predictor_v2_source_manifest_v1",
        "total_source_prompts_audited": len(source_records),
        "historically_consumed_prompts_count": len(pilot_consumed),
        "source_files": DATASET_PATHS,
        "records": [
            {
                "prompt_id": r["prompt_id"],
                "domain": r["domain"],
                "source_dataset": r["source_dataset"],
                "historical_usage_status": r["historical_usage_status"],
                "eligible_for_train": r["eligible_for_train"],
                "eligible_for_dev": r["eligible_for_dev"],
                "eligible_for_final": r["eligible_for_final"],
                "reason": r["reason"],
            }
            for r in source_records
        ],
    }
    with open(OUT_SOURCE_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(source_summary, f, indent=2)
    print(f"Saved source manifest to {OUT_SOURCE_MANIFEST}")

    # Generate Markdown Documentation
    md_lines = [
        "# Predictor V2 Scaled Dataset Specification & Split Manifest",
        "",
        "**Date:** 2026-09-28  ",
        "**Branch:** `codex/predictor-v2-scale-and-candidate-recall`  ",
        f"**Dataset Manifest SHA-256:** `{manifest_hash}`  ",
        f"**Deterministic Seed:** `{seed}`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "- **Target Dataset Scale:** **900 distinct prompt contexts** (reaching the preferred target beyond the 600 minimum).",
        "- **Domain Balance:** Exactly **300 Code (MBPP)**, **300 Reasoning (GSM8K)**, and **300 Instruction (Alpaca)**.",
        "- **Split Partitioning:**",
        "  - **TRAIN (70%):** 630 prompts (210 Code, 210 Reasoning, 210 Instruction) drawn strictly from `data/train.jsonl`.",
        "  - **DEV (15%):** 135 prompts (45 Code, 45 Reasoning, 45 Instruction) drawn strictly from unconsumed items in `data/val.jsonl`.",
        "  - **FINAL (15%):** 135 prompts (45 Code, 45 Reasoning, 45 Instruction) drawn strictly from `data/test.jsonl`.",
        "- **Untouched Holdout Integrity:** **FINAL IS GENUINELY UNTOUCHED.** 100% of FINAL prompts originate from `data/test.jsonl`, having zero overlap with training data, validation data, or the 60-prompt V1 pilot.",
        "- **Zero Overlap:** Pairwise overlap between TRAIN, DEV, FINAL, and the 60 historically consumed pilot prompts is strictly **0**.",
        "",
        "## 2. Split by Domain Matrix",
        "",
        "| Split | Code (MBPP) | Reasoning (GSM8K) | Instruction (Alpaca) | Total Prompts | Split % | Source File | Eligible Operations |",
        "|---|---|---|---|---|---|---|---|",
        "| **TRAIN** | 210 | 210 | 210 | **630** | 70.0% | `data/train.jsonl` | Token-association index, candidate tuning, ranker training |",
        "| **DEV** | 45 | 45 | 45 | **135** | 15.0% | `data/val.jsonl` | Candidate strategy selection, architecture selection |",
        "| **FINAL** | 45 | 45 | 45 | **135** | 15.0% | `data/test.jsonl` | Held-out frozen evaluation ONLY (zero tuning) |",
        "| **TOTAL** | **300** | **300** | **300** | **900** | **100.0%** | Multi-source | Full Scaled Dataset |",
        "",
        "## 3. Historical Pilot Benchmark Prompts Status",
        "",
        "- The 60 prompts from `data/cached_pure_pred_val_60.json` (20 code, 20 reasoning, 20 instruction) are designated as **`HISTORICALLY_CONSUMED_PILOT`**.",
        "- They are explicitly **excluded** from TRAIN, DEV, and FINAL to prevent any diagnostic look-ahead bias.",
        "",
        "## 4. Canonical Vanilla Generation Specifications",
        "",
        "- **Base Model:** `microsoft/Phi-3.5-mini-instruct` (3.8B)",
        "- **Base Revision:** `2fe192450127e6a83f7441aef6e3ca586c338b77`",
        "- **Tokenizer:** `microsoft/Phi-3.5-mini-instruct`",
        "- **Generation Parameters:**",
        "  - `do_sample`: `False` (deterministic greedy decoding)",
        "  - `temperature`: `0.0`",
        "  - `max_new_tokens`: `300`",
        "  - `pad_token_id`: `32000` (`eos_token_id`)",
        "- **Prompt Template Code Version:** `mbpp_task_signature_v2` (assertions formatted into prompt for code; raw instruction prompt for GSM8K and Alpaca)",
        "- **Label Contract:** Strictly derived from frozen base model token emissions. Human reference solutions are retained exclusively for task quality evaluation.",
    ]

    with open(OUT_DOC_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved documentation to {OUT_DOC_MD}")

    return split_manifest


if __name__ == "__main__":
    build_manifests()
