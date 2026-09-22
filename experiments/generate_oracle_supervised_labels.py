"""Phase 4: Create Oracle Training Labels (TRAIN Split Only).

Extracts quality-aware hypertoken supervision data from data/train.jsonl.
Strict data discipline:
- ONLY data/train.jsonl is used for generating training labels.
- Validation split (data/cached_pure_pred_val_60.json) is audited only to verify
  oracle target distributions, and NEVER leaked into training.
- Test split is UNTOUCHED.

Generates:
- data/oracle_supervised_train_labels.jsonl
- experiments/checkpoints/quality_benchmark/val_oracle_audit.json
"""

import heapq
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from transformers import AutoTokenizer
from experiments.build_quality_aware_oracle import (
    compute_safety_prior,
    extract_all_candidate_phrases,
)
from src.evaluation.offline_segmenter import segment_tokens_dp

TRAIN_DATA_PATH = "data/train.jsonl"
VAL_DATA_PATH = "data/cached_pure_pred_val_60.json"
OUT_TRAIN_LABELS = "data/oracle_supervised_train_labels.jsonl"
OUT_VAL_AUDIT = "experiments/checkpoints/quality_benchmark/val_oracle_audit.json"

K_CODEBOOK = 32
CANDIDATE_POOL_LIMIT = 40


def lazy_greedy_quality_oracle(
    r_ids: List[int],
    prompt_text: str,
    p_ids: List[int],
    domain: str,
    tokenizer: Any,
    k: int = K_CODEBOOK,
) -> Tuple[Set[Tuple[int, ...]], int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Computes Quality-Aware Oracle codebook using accelerated lazy greedy submodular optimization.
    
    Returns:
        (selected_phrases, total_steps_saved, selected_phrase_records, all_candidate_records)
    """
    ngram_counts = extract_all_candidate_phrases(r_ids, min_len=2, max_len=4)
    if not ngram_counts:
        return set(), 0, [], []

    # 1. Compute safety prior and value for all n-grams
    scored_candidates = []
    all_cand_map = {}
    for p_tup, count in ngram_counts.items():
        p_text = tokenizer.decode(list(p_tup))
        safety = compute_safety_prior(p_text, list(p_tup), prompt_text, p_ids, domain)
        isolated_saved = (len(p_tup) - 1) * count
        base_val = isolated_saved * safety["safety_prior"]
        
        record = {
            "phrase_tokens": list(p_tup),
            "phrase_text": p_text,
            "length": len(p_tup),
            "target_count": count,
            "isolated_steps_saved": isolated_saved,
            "safety_prior": safety["safety_prior"],
            "is_safe": safety["is_safe"],
            "reasons": safety["reasons"],
            "has_trailing_space": safety["has_trailing_space"],
            "base_value": round(base_val, 4),
            "selected_in_oracle": False,
            "marginal_steps_saved": 0,
            "target_value": 0.0,
        }
        all_cand_map[p_tup] = record

        # Filter candidates for forward selection:
        # Must have safety_prior >= 0.50 and no trailing whitespace hazard
        if safety["safety_prior"] >= 0.50 and not safety["has_trailing_space"]:
            scored_candidates.append((base_val, p_tup, safety["safety_prior"]))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    filtered = scored_candidates[:CANDIDATE_POOL_LIMIT]

    if not filtered:
        return set(), 0, [], list(all_cand_map.values())

    # 2. Accelerated Lazy Greedy Forward Selection
    # Initial upper bound marginal gain from singleton DP
    heap = []
    for base_val, p_tup, s_prior in filtered:
        _, _, st = segment_tokens_dp(r_ids, {p_tup})
        gain = st["tokens_saved"]
        if gain > 0:
            heapq.heappush(heap, (-gain * s_prior, -gain, p_tup, s_prior))

    selected: Set[Tuple[int, ...]] = set()
    curr_saved = 0
    selected_records = []

    for _ in range(k):
        found = False
        while heap:
            neg_val, neg_gain, best_p, s_prior = heapq.heappop(heap)
            _, _, st = segment_tokens_dp(r_ids, selected | {best_p})
            new_gain = st["tokens_saved"] - curr_saved
            if new_gain <= 0:
                continue
            new_val = new_gain * s_prior

            # Check if this gain is >= next upper bound in heap
            if not heap or new_val >= -heap[0][0]:
                selected.add(best_p)
                curr_saved += new_gain
                rec = all_cand_map[best_p]
                rec["selected_in_oracle"] = True
                rec["marginal_steps_saved"] = new_gain
                rec["target_value"] = round(new_val, 4)
                selected_records.append(rec)
                found = True
                break
            else:
                heapq.heappush(heap, (-new_val, -new_gain, best_p, s_prior))
        if not found:
            break

    return selected, curr_saved, selected_records, list(all_cand_map.values())


def run_phase4_label_generation():
    t0 = time.perf_counter()
    print("=" * 80)
    print("PHASE 4: CREATE ORACLE TRAINING LABELS (TRAIN SPLIT ONLY)")
    print("=" * 80)

    # 1. Load Tokenizer
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

    # 2. Audit Validation Split FIRST (No leakage verification)
    print(f"\n--- 1. AUDITING VALIDATION SPLIT ({VAL_DATA_PATH}) ---", flush=True)
    with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
        val_samples = json.load(f)

    val_prompt_ids = set()
    val_stats = {
        "num_samples": len(val_samples),
        "total_target_tokens": 0,
        "total_steps_saved": 0,
        "total_slots_allocated": 0,
        "safe_slots": 0,
        "unsafe_slots": 0,
        "by_domain": defaultdict(lambda: {"samples": 0, "tokens": 0, "saved": 0, "slots": 0, "safe_slots": 0}),
        "top_phrases": defaultdict(Counter),
    }

    for s in val_samples:
        pid = s["id"]
        val_prompt_ids.add(pid)
        dom = s["domain"]
        p_text = s["prompt"]
        r_text = s.get("ground_truth_response", s.get("response", ""))
        p_ids = tokenizer.encode(p_text, add_special_tokens=False)
        r_ids = tokenizer.encode(r_text, add_special_tokens=False)

        val_stats["total_target_tokens"] += len(r_ids)
        val_stats["by_domain"][dom]["samples"] += 1
        val_stats["by_domain"][dom]["tokens"] += len(r_ids)

        if len(r_ids) >= 2:
            sel_cb, saved, sel_recs, _ = lazy_greedy_quality_oracle(
                r_ids, p_text, p_ids, dom, tokenizer, k=K_CODEBOOK
            )
            val_stats["total_steps_saved"] += saved
            val_stats["total_slots_allocated"] += len(sel_cb)
            val_stats["by_domain"][dom]["saved"] += saved
            val_stats["by_domain"][dom]["slots"] += len(sel_cb)

            for rec in sel_recs:
                if rec["is_safe"]:
                    val_stats["safe_slots"] += 1
                    val_stats["by_domain"][dom]["safe_slots"] += 1
                else:
                    val_stats["unsafe_slots"] += 1
                val_stats["top_phrases"][dom][rec["phrase_text"]] += rec["marginal_steps_saved"]

    val_micro = (val_stats["total_steps_saved"] / val_stats["total_target_tokens"]) * 100
    print(f"Validation Audit Complete: {val_stats['num_samples']} samples, "
          f"{val_stats['total_target_tokens']} tokens, "
          f"{val_stats['total_steps_saved']} steps saved ({val_micro:.2f}% micro compression), "
          f"{val_stats['safe_slots']} safe slots ({val_stats['safe_slots'] / max(1, val_stats['total_slots_allocated'])*100:.1f}%)")

    # Format top phrases for audit JSON
    audit_output = {
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation_samples": val_stats["num_samples"],
        "validation_tokens": val_stats["total_target_tokens"],
        "total_steps_saved": val_stats["total_steps_saved"],
        "micro_compression_pct": round(val_micro, 2),
        "total_slots_allocated": val_stats["total_slots_allocated"],
        "safe_slots": val_stats["safe_slots"],
        "unsafe_slots": val_stats["unsafe_slots"],
        "safe_slot_pct": round(val_stats["safe_slots"] / max(1, val_stats["total_slots_allocated"]) * 100, 2),
        "by_domain": {
            d: {
                "samples": val_stats["by_domain"][d]["samples"],
                "tokens": val_stats["by_domain"][d]["tokens"],
                "saved": val_stats["by_domain"][d]["saved"],
                "micro_compression_pct": round(val_stats["by_domain"][d]["saved"] / max(1, val_stats["by_domain"][d]["tokens"]) * 100, 2),
                "slots": val_stats["by_domain"][d]["slots"],
                "safe_slots": val_stats["by_domain"][d]["safe_slots"],
                "top_phrases": [
                    {"phrase": p, "steps_saved": c}
                    for p, c in val_stats["top_phrases"][d].most_common(10)
                ],
            }
            for d in ["code", "reasoning", "instruction"]
        },
    }

    os.makedirs(os.path.dirname(OUT_VAL_AUDIT), exist_ok=True)
    with open(OUT_VAL_AUDIT, "w", encoding="utf-8") as f:
        json.dump(audit_output, f, indent=2)
    print(f"Saved validation audit to {OUT_VAL_AUDIT}", flush=True)

    # 3. Process Training Split (data/train.jsonl)
    print(f"\n--- 2. EXTRACTING LABELS FROM TRAIN SPLIT ({TRAIN_DATA_PATH}) ---", flush=True)
    
    # Load all train samples
    train_samples_by_domain = defaultdict(list)
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            dom = item.get("domain", "general")
            train_samples_by_domain[dom].append(item)

    print("Loaded training samples by domain:")
    for dom, s_list in train_samples_by_domain.items():
        print(f"  {dom:12s}: {len(s_list)} samples")

    # Select a balanced, high-quality stratified training set:
    # All 779 code samples, 1000 instruction samples, 1000 reasoning samples
    # Total = 2,779 training samples
    # (Covers the full code diversity while maintaining balanced domain representation)
    selected_train_samples = []
    for dom in ["code", "instruction", "reasoning"]:
        s_list = train_samples_by_domain[dom]
        limit = 779 if dom == "code" else 1000
        selected_train_samples.extend(s_list[:limit])

    print(f"Selected {len(selected_train_samples)} stratified training samples for label generation", flush=True)

    # Verify zero overlap with validation prompt IDs
    train_prompt_ids = set(s["id"] for s in selected_train_samples)
    overlap = train_prompt_ids & val_prompt_ids
    assert len(overlap) == 0, f"DATA CONTAMINATION DETECTED: {len(overlap)} IDs overlap between Train and Val!"
    print(f"Verified strict data isolation: 0 overlapping prompt IDs with validation set.", flush=True)

    # Process samples and stream out to OUT_TRAIN_LABELS
    os.makedirs(os.path.dirname(OUT_TRAIN_LABELS), exist_ok=True)
    out_file = open(OUT_TRAIN_LABELS, "w", encoding="utf-8")

    total_train_tokens = 0
    total_train_saved = 0
    total_train_slots = 0
    total_safe_slots = 0
    domain_train_stats = defaultdict(lambda: {"samples": 0, "tokens": 0, "saved": 0, "slots": 0, "safe_slots": 0})

    t_start_loop = time.perf_counter()
    report_interval = 250

    for idx, sample in enumerate(selected_train_samples):
        pid = sample["id"]
        dom = sample.get("domain", "general")
        p_text = sample["prompt"]
        r_text = sample.get("response", "")

        p_ids = tokenizer.encode(p_text, add_special_tokens=False)
        r_ids = tokenizer.encode(r_text, add_special_tokens=False)

        total_train_tokens += len(r_ids)
        domain_train_stats[dom]["samples"] += 1
        domain_train_stats[dom]["tokens"] += len(r_ids)

        if len(r_ids) < 2:
            continue

        # Extract prompt candidate n-grams as negative / distractor pool
        prompt_ngrams = extract_all_candidate_phrases(p_ids, min_len=2, max_len=4)

        # Run Quality-Aware Oracle
        sel_cb, saved, sel_recs, all_target_cands = lazy_greedy_quality_oracle(
            r_ids, p_text, p_ids, dom, tokenizer, k=K_CODEBOOK
        )

        total_train_saved += saved
        total_train_slots += len(sel_cb)
        domain_train_stats[dom]["saved"] += saved
        domain_train_stats[dom]["slots"] += len(sel_cb)

        for r in sel_recs:
            if r["is_safe"]:
                total_safe_slots += 1
                domain_train_stats[dom]["safe_slots"] += 1

        # Build candidate supervision records:
        # 1. All selected oracle phrases (positive targets with marginal_steps_saved and target_value > 0)
        # 2. Top unselected target candidates (had frequency, but marginal gain was 0 or unsafe)
        # 3. Prompt distractors not occurring in target (target_value = 0.0)
        target_phrases_set = set(tuple(r["phrase_tokens"]) for r in all_target_cands)
        prompt_distractors = []
        for p_cand_tup in prompt_ngrams.keys():
            if p_cand_tup not in target_phrases_set:
                p_cand_text = tokenizer.decode(list(p_cand_tup))
                p_safety = compute_safety_prior(p_cand_text, list(p_cand_tup), p_text, p_ids, dom)
                prompt_distractors.append({
                    "phrase_tokens": list(p_cand_tup),
                    "phrase_text": p_cand_text,
                    "length": len(p_cand_tup),
                    "target_count": 0,
                    "isolated_steps_saved": 0,
                    "marginal_steps_saved": 0,
                    "safety_prior": p_safety["safety_prior"],
                    "is_safe": p_safety["is_safe"],
                    "reasons": p_safety["reasons"],
                    "has_trailing_space": p_safety["has_trailing_space"],
                    "selected_in_oracle": False,
                    "target_value": 0.0,
                })
                if len(prompt_distractors) >= 15:
                    break

        train_record = {
            "id": pid,
            "domain": dom,
            "prompt_text": p_text,
            "prompt_token_ids": p_ids,
            "target_response_len": len(r_ids),
            "oracle_steps_saved": saved,
            "selected_codebook_size": len(sel_cb),
            "selected_phrases": sel_recs,
            "other_target_candidates": [c for c in all_target_cands if not c["selected_in_oracle"]][:15],
            "prompt_distractors": prompt_distractors,
        }

        out_file.write(json.dumps(train_record) + "\n")

        if (idx + 1) % report_interval == 0 or (idx + 1) == len(selected_train_samples):
            elapsed = time.perf_counter() - t_start_loop
            rate = (idx + 1) / elapsed
            print(f"  Processed {idx + 1}/{len(selected_train_samples)} train samples "
                  f"({rate:.1f} samples/s) | Net saved: {total_train_saved} tokens", flush=True)

    out_file.close()

    train_micro = (total_train_saved / max(1, total_train_tokens)) * 100
    safe_pct = (total_safe_slots / max(1, total_train_slots)) * 100
    print(f"\nLabel Generation Complete:")
    print(f"  Output File: {OUT_TRAIN_LABELS}")
    print(f"  Total Processed Samples: {len(selected_train_samples)}")
    print(f"  Total Target Tokens: {total_train_tokens}")
    print(f"  Total Oracle Steps Saved: {total_train_saved} ({train_micro:.2f}% micro compression)")
    print(f"  Total Slots Allocated: {total_train_slots}")
    print(f"  Safe Slots: {total_safe_slots} ({safe_pct:.1f}%)")

    for dom in ["code", "instruction", "reasoning"]:
        dst = domain_train_stats[dom]
        d_micro = (dst["saved"] / max(1, dst["tokens"])) * 100
        d_safe_pct = (dst["safe_slots"] / max(1, dst["slots"])) * 100
        print(f"  Domain {dom:12s}: {dst['samples']} samples | {dst['tokens']} tokens | "
              f"{dst['saved']} saved ({d_micro:.2f}%) | Safe Slots: {dst['safe_slots']}/{dst['slots']} ({d_safe_pct:.1f}%)")

    total_time = time.perf_counter() - t0
    print(f"\nPhase 4 complete in {total_time:.2f}s", flush=True)


if __name__ == "__main__":
    run_phase4_label_generation()
