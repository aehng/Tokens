"""
Prepare and Pre-segment Training and Multi-Domain Validation Sets for Predictive Calibration.

Requirements:
1. Strict prompt-only causality:
   Prompt -> Predictor (prompt tokens ONLY) -> K=32 Codebook
   -> DP Segment Prompt
   -> DP Segment Response with the exact SAME codebook
2. Training Set:
   - Sample 2,000 diverse prompt-response pairs from data/train.jsonl
   - Stratified: Code (500), Instruction (750), Reasoning (750)
   - Filter / truncate to max total length 512 to maintain fast CPU training throughput
3. Validation Set:
   - 60 held-out prompts from data/val.jsonl
   - Stratified: Code (20), Instruction (20), Reasoning (20)
   - Diverse prompt length tiers
4. Cache all segmented inputs, target labels, codebooks, and metadata to disk.
"""

import json
import os
import pickle
import random
import sys
import time
from typing import Dict, List, Tuple
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.evaluation.offline_segmenter import segment_tokens_dp


def prepare_datasets(
    train_path: str = "data/train.jsonl",
    val_path: str = "data/val.jsonl",
    predictor_path: str = "experiments/checkpoints/cached_predictor.pkl",
    output_train_path: str = "data/cached_pure_pred_train_2k.pkl",
    output_val_path: str = "data/cached_pure_pred_val_60.json",
    budget_k: int = 32,
    max_seq_len: int = 512,
    seed: int = 42,
):
    print("=" * 80)
    print("PREPARING PREDICTIVE CALIBRATION DATASETS")
    print(f"K = {budget_k}, Max Seq Len = {max_seq_len}, Seed = {seed}")
    print("=" * 80)

    random.seed(seed)
    initial_vocab_size = 32011

    # Load Tokenizer & Predictor
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    with open(predictor_path, "rb") as f:
        predictor = pickle.load(f)
    print("Loaded tokenizer and predictor.")

    # 1. Load data/train.jsonl by domain
    train_by_domain = {"code": [], "instruction": [], "reasoning": []}
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            d = item.get("domain", "instruction")
            if d in train_by_domain:
                train_by_domain[d].append(item)

    print(f"Loaded train items by domain:")
    for d, items in train_by_domain.items():
        print(f"  {d}: {len(items)}")

    # Target: 500 code, 750 instruction, 750 reasoning
    random.shuffle(train_by_domain["code"])
    random.shuffle(train_by_domain["instruction"])
    random.shuffle(train_by_domain["reasoning"])

    selected_train_raw = (
        train_by_domain["code"][:500]
        + train_by_domain["instruction"][:750]
        + train_by_domain["reasoning"][:750]
    )
    random.shuffle(selected_train_raw)
    print(f"Selected {len(selected_train_raw)} training candidates.")

    # 2. Process and pre-segment training samples
    processed_train = []
    total_prompt_base_tokens = 0
    total_prompt_comp_tokens = 0
    total_response_base_tokens = 0
    total_response_comp_tokens = 0
    total_hyper_in_response = 0

    t0 = time.time()
    for s_idx, item in enumerate(selected_train_raw):
        p_text = item["prompt"]
        r_text = item["response"]

        p_ids = tokenizer.encode(p_text, add_special_tokens=False)
        r_ids = tokenizer.encode(r_text, add_special_tokens=False)

        # Strict prompt-only causality: predictor ONLY sees p_ids
        p_dict, _ = predictor.select_prompt_conditioned(p_ids, budget=budget_k)
        pred_phrases = list(p_dict.keys())
        seeded_dict = {p: initial_vocab_size + i for i, p in enumerate(pred_phrases)}

        # Segment prompt
        _, p_tiles, _ = segment_tokens_dp(p_ids, set(pred_phrases))
        p_comp = [tile[0] if len(tile) == 1 else seeded_dict[tile] for tile in p_tiles]

        # Segment response with the SAME codebook
        _, r_tiles, _ = segment_tokens_dp(r_ids, set(pred_phrases))
        r_comp = [tile[0] if len(tile) == 1 else seeded_dict[tile] for tile in r_tiles]

        # Count hypertoken emissions in response
        n_hypers = sum(1 for tid in r_comp if tid >= initial_vocab_size)

        # Truncate if sequence exceeds max_seq_len
        # Format: [prompt_ids] + [response_ids] + [eos_token_id]
        eos_id = tokenizer.eos_token_id or 32000
        full_input_ids = p_comp + r_comp + [eos_id]
        full_labels = [-100] * len(p_comp) + r_comp + [eos_id]

        if len(full_input_ids) > max_seq_len:
            # Keep prompt intact if possible, truncate response
            if len(p_comp) >= max_seq_len - 10:
                p_comp = p_comp[: max_seq_len - 64]
            r_comp = r_comp[: max_seq_len - len(p_comp) - 1]
            full_input_ids = p_comp + r_comp + [eos_id]
            full_labels = [-100] * len(p_comp) + r_comp + [eos_id]
            n_hypers = sum(1 for tid in r_comp if tid >= initial_vocab_size)

        total_prompt_base_tokens += len(p_ids)
        total_prompt_comp_tokens += len(p_comp)
        total_response_base_tokens += len(r_ids)
        total_response_comp_tokens += len(r_comp)
        total_hyper_in_response += n_hypers

        processed_train.append({
            "id": item.get("id", f"train_{s_idx}"),
            "domain": item.get("domain", "general"),
            "input_ids": full_input_ids,
            "labels": full_labels,
            "prompt_len": len(p_comp),
            "response_len": len(r_comp),
            "n_hypers_in_target": n_hypers,
            "seeded_dict": seeded_dict,
        })

    print(f"Pre-segmented {len(processed_train)} training samples in {time.time() - t0:.2f}s")
    print(f"Prompt base tokens:   {total_prompt_base_tokens:,} -> {total_prompt_comp_tokens:,} (saved {(total_prompt_base_tokens - total_prompt_comp_tokens)/total_prompt_base_tokens*100:.2f}%)")
    print(f"Response base tokens: {total_response_base_tokens:,} -> {total_response_comp_tokens:,} (saved {(total_response_base_tokens - total_response_comp_tokens)/total_response_base_tokens*100:.2f}%)")
    print(f"Total hypertoken target instances in training responses: {total_hyper_in_response:,} ({total_hyper_in_response / len(processed_train):.1f} per sample)")
    print(f"Samples with >= 1 hypertoken in target: {sum(1 for s in processed_train if s['n_hypers_in_target'] > 0)} / {len(processed_train)} ({sum(1 for s in processed_train if s['n_hypers_in_target'] > 0)/len(processed_train)*100:.1f}%)")

    # Save training dataset
    with open(output_train_path, "wb") as f:
        pickle.dump(processed_train, f)
    print(f"Saved cached training data to {output_train_path}")

    # 3. Process Validation Set (60 held-out prompts from val.jsonl)
    val_by_domain = {"code": [], "instruction": [], "reasoning": []}
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            d = item.get("domain", "instruction")
            if d in val_by_domain:
                val_by_domain[d].append(item)

    print("\nLoaded val items by domain:")
    for d, items in val_by_domain.items():
        print(f"  {d}: {len(items)}")

    random.shuffle(val_by_domain["code"])
    random.shuffle(val_by_domain["instruction"])
    random.shuffle(val_by_domain["reasoning"])

    selected_val = (
        val_by_domain["code"][:20]
        + val_by_domain["instruction"][:20]
        + val_by_domain["reasoning"][:20]
    )

    val_records = []
    for item in selected_val:
        p_text = item["prompt"]
        p_ids = tokenizer.encode(p_text, add_special_tokens=False)
        val_records.append({
            "id": item.get("id"),
            "domain": item.get("domain"),
            "prompt": p_text,
            "ground_truth_response": item.get("response", ""),
            "prompt_token_ids": p_ids,
            "base_prompt_len": len(p_ids),
        })

    with open(output_val_path, "w", encoding="utf-8") as f:
        json.dump(val_records, f, indent=2)
    print(f"Saved 60 frozen validation records to {output_val_path}")


if __name__ == "__main__":
    prepare_datasets()
