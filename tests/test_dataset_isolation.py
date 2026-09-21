import hashlib
import json
import os
import pytest


def test_stage1_dataset_isolation():
    splits = {}
    for s in ["train", "val", "test"]:
        p = f"data/corpus_stage1_{s}.jsonl"
        assert os.path.exists(p), f"Missing split file {p}"
        with open(p, "r", encoding="utf-8") as f:
            splits[s] = [json.loads(line) for line in f]

    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0

    # 1. Thread ID isolation: sets of thread IDs must be completely disjoint
    train_threads = {x["thread_id"] for x in splits["train"]}
    val_threads = {x["thread_id"] for x in splits["val"]}
    test_threads = {x["thread_id"] for x in splits["test"]}

    assert len(train_threads.intersection(val_threads)) == 0, "Leakage between train and val threads!"
    assert len(train_threads.intersection(test_threads)) == 0, "Leakage between train and test threads!"
    assert len(val_threads.intersection(test_threads)) == 0, "Leakage between val and test threads!"

    # 2. Prompt text exact deduplication across splits
    def p_hash(prompt_text):
        return hashlib.sha256(prompt_text.strip().lower().encode("utf-8")).hexdigest()

    train_prompts = {p_hash(x["prompt"]) for x in splits["train"]}
    val_prompts = {p_hash(x["prompt"]) for x in splits["val"]}
    test_prompts = {p_hash(x["prompt"]) for x in splits["test"]}

    assert len(train_prompts.intersection(val_prompts)) == 0, "Duplicate prompt between train and val!"
    assert len(train_prompts.intersection(test_prompts)) == 0, "Duplicate prompt between train and test!"
    assert len(val_prompts.intersection(test_prompts)) == 0, "Duplicate prompt between val and test!"

    # 3. Valid token ID sequences
    for s, items in splits.items():
        for item in items:
            assert len(item["prompt_token_ids"]) > 0
            assert len(item["response_token_ids"]) > 0
            assert item["prompt_length"] == len(item["prompt_token_ids"])
            assert item["response_length"] == len(item["response_token_ids"])
            assert item["domain"] in ["conversation", "code", "reasoning"]
