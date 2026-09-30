"""Unit tests for phrase dataset generation, split isolation, and safety checks."""

import json
from pathlib import Path
import pytest

from experiments.build_phrase_dataset import (
    extract_phrases_from_continuation,
    is_special_token,
)


def test_is_special_token_filter():
    """Verify that token IDs >= 32000 and canonical EOS are classified as special."""
    assert is_special_token(32000) is True
    assert is_special_token(32007) is True  # Main Phi EOS
    assert is_special_token(32001) is True
    assert is_special_token(32063) is True
    assert is_special_token(100) is False
    assert is_special_token(29871) is False


def test_extract_phrases_from_continuation_length_and_constraints():
    """Verify that extracted phrases satisfy all length and token constraints."""
    # Synthetic tokens: 100 normal tokens
    tokens = list(range(100, 200))
    # Insert special token at pos 50
    tokens[50] = 32007

    examples = extract_phrases_from_continuation(
        prompt_id="test_prompt",
        domain="code",
        continuation_tokens=tokens,
        min_context=32,
        max_context=64,
        min_future=16,
        max_future=32,
        max_examples_per_prompt=5,
        context_step=8,
    )

    assert len(examples) > 0
    assert len(examples) <= 5

    for ex in examples:
        assert ex["prompt_id"] == "test_prompt"
        assert ex["domain"] == "code"
        assert ex["token_a"] < 32000
        assert ex["token_b"] < 32000
        assert len(ex["context_token_ids"]) >= 32
        assert len(ex["future_token_ids"]) >= 16


def test_phrase_dataset_manifest_and_splits_isolation():
    """Verify that generated phrase datasets exist and have zero overlap between subtrain and val."""
    manifest_path = Path("data/phrase_training_dataset/phrase_dataset_manifest.json")
    if not manifest_path.exists():
        pytest.skip("Dataset manifest not yet generated")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["final_split_accessed"] is False
    assert manifest["subtrain_example_count"] > 5000
    assert manifest["val_example_count"] > 500
    assert manifest["dev_benchmark_example_count"] == 48

    subtrain_file = Path("data/phrase_training_dataset/train_phrases_subtrain.jsonl")
    val_file = Path("data/phrase_training_dataset/train_phrases_val.jsonl")

    subtrain_pids = set()
    with subtrain_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                subtrain_pids.add(json.loads(line)["prompt_id"])

    val_pids = set()
    with val_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_pids.add(json.loads(line)["prompt_id"])

    # Strict disjointness
    overlap = subtrain_pids.intersection(val_pids)
    assert len(overlap) == 0, f"Subtrain and validation prompt IDs overlap: {overlap}"
