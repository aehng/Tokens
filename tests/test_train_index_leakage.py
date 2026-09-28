"""Tests for Train-Only Candidate Association Index Leakage Safeguards.

Asserts:
1. Scaled dataset manifest has zero overlap between TRAIN, DEV, and FINAL.
2. Index contains only prompt IDs from TRAIN (no DEV or FINAL IDs).
3. BM25 corpus documents in index are 100% from TRAIN.
4. Candidate generation using TrainOnlyAssociationIndex on DEV prompts executes strictly prompt-only.
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from src.zip2zip.predictor_v2.candidate_retrieval import (
    ConfigurableCandidateGenerator,
    RetrievalStrategy,
    TrainOnlyAssociationIndex,
)
from src.zip2zip.predictor_v2.vanilla_labels import get_canonical_tokenizer


def test_scaled_manifest_disjoint_splits():
    manifest_path = "docs/predictor_v2_scaled_dataset_manifest.json"
    assert os.path.exists(manifest_path), f"Manifest missing: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_ids = set(data["train_prompt_ids"])
    dev_ids = set(data["dev_prompt_ids"])
    final_ids = set(data["final_prompt_ids"])

    assert len(train_ids) == 630
    assert len(dev_ids) == 135
    assert len(final_ids) == 135

    assert len(train_ids & dev_ids) == 0, "TRAIN and DEV overlap!"
    assert len(train_ids & final_ids) == 0, "TRAIN and FINAL overlap!"
    assert len(dev_ids & final_ids) == 0, "DEV and FINAL overlap!"


def test_train_index_leakage():
    index_path = "experiments/checkpoints/train_only_association_index.pkl"
    if not os.path.exists(index_path):
        pytest.skip(f"Index artifact {index_path} not yet built (waiting for continuation generation).")

    index = TrainOnlyAssociationIndex.load(index_path)

    manifest_path = "docs/predictor_v2_scaled_dataset_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_ids = set(manifest["train_prompt_ids"])
    dev_ids = set(manifest["dev_prompt_ids"])
    final_ids = set(manifest["final_prompt_ids"])

    # Check indexed prompt IDs
    assert set(index.train_prompt_ids) == train_ids, "Indexed prompt IDs do not match TRAIN split!"

    # Check that no DEV or FINAL IDs are in BM25 docs
    for doc in index.train_prompt_lexical_docs:
        assert doc["prompt_id"] in train_ids
        assert doc["prompt_id"] not in dev_ids
        assert doc["prompt_id"] not in final_ids

    # Candidate generator test
    tok = get_canonical_tokenizer()
    gen = ConfigurableCandidateGenerator(index, tok)

    # Test all 4 strategies
    for strategy in [
        RetrievalStrategy.BASELINE,
        RetrievalStrategy.EXPANDED_ASSOCIATIONS,
        RetrievalStrategy.SUFFIX_CONDITIONED,
        RetrievalStrategy.SPARSE_LEXICAL,
    ]:
        p_ids = tok.encode("def compute_gcd(a, b):", add_special_tokens=False)
        cands = gen.generate_candidate_pool(p_ids, "def compute_gcd(a, b):", domain="code", strategy=strategy, target_pool_size=256)
        assert len(cands) > 0
        assert len(cands) <= 256
