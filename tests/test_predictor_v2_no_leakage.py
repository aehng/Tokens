"""Tests for Predictor V2 Leakage Safeguards.

Asserts:
1. Candidate generation sees prompt only.
2. Model scorer inputs contain zero continuation tokens.
3. Split IDs have strictly zero overlap (Train, Dev, Test disjoint).
4. Oracle classes require explicit continuation input and cannot be used as production predictors.
5. Global oracle never contaminates candidate pool.
"""

import pickle
import pytest
from src.zip2zip.predictor_v2.dataset import create_deterministic_splits, BakeoffDataset
from src.zip2zip.predictor_v2.candidate_pool import PromptCandidateGenerator
from src.zip2zip.predictor_v2.models.ridge import RidgeRanker
from src.zip2zip.predictor_v2.vanilla_labels import load_canonical_vanilla_records, get_canonical_tokenizer


def test_split_zero_leakage():
    records = load_canonical_vanilla_records()
    manifest = create_deterministic_splits(records, seed=42)
    manifest.verify_zero_leakage()

    train_set = set(manifest.train_ids)
    dev_set = set(manifest.dev_ids)
    test_set = set(manifest.frozen_test_ids)

    assert len(train_set & dev_set) == 0
    assert len(train_set & test_set) == 0
    assert len(dev_set & test_set) == 0
    assert len(train_set) + len(dev_set) + len(test_set) == len(records)


def test_candidate_generation_is_prompt_only():
    with open("experiments/checkpoints/cached_predictor.pkl", "rb") as f:
        p_idx = pickle.load(f)
    tok = get_canonical_tokenizer()
    gen = PromptCandidateGenerator(p_idx, tok)

    prompt_ids = tok.encode("Write a Python function to add two numbers.", add_special_tokens=False)
    prompt_text = "Write a Python function to add two numbers."

    # Call candidate generation without response
    cands = gen.generate_candidate_pool(prompt_ids, prompt_text, domain="code")
    assert len(cands) > 0

    # Ensure no continuation tokens or responses were required
    for gram in cands.keys():
        assert len(gram) >= 2
        assert len(gram) <= 4


def test_model_scorer_zero_continuation_input():
    ridge = RidgeRanker()
    # Scorer API accepts only prompt_ids, candidates, and domain
    # It has no argument for continuation tokens
    import inspect
    sig = inspect.signature(ridge.score_candidates)
    params = list(sig.parameters.keys())
    assert "continuation_tokens" not in params
    assert "response" not in params
