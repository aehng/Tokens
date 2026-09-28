"""Tests for Predictor V2 Candidate Pool Generator."""

import pickle
import pytest
from src.zip2zip.predictor_v2.candidate_pool import PromptCandidateGenerator, is_bare_punctuation, extract_handcrafted_features
from src.zip2zip.predictor_v2.vanilla_labels import get_canonical_tokenizer, load_canonical_vanilla_records


def test_candidate_pool_generation():
    with open("experiments/checkpoints/cached_predictor.pkl", "rb") as f:
        p_idx = pickle.load(f)
    tok = get_canonical_tokenizer()
    gen = PromptCandidateGenerator(p_idx, tok)

    recs = load_canonical_vanilla_records()
    sample = recs[0]

    cands = gen.build_candidate_records(sample)
    assert len(cands) >= 50
    assert len(cands[0].features) == 21
    assert cands[0].length in (2, 3, 4)


def test_bare_punctuation_filter():
    tok = get_canonical_tokenizer()
    dot_comma = tok.encode("...,", add_special_tokens=False)
    assert is_bare_punctuation(tuple(dot_comma), tok) is True

    word_phrase = tok.encode(" hello world", add_special_tokens=False)
    assert is_bare_punctuation(tuple(word_phrase), tok) is False
