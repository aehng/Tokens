"""Tests for Predictor V2 Oracles."""

import pytest
from src.zip2zip.predictor_v2.oracle_global import GlobalOccurrenceOracle
from src.zip2zip.predictor_v2.oracle_candidate_pool import CandidatePoolOracle
from src.zip2zip.predictor_v2.candidate_pool import CandidateRecord
from src.zip2zip.predictor_v2.oracle_exact import ExactHypertokenOracle
import numpy as np


def test_global_occurrence_oracle_basic():
    oracle = GlobalOccurrenceOracle(min_len=2, max_len=4)
    # Repeating phrase: [10, 20, 30] repeated 3 times
    tokens = [10, 20, 30, 40, 10, 20, 30, 50, 10, 20, 30]
    res = oracle.solve(tokens, k=4)
    assert res.steps_saved > 0
    assert (10, 20, 30) in res.selected_phrases
    assert res.emissions == 3


def test_candidate_pool_oracle_basic():
    oracle = CandidatePoolOracle()
    tokens = [1, 2, 3, 4, 1, 2, 3, 5]
    cand = CandidateRecord(
        prompt_id="test",
        tokens=(1, 2, 3),
        text="1 2 3",
        length=3,
        sources=["test"],
        raw_association_weight=1.0,
        features=np.zeros(21, dtype=np.float32),
        occurs_in_vanilla=True,
        occurrence_count=2,
        first_occurrence_index=0,
        first_occurrence_bucket=0,
        isolated_steps_saved=4,
    )
    res = oracle.solve([cand], tokens, k=4, global_oracle_steps=4)
    assert res.steps_saved == 4
    assert res.candidate_generation_capture == 1.0


def test_fixed_pool_occurrence_scan_matches_allowed_spans():
    oracle = ExactHypertokenOracle(min_len=2, max_len=4)
    occurrences = oracle.extract_occurrences(
        [1, 2, 1, 2, 3],
        allowed_phrases={(1, 2), (2, 1), (1, 2, 3), (9, 9), (1, 2, 3, 4)},
    )
    assert occurrences == {
        (1, 2): [(0, 2), (2, 4)],
        (2, 1): [(1, 3)],
        (1, 2, 3): [(2, 5)],
    }
