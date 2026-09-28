"""Tests for Predictor V2 Scorer Architectures."""

import pytest
import numpy as np
from src.zip2zip.predictor_v2.models.ridge import RidgeRanker
from src.zip2zip.predictor_v2.models.pooled_mlp import PooledMLPRanker
from src.zip2zip.predictor_v2.models.cnn_ranker import CNNRanker
from src.zip2zip.predictor_v2.models.gru_ranker import GRURanker
from src.zip2zip.predictor_v2.models.transformer_ranker import TransformerRanker
from src.zip2zip.predictor_v2.candidate_pool import CandidateRecord


def _make_mock_candidate():
    return CandidateRecord(
        prompt_id="p1",
        tokens=(100, 200),
        text="test phrase",
        length=2,
        sources=["ngram"],
        raw_association_weight=5.0,
        features=np.ones(21, dtype=np.float32),
        occurs_in_vanilla=True,
        occurrence_count=3,
        first_occurrence_index=10,
        first_occurrence_bucket=0,
        isolated_steps_saved=3,
    )


@pytest.mark.parametrize("model_cls", [
    RidgeRanker,
    PooledMLPRanker,
    CNNRanker,
    GRURanker,
    TransformerRanker,
])
def test_model_forward_and_scoring(model_cls):
    model = model_cls()
    cand = _make_mock_candidate()
    prompt_ids = [10, 20, 30, 40, 50]

    # For Ridge, initialize dummy weights so scoring works without fit
    if isinstance(model, RidgeRanker):
        model.weights_occur = np.ones(21, dtype=np.float32)
        model.weights_count = np.ones(21, dtype=np.float32)

    preds = model.score_candidates(prompt_ids, [cand], domain="code")
    assert len(preds.p_occurs) == 1
    assert 0.0 <= preds.p_occurs[0] <= 1.0
    assert preds.expected_count[0] >= 0.0
    assert preds.horizon_logits.shape == (1, 5)
    assert len(preds.ranking_scores) == 1

    ranked = model.rank_codebook(prompt_ids, [cand], domain="code", k=1)
    assert len(ranked) == 1
    assert ranked[0][0] == cand
