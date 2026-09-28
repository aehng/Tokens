"""Architecture A: Retrained Ridge Baseline for Predictor V2.

Trains a feature-weighted closed-form Ridge regression model on the 21 handcrafted features
using objective occurrence labels strictly on the TRAIN split.
"""

from __future__ import annotations

import pickle
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.zip2zip.predictor_v2.candidate_pool import CandidateRecord
from src.zip2zip.predictor_v2.interfaces import MultiTaskPredictions, PredictorScorer
from src.zip2zip.predictor_v2.utility import compute_expected_occurrence_value


class RidgeRanker(PredictorScorer):
    """Architecture A: Retrained Ridge Baseline on 21 Handcrafted Features."""

    def __init__(self, l2_reg: float = 1.0):
        self.l2_reg = l2_reg
        self.weights_occur: Optional[np.ndarray] = None
        self.bias_occur: float = 0.0
        self.weights_count: Optional[np.ndarray] = None
        self.bias_count: float = 0.0
        self.trained_at: Optional[float] = None

    def fit(
        self,
        train_records: Sequence[Any],
        train_candidates: Dict[str, List[CandidateRecord]],
        dev_records: Optional[Sequence[Any]] = None,
        dev_candidates: Optional[Dict[str, List[CandidateRecord]]] = None,
        epochs: int = 1,
        lr: float = 0.0,
    ) -> Dict[str, Any]:
        """Fits closed-form Ridge regression on TRAIN split."""
        t0 = time.perf_counter()
        X_list = []
        y_occur_list = []
        y_count_list = []

        for r in train_records:
            cands = train_candidates[r.prompt_id]
            for c in cands:
                X_list.append(c.features)
                y_occur_list.append(1.0 if c.occurs_in_vanilla else 0.0)
                y_count_list.append(float(c.occurrence_count))

        X = np.array(X_list, dtype=np.float32)
        y_occur = np.array(y_occur_list, dtype=np.float32)
        y_count = np.array(y_count_list, dtype=np.float32)

        N, D = X.shape
        X_ext = np.hstack([X, np.ones((N, 1), dtype=np.float32)])

        reg = self.l2_reg * np.eye(D + 1, dtype=np.float32)
        reg[-1, -1] = 0.0  # Do not regularize bias

        A = np.dot(X_ext.T, X_ext) + reg

        # 1. Occurrence Head (Linear Probability)
        b_occur = np.dot(X_ext.T, y_occur)
        theta_occur = np.linalg.solve(A, b_occur)
        self.weights_occur = theta_occur[:-1]
        self.bias_occur = float(theta_occur[-1])

        # 2. Count Head (Expected Count)
        b_count = np.dot(X_ext.T, y_count)
        theta_count = np.linalg.solve(A, b_count)
        self.weights_count = theta_count[:-1]
        self.bias_count = float(theta_count[-1])

        self.trained_at = time.time()
        elapsed = time.perf_counter() - t0

        return {
            "model": "RidgeRanker",
            "samples_trained": N,
            "features_dim": D,
            "train_time_sec": round(elapsed, 4),
        }

    def score_candidates(
        self,
        prompt_ids: Sequence[int],
        candidates: Sequence[CandidateRecord],
        domain: str = "general",
    ) -> MultiTaskPredictions:
        """Scores candidate pool using trained Ridge weights."""
        if not candidates:
            return MultiTaskPredictions(
                p_occurs=np.array([]),
                expected_count=np.array([]),
                horizon_logits=np.zeros((0, 5)),
                p_safe=np.array([]),
                ranking_scores=np.array([]),
            )

        X = np.array([c.features for c in candidates], dtype=np.float32)
        lens = [c.length for c in candidates]

        # 1. P(occurs): Linear + Sigmoid
        raw_occur = np.dot(X, self.weights_occur) + self.bias_occur
        p_occurs = 1.0 / (1.0 + np.exp(-np.clip(raw_occur, -15.0, 15.0)))

        # 2. Expected Count: Softplus
        raw_count = np.dot(X, self.weights_count) + self.bias_count
        expected_count = np.log1p(np.exp(np.clip(raw_count, -15.0, 15.0)))

        # 3. Horizon Logits: 5 buckets, simplified distribution
        # High occurrence probability leans earlier
        horizon_logits = np.zeros((len(candidates), 5), dtype=np.float32)
        horizon_logits[:, 0] = raw_occur * 0.5
        horizon_logits[:, 1] = raw_occur * 0.3
        horizon_logits[:, 2] = raw_occur * 0.1
        horizon_logits[:, 3] = raw_occur * 0.05
        horizon_logits[:, 4] = -raw_occur * 0.8  # Never bucket

        # 4. Safe prior default
        p_safe = np.ones(len(candidates), dtype=np.float32)

        # 5. Ranking score: P(occur) * E[count] * (len - 1)
        ranking_scores = compute_expected_occurrence_value(p_occurs, expected_count, lens)

        return MultiTaskPredictions(
            p_occurs=p_occurs,
            expected_count=expected_count,
            horizon_logits=horizon_logits,
            p_safe=p_safe,
            ranking_scores=ranking_scores,
        )

    def get_parameter_count(self) -> int:
        d = len(self.weights_occur) if self.weights_occur is not None else 21
        return (d + 1) * 2  # 2 heads (occur, count)

    def get_model_size_bytes(self) -> int:
        return len(pickle.dumps(self))
