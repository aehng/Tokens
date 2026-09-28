"""Common PredictorScorer Interfaces & Prediction Data Structures for Predictor V2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.zip2zip.predictor_v2.candidate_pool import CandidateRecord


@dataclass
class MultiTaskPredictions:
    p_occurs: np.ndarray  # (N,) float in [0, 1]
    expected_count: np.ndarray  # (N,) float >= 0
    horizon_logits: np.ndarray  # (N, 5) raw logits
    p_safe: np.ndarray  # (N,) float in [0, 1]
    ranking_scores: np.ndarray  # (N,) float


class PredictorScorer(ABC):
    """Common interface for all Predictor V2 candidate ranker architectures."""

    @abstractmethod
    def fit(
        self,
        train_records: Sequence[Any],
        train_candidates: Dict[str, List[CandidateRecord]],
        dev_records: Optional[Sequence[Any]] = None,
        dev_candidates: Optional[Dict[str, List[CandidateRecord]]] = None,
        epochs: int = 15,
        lr: float = 1e-3,
    ) -> Dict[str, Any]:
        """Fits the architecture on the TRAIN split with DEV early stopping."""
        pass

    @abstractmethod
    def score_candidates(
        self,
        prompt_ids: Sequence[int],
        candidates: Sequence[CandidateRecord],
        domain: str = "general",
    ) -> MultiTaskPredictions:
        """Scores candidate pool for a given prompt and returns multi-task predictions."""
        pass

    def rank_codebook(
        self,
        prompt_ids: Sequence[int],
        candidates: Sequence[CandidateRecord],
        domain: str = "general",
        k: int = 32,
    ) -> List[Tuple[CandidateRecord, float]]:
        """Ranks candidates by primary occurrence utility and returns top-K."""
        if not candidates or k <= 0:
            return []
        preds = self.score_candidates(prompt_ids, candidates, domain=domain)
        indexed = list(zip(candidates, preds.ranking_scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:k]

    @abstractmethod
    def get_parameter_count(self) -> int:
        """Returns total trainable/active parameter count."""
        pass

    @abstractmethod
    def get_model_size_bytes(self) -> int:
        """Returns serialized size in bytes."""
        pass
