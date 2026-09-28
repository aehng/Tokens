"""Ranking Utility & Factorized Value Formulations for Predictor V2.

Objective ranking utility formulation:
    expected_occurrence_value = P_occurs * expected_occurrence_count * (phrase_length - 1)

Safety-adjusted reporting utility:
    expected_safe_value = expected_occurrence_value * P_empirical_safe
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


def compute_expected_occurrence_value(
    p_occurs: np.ndarray,
    expected_count: np.ndarray,
    phrase_lengths: Sequence[int],
) -> np.ndarray:
    """Computes expected occurrence utility: P(occur) * E[count] * (len - 1)."""
    lens = np.array(phrase_lengths, dtype=np.float32)
    step_savings = np.maximum(0.0, lens - 1.0)
    return p_occurs * expected_count * step_savings


def compute_expected_safe_value(
    occurrence_value: np.ndarray,
    p_safe: np.ndarray,
) -> np.ndarray:
    """Computes expected safe value: occurrence_value * P(safe)."""
    return occurrence_value * p_safe
