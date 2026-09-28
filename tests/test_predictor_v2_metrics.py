"""Tests for Predictor V2 Metrics Suite."""

import pytest
import numpy as np
from src.zip2zip.predictor_v2.metrics import compute_auroc_auprc, compute_brier_score, compute_macro_f1


def test_metrics_calculation():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([0.9, 0.8, 0.2, 0.1, 0.4, 0.3])

    auroc, auprc = compute_auroc_auprc(y_true, y_pred)
    assert 0.0 <= auroc <= 1.0
    assert 0.0 <= auprc <= 1.0

    brier = compute_brier_score(y_true, y_pred)
    assert 0.0 <= brier <= 1.0

    y_class_true = np.array([0, 1, 2, 3, 4])
    y_class_pred = np.array([0, 1, 2, 3, 4])
    f1 = compute_macro_f1(y_class_true, y_class_pred, num_classes=5)
    assert f1 == 1.0
