"""Evaluation Metrics for Predictor V2 Architectures.

Implements all Part 13 evaluation metrics:
1. Ranking metrics (Precision@K, Recall@K, Dead-slot rate, DP steps saved, Capture ratios)
2. Occurrence head metrics (AUPRC, AUROC, Brier score)
3. Count head metrics (MAE, log-MAE)
4. Horizon head metrics (Accuracy, Macro-F1)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from src.evaluation.offline_segmenter import segment_tokens_dp
from src.zip2zip.predictor_v2.candidate_pool import CandidateRecord
from src.zip2zip.predictor_v2.interfaces import MultiTaskPredictions, PredictorScorer


def compute_auroc_auprc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Computes AUROC and AUPRC without external scikit-learn dependency."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    # Sort descending by predicted score
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    
    n_pos = int(np.sum(y_true_sorted == 1))
    n_neg = int(np.sum(y_true_sorted == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    # AUROC via Wilcoxon-Mann-Whitney rank sum
    ranks = np.arange(len(y_score), 0, -1)
    pos_ranks = ranks[y_true_sorted == 1]
    rank_sum = np.sum(pos_ranks)
    u = rank_sum - (n_pos * (n_pos + 1)) / 2.0
    auroc = float(u / (n_pos * n_neg))

    # AUPRC via trapezoidal approximation
    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)
    precision = tp / (tp + fp)
    recall = tp / n_pos

    # Prepend (recall=0, precision=precision[0])
    recall_full = np.concatenate([[0.0], recall])
    precision_full = np.concatenate([[precision[0]], precision])
    auprc = float(np.trapz(precision_full, recall_full))

    return round(auroc, 4), round(auprc, 4)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> float:
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def evaluate_architecture_ranking(
    model: PredictorScorer,
    records: Sequence[Any],
    candidates_by_prompt: Dict[str, List[CandidateRecord]],
    k_values: Sequence[int] = (8, 16, 32),
    global_oracle_steps_by_k: Optional[Dict[int, int]] = None,
    candidate_oracle_steps_by_k: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """Evaluates ranking performance and multi-task head metrics across a split."""
    results: Dict[str, Any] = {"ranking_by_k": {}, "head_metrics": {}}

    all_y_occur: List[int] = []
    all_p_occur: List[float] = []
    all_y_count: List[int] = []
    all_p_count: List[float] = []
    all_y_horizon: List[int] = []
    all_p_horizon: List[int] = []

    for k in k_values:
        total_selected = 0
        total_useful_selected = 0
        total_pool_useful = 0
        total_dp_steps = 0
        per_prompt_dp = []

        for r in records:
            cands = candidates_by_prompt[r.prompt_id]
            pool_useful = sum(1 for c in cands if c.occurs_in_vanilla)
            total_pool_useful += pool_useful

            # Rank top-K
            ranked = model.rank_codebook(r.prompt_token_ids, cands, domain=r.domain, k=k)
            selected_cands = [c for c, s in ranked]
            total_selected += len(selected_cands)

            useful = sum(1 for c in selected_cands if c.occurs_in_vanilla)
            total_useful_selected += useful

            # Compute realized DP steps saved on Vanilla continuation
            sel_phrases = {c.tokens for c in selected_cands}
            _, _, st = segment_tokens_dp(r.continuation_token_ids, sel_phrases)
            dp_saved = st["tokens_saved"]
            total_dp_steps += dp_saved
            per_prompt_dp.append(dp_saved)

        precision = (total_useful_selected / total_selected) if total_selected > 0 else 0.0
        recall = (total_useful_selected / total_pool_useful) if total_pool_useful > 0 else 0.0
        dead_slot_rate = 1.0 - precision

        cand_oracle_steps = candidate_oracle_steps_by_k.get(k, total_dp_steps) if candidate_oracle_steps_by_k else total_dp_steps
        global_oracle_steps = global_oracle_steps_by_k.get(k, total_dp_steps) if global_oracle_steps_by_k else total_dp_steps

        cand_capture = (total_dp_steps / cand_oracle_steps) if cand_oracle_steps > 0 else 0.0
        global_capture = (total_dp_steps / global_oracle_steps) if global_oracle_steps > 0 else 0.0

        results["ranking_by_k"][k] = {
            "k": k,
            "precision_at_k": round(precision, 4),
            "recall_at_k": round(recall, 4),
            "dead_slot_rate": round(dead_slot_rate, 4),
            "useful_phrases_selected": total_useful_selected,
            "total_slots_allocated": total_selected,
            "realized_dp_steps": total_dp_steps,
            "candidate_oracle_capture_pct": round(cand_capture * 100.0, 2),
            "global_oracle_capture_pct": round(global_capture * 100.0, 2),
            "mean_dp_steps_per_prompt": round(float(np.mean(per_prompt_dp)), 2),
        }

    # Head metrics evaluation
    for r in records:
        cands = candidates_by_prompt[r.prompt_id]
        preds = model.score_candidates(r.prompt_token_ids, cands, domain=r.domain)
        for idx, c in enumerate(cands):
            all_y_occur.append(1 if c.occurs_in_vanilla else 0)
            all_p_occur.append(float(preds.p_occurs[idx]))
            all_y_count.append(c.occurrence_count)
            all_p_count.append(float(preds.expected_count[idx]))
            all_y_horizon.append(c.first_occurrence_bucket)
            pred_h = int(np.argmax(preds.horizon_logits[idx])) if preds.horizon_logits is not None else 4
            all_p_horizon.append(pred_h)

    y_occ = np.array(all_y_occur)
    p_occ = np.array(all_p_occur)
    y_cnt = np.array(all_y_count, dtype=np.float32)
    p_cnt = np.array(all_p_count, dtype=np.float32)
    y_hor = np.array(all_y_horizon)
    p_hor = np.array(all_p_horizon)

    auroc, auprc = compute_auroc_auprc(y_occ, p_occ)
    brier = compute_brier_score(y_occ, p_occ)
    count_mae = float(np.mean(np.abs(y_cnt - p_cnt)))
    log_count_mae = float(np.mean(np.abs(np.log1p(y_cnt) - np.log1p(p_cnt))))
    horizon_acc = float(np.mean(y_hor == p_hor))
    horizon_f1 = compute_macro_f1(y_hor, p_hor, num_classes=5)

    results["head_metrics"] = {
        "occurrence_head": {
            "auroc": auroc,
            "auprc": auprc,
            "brier_score": round(brier, 4),
        },
        "count_head": {
            "mae": round(count_mae, 4),
            "log_mae": round(log_count_mae, 4),
        },
        "horizon_head": {
            "accuracy": round(horizon_acc, 4),
            "macro_f1": round(horizon_f1, 4),
        },
    }

    return results
