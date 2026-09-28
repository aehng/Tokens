"""Safety Labeling & Heuristic vs Empirical Safety Audit for Predictor V2.

Contains:
1. `compute_heuristic_safety_prior`: The legacy handcrafted safety prior renamed/clarified.
2. `ContinuationProbeResult`: Detailed schema for contextual base vs H continuation probes.
3. `load_historical_empirical_probes`: Loads verified empirical probe records.
4. `audit_heuristic_safety`: Benchmarks heuristic prior against empirical probes.
5. `StratifiedProbeSampler`: Constructs stratified probe candidate sets for empirical measurement.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Grammatical glue patterns
GRAMMATICAL_GLUE = {
    " of the", " to the", " in the", " for the", " on the", " with the", " by the",
    " at the", " from the", " into the", " as well", " as well as", " is a", " was a",
    " to be", " can be", " will be", " would be", " should be", " has been", " have been",
    " that the", " that is", " there is", " there are", " it is", " in order", " in order to",
    " such as", " based on", " due to", " according to", " in addition", " for example",
    " as a", " with a", " of a", " in a", " to a",
}

# Dangerous code syntax fragments
CODE_SYNTAX_FRAGMENTS = {
    "):\n", "): \n", "):", "):    ", "]:", "):\n    ", "\n    return ", "    return",
    "():", "[]", "{}", "()", "len(", "range(", "int(", "str(", "float(", "list(", "dict(",
}


def compute_heuristic_safety_prior(
    phrase_text: str,
    phrase_tokens: Sequence[int],
    prompt_text: str,
    prompt_tokens: Sequence[int],
    domain: str = "general",
) -> Dict[str, Any]:
    """Computes the legacy continuous heuristic SafetyPrior in [0.01, 1.0].
    
    Explicitly labeled as a heuristic teacher / rule-based prior, NOT empirical ground truth.
    """
    reasons = []
    score = 0.80

    # 1. Trailing whitespace hazard
    has_trailing_space = phrase_text.endswith(" ") or phrase_text.endswith("\t")
    if has_trailing_space:
        score *= 0.20
        reasons.append("trailing_whitespace_hazard")

    # 2. Leading boundary alignment
    has_leading_space = phrase_text.startswith(" ") or phrase_text.startswith("\n")
    has_leading_punct = len(phrase_text) > 0 and phrase_text[0] in ".,;:!?()[]{}\"'-"
    boundary_aligned = has_leading_space or has_leading_punct
    if boundary_aligned:
        score *= 1.05
    else:
        if len(phrase_text) > 0 and phrase_text[0].isalnum():
            score *= 0.70
            reasons.append("mid_word_boundary_hazard")

    # 3. Prompt Grounding
    is_exact_in_prompt = (phrase_text in prompt_text) or (phrase_text.strip() in prompt_text)
    prompt_words = set(re.findall(r"\b\w+\b", prompt_text.lower()))
    phrase_words = re.findall(r"\b\w+\b", phrase_text.lower())

    if is_exact_in_prompt:
        score *= 1.25
        reasons.append("prompt_grounded_exact")
    elif phrase_words:
        grounded_word_count = sum(1 for w in phrase_words if w in prompt_words)
        grounded_ratio = grounded_word_count / len(phrase_words)
        if grounded_ratio >= 0.67:
            score *= 1.10
            reasons.append("prompt_grounded_constituent")
        elif grounded_ratio == 0.0 and len(phrase_words) >= 2:
            score *= 0.85
            reasons.append("prompt_novel_all_words")

    # 4. Numeric Safety
    digits_in_phrase = re.findall(r"\d+", phrase_text)
    if digits_in_phrase:
        prompt_digits = set(re.findall(r"\d+", prompt_text))
        all_digits_grounded = all(d in prompt_digits for d in digits_in_phrase)
        if all_digits_grounded:
            score *= 0.90
            reasons.append("prompt_grounded_numeric")
        else:
            score *= 0.10
            reasons.append("hallucinated_ungrounded_numeric")

    # 5. Function Names & Code Identifiers
    fn_def_match = re.search(r"def\s+([a-zA-Z_]\w*)", phrase_text)
    fn_call_match = re.search(r"([a-zA-Z_]\w*)\s*\(", phrase_text)
    fn_name = None
    if fn_def_match:
        fn_name = fn_def_match.group(1)
    elif fn_call_match and len(fn_call_match.group(1)) > 1:
        fn_name = fn_call_match.group(1)

    if fn_name:
        if fn_name.lower() in prompt_text.lower():
            score *= 1.15
            reasons.append("prompt_grounded_function_name")
        else:
            score *= 0.15
            reasons.append("ungrounded_function_name_hazard")

    # 6. Syntax-Sensitive Code Delimiters & Unbalanced Brackets
    unbalanced = False
    for open_b, close_b in [("(", ")"), ("[", "]"), ("{", "}")]:
        if phrase_text.count(open_b) != phrase_text.count(close_b):
            unbalanced = True
            break
    if unbalanced:
        score *= 0.30
        reasons.append("unbalanced_bracket_syntax_hazard")

    phrase_clean = phrase_text.strip()
    if phrase_clean in CODE_SYNTAX_FRAGMENTS:
        score *= 0.40
        reasons.append("isolated_syntax_fragment")

    # 7. High-Frequency Grammatical Glue Reward
    if phrase_text in GRAMMATICAL_GLUE or (" " + phrase_clean) in GRAMMATICAL_GLUE:
        score *= 1.30
        reasons.append("high_frequency_grammatical_glue")

    # Clean structural indentation
    if phrase_text.startswith("\n    ") and not has_trailing_space:
        score *= 1.15
        reasons.append("clean_structural_indentation")

    final_score = max(0.01, min(1.0, score))
    return {
        "heuristic_safety_prior": round(final_score, 4),
        "is_safe_heuristic": final_score >= 0.65,
        "reasons": reasons,
        "has_trailing_space": has_trailing_space,
        "is_exact_in_prompt": is_exact_in_prompt,
        "has_digits": bool(digits_in_phrase),
    }


@dataclass
class ContinuationProbeRecord:
    context: str
    phrase: str
    category: str
    phrase_tokens: List[int]
    phrase_len: int
    kl_divergence: float
    top1_match: bool
    top5_overlap: float
    top10_overlap: float
    hidden_cosine_sim: float
    continuation_match: bool
    heuristic_safety_prior: float
    is_safe_heuristic: bool
    is_safe_empirical: bool  # Defined as kl < 2.0 and top1_match or top5_overlap >= 0.40


def load_historical_empirical_probes(
    probe_path: str = "experiments/checkpoints/continuation_equivalence_baseline.json",
) -> List[ContinuationProbeRecord]:
    """Loads historical empirical continuation probe records and annotates with heuristic safety."""
    if not os.path.exists(probe_path):
        return []

    with open(probe_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    probes = data.get("probes", [])
    records: List[ContinuationProbeRecord] = []

    for p in probes:
        ctx = p.get("context", "")
        phr = p.get("phrase", "")
        cat = p.get("category", "general")
        toks = p.get("phrase_tokens", [])
        kl = p.get("kl_divergence", 0.0)
        top1 = p.get("top1_match", False)
        top5 = p.get("top5_overlap", 0.0)
        top10 = p.get("top10_overlap", 0.0)
        cos = p.get("hidden_cosine_sim", 0.0)
        cont_match = p.get("continuation_match", False)

        # Evaluate heuristic safety prior on probe
        heur = compute_heuristic_safety_prior(
            phrase_text=phr,
            phrase_tokens=toks,
            prompt_text=ctx,
            prompt_tokens=[],
            domain=cat,
        )

        # Empirical ground-truth safety rule: low KL (< 2.0) and preserved distribution (top5 >= 0.40 or top1)
        is_safe_emp = (kl < 2.0) and (top1 or top5 >= 0.40)

        records.append(
            ContinuationProbeRecord(
                context=ctx,
                phrase=phr,
                category=cat,
                phrase_tokens=toks,
                phrase_len=len(toks),
                kl_divergence=kl,
                top1_match=top1,
                top5_overlap=top5,
                top10_overlap=top10,
                hidden_cosine_sim=cos,
                continuation_match=cont_match,
                heuristic_safety_prior=heur["heuristic_safety_prior"],
                is_safe_heuristic=heur["is_safe_heuristic"],
                is_safe_empirical=is_safe_emp,
            )
        )

    return records


def audit_heuristic_safety(
    records: Sequence[ContinuationProbeRecord],
) -> Dict[str, Any]:
    """Audits correlation, false-safe rate, false-unsafe rate between heuristic prior and empirical probes."""
    if not records:
        return {"error": "no_records"}

    heur_scores = np.array([r.heuristic_safety_prior for r in records])
    kl_values = np.array([r.kl_divergence for r in records])
    top1_matches = np.array([1.0 if r.top1_match else 0.0 for r in records])
    top5_overlaps = np.array([r.top5_overlap for r in records])
    emp_safe = np.array([1.0 if r.is_safe_empirical else 0.0 for r in records])
    heur_safe = np.array([1.0 if r.is_safe_heuristic else 0.0 for r in records])

    # Pearson correlation with negative KL (higher safety should mean lower KL)
    corr_kl = float(np.corrcoef(heur_scores, -kl_values)[0, 1]) if len(records) > 2 else 0.0
    corr_top1 = float(np.corrcoef(heur_scores, top1_matches)[0, 1]) if len(records) > 2 else 0.0
    corr_top5 = float(np.corrcoef(heur_scores, top5_overlaps)[0, 1]) if len(records) > 2 else 0.0

    # Confusion matrix
    # True Positive: Heuristic says safe, Empirical is safe
    # False Positive (False Safe): Heuristic says safe, Empirical is UNSAFE (Hazard!)
    # False Negative (False Unsafe): Heuristic says unsafe, Empirical is SAFE (Lost Opportunity)
    # True Negative: Heuristic says unsafe, Empirical is unsafe
    tp = int(np.sum((heur_safe == 1.0) & (emp_safe == 1.0)))
    fp = int(np.sum((heur_safe == 1.0) & (emp_safe == 0.0)))
    fn = int(np.sum((heur_safe == 0.0) & (emp_safe == 1.0)))
    tn = int(np.sum((heur_safe == 0.0) & (emp_safe == 0.0)))

    n_total = len(records)
    false_safe_rate = (fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    false_unsafe_rate = (fn / (tn + fn)) if (tn + fn) > 0 else 0.0

    # Domain / Category breakdown
    by_cat: Dict[str, Any] = {}
    cats = sorted(list(set(r.category for r in records)))
    for c in cats:
        cat_recs = [r for r in records if r.category == c]
        by_cat[c] = {
            "count": len(cat_recs),
            "mean_kl": round(float(np.mean([r.kl_divergence for r in cat_recs])), 3),
            "mean_heuristic_safety": round(float(np.mean([r.heuristic_safety_prior for r in cat_recs])), 3),
            "top1_agreement": round(float(np.mean([1.0 if r.top1_match else 0.0 for r in cat_recs])), 3),
            "empirical_safe_count": sum(1 for r in cat_recs if r.is_safe_empirical),
            "heuristic_safe_count": sum(1 for r in cat_recs if r.is_safe_heuristic),
        }

    return {
        "num_probes": n_total,
        "corr_heuristic_vs_neg_kl": round(corr_kl, 4),
        "corr_heuristic_vs_top1": round(corr_top1, 4),
        "corr_heuristic_vs_top5": round(corr_top5, 4),
        "confusion_matrix": {
            "true_safe": tp,
            "false_safe": fp,
            "false_unsafe": fn,
            "true_unsafe": tn,
        },
        "false_safe_rate": round(false_safe_rate, 4),
        "false_unsafe_rate": round(false_unsafe_rate, 4),
        "category_breakdown": by_cat,
        "findings": [
            f"Heuristic Safety Prior correlates at r={corr_kl:.3f} with lower continuation KL.",
            f"False-safe rate is {false_safe_rate*100:.1f}% (heuristic deemed safe but empirical probe diverged).",
            f"False-unsafe rate is {false_unsafe_rate*100:.1f}% (heuristic penalized but empirical probe was stable).",
        ],
    }
