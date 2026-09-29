"""Offline causal-attribution accounting for committed Phi Stage 1 records.

This module reads only the supplied DEV attribution records. It never opens
the canonical dataset (which also contains FINAL) and never runs a model.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from src.evaluation.offline_segmenter import segment_tokens_dp
from src.evaluation.oracle_v2 import OracleV2


VOCAB_SIZE = 32011
HYPER_START_ID = 32011
K = 32
DISABLED_TOKEN_IDS = frozenset({0, 1, 2, *range(32000, 32011)})
CONDITIONS = ("A_vanilla", "B_h_disabled", "C_oracle", "D_real_predictor")


def codebook_from_record(record: Mapping[str, Any]) -> List[Tuple[int, ...]]:
    """Recover the exact phrase list serialized by the Stage 1 live runner."""
    phrases = [tuple(int(token) for token in phrase) for phrase in record.get("selected_h_slots", [])]
    if len(phrases) > K:
        raise ValueError(f"{record.get('prompt_id')}: codebook has {len(phrases)} slots, K={K}")
    if len(set(phrases)) != len(phrases):
        raise ValueError(f"{record.get('prompt_id')}: duplicate phrase in supplied codebook")
    for phrase in phrases:
        if not 2 <= len(phrase) <= 4:
            raise ValueError(f"{record.get('prompt_id')}: phrase length {len(phrase)} is outside 2..4")
        if any(token < 0 or token >= VOCAB_SIZE or token in DISABLED_TOKEN_IDS for token in phrase):
            raise ValueError(f"{record.get('prompt_id')}: phrase contains an invalid or disabled token ID")
    return phrases


def count_phrase_occurrences(
    tokens: Sequence[int], phrases: Iterable[Sequence[int]]
) -> Dict[str, Any]:
    """Count overlapping occurrences, matching ordinary n-gram occurrence semantics."""
    counts: Dict[Tuple[int, ...], int] = {}
    seq = tuple(int(token) for token in tokens)
    for raw_phrase in phrases:
        phrase = tuple(int(token) for token in raw_phrase)
        width = len(phrase)
        counts[phrase] = sum(
            1 for start in range(max(0, len(seq) - width + 1))
            if seq[start : start + width] == phrase
        )
    occurring = {phrase: count for phrase, count in counts.items() if count}
    return {
        "codebook_phrase_types": len(counts),
        "phrase_types_that_occur": len(occurring),
        "occurrence_count_overlapping": sum(occurring.values()),
        "frequency_weighted_potential_savings": sum(
            count * (len(phrase) - 1) for phrase, count in occurring.items()
        ),
        "phrase_length_distribution_that_occurs": _length_histogram(occurring),
    }


def codebook_opportunity(tokens: Sequence[int], phrases: Sequence[Sequence[int]]) -> Dict[str, Any]:
    """Measure exact DP opportunity using one fixed, supplied codebook."""
    normalized = [tuple(int(token) for token in phrase) for phrase in phrases]
    _validate_phrases(normalized)
    base_count = len(tokens)
    compressed_count, tiles, dp = segment_tokens_dp(list(tokens), set(normalized))
    phrase_tiles = [tuple(tile) for tile in tiles if len(tile) > 1]
    occurrence_stats = count_phrase_occurrences(tokens, normalized)
    return {
        "base_token_count": base_count,
        "oracle_compressed_step_count": compressed_count,
        "tokens_saved": base_count - compressed_count,
        "compression_pct": dp.get("compression_pct", 0.0),
        "codebook_size": len(normalized),
        "codebook_phrase_length_distribution": _length_histogram(normalized),
        "supplied_phrases_that_occur": occurrence_stats["phrase_types_that_occur"],
        "supplied_phrase_occurrences_overlapping": occurrence_stats["occurrence_count_overlapping"],
        "optimal_h_substitutions": dp.get("hypertoken_emissions", len(phrase_tiles)),
        "optimal_tiling_phrase_length_distribution": _length_histogram(phrase_tiles),
        "occurrence_details": occurrence_stats,
    }


def oracle_ceiling(
    tokens: Sequence[int], *, k: int = K, min_length: int = 2, max_length: int = 3
) -> Dict[str, Any]:
    """Run Oracle V2 with the live base-vocabulary and disabled-ID constraints."""
    phrases, oracle_stats = OracleV2.compute_codebook(
        list(tokens),
        k=k,
        min_length=min_length,
        max_length=max_length,
        beam_width=4,
        candidate_limit=80,
        disabled_token_ids=set(DISABLED_TOKEN_IDS),
        valid_token_min=0,
        valid_token_max_exclusive=VOCAB_SIZE,
    )
    selected = sorted(phrases, key=lambda phrase: (len(phrase), phrase))
    opportunity = codebook_opportunity(tokens, selected)
    if len(selected) > k:
        raise AssertionError("Oracle V2 exceeded the requested K")
    return {
        **opportunity,
        "oracle_v2_candidate_limit": 80,
        "oracle_v2_beam_width": 4,
        "oracle_v2_reported_tokens_saved": oracle_stats.get("tokens_saved", 0),
        "oracle_v2_codebook_size": oracle_stats.get("codebook_size", len(selected)),
        "selected_phrases": [list(phrase) for phrase in selected],
    }


def live_realization(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and measure realized H savings from one serialized live record."""
    phrases = codebook_from_record(record)
    generated_ids = record.get("generated_token_ids", [])
    expanded_ids = record.get("expanded_token_ids", [])
    if int(record.get("generated_token_count", -1)) != len(generated_ids):
        raise ValueError(f"{record.get('prompt_id')}: generated token count does not match IDs")
    if int(record.get("expanded_token_count", -1)) != len(expanded_ids):
        raise ValueError(f"{record.get('prompt_id')}: expanded token count does not match IDs")
    if int(record.get("transformer_decode_calls", -1)) != len(generated_ids):
        raise ValueError(f"{record.get('prompt_id')}: Stage 1 decode calls are not len(generated_token_ids)")

    actual_savings = len(expanded_ids) - len(generated_ids)
    emissions = record.get("h_emissions", [])
    indexed = {HYPER_START_ID + index: phrase for index, phrase in enumerate(phrases)}
    expansion_savings = 0
    realized_lengths: Counter[int] = Counter()
    for emission in emissions:
        hyper_id = int(emission["id"])
        phrase = indexed.get(hyper_id)
        serialized_phrase = tuple(int(token) for token in emission.get("subtokens", []))
        if phrase is None or phrase != serialized_phrase:
            raise ValueError(f"{record.get('prompt_id')}: H emission does not map to its supplied slot")
        expansion_savings += len(phrase) - 1
        realized_lengths[len(phrase)] += 1
    if expansion_savings != actual_savings:
        raise ValueError(
            f"{record.get('prompt_id')}: H-span savings {expansion_savings} != expanded/generated delta {actual_savings}"
        )
    expanded_count = len(expanded_ids)
    return {
        "base_equivalent_tokens": expanded_count,
        "model_decode_calls": len(generated_ids),
        "h_emissions": len(emissions),
        "realized_tokens_saved": actual_savings,
        "realized_compression_pct": 100.0 * actual_savings / expanded_count if expanded_count else 0.0,
        "mean_h_span_length": (
            sum(length * count for length, count in realized_lengths.items()) / len(emissions)
            if emissions else 0.0
        ),
        "realized_h_phrase_length_distribution": dict(sorted(realized_lengths.items())),
    }


def analyze_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Build per-prompt and aggregate attribution metrics from DEV-only records."""
    by_prompt: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for record in records:
        if str(record.get("split", "")).upper() != "DEV":
            raise ValueError("Attribution reanalysis accepts DEV records only")
        condition = str(record.get("condition", ""))
        if condition not in CONDITIONS:
            raise ValueError(f"Unexpected Stage 1 condition {condition!r}")
        prompt_id = str(record.get("prompt_id", ""))
        if not prompt_id or condition in by_prompt.setdefault(prompt_id, {}):
            raise ValueError(f"Missing or duplicate record for {prompt_id!r}/{condition}")
        by_prompt[prompt_id][condition] = record
    if not by_prompt:
        raise ValueError("No DEV records supplied")
    incomplete = {pid: sorted(set(CONDITIONS) - set(row)) for pid, row in by_prompt.items() if set(row) != set(CONDITIONS)}
    if incomplete:
        raise ValueError(f"Incomplete Stage 1 condition matrix: {incomplete}")

    prompt_rows: List[Dict[str, Any]] = []
    for prompt_id in sorted(by_prompt):
        row = by_prompt[prompt_id]
        a = row["A_vanilla"]
        b = row["B_h_disabled"]
        c = row["C_oracle"]
        d = row["D_real_predictor"]
        a_tokens = a.get("expanded_token_ids", [])
        b_tokens = b.get("expanded_token_ids", [])
        c_tokens = c.get("expanded_token_ids", [])
        d_tokens = d.get("expanded_token_ids", [])
        a_codebook = codebook_from_record(c)
        c_codebook = codebook_from_record(c)
        d_codebook = codebook_from_record(d)
        # C's ordered slot assignment is recoverable from the record. Verify it
        # against the original A-continuation frequency builder's deterministic rule.
        derived_from_a = _derive_stage1_c_codebook(a_tokens)
        c_reconstruction_match = c_codebook == derived_from_a

        a_ceiling_23 = oracle_ceiling(a_tokens, min_length=2, max_length=3)
        a_ceiling_24 = oracle_ceiling(a_tokens, min_length=2, max_length=4)
        b_ceiling_23 = oracle_ceiling(b_tokens, min_length=2, max_length=3)
        b_ceiling_24 = oracle_ceiling(b_tokens, min_length=2, max_length=4)
        c_ceiling_23 = oracle_ceiling(c_tokens, min_length=2, max_length=3)
        c_ceiling_24 = oracle_ceiling(c_tokens, min_length=2, max_length=4)
        d_ceiling_23 = oracle_ceiling(d_tokens, min_length=2, max_length=3)
        d_ceiling_24 = oracle_ceiling(d_tokens, min_length=2, max_length=4)
        c_opportunity = codebook_opportunity(c_tokens, c_codebook)
        c_realization = live_realization(c)
        d_opportunity = codebook_opportunity(d_tokens, d_codebook)
        d_realization = live_realization(d)
        prompt_row = {
            "prompt_id": prompt_id,
            "domain": a.get("domain"),
            "base_token_counts": {
                "A_vanilla": len(a_tokens),
                "B_h_disabled": len(b_tokens),
                "C_oracle_codebook_live": len(c_tokens),
                "D_real_predictor": len(d_tokens),
            },
            "oracle_a_ceiling": {"k32_len2_3": a_ceiling_23, "k32_len2_4": a_ceiling_24},
            "oracle_b_ceiling": {"k32_len2_3": b_ceiling_23, "k32_len2_4": b_ceiling_24},
            "oracle_c_ceiling": {"k32_len2_3": c_ceiling_23, "k32_len2_4": c_ceiling_24},
            "oracle_d_ceiling": {"k32_len2_3": d_ceiling_23, "k32_len2_4": d_ceiling_24},
            "a_derived_c_codebook_phrases": [list(phrase) for phrase in derived_from_a],
            "c_exact_supplied_codebook_phrases": [list(phrase) for phrase in c_codebook],
            "d_exact_supplied_codebook_phrases": [list(phrase) for phrase in d_codebook],
            "a_derived_c_codebook_cross_output": {
                "A_vanilla": codebook_opportunity(a_tokens, a_codebook),
                "B_h_disabled": codebook_opportunity(b_tokens, a_codebook),
                "C_oracle_codebook_live": codebook_opportunity(c_tokens, a_codebook),
            },
            "c_codebook_reconstruction_matches_stage1_builder_on_a": c_reconstruction_match,
            "c_exact_supplied_codebook_opportunity": c_opportunity,
            "c_live_realization": c_realization,
            "c_opportunity_to_realization_ratio": _ratio(c_realization["realized_tokens_saved"], c_opportunity["tokens_saved"]),
            "d_exact_supplied_codebook_opportunity": d_opportunity,
            "d_live_realization": d_realization,
            "d_opportunity_to_realization_ratio": _ratio(d_realization["realized_tokens_saved"], d_opportunity["tokens_saved"]),
            "quality_pass": {condition: bool(row[condition].get("quality_gate_pass")) for condition in CONDITIONS},
        }
        prompt_rows.append(prompt_row)

    condition_records = {condition: [row[condition] for row in by_prompt.values()] for condition in CONDITIONS}
    pairwise_quality = _paired_quality_summary(by_prompt)
    aggregate = {
        "oracle_a_ceiling": {
            "k32_len2_3": _aggregate_opportunities([r["oracle_a_ceiling"]["k32_len2_3"] for r in prompt_rows]),
            "k32_len2_4": _aggregate_opportunities([r["oracle_a_ceiling"]["k32_len2_4"] for r in prompt_rows]),
        },
        "oracle_b_ceiling": {
            "k32_len2_3": _aggregate_opportunities([r["oracle_b_ceiling"]["k32_len2_3"] for r in prompt_rows]),
            "k32_len2_4": _aggregate_opportunities([r["oracle_b_ceiling"]["k32_len2_4"] for r in prompt_rows]),
        },
        "oracle_c_ceiling": {
            "k32_len2_3": _aggregate_opportunities([r["oracle_c_ceiling"]["k32_len2_3"] for r in prompt_rows]),
            "k32_len2_4": _aggregate_opportunities([r["oracle_c_ceiling"]["k32_len2_4"] for r in prompt_rows]),
        },
        "oracle_d_ceiling": {
            "k32_len2_3": _aggregate_opportunities([r["oracle_d_ceiling"]["k32_len2_3"] for r in prompt_rows]),
            "k32_len2_4": _aggregate_opportunities([r["oracle_d_ceiling"]["k32_len2_4"] for r in prompt_rows]),
        },
        "a_derived_c_codebook_cross_output": {
            condition: _aggregate_opportunities([r["a_derived_c_codebook_cross_output"][condition] for r in prompt_rows])
            for condition in ("A_vanilla", "B_h_disabled", "C_oracle_codebook_live")
        },
        "c_exact_supplied_codebook_opportunity": _aggregate_opportunities(
            [r["c_exact_supplied_codebook_opportunity"] for r in prompt_rows]
        ),
        "c_live_realization": _aggregate_realization([r["c_live_realization"] for r in prompt_rows]),
        "c_opportunity_to_realization_ratio": _ratio(
            sum(r["c_live_realization"]["realized_tokens_saved"] for r in prompt_rows),
            sum(r["c_exact_supplied_codebook_opportunity"]["tokens_saved"] for r in prompt_rows),
        ),
        "d_exact_supplied_codebook_opportunity": _aggregate_opportunities(
            [r["d_exact_supplied_codebook_opportunity"] for r in prompt_rows]
        ),
        "d_live_realization": _aggregate_realization([r["d_live_realization"] for r in prompt_rows]),
        "d_opportunity_to_realization_ratio": _ratio(
            sum(r["d_live_realization"]["realized_tokens_saved"] for r in prompt_rows),
            sum(r["d_exact_supplied_codebook_opportunity"]["tokens_saved"] for r in prompt_rows),
        ),
        "live_condition_quality": {
            condition: _aggregate_quality(condition_records[condition]) for condition in CONDITIONS
        },
        "paired_quality_comparisons": pairwise_quality,
    }
    if not all(row["c_codebook_reconstruction_matches_stage1_builder_on_a"] for row in prompt_rows):
        raise ValueError("C supplied slot list did not reproduce from the A records")
    return {
        "schema": "phi_attribution_offline_reanalysis_v1",
        "status_labels": ["MEASURED", "INFERENCE", "HYPOTHESIS", "NOT TESTED"],
        "split": "DEV",
        "prompt_count": len(prompt_rows),
        "condition_record_counts": {condition: len(condition_records[condition]) for condition in CONDITIONS},
        "methodology": {
            "source_records": "docs/stage1_attribution_records.jsonl",
            "stage1_source_commit": "8af6589ba0711018637c8ce41f8584642258cc49",
            "stage1_current_raw_condition_names": list(CONDITIONS),
            "reanalysis_never_opens_canonical_dataset": True,
            "final_accessed": False,
            "oracle_v2": {
                "k": K,
                "candidate_limit": 80,
                "beam_width": 4,
                "disabled_token_ids": sorted(DISABLED_TOKEN_IDS),
                "valid_token_ids": [0, VOCAB_SIZE - 1],
                "max_phrase_lengths_reported": [3, 4],
            },
        },
        "aggregate": aggregate,
        "per_prompt": prompt_rows,
        "not_tested": [
            "A versus B0 (wrapper with Vanilla weights)",
            "CF FORCED ORACLE live continuation/state test",
            "direct base-logit Vanilla versus B0 versus B1 fidelity metrics",
            "TRAIN continuations generated by B1 and P-VANILLA versus P-LORA target-source comparison",
            "CUDA matched-path timing and VRAM decomposition",
        ],
    }


def _derive_stage1_c_codebook(tokens: Sequence[int]) -> List[Tuple[int, ...]]:
    counts: Counter[Tuple[int, ...]] = Counter()
    for length in range(2, min(4, len(tokens)) + 1):
        for start in range(len(tokens) - length + 1):
            phrase = tuple(int(token) for token in tokens[start : start + length])
            if all(0 <= token < VOCAB_SIZE and token not in DISABLED_TOKEN_IDS for token in phrase):
                counts[phrase] += 1
    ordered = sorted(
        counts,
        key=lambda phrase: (counts[phrase] * (len(phrase) - 1), counts[phrase], -len(phrase)),
        reverse=True,
    )
    return ordered[:K]


def _validate_phrases(phrases: Sequence[Tuple[int, ...]]) -> None:
    if len(phrases) > K:
        raise ValueError(f"Codebook size {len(phrases)} exceeds K={K}")
    for phrase in phrases:
        if not 2 <= len(phrase) <= 4:
            raise ValueError(f"Phrase length {len(phrase)} is outside 2..4")
        if any(token < 0 or token >= VOCAB_SIZE or token in DISABLED_TOKEN_IDS for token in phrase):
            raise ValueError("Codebook contains invalid or disabled token IDs")


def _length_histogram(phrases: Iterable[Sequence[int]]) -> Dict[str, int]:
    counts = Counter(str(len(phrase)) for phrase in phrases)
    return {length: counts[length] for length in sorted(counts, key=int)}


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _aggregate_opportunities(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    base = sum(int(row["base_token_count"]) for row in rows)
    compressed = sum(int(row["oracle_compressed_step_count"]) for row in rows)
    lengths: Counter[str] = Counter()
    tiling_lengths: Counter[str] = Counter()
    codebook_total = 0
    phrase_types = 0
    occurrences = 0
    substitutions = 0
    for row in rows:
        codebook_total += int(row["codebook_size"])
        phrase_types += int(row["supplied_phrases_that_occur"])
        occurrences += int(row["supplied_phrase_occurrences_overlapping"])
        substitutions += int(row["optimal_h_substitutions"])
        lengths.update(row["codebook_phrase_length_distribution"])
        tiling_lengths.update(row["optimal_tiling_phrase_length_distribution"])
    return {
        "prompt_count": len(rows),
        "base_token_count": base,
        "oracle_compressed_step_count": compressed,
        "tokens_saved": base - compressed,
        "compression_pct": 100.0 * (base - compressed) / base if base else 0.0,
        "codebook_size_total": codebook_total,
        "mean_codebook_size": codebook_total / len(rows) if rows else 0.0,
        "codebook_phrase_length_distribution_total": dict(sorted(lengths.items(), key=lambda kv: int(kv[0]))),
        "supplied_phrases_that_occur_total": phrase_types,
        "supplied_phrase_occurrences_overlapping_total": occurrences,
        "optimal_h_substitutions_total": substitutions,
        "optimal_tiling_phrase_length_distribution_total": dict(sorted(tiling_lengths.items(), key=lambda kv: int(kv[0]))),
    }


def _aggregate_realization(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    base = sum(int(row["base_equivalent_tokens"]) for row in rows)
    calls = sum(int(row["model_decode_calls"]) for row in rows)
    saved = sum(int(row["realized_tokens_saved"]) for row in rows)
    emissions = sum(int(row["h_emissions"]) for row in rows)
    lengths: Counter[str] = Counter()
    for row in rows:
        lengths.update({str(k): int(v) for k, v in row["realized_h_phrase_length_distribution"].items()})
    return {
        "prompt_count": len(rows),
        "base_equivalent_tokens": base,
        "model_decode_calls": calls,
        "h_emissions": emissions,
        "realized_tokens_saved": saved,
        "realized_compression_pct": 100.0 * saved / base if base else 0.0,
        "mean_h_span_length": sum(row["mean_h_span_length"] * row["h_emissions"] for row in rows) / emissions if emissions else 0.0,
        "realized_h_phrase_length_distribution_total": dict(sorted(lengths.items(), key=lambda kv: int(kv[0]))),
    }


def _aggregate_quality(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    domain_counts: Dict[str, Dict[str, int]] = {}
    for record in records:
        domain = str(record.get("domain", "unknown"))
        stats = domain_counts.setdefault(domain, {"prompts": 0, "passes": 0})
        stats["prompts"] += 1
        stats["passes"] += int(bool(record.get("quality_gate_pass")))
    return {
        "prompts": len(records),
        "passes": sum(int(bool(record.get("quality_gate_pass"))) for record in records),
        "quality_rate": sum(int(bool(record.get("quality_gate_pass"))) for record in records) / len(records) if records else 0.0,
        "by_domain": {
            domain: {**values, "rate": values["passes"] / values["prompts"] if values["prompts"] else 0.0}
            for domain, values in sorted(domain_counts.items())
        },
    }


def _paired_quality_summary(by_prompt: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for left, right in (("A_vanilla", "B_h_disabled"), ("B_h_disabled", "C_oracle"), ("C_oracle", "D_real_predictor")):
        transitions = Counter()
        for row in by_prompt.values():
            transitions[f"{int(bool(row[left].get('quality_gate_pass')))}->{int(bool(row[right].get('quality_gate_pass')))}"] += 1
        result[f"{left}_to_{right}"] = {
            "paired_prompts": len(by_prompt),
            "pass_to_pass": transitions["1->1"],
            "pass_to_fail": transitions["1->0"],
            "fail_to_pass": transitions["0->1"],
            "fail_to_fail": transitions["0->0"],
            "aggregate_quality_rate_delta_pp": 100.0 * (
                sum(bool(row[right].get("quality_gate_pass")) for row in by_prompt.values())
                - sum(bool(row[left].get("quality_gate_pass")) for row in by_prompt.values())
            ) / max(len(by_prompt), 1),
        }
    return result
