"""Lightweight, generation-local runtime metrics for live quality benchmarks.

This module intentionally does not profile tensors or synchronize devices.
Callers add measured timing/event fields while generating, then use these pure
helpers to derive comparable rates and paired ratios.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


RUNTIME_DIAGNOSTICS_SCHEMA = "phi_runtime_diagnostics_v1"
_FINAL_ANSWER_MARKER = re.compile(
    r"(?:####\s*(-?\d[\d,]*(?:\.\d+)?)|\\boxed\{\s*(-?\d[\d,]*(?:\.\d+)?)\s*\})"
)


def _ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)


def make_hypertoken_event(
    *,
    position: int,
    token_id: int,
    phrase_token_ids: Sequence[int],
    phrase_text: str,
) -> dict[str, Any]:
    """Build a compact per-emission event; never include logits/tensors."""
    token_ids = [int(token_id) for token_id in phrase_token_ids]
    return {
        "position": int(position),
        "token_id": int(token_id),
        "phrase_token_ids": token_ids,
        "phrase_length": len(token_ids),
        "phrase_text": phrase_text,
        "base_positions_saved": max(0, len(token_ids) - 1),
    }


def first_deterministic_answer_position(prefix_texts: Sequence[str]) -> int | None:
    """Find the first prefix with an explicit GSM-style final-answer marker."""
    for position, prefix in enumerate(prefix_texts):
        if _FINAL_ANSWER_MARKER.search(prefix):
            return position
    return None


def first_repeated_trigram_position(token_ids: Sequence[int]) -> int | None:
    """Return the start of the first repeated token trigram, if any."""
    first_seen: dict[tuple[int, int, int], int] = {}
    for start in range(max(0, len(token_ids) - 2)):
        trigram = tuple(int(token) for token in token_ids[start : start + 3])
        if trigram in first_seen:
            return start
        first_seen[trigram] = start
    return None


def add_runtime_derived_metrics(
    record: Mapping[str, Any],
    *,
    vanilla_record: Mapping[str, Any] | None = None,
    quality_pass: bool | None = None,
) -> dict[str, Any]:
    """Return a copy with standard per-prompt derived values attached.

    Missing timings stay null. Ratios are not clamped; negative decode savings
    or slower-than-Vanilla behavior must remain visible.
    """
    out = dict(record)
    out["runtime_diagnostics_schema"] = RUNTIME_DIAGNOSTICS_SCHEMA
    steps = out.get("transformer_decode_iterations", out.get("decode_steps"))
    expanded = out.get("expanded_base_equivalent_output_tokens", out.get("expanded_output_tokens"))
    decode_wall = out.get("decode_wall_time_s", out.get("decode_time_s"))
    out["decode_wall_time_s"] = decode_wall
    out["total_request_wall_time_s"] = out.get("total_request_wall_time_s", out.get("wall_time_s"))
    out["transformer_decode_iterations"] = steps
    out["expanded_base_equivalent_output_tokens"] = expanded

    if expanded is not None and steps is not None:
        out["net_decode_steps_saved"] = int(expanded) - int(steps)
        out["raw_decode_reduction"] = _ratio(int(expanded) - int(steps), expanded)
    else:
        out["net_decode_steps_saved"] = None
        out["raw_decode_reduction"] = None

    out["transformer_steps_per_second"] = _ratio(steps, decode_wall)
    out["expanded_tokens_per_second"] = _ratio(expanded, decode_wall)
    out["wall_time_per_decode_step_s"] = _ratio(decode_wall, steps)
    out["wall_time_per_expanded_token_s"] = _ratio(decode_wall, expanded)

    events = out.get("hypertoken_events", out.get("hypertokens_emitted", [])) or []
    represented = sum(
        len(event.get("phrase_token_ids", event.get("subtokens", [])) or [])
        for event in events
    )
    out["hypertokens_emitted_count"] = len(events)
    out["base_tokens_represented_by_hypertokens"] = represented

    out_text = out.get("output_text") or ""
    out["generated_word_count"] = len(out_text.split())
    out["generated_character_count"] = len(out_text)

    answer_position = out.get("first_answer_decode_position")
    if answer_position is None:
        out["post_answer_decode_iterations"] = None
        out["post_answer_output_tokens"] = None
        out["hypertokens_before_answer"] = None
        out["hypertokens_after_answer"] = None
    else:
        out["post_answer_decode_iterations"] = max(0, int(steps or 0) - int(answer_position) - 1)
        answer_expanded_position = out.get("first_answer_expanded_position")
        out["post_answer_output_tokens"] = (
            max(0, int(expanded or 0) - int(answer_expanded_position) - 1)
            if answer_expanded_position is not None
            else None
        )
        out["hypertokens_before_answer"] = sum(int(event.get("position", event.get("pos", -1))) <= int(answer_position) for event in events)
        out["hypertokens_after_answer"] = sum(int(event.get("position", event.get("pos", -1))) > int(answer_position) for event in events)

    if events and steps is not None:
        out["hypertoken_density"] = len(events) / max(int(steps), 1)
        last_position = max(int(event.get("position", event.get("pos", -1))) for event in events)
        out["decode_iterations_after_last_hypertoken"] = max(0, int(steps) - last_position - 1)
    else:
        out["hypertoken_density"] = 0.0 if steps is not None else None
        out["decode_iterations_after_last_hypertoken"] = None

    step_samples = out.get("decode_step_intervals_s") or []
    if step_samples:
        ordered = sorted(float(value) for value in step_samples)
        out["mean_decode_step_time_s"] = sum(ordered) / len(ordered)
        out["median_decode_step_time_s"] = _percentile(ordered, 0.5)
        out["p95_decode_step_time_s"] = _percentile(ordered, 0.95)
    else:
        out.setdefault("mean_decode_step_time_s", None)
        out.setdefault("median_decode_step_time_s", None)
        out.setdefault("p95_decode_step_time_s", None)

    repeat_position = out.get("first_repeated_trigram_position")
    if repeat_position is not None:
        out["repetition_begins_within_16_steps_after_hypertoken"] = any(
            0 <= int(repeat_position) - int(event.get("position", event.get("pos", -1))) <= 16
            for event in events
        )
    else:
        out["repetition_begins_within_16_steps_after_hypertoken"] = None

    if vanilla_record is not None:
        out["output_length_ratio_vs_vanilla"] = _ratio(
            expanded,
            vanilla_record.get("expanded_base_equivalent_output_tokens", vanilla_record.get("expanded_output_tokens")),
        )
        out["latency_ratio_vs_vanilla"] = _ratio(
            out.get("total_request_wall_time_s", out.get("wall_time_s")),
            vanilla_record.get("total_request_wall_time_s", vanilla_record.get("wall_time_s")),
        )
    else:
        out["output_length_ratio_vs_vanilla"] = None
        out["latency_ratio_vs_vanilla"] = None

    if quality_pass is True and expanded is not None and steps is not None:
        out["quality_preserved_decode_steps_saved"] = int(expanded) - int(steps)
        out["quality_preserved_expanded_tokens"] = int(expanded)
    elif quality_pass is False:
        out["quality_preserved_decode_steps_saved"] = 0
        out["quality_preserved_expanded_tokens"] = 0
    else:
        out["quality_preserved_decode_steps_saved"] = None
        out["quality_preserved_expanded_tokens"] = None
    return out


def aggregate_runtime_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute pooled condition-level rates and preserve paired null counts."""
    decode_steps = sum(int(r.get("transformer_decode_iterations") or 0) for r in records)
    expanded = sum(int(r.get("expanded_base_equivalent_output_tokens") or 0) for r in records)
    decode_wall = sum(float(r.get("decode_wall_time_s") or 0.0) for r in records)
    request_wall = sum(float(r.get("total_request_wall_time_s", r.get("wall_time_s")) or 0.0) for r in records)
    q_saved = sum(int(r.get("quality_preserved_decode_steps_saved") or 0) for r in records)
    q_expanded = sum(int(r.get("quality_preserved_expanded_tokens") or 0) for r in records)
    all_step_samples = sorted(
        float(value)
        for record in records
        for value in (record.get("decode_step_intervals_s") or [])
    )
    return {
        "prompt_count": len(records),
        "transformer_decode_iterations": decode_steps,
        "expanded_base_equivalent_output_tokens": expanded,
        "decode_wall_time_s": decode_wall,
        "total_request_wall_time_s": request_wall,
        "total_hypertokens_emitted": sum(int(r.get("hypertokens_emitted_count") or 0) for r in records),
        "mean_hypertoken_density": _mean_present(records, "hypertoken_density"),
        "mean_decode_iterations_after_last_hypertoken": _mean_present(records, "decode_iterations_after_last_hypertoken"),
        "eos_count": sum(bool(r.get("eos_reached")) for r in records),
        "truncation_count": sum(bool(r.get("truncated")) for r in records),
        "severe_repetition_count": sum(bool(r.get("severe_repetition_detected")) for r in records),
        "quality_preserved_decode_steps_saved": q_saved,
        "quality_preserved_expanded_tokens": q_expanded,
        "raw_decode_reduction": _ratio(expanded - decode_steps, expanded),
        "quality_preserved_decode_reduction": _ratio(q_saved, q_expanded),
        "transformer_steps_per_second": _ratio(decode_steps, decode_wall),
        "expanded_tokens_per_second": _ratio(expanded, decode_wall),
        "wall_time_per_decode_step_s": _ratio(decode_wall, decode_steps),
        "mean_decode_step_time_s": (
            sum(all_step_samples) / len(all_step_samples) if all_step_samples else None
        ),
        "median_decode_step_time_s": _percentile(all_step_samples, 0.5) if all_step_samples else None,
        "p95_decode_step_time_s": _percentile(all_step_samples, 0.95) if all_step_samples else None,
        "paired_output_length_ratio_mean": _mean_present(records, "output_length_ratio_vs_vanilla"),
        "paired_latency_ratio_mean": _mean_present(records, "latency_ratio_vs_vanilla"),
        "mean_post_answer_decode_iterations": _mean_present(records, "post_answer_decode_iterations"),
        "paired_prompt_count": sum(r.get("output_length_ratio_vs_vanilla") is not None for r in records),
        "answer_tail_prompt_count": sum(r.get("post_answer_decode_iterations") is not None for r in records),
    }


def _mean_present(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in records if r.get(key) is not None]
    return sum(values) / len(values) if values else None


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction
