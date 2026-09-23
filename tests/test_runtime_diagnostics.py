from experiments.runtime_diagnostics import (
    add_runtime_derived_metrics,
    aggregate_runtime_metrics,
    first_deterministic_answer_position,
    first_repeated_trigram_position,
    make_hypertoken_event,
)
import pytest


def test_hypertoken_event_is_compact_and_counts_saved_base_positions():
    event = make_hypertoken_event(
        position=3,
        token_id=32012,
        phrase_token_ids=[10, 11, 12],
        phrase_text=" phrase",
    )
    assert event == {
        "position": 3,
        "token_id": 32012,
        "phrase_token_ids": [10, 11, 12],
        "phrase_length": 3,
        "phrase_text": " phrase",
        "base_positions_saved": 2,
    }


def test_answer_tail_heuristic_requires_explicit_final_marker():
    assert first_deterministic_answer_position(
        ["work out 5 + 2", "answer is 7", "#### 7"]
    ) == 2
    assert first_deterministic_answer_position(["the answer is 7"]) is None


def test_first_repeated_trigram_position():
    assert first_repeated_trigram_position([1, 2, 3, 8, 1, 2, 3]) == 4
    assert first_repeated_trigram_position([1, 2, 3, 4]) is None


def test_derived_metrics_preserve_negative_savings_and_pair_ratios():
    vanilla = {"expanded_output_tokens": 8, "wall_time_s": 4.0}
    predictive = add_runtime_derived_metrics(
        {
            "decode_steps": 10,
            "expanded_output_tokens": 8,
            "decode_time_s": 5.0,
            "wall_time_s": 6.0,
            "output_text": "one two",
            "hypertokens_emitted": [{"pos": 2, "subtokens": [4, 5, 6]}],
            "first_answer_decode_position": 4,
            "first_repeated_trigram_position": 6,
        },
        vanilla_record=vanilla,
        quality_pass=True,
    )

    assert predictive["net_decode_steps_saved"] == -2
    assert predictive["raw_decode_reduction"] == -0.25
    assert predictive["wall_time_per_decode_step_s"] == 0.5
    assert predictive["transformer_steps_per_second"] == 2.0
    assert predictive["expanded_tokens_per_second"] == 1.6
    assert predictive["output_length_ratio_vs_vanilla"] == 1.0
    assert predictive["latency_ratio_vs_vanilla"] == 1.5
    assert predictive["base_tokens_represented_by_hypertokens"] == 3
    assert predictive["post_answer_decode_iterations"] == 5
    assert predictive["quality_preserved_decode_steps_saved"] == -2
    assert predictive["repetition_begins_within_16_steps_after_hypertoken"] is True


def test_aggregate_uses_pooled_savings_and_keeps_missing_pairs_null():
    rows = [
        add_runtime_derived_metrics(
            {
                "decode_steps": 3,
                "expanded_output_tokens": 4,
                "decode_time_s": 2.0,
                "wall_time_s": 2.5,
                "decode_step_intervals_s": [0.2, 0.4],
            },
            quality_pass=True,
        ),
        add_runtime_derived_metrics(
            {
                "decode_steps": 5,
                "expanded_output_tokens": 5,
                "decode_time_s": 3.0,
                "wall_time_s": 3.5,
            },
            quality_pass=False,
        ),
    ]
    summary = aggregate_runtime_metrics(rows)
    assert summary["raw_decode_reduction"] == 1 / 9
    assert summary["quality_preserved_decode_reduction"] == 0.25
    assert summary["transformer_steps_per_second"] == 8 / 5
    assert summary["paired_output_length_ratio_mean"] is None
    assert summary["paired_prompt_count"] == 0
    assert summary["median_decode_step_time_s"] == pytest.approx(0.3)
