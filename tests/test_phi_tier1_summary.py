import json

import pytest

from experiments.run_phi_tier1 import DEFAULT_PREDICTOR, _write_tier1_summary


def test_tier1_defaults_to_the_canonical_oracle_guided_predictor():
    assert DEFAULT_PREDICTOR.name == "oracle_guided_predictor.pkl"


def _pair(prompt_id, codebook_sha="same"):
    common = {
        "prompt_id": prompt_id,
        "domain": "code",
        "codebook_sha256": codebook_sha,
        "decode_steps": 5,
        "expanded_output_tokens": 7,
        "wall_time_s": 1.5,
        "problem_pass": True,
        "syntax_valid": True,
    }
    raw = {
        **common,
        "condition": "predictive_step_100",
        "model_prefill_tokens": 20,
        "prompt_compression_pct": 0.0,
    }
    compressed = {
        **common,
        "condition": "predictive_step_100_compressed_prompt",
        "model_prefill_tokens": 14,
        "prompt_compression_pct": 30.0,
    }
    return raw, compressed


def test_summary_reports_matched_prompt_raw_vs_compressed(tmp_path):
    sample = {"id": "mbpp_1", "domain": "code", "prompt": "test"}
    records = list(_pair("mbpp_1"))
    _write_tier1_summary(
        tmp_path, [sample], records,
        ["predictive_step_100", "predictive_step_100_compressed_prompt"],
        "a" * 40,
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    paired = summary["matched_prompt_representation"]
    assert paired["pair_count"] == 1
    assert paired["all_codebooks_match"] is True
    assert paired["per_prompt"][0]["compressed_model_prefill_tokens"] == 14


def test_summary_rejects_raw_compressed_pair_with_different_codebooks(tmp_path):
    sample = {"id": "mbpp_1", "domain": "code", "prompt": "test"}
    records = list(_pair("mbpp_1"))
    records[1]["codebook_sha256"] = "different"

    with pytest.raises(RuntimeError, match="different codebooks"):
        _write_tier1_summary(
            tmp_path, [sample], records,
            ["predictive_step_100", "predictive_step_100_compressed_prompt"],
            "a" * 40,
        )
