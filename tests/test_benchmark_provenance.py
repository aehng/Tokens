import json

import pytest

from experiments.benchmark_provenance import (
    build_generation_cache_key,
    create_or_verify_run_manifest,
    select_prompt_subset,
    validate_generation_cache_record,
)


def test_prompt_subset_preserves_order_and_checks_domain_counts(tmp_path):
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps([{"id": "math_b"}, "code_a"]), encoding="utf-8")
    samples = [
        {"id": "code_a", "domain": "code"},
        {"id": "math_b", "domain": "reasoning"},
        {"id": "other", "domain": "instruction"},
    ]

    selected = select_prompt_subset(
        samples,
        ids_path,
        required_domain_counts={"code": 1, "reasoning": 1},
    )

    assert [sample["id"] for sample in selected] == ["math_b", "code_a"]


@pytest.mark.parametrize(
    "entries",
    [
        [],
        ["missing"],
        ["code_a", "code_a"],
        [None],
    ],
)
def test_prompt_subset_rejects_bad_ids(tmp_path, entries):
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps(entries), encoding="utf-8")
    samples = [{"id": "code_a", "domain": "code"}]

    with pytest.raises(ValueError):
        select_prompt_subset(samples, ids_path)


def test_generation_cache_key_covers_k_prompt_and_generation_settings():
    base = {
        "condition": {"name": "predictive", "k": 8},
        "prompt_id": "p1",
        "prompt_text": "question",
        "reference_text": "answer",
        "generation": {"max_new_tokens": 100, "eos": "last_token"},
        "evaluator_sha256": "eval-a",
        "environment": {"device": "cuda:0", "gpu": "T4"},
    }
    first = build_generation_cache_key(**base)

    assert first != build_generation_cache_key(**{**base, "condition": {"name": "predictive", "k": 16}})
    assert first != build_generation_cache_key(**{**base, "prompt_text": "changed question"})
    assert first != build_generation_cache_key(**{**base, "generation": {"max_new_tokens": 200, "eos": "last_token"}})
    assert first == build_generation_cache_key(**base)


def test_run_manifest_is_resumable_only_for_identical_identity(tmp_path):
    path = tmp_path / "manifest.json"
    identity = {"tested_commit": "a" * 40, "k_values": [4, 8, 16]}

    first, resumed_first = create_or_verify_run_manifest(path, identity)
    second, resumed_second = create_or_verify_run_manifest(path, identity)

    assert not resumed_first
    assert resumed_second
    assert first["identity_sha256"] == second["identity_sha256"]
    with pytest.raises(ValueError, match="does not match"):
        create_or_verify_run_manifest(path, {**identity, "k_values": [4, 8, 16, 24]})

    sweep_path = tmp_path / "sweep_manifest.json"
    create_or_verify_run_manifest(
        sweep_path, identity, schema="tokens_k_sweep_manifest_v2"
    )
    with pytest.raises(ValueError, match="does not match"):
        create_or_verify_run_manifest(sweep_path, identity)


def test_generation_cache_record_requires_exact_key_and_schema():
    row = {"record_schema": "phi_generation_record_v2", "generation_cache_key": "abc"}
    assert validate_generation_cache_record(row, "abc")
    assert validate_generation_cache_record(row, "abc", "phi_generation_record_v2")
    assert not validate_generation_cache_record(row, "abc", "k_sweep_generation_v3")
    sweep_row = {"record_schema": "k_sweep_generation_v3", "generation_cache_key": "abc"}
    assert validate_generation_cache_record(sweep_row, "abc", "k_sweep_generation_v3")
    assert not validate_generation_cache_record(row, "different")
    assert not validate_generation_cache_record({"generation_cache_key": "abc"}, "abc")
