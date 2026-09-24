import pytest

from experiments import validate_predictive_fast_path as harness


def test_fast_path_harness_defaults_to_non_loading_dry_run():
    args = harness.build_parser().parse_args(["--dry-run"])
    harness.validate_args(args)
    report = harness.dry_run_report(args)

    assert report["execution_performed"] is False
    assert report["gpu_execution_enabled"] is False
    assert report["conditions"] == list(harness.CONDITIONS)
    assert report["predictor_max_subtokens"] == 3
    assert report["manager_max_subtokens"] == 4
    assert report["kv_cache_length"] == 256
    assert report["real_phi_additional_table_mib"] > 300
    assert report["source_prompt_sha256"]
    assert report["lifecycle"].index(
        "legacy mode: old prepare_prompt_input_ids -> model.generate -> manager.decode_sequence"
    ) < report["lifecycle"].index(
        "fast mode: prepare_inference_tables -> prepare_input_sequence -> generate -> decode_sequence"
    )


def test_fast_path_harness_rejects_unsupported_batch_and_implicit_cuda():
    parser = harness.build_parser()
    with pytest.raises(ValueError, match="batch-size 1"):
        harness.validate_args(parser.parse_args(["--batch-size", "2"]))
    with pytest.raises(ValueError, match="--allow-gpu"):
        harness.validate_args(parser.parse_args(["--device", "cuda"]))


def test_fast_path_harness_cuda_execution_requires_explicit_flags_but_is_not_run():
    args = harness.build_parser().parse_args(
        ["--execute", "--device", "cuda", "--allow-gpu"]
    )
    harness.validate_args(args)
    assert args.execute
    assert args.allow_gpu
    # Deliberately do not invoke harness.execute(): this test is CPU-only.


def test_smoke_comparison_limits_logits_to_shared_generation_prefix():
    common = {
        "codebook_sha256": "same-codebook",
        "source_prompt_sha256": "same-prompt",
        "original_prompt_token_ids_sha256": "same-token-ids",
        "tokenizer_revision": "pinned-rev",
        "do_sample": False,
        "max_new_tokens": 4,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "cache_implementation": "static",
        "fixed_kv_cache_length": 256,
        "generation_score_tensors": (
            harness.torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]),
            harness.torch.tensor([[0.0, 5.0, 4.0, 3.0, 2.0, 1.0]]),
            harness.torch.zeros((1, 6)),
        ),
        "generation_logit_tensors": (
            harness.torch.zeros((1, 6)),
            harness.torch.ones((1, 6)),
            harness.torch.full((1, 6), 2.0),
        ),
    }
    legacy = {**common, "condition": "predictive_legacy_merged", "raw_decode_ids": [5, 1, 2]}
    fast = {
        **common,
        "condition": "predictive_fast_merged",
        "raw_decode_ids": [5, 4, 3],
        "generation_score_tensors": common["generation_score_tensors"],
    }

    result = harness._compare_smoke_runs(legacy, fast)
    assert result["first_divergent_step"] == 1
    assert result["aligned_score_steps_before_context_diverges"] == 2
    assert result["same_codebook_sha256"] is True
