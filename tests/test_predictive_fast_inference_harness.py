import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.cache_utils import DynamicCache
from zip2zip.predictor_policy import CappedPredictorPolicy

from experiments import validate_predictive_fast_path as harness


def test_dry_run_describes_two_separate_protocols_without_model_or_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not initialize or query CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    args = harness.build_parser().parse_args(["--dry-run"])
    harness.validate_args(args)
    report = harness.dry_run_report(args)
    alias_args = harness.build_parser().parse_args(["--dry-run", "--kv-cache-length", "256"])
    harness.validate_args(alias_args)
    assert alias_args.static_cache_capacity == 256

    assert report["execution_performed"] is False
    assert report["gpu_execution_enabled"] is False
    assert report["conditions"] == [
        "vanilla",
        "predictive_legacy_merged",
        "predictive_fast_merged",
    ]
    assert report["predictor_max_subtokens"] == 3
    assert report["manager_max_subtokens"] == 4
    behavioral = report["protocols"]["behavioral_generation_smoke"]
    fixed = report["protocols"]["fixed_kv_microbenchmark"]
    assert behavioral["cache_implementation"] == "static"
    assert behavioral["static_cache_capacity"] == 256
    assert behavioral["active_context_grows_during_generation"] is True
    assert "not fixed-KV" in behavioral["interpretation"]
    assert behavioral["per_forward_cuda_timing_authoritative"] is False
    assert behavioral["transformer_forward_timing"] is None
    assert behavioral["authoritative_per_step_timing"] == "fixed_kv_microbenchmark"
    assert fixed["active_kv_length"] == 256
    assert fixed["warmup_iterations"] >= 20
    assert fixed["measured_iterations"] == 100
    assert fixed["execution_performed"] is False
    assert "cache_strategy" in fixed
    assert report["real_phi_additional_table_mib"] > 300
    assert report["source_prompt_sha256"]
    assert report["lifecycle"].index(
        "behavioral smoke: legacy mode uses old prepare_prompt_input_ids -> model.generate -> manager.decode_sequence"
    ) < report["lifecycle"].index(
        "behavioral smoke: fast mode uses prepare_inference_tables -> prepare_input_sequence -> generate -> decode_sequence"
    )
    assert any("fixed-KV" in step for step in report["lifecycle"])


def test_partial_condition_artifacts_are_atomic_and_survive_later_failure(
    tmp_path, monkeypatch
):
    provenance = {"torchao_initial_version": "0.10.0", "torchao_final_state": "absent"}
    vanilla_payload = {
        "condition_order": ["vanilla"],
        "behavioral_generation_smoke": {"conditions": {"vanilla": {"wall_s": 1.0}}},
        "fixed_kv_microbenchmark": {"conditions": {"vanilla": {"median_ms": 58.0}}},
        "environment_provenance": provenance,
    }
    vanilla_path, aggregate_path = harness._write_partial_condition_result(
        output_dir=tmp_path,
        condition="vanilla",
        payload=vanilla_payload,
    )
    assert json.loads(vanilla_path.read_text(encoding="utf-8"))["partial_result"][
        "completed_conditions"
    ] == ["vanilla"]

    legacy_payload = {
        **vanilla_payload,
        "condition_order": ["vanilla", "predictive_legacy_merged"],
        "behavioral_generation_smoke": {
            "conditions": {
                "vanilla": {"wall_s": 1.0},
                "predictive_legacy_merged": {"wall_s": 2.0},
            }
        },
        "fixed_kv_microbenchmark": {
            "conditions": {
                "vanilla": {"median_ms": 58.0},
                "predictive_legacy_merged": {"median_ms": 100.0},
            }
        },
    }
    legacy_path, aggregate_path = harness._write_partial_condition_result(
        output_dir=tmp_path,
        condition="predictive_legacy_merged",
        payload=legacy_payload,
    )
    # Simulate a failure while publishing a later condition. The earlier
    # aggregate remains intact and the completed Fast condition file is atomic.
    original_writer = harness.write_json_atomic

    def fail_latest_aggregate(path, payload):
        if Path(path).name == "partial_results.json" and payload["partial_result"][
            "completed_condition"
        ] == "predictive_fast_merged":
            raise OSError("simulated Fast artifact publish failure")
        original_writer(path, payload)

    fast_payload = {
        **legacy_payload,
        "condition_order": [
            "vanilla",
            "predictive_legacy_merged",
            "predictive_fast_merged",
        ],
    }
    monkeypatch.setattr(harness, "write_json_atomic", fail_latest_aggregate)
    with pytest.raises(OSError, match="simulated Fast artifact publish failure"):
        harness._write_partial_condition_result(
            output_dir=tmp_path,
            condition="predictive_fast_merged",
            payload=fast_payload,
        )

    for path in (vanilla_path, legacy_path, aggregate_path):
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert "behavioral_generation_smoke" in saved
        assert "fixed_kv_microbenchmark" in saved
        assert saved["environment_provenance"] == provenance
    assert json.loads(aggregate_path.read_text(encoding="utf-8"))["partial_result"][
        "completed_conditions"
    ] == ["vanilla", "predictive_legacy_merged"]
    assert (tmp_path / "partial_predictive_fast_merged.json").is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_predictive_pair_publishes_each_condition_before_continuing():
    pair_source = inspect.getsource(harness._run_predictive_pair)
    execute_source = inspect.getsource(harness.execute)
    assert "on_condition_complete(" in pair_source
    assert pair_source.index("prior_manager_bindings_restored_after_detach") < pair_source.index(
        "on_condition_complete("
    )
    assert "on_condition_complete=on_predictive_condition_complete" in execute_source
    assert "persist_partial(\n                \"vanilla\"" in execute_source


def test_execution_and_fast_path_batch_are_explicitly_gated(monkeypatch):
    parser = harness.build_parser()
    with pytest.raises(ValueError, match="batch-size 1"):
        harness.validate_args(parser.parse_args(["--batch-size", "2"]))
    with pytest.raises(ValueError, match="--allow-gpu"):
        harness.validate_args(parser.parse_args(["--device", "cuda"]))
    with pytest.raises(ValueError, match="--allow-gpu"):
        harness.validate_args(parser.parse_args(["--execute", "--device", "cuda"]))

    # Calling execute without all three acknowledgements must stop before any
    # CUDA availability query, device initialization, or model loading.
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA path was reached without explicit authorization")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    with pytest.raises(RuntimeError, match="--execute"):
        harness.execute(parser.parse_args(["--dry-run"]))

    authorized = parser.parse_args(
        ["--execute", "--device", "cuda:0", "--allow-gpu"]
    )
    harness.validate_args(authorized)
    assert authorized.execute and authorized.allow_gpu
    # The test deliberately does not invoke execute with these flags.


def test_behavioral_warmup_uses_separate_state_and_masks_unseeded_h_ids():
    marker = object()

    class WarmupManager:
        def __init__(self):
            self.reset_calls = 0

        def get_logits_processor(self):
            return marker

        def reset(self):
            self.reset_calls += 1

    class WarmupModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return torch.tensor([[3, 4, 5]])

    model = WarmupModel()
    manager = WarmupManager()
    result = harness._warmup_generation(
        model=model,
        input_ids=torch.tensor([[3]]),
        tokenizer=SimpleNamespace(eos_token_id=1),
        manager=manager,
        device=torch.device("cpu"),
        static_cache_capacity=256,
    )
    assert result["performed"] is True
    assert result["included_in_recorded_generation_timing"] is False
    assert model.kwargs["max_new_tokens"] == model.kwargs["min_new_tokens"] == 2
    assert model.kwargs["cache_implementation"] == "static"
    assert model.kwargs["max_cache_len"] == 256
    assert model.kwargs["logits_processor"][0] is marker
    assert manager.reset_calls == 1


def test_behavioral_generation_needs_no_optional_forward_events(monkeypatch):
    class Model(torch.nn.Module):
        def register_forward_pre_hook(self, *args, **kwargs):
            raise AssertionError("behavioral smoke must not install forward timing hooks")

        def register_forward_hook(self, *args, **kwargs):
            raise AssertionError("behavioral smoke must not install forward timing hooks")

        def generate(self, **kwargs):
            if kwargs.get("return_dict_in_generate"):
                return SimpleNamespace(sequences=torch.tensor([[7, 1]]), scores=(), logits=())
            return torch.tensor([[7, 8, 9]])

    def unexpected_cuda_event(*args, **kwargs):
        raise AssertionError("behavioral smoke must not create CUDA forward events")

    monkeypatch.setattr(torch.cuda, "Event", unexpected_cuda_event)
    args = harness.build_parser().parse_args(["--dry-run"])
    record, _ = harness._generation_result(
        condition="vanilla",
        model=Model(),
        tokenizer=SimpleNamespace(
            eos_token_id=1,
            decode=lambda token_ids, skip_special_tokens=True: "answer",
        ),
        prompt_ids=[7],
        input_ids=[7],
        manager=None,
        device=torch.device("cpu"),
        args=args,
        prompt_metadata={"id": "fixture", "prompt_text": "fixture prompt"},
    )
    assert record["raw_decode_ids"] == [1]
    assert record["expanded_base_ids"] == [1]
    assert record["output_text"] == "answer"
    assert record["eos_emitted"] is True
    assert record["prompt_prefill_ttft_s"] is None
    assert record["generation_wall_time_s"] >= 0
    assert record["per_forward_cuda_timing_authoritative"] is False
    assert record["transformer_forward_timing"] is None
    assert "fixed_kv_microbenchmark is the authoritative" in record["per_forward_timing_note"]


def test_fixed_kv_cuda_event_pairs_are_synchronized_and_finalized_strictly(monkeypatch):
    sync_calls = []

    class Event:
        def __init__(self, duration_ms):
            self.duration_ms = duration_ms

        def query(self):
            return True

        def elapsed_time(self, other):
            assert isinstance(other, Event)
            return self.duration_ms

    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: sync_calls.append(device))
    device = torch.device("cuda:0")
    pairs = [
        {
            "start_event": Event(1.25),
            "end_event": Event(2.5),
            "start_recorded": True,
            "end_recorded": True,
        }
    ]
    durations = harness._fixed_kv_cuda_event_durations(
        pairs, device=device, expected_iterations=1
    )
    assert durations == [1.25]
    assert sync_calls == [device]


@pytest.mark.parametrize(
    "pair, expected_iterations, message",
    [
        (
            {
                "start_event": object(),
                "end_event": object(),
                "start_recorded": True,
                "end_recorded": False,
            },
            1,
            "not fully recorded",
        ),
        (
            {
                "start_event": SimpleNamespace(query=lambda: False),
                "end_event": SimpleNamespace(query=lambda: True),
                "start_recorded": True,
                "end_recorded": True,
            },
            1,
            "not recorded/completed",
        ),
        (None, 1, "event pair 0 is missing"),
    ],
)
def test_fixed_kv_invalid_event_pairs_raise_without_partial_timings(
    monkeypatch, pair, expected_iterations, message
):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    with pytest.raises(RuntimeError, match=message):
        harness._fixed_kv_cuda_event_durations(
            [pair], device=torch.device("cuda:0"), expected_iterations=expected_iterations
        )


def test_fixed_kv_elapsed_time_failure_is_invalid_not_a_wall_clock_fallback(monkeypatch):
    class UnfinalizableEvent:
        def query(self):
            return True

        def elapsed_time(self, other):
            raise ValueError("Both events must be recorded before calculating elapsed time.")

    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    pair = {
        "start_event": UnfinalizableEvent(),
        "end_event": UnfinalizableEvent(),
        "start_recorded": True,
        "end_recorded": True,
    }
    with pytest.raises(RuntimeError, match="could not be finalized.*no performance timings"):
        harness._fixed_kv_cuda_event_durations(
            [pair], device=torch.device("cuda:0"), expected_iterations=1
        )


def test_cache_copy_preserves_representation_metadata_and_independent_storage():
    cache = DynamicCache()
    key = torch.arange(1 * 2 * 256 * 4, dtype=torch.float32).reshape(1, 2, 256, 4)
    value = key + 1
    cache.update(key, value, 0)
    cache.update(key + 2, value + 2, 1)

    copied = harness._clone_past_key_values(cache)
    assert type(copied) is type(cache)
    assert copied is not cache
    assert harness._cache_sequence_length(copied) == 256
    assert harness._cache_layer_count(copied) == harness._cache_layer_count(cache) == 2
    source_tensors = harness._cache_tensors(cache)
    copied_tensors = harness._cache_tensors(copied)
    source_snapshots = [tensor.clone() for tensor in source_tensors]
    assert len(source_tensors) == len(copied_tensors) == 4
    for source, clone in zip(source_tensors, copied_tensors):
        assert source.shape == clone.shape
        assert source.dtype == clone.dtype
        assert source.device == clone.device
        assert source.data_ptr() != clone.data_ptr()
        assert torch.equal(source, clone)

    copied.update(
        torch.zeros((1, 2, 1, 4), dtype=torch.float32),
        torch.zeros((1, 2, 1, 4), dtype=torch.float32),
        0,
    )
    assert harness._cache_sequence_length(copied) == 257
    assert harness._cache_sequence_length(cache) == 256
    assert all(torch.equal(current, old) for current, old in zip(harness._cache_tensors(cache), source_snapshots))


def test_cache_copy_supports_legacy_tuple_and_rejects_unknown_representation():
    key = torch.arange(512, dtype=torch.float16).reshape(1, 2, 256, 1)
    value = key + 1
    legacy = ((key, value),)
    copied = harness._clone_past_key_values(legacy)
    assert type(copied) is tuple
    assert copied is not legacy
    assert harness._cache_sequence_length(copied) == 256
    assert harness._cache_layer_count(copied) == 1
    for source, clone in zip(harness._cache_tensors(legacy), harness._cache_tensors(copied)):
        assert source.dtype == clone.dtype == torch.float16
        assert source.device == clone.device
        assert source.shape == clone.shape
        assert source.data_ptr() != clone.data_ptr()
        assert torch.equal(source, clone)
    with pytest.raises(TypeError, match="unsupported cache class"):
        harness._clone_past_key_values(object())


def test_fixed_kv_mutable_cache_uses_independent_samples_and_excludes_copy_time(monkeypatch):
    class FakeClock:
        now = 0.0

        def perf_counter(self):
            return self.now

    clock = FakeClock()

    class TinyCachedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cache_inputs = []
            self.reference_cache = None

        def forward(
            self,
            *,
            input_ids,
            attention_mask,
            position_ids,
            cache_position,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        ):
            cache = past_key_values if past_key_values is not None else DynamicCache()
            if past_key_values is not None:
                before = harness._cache_sequence_length(cache)
                first_tensor = harness._cache_tensors(cache)[0]
                self.cache_inputs.append((cache, before, first_tensor))
            else:
                self.reference_cache = cache
            if input_ids.shape[-1] == 1:
                clock.now += 0.007
            key = torch.zeros((1, 1, input_ids.shape[-1], 2), dtype=torch.float32)
            cache.update(key, key, 0)
            return SimpleNamespace(past_key_values=cache)

    monkeypatch.setattr(harness.time, "perf_counter", clock.perf_counter)
    actual_clone = harness._clone_past_key_values

    def expensive_clone(cache):
        result = actual_clone(cache)
        clock.now += 9.0  # if included, this would dominate measured samples
        return result

    monkeypatch.setattr(harness, "_clone_past_key_values", expensive_clone)
    model = TinyCachedModel()
    result = harness._fixed_kv_microbenchmark(
        condition="synthetic",
        model=model,
        manager=None,
        prefix_ids=[5, 6],
        source_prompt_ids=[5, 6, 7],
        device=torch.device("cpu"),
    )
    assert len(model.cache_inputs) == 1 + 20 + 100  # probe, warmup, measured
    assert all(length == 256 for _, length, _ in model.cache_inputs)
    assert len({id(cache) for cache, _, _ in model.cache_inputs}) == len(model.cache_inputs)
    reference_ptrs = {tensor.data_ptr() for tensor in harness._cache_tensors(model.reference_cache)}
    assert all(cache is not model.reference_cache for cache, _, _ in model.cache_inputs)
    assert all(tensor.data_ptr() not in reference_ptrs for _, _, tensor in model.cache_inputs)
    assert result["active_kv_length"] == 256
    assert result["warmup_iterations"] == 20
    assert result["measured_iterations"] == 100
    assert result["reference_cache_sequence_length_before_probe"] == 256
    assert result["reference_cache_sequence_length_after_probe"] == 256
    assert result["reference_cache_sequence_length_before_timing"] == 256
    assert result["reference_cache_sequence_length_after_timing"] == 256
    assert result["probe_cache_input_length_before_forward"] == 256
    assert result["probe_cache_input_length_after_forward"] == 257
    assert result["cache_forward_mutates_input_in_place"] is True
    assert result["timed_past_cache_sequence_lengths"] == [256]
    assert result["all_timed_past_cache_lengths_equal_256"] is True
    assert result["cache_copy_time_included"] is False
    assert result["cache_reused_by_copy"] is True
    assert result["mean_forward_ms"] == pytest.approx(7.0, abs=1e-6)
    assert result["median_forward_ms"] == pytest.approx(7.0, abs=1e-6)
    assert result["stddev_forward_ms"] == pytest.approx(0.0, abs=1e-6)
    assert result["measured_iterations"] == 100
    assert result["reference_cache_layer_count"] == 1
    assert result["reference_cache_kv_tensor_metadata"][0] == {
        "shape": [1, 1, 256, 2],
        "dtype": "torch.float32",
        "device": "cpu",
    }


def test_fixed_kv_reuses_cache_only_after_immutable_probe():
    class ImmutableTupleCacheModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(
            self,
            *,
            input_ids,
            attention_mask,
            position_ids,
            cache_position,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        ):
            token_count = input_ids.shape[-1]
            if past_key_values is None:
                key = torch.zeros((1, 1, token_count, 2))
                return SimpleNamespace(past_key_values=((key, key.clone()),))
            self.inputs.append((past_key_values, harness._cache_sequence_length(past_key_values)))
            key, value = past_key_values[0]
            new_key = torch.cat([key, torch.zeros((1, 1, 1, 2))], dim=-2)
            new_value = torch.cat([value, torch.zeros((1, 1, 1, 2))], dim=-2)
            return SimpleNamespace(past_key_values=((new_key, new_value),))

    model = ImmutableTupleCacheModel()
    result = harness._fixed_kv_microbenchmark(
        condition="synthetic_immutable",
        model=model,
        manager=None,
        prefix_ids=[4, 5],
        source_prompt_ids=[4, 5],
        device=torch.device("cpu"),
    )
    assert result["cache_forward_mutates_input_in_place"] is False
    assert result["cache_reused_by_copy"] is False
    assert result["working_cache_copy"].startswith("reused immutable reference cache")
    assert len(model.inputs) == 1 + 20 + 100  # probe, warmup, measured
    assert all(length == 256 for _, length in model.inputs)
    # The first call is the mutation probe's independent copy; all actual
    # warmup and measured samples reuse the unmodified reference object.
    assert all(cache is model.inputs[1][0] for cache, _ in model.inputs[1:])
    assert model.inputs[0][0] is not model.inputs[1][0]
    assert result["reference_cache_sequence_length_after_timing"] == 256


def test_fixed_kv_uses_condition_correct_semantic_positions():
    vocab = 100

    class PositionManager:
        initial_vocab_size = vocab
        max_codebook_size = 32
        hyper_to_subtokens = {vocab: [8, 9, 10]}

        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

        def prepare_input_ids(self, context, attention_mask=None):
            return torch.arange(context.shape[-1], dtype=torch.long).unsqueeze(0) * 2

    context = torch.tensor([[5, vocab, 6]])
    mask = torch.ones_like(context)
    vanilla = harness._fixed_kv_position_setup(
        SimpleNamespace(), None, context, mask, [5, vocab, 6], torch.device("cpu")
    )
    assert vanilla[1] == 256
    assert vanilla[2] == "vanilla_absolute"
    assert vanilla[0].shape == (1, 256)

    compressed_model = SimpleNamespace(
        zip2zip_config=SimpleNamespace(position_mode="compressed")
    )
    compressed = harness._fixed_kv_position_setup(
        compressed_model,
        PositionManager(),
        context,
        mask,
        [5, vocab, 6],
        torch.device("cpu"),
    )
    assert compressed[1] == 256
    assert compressed[2] == "compressed"

    base_model = SimpleNamespace(
        zip2zip_config=SimpleNamespace(position_mode="base_token_end")
    )
    manager = PositionManager()
    base = harness._fixed_kv_position_setup(
        base_model,
        manager,
        context,
        mask,
        [5, vocab, 6],
        torch.device("cpu"),
    )
    assert base[0].tolist() == [[0, 2, 4]]
    assert base[1] == 5  # one + three constituent positions + one
    assert base[2] == "base_token_end"
    assert manager.reset_count == 1


def test_fixed_kv_rejects_wrong_semantic_position_vector_before_prefill():
    class BadPositionManager:
        initial_vocab_size = 100
        max_codebook_size = 32
        hyper_to_subtokens = {}

        def reset(self):
            return None

        def prepare_input_ids(self, context, attention_mask=None):
            return torch.zeros((1, 255), dtype=torch.long)

    class NeverRunModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_calls = 0
            self.zip2zip_config = SimpleNamespace(position_mode="base_token_end")

        def forward(self, **kwargs):
            self.forward_calls += 1
            raise AssertionError("bad position shape must be rejected before model prefill")

    model = NeverRunModel()
    with pytest.raises(RuntimeError, match="refusing invalid timing"):
        harness._fixed_kv_microbenchmark(
            condition="bad_positions",
            model=model,
            manager=BadPositionManager(),
            prefix_ids=[1, 2],
            source_prompt_ids=[1, 2],
            device=torch.device("cpu"),
        )
    assert model.forward_calls == 0


def _canonical_codebook_for_test():
    phrases = [(101, 102), (201, 202, 203)] + [
        (100 + index, 200 + index) for index in range(2, 32)
    ]
    return {
        phrases[index]: 32011 + index
        for index in reversed(range(len(phrases)))
    }


def test_canonical_codebook_serialization_round_trips_and_hashes_deterministically():
    codebook = _canonical_codebook_for_test()

    serialized, digest = harness._serialize_codebook(codebook)
    serialized_again, digest_again = harness._serialize_codebook(
        dict(reversed(list(codebook.items())))
    )

    assert serialized[:2] == [
        {"hyper_id": 32011, "subtoken_ids": [101, 102]},
        {"hyper_id": 32012, "subtoken_ids": [201, 202, 203]},
    ]
    reconstructed = {
        tuple(item["subtoken_ids"]): item["hyper_id"] for item in serialized
    }
    assert reconstructed == codebook
    assert serialized_again == serialized
    assert digest_again == digest
    assert digest == hashlib.sha256(
        json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("reversed_orientation", "phrase tuples"),
        ("wrong_k", "exactly K=32"),
        ("short_phrase", "phrase length"),
        ("long_phrase", "phrase length"),
        ("non_integer_phrase_id", "phrase IDs must be integers"),
        ("tail_token_id", "outside base vocabulary"),
        ("disabled_token_id", "disabled token ID"),
        ("non_integer_hyper_id", "must be an integer"),
        ("out_of_range_hyper_id", "outside the required absolute range"),
        ("duplicate_hyper_id", "duplicate H ID"),
    ],
)
def test_predictor_codebook_contract_rejects_malformed_results(case, error):
    codebook = _canonical_codebook_for_test()
    first_phrase = next(iter(codebook))
    if case == "reversed_orientation":
        codebook = {hyper_id: phrase for phrase, hyper_id in codebook.items()}
    elif case == "wrong_k":
        codebook.pop(first_phrase)
    elif case == "short_phrase":
        codebook[(10,)] = codebook.pop(first_phrase)
    elif case == "long_phrase":
        codebook[(10, 11, 12, 13)] = codebook.pop(first_phrase)
    elif case == "non_integer_phrase_id":
        codebook[(10, "11")] = codebook.pop(first_phrase)
    elif case == "tail_token_id":
        codebook[(10, 32011)] = codebook.pop(first_phrase)
    elif case == "disabled_token_id":
        codebook[(101, 102)] = 32011
    elif case == "non_integer_hyper_id":
        codebook[first_phrase] = "32011"
    elif case == "out_of_range_hyper_id":
        codebook[first_phrase] = 32043
    elif case == "duplicate_hyper_id":
        second_phrase = next(phrase for phrase in codebook if phrase != first_phrase)
        codebook[first_phrase] = codebook[second_phrase]

    with pytest.raises((TypeError, ValueError), match=error):
        harness._serialize_codebook(
            codebook,
            disabled_ids=(101,) if case == "disabled_token_id" else (),
        )


def test_real_policy_orientation_flows_through_serializer_and_manager_unchanged():
    class TinyTokenizer:
        @staticmethod
        def decode(token_ids):
            return " ".join(f"piece{token_id}" for token_id in token_ids)

    predictor_index = SimpleNamespace(
        disabled_ids=[],
        max_subtokens=3,
        token_associations={},
    )
    policy = CappedPredictorPolicy(
        predictor_index,
        TinyTokenizer(),
        budget=32,
        max_structural_slots=0,
        filter_bare_punctuation=True,
    )
    prompt_ids = list(range(100, 140))
    selected, metadata = policy.select_codebook(prompt_ids)

    assert metadata["total_phrases"] == 32
    assert len(selected) == 32
    assert all(isinstance(phrase, tuple) for phrase in selected)
    assert all(type(hyper_id) is int for hyper_id in selected.values())
    assert set(selected.values()) == set(range(32011, 32043))
    serialized, digest = harness._serialize_codebook(
        selected, disabled_ids=policy.disabled_ids
    )
    assert len(serialized) == 32
    assert digest == harness._serialize_codebook(
        selected, disabled_ids=policy.disabled_ids
    )[1]

    predictive_conditions = harness._predictive_condition_codebooks(
        selected,
        ["predictive_legacy_merged", "predictive_fast_merged"],
    )
    assert predictive_conditions[0][1] is predictive_conditions[1][1] is selected

    manager = harness.StaticCodebookManager(
        initial_vocab_size=32011,
        max_codebook_size=32,
        max_subtokens=4,
        embedding_dim=8,
        pad_token_id=0,
        disabled_ids=policy.disabled_ids,
    )
    # Pass the original policy object directly; runtime orientation is not reversed.
    manager.set_seeded_codebook(selected, batch_size=1, device=torch.device("cpu"))
    assert manager.subtokens_to_hyper == selected
    assert manager.hyper_to_subtokens == {
        hyper_id: list(phrase) for phrase, hyper_id in selected.items()
    }


def test_fresh_manager_detaches_and_restores_all_prior_condition_bindings():
    prior = object()

    class Wrapper:
        codebook_manager = prior

    class Base:
        def __init__(self):
            self.input = Wrapper()
            self.output = Wrapper()

        def get_input_embeddings(self):
            return self.input

        def get_output_embeddings(self):
            return self.output

    class Model:
        def __init__(self):
            self.base_model = Base()
            self.codebook_manager = prior

    model = Model()
    original = harness._model_codebook_manager_bindings(model)
    manager = harness.StaticCodebookManager(
        initial_vocab_size=100,
        max_codebook_size=32,
        max_subtokens=4,
        embedding_dim=8,
        pad_token_id=0,
    )
    manager.attach_to_model(model)
    assert all(binding is manager for binding in harness._model_codebook_manager_bindings(model))
    manager.fast_inference_ready = True
    manager.detach_from_model(model)
    harness._assert_codebook_manager_bindings(model, original)
    assert all(binding is prior for binding in harness._model_codebook_manager_bindings(model))


def _smoke_run(condition, *, codebook_hash="same-codebook", fixed_position=256):
    fixed = {
        "reference_context_input_ids_sha256": "same-context",
        "position_mode": "compressed",
        "prefill_position_ids_sha256": "same-prefill-positions",
        "next_position_id": fixed_position,
        "next_input_token_id": 77,
        "next_cache_position": 256,
        "position_semantics": "ordinary compressed-sequence position after 0..255",
    }
    return {
        "condition": condition,
        "codebook_sha256": codebook_hash,
        "source_prompt_sha256": "same-prompt",
        "original_prompt_token_ids_sha256": "same-original-ids",
        "input_token_ids_sha256": "same-compressed-ids",
        "tokenizer_revision": "pinned-rev",
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "min_new_tokens": 0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "use_cache": True,
        "max_new_tokens": 4,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "cache_implementation": "static",
        "static_cache_capacity": 256,
        "generation_context_grows_during_decode": True,
        "fixed_kv_microbenchmark": fixed,
        "raw_decode_ids": [5, 1, 2] if condition.endswith("legacy_merged") else [5, 4, 3],
        "generation_score_tensors": (
            torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]),
            torch.tensor([[0.0, 5.0, 4.0, 3.0, 2.0, 1.0]]),
            torch.zeros((1, 6)),
        ),
        "generation_logit_tensors": (
            torch.zeros((1, 6)),
            torch.ones((1, 6)),
            torch.full((1, 6), 2.0),
        ),
    }


def test_smoke_comparison_checks_codebook_context_and_ids():
    legacy = _smoke_run("predictive_legacy_merged")
    fast = _smoke_run("predictive_fast_merged")
    result = harness._compare_smoke_runs(legacy, fast)
    assert result["first_divergent_step"] == 1
    assert result["aligned_score_steps_before_context_diverges"] == 2
    assert result["same_codebook_sha256"] is True
    assert result["same_original_prompt_token_ids"] is True
    assert result["same_compressed_predictive_input_ids"] is True
    assert result["same_fixed_kv_position_setup"] is True
    with pytest.raises(RuntimeError, match="different predictive codebooks"):
        harness._compare_smoke_runs(
            legacy, _smoke_run("predictive_fast_merged", codebook_hash="other")
        )
    with pytest.raises(RuntimeError, match="next_position_id"):
        harness._compare_smoke_runs(
            legacy, _smoke_run("predictive_fast_merged", fixed_position=257)
        )


def test_payload_separates_behavioral_and_fixed_kv_metrics_and_derives_comparisons():
    args = harness.build_parser().parse_args(["--dry-run"])
    fixed_medians = {"vanilla": 50.0, "predictive_legacy_merged": 80.0, "predictive_fast_merged": 60.0}
    records = {}
    for condition, median in fixed_medians.items():
        records[condition] = {
            "condition": condition,
            "output_text": "answer",
            "generation_wall_time_s": 1.5,
            "per_forward_cuda_timing_authoritative": False,
            "transformer_forward_timing": None,
            "fixed_kv_microbenchmark": {
                "median_forward_ms": median,
                "mean_forward_ms": median,
                "p95_forward_ms": median + 1,
                "stddev_forward_ms": 0.5,
                "min_forward_ms": median - 1,
                "max_forward_ms": median + 1,
                "forward_steps_per_second": 1000 / median,
                "next_position_id": 256,
            },
            "setup": {
                "predictor_latency_s": 0.04,
                "codebook_manager_setup_s": 0.01,
                "h_vector_synthesis_ms": 0.2,
                "effective_input_table_build_ms": 0.3,
                "effective_output_table_build_ms": 0.4,
                "total_table_preparation_ms": 0.9,
                "effective_table_memory": {"additional_bytes": 1024},
            },
        }
    payload = harness._build_validation_payload(
        args=args,
        prompt={"id": "gsm_2956"},
        records=records,
        smoke={"first_divergent_step": None},
        original_ids=[10, 11],
        device=torch.device("cpu"),
        gpu_name="CPU dry fixture",
        serialized_codebook=None,
        codebook_sha256="same-hash",
    )
    assert payload["condition_labels"] == [
        "vanilla",
        "predictive_legacy_merged",
        "predictive_fast_merged",
    ]
    assert set(payload) >= {"behavioral_generation_smoke", "fixed_kv_microbenchmark"}
    assert payload["behavioral_generation_smoke"]["timings_are_fixed_kv_measurements"] is False
    assert payload["behavioral_generation_smoke"]["per_forward_cuda_timing_authoritative"] is False
    assert payload["behavioral_generation_smoke"]["transformer_forward_timing"] is None
    assert "fixed_kv_microbenchmark" not in payload["behavioral_generation_smoke"]["conditions"]["vanilla"]
    assert payload["behavioral_generation_smoke"]["conditions"]["vanilla"]["transformer_forward_timing"] is None
    assert payload["fixed_kv_microbenchmark"]["protocol"]["active_kv_length"] == 256
    comparisons = payload["fixed_kv_microbenchmark"]["derived_comparisons"]
    assert comparisons["legacy_overhead_vs_vanilla_pct"] == pytest.approx(60.0)
    assert comparisons["fast_overhead_vs_vanilla_pct"] == pytest.approx(20.0)
    assert comparisons["fast_speedup_vs_legacy_pct"] == pytest.approx(25.0)
    assert comparisons["legacy_minus_vanilla_ms"] == 30.0
    assert comparisons["fast_minus_vanilla_ms"] == 10.0
    assert comparisons["legacy_minus_fast_ms"] == 20.0
    assert comparisons["fast_median_minus_vanilla_median_ms_residual_per_step_overhead"] == 10.0
    setup = payload["fixed_kv_microbenchmark"]["conditions"]["predictive_fast_merged"]["request_setup"]
    assert setup["included_in_fixed_kv_forward_timing"] is False
    assert setup["codebook_manager_setup_s"] == 0.01
    assert setup["effective_table_bytes"]["additional_bytes"] == 1024
    summary = harness._render_human_readable_summary(payload)
    assert summary.index("## Behavioral Generation Smoke") < summary.index("## Fixed-KV Microbenchmark")
    assert "`static` cache, capacity 256" in summary
    assert "outside the timed forward" in summary


def test_fixed_context_ids_are_deterministic_and_exactly_256_positions():
    ids = harness._fixed_context_ids([9, 8, 7], [21, 22, 23])
    assert len(ids) == 256
    assert ids[:3] == [9, 8, 7]
    assert ids[3:] == [21] * 253
    assert ids == harness._fixed_context_ids([9, 8, 7], [21, 22, 23])
