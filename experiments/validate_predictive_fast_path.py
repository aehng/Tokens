"""Explicitly gated real-Phi comparison for legacy and prepared fast inference.

The default is a local, non-loading dry run. Real model execution requires all
three flags: ``--execute --device cuda --allow-gpu``. This harness is separate
from ``run_quality_benchmark.py`` so that benchmark remains the legacy control.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from zip2zip import StaticCodebookManager, prepare_model_for_inference
from zip2zip.static_codebook import estimate_effective_table_memory
from experiments.benchmark_provenance import write_json_atomic
from experiments import run_quality_benchmark as benchmark


CONDITIONS = (
    "vanilla",
    "predictive_legacy_merged",
    "predictive_fast_merged",
)
DEFAULT_PROMPT_ID = "gsm_2956"
DEFAULT_MAX_NEW_TOKENS = 101
STATIC_CACHE_CAPACITY = 256
FIXED_KV_CONTEXT_LENGTH = 256
FIXED_KV_WARMUP_ITERATIONS = 20
FIXED_KV_MEASURED_ITERATIONS = 100
PREDICTOR_MAX_SUBTOKENS = 3
MANAGER_MAX_SUBTOKENS = 4
MAX_CODEBOOK_SIZE = 32


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run the dedicated fast-inference comparison. Real execution is "
            "CUDA-gated and does not use the authoritative legacy benchmark path."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Only validate imports/configuration (default).")
    mode.add_argument("--execute", action="store_true", help="Load models and run the selected comparison.")
    parser.add_argument("--allow-gpu", action="store_true", help="Required acknowledgement for CUDA execution.")
    parser.add_argument("--device", default="cpu", help="Execution device; real runs require cuda or cuda:N.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--prompt-id", default=DEFAULT_PROMPT_ID)
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / benchmark.CKPT_100_PATH))
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--static-cache-capacity",
        "--kv-cache-length",
        dest="static_cache_capacity",
        type=int,
        default=STATIC_CACHE_CAPACITY,
        help="Behavioral generate() cache capacity (legacy alias: --kv-cache-length); not fixed context length.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "experiments" / "checkpoints" / "fast_inference_validation"),
    )
    parser.add_argument(
        "--environment-provenance",
        help="Optional JSON provenance file copied into each durable partial result.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size != 1:
        raise ValueError("the predictive fast validation harness requires --batch-size 1")
    if args.static_cache_capacity != STATIC_CACHE_CAPACITY:
        raise ValueError("the behavioral smoke protocol uses static cache capacity 256")
    if args.max_new_tokens < 2 or args.max_new_tokens > args.static_cache_capacity:
        raise ValueError("max-new-tokens must be between 2 and the static cache capacity")
    device = torch.device(args.device)
    is_cuda = device.type == "cuda"
    if is_cuda and not args.allow_gpu:
        raise ValueError("CUDA is disabled unless --allow-gpu is also supplied")
    if args.execute and not is_cuda:
        raise ValueError("real-model execution is CUDA-only; use the default dry run on CPU")
    if args.execute and not args.allow_gpu:
        raise ValueError("real-model execution requires both --device cuda and --allow-gpu")


def _shape_only_phi_memory_estimate() -> Dict[str, int]:
    # Pinned Phi-3.5 Mini dimensions; shape/dtype arithmetic only, no weights loaded.
    return estimate_effective_table_memory(
        (32064, 3072),
        torch.float16,
        (32064, 3072),
        torch.float16,
        codebook_size=MAX_CODEBOOK_SIZE,
    )


def dry_run_report(args: argparse.Namespace) -> Dict[str, Any]:
    prompt = _load_prompt(args.prompt_id)
    lifecycle = [
        "run Vanilla behavioral smoke and fixed-KV protocol first, then release its model",
        "load pinned Step-100 predictive bundle and canonical predictor policy",
        "tokenize the predictive source prompt once with the pinned Phi tokenizer",
        "select one K=32 codebook once and reuse its exact SHA and entries for legacy and fast",
        "merge LoRA explicitly; preserve predictor max_subtokens=3 and manager max_subtokens=4",
        "behavioral smoke: legacy mode uses old prepare_prompt_input_ids -> model.generate -> manager.decode_sequence",
        "legacy fixed-KV protocol: prefill 256 positions and run warmup/measured cached forwards",
        "detach the fresh legacy manager and verify prior model/embedding/output bindings are restored",
        "behavioral smoke: fast mode uses prepare_inference_tables -> prepare_input_sequence -> generate -> decode_sequence",
        "fast fixed-KV protocol: prefill 256 positions and run warmup/measured cached forwards",
        "detach the fresh fast manager and verify prior model/embedding/output bindings are restored",
        "compare predictive smoke records, codebook SHA, original IDs, compressed IDs, and semantic positions",
    ]
    assert lifecycle.index("behavioral smoke: legacy mode uses old prepare_prompt_input_ids -> model.generate -> manager.decode_sequence") < lifecycle.index("behavioral smoke: fast mode uses prepare_inference_tables -> prepare_input_sequence -> generate -> decode_sequence")
    memory = _shape_only_phi_memory_estimate()
    return {
        "mode": "dry-run",
        "execution_performed": False,
        "gpu_execution_enabled": False,
        "conditions": list(args.conditions),
        "prompt_id": args.prompt_id,
        "source_prompt_sha256": hashlib.sha256(prompt["prompt_text"].encode("utf-8")).hexdigest(),
        "source_prompt_characters": len(prompt["prompt_text"]),
        "batch_size": args.batch_size,
        "device_requested": args.device,
        "checkpoint": args.checkpoint,
        "max_new_tokens": args.max_new_tokens,
        "protocols": {
            "behavioral_generation_smoke": {
                "cache_implementation": "static",
                "static_cache_capacity": args.static_cache_capacity,
                "active_context_grows_during_generation": True,
                "interpretation": "behavioral/end-to-end only; not fixed-KV timing",
                "per_forward_cuda_timing_authoritative": False,
                "transformer_forward_timing": None,
                "authoritative_per_step_timing": "fixed_kv_microbenchmark",
            },
            "fixed_kv_microbenchmark": {
                "active_kv_length": FIXED_KV_CONTEXT_LENGTH,
                "warmup_iterations": FIXED_KV_WARMUP_ITERATIONS,
                "measured_iterations": FIXED_KV_MEASURED_ITERATIONS,
                "execution_performed": False,
                "cache_strategy": "copy outside timing if the mutation probe detects changes; otherwise reuse only after proving the reference remains unchanged",
                "note": "Separate from generate(); no model or cache was loaded in dry-run mode.",
            },
        },
        "predictor_max_subtokens": PREDICTOR_MAX_SUBTOKENS,
        "manager_max_subtokens": MANAGER_MAX_SUBTOKENS,
        "lifecycle": lifecycle,
        "real_phi_fp16_table_memory_estimate": memory,
        "real_phi_effective_input_mib": memory["effective_input_embedding_bytes"] / (1024**2),
        "real_phi_effective_output_mib": memory["effective_output_head_bytes"] / (1024**2),
        "real_phi_additional_table_mib": memory["additional_bytes"] / (1024**2),
        "note": "No checkpoint, tokenizer, model weights, CUDA context, or remote compute was loaded.",
    }


def _load_prompt(prompt_id: str) -> Dict[str, Any]:
    data_path = Path(benchmark.VAL_DATA_PATH)
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    with data_path.open("r", encoding="utf-8") as source:
        samples = json.load(source)
    for sample in samples:
        if sample["id"] == prompt_id:
            prompt_text = (
                benchmark.build_mbpp_prompt(sample)
                if sample.get("domain") == "code"
                else sample["prompt"]
            )
            return {**sample, "prompt_text": prompt_text}
    raise ValueError(f"prompt ID {prompt_id!r} is absent from {data_path}")


def _fixed_kv_cuda_event_durations(
    measured_events: Sequence[Dict[str, Any]],
    *,
    device: torch.device,
    expected_iterations: int,
) -> List[float]:
    """Finalize strict fixed-KV CUDA event pairs; never fall back to wall time."""
    if len(measured_events) != expected_iterations:
        raise RuntimeError(
            "fixed-KV CUDA timing invalid: expected "
            f"{expected_iterations} recorded event pairs, found {len(measured_events)}; "
            "no performance timings will be reported"
        )

    # One batch-boundary synchronization preserves the event timing protocol.
    torch.cuda.synchronize(device)
    durations_ms: List[float] = []
    for sample_index, pair in enumerate(measured_events):
        if not isinstance(pair, dict):
            raise RuntimeError(
                f"fixed-KV CUDA timing invalid: event pair {sample_index} is missing; "
                "no performance timings will be reported"
            )
        start_event = pair.get("start_event")
        end_event = pair.get("end_event")
        if start_event is None or end_event is None:
            raise RuntimeError(
                f"fixed-KV CUDA timing invalid: event pair {sample_index} is incomplete; "
                "no performance timings will be reported"
            )
        if pair.get("start_recorded") is not True or pair.get("end_recorded") is not True:
            raise RuntimeError(
                f"fixed-KV CUDA timing invalid: event pair {sample_index} was not fully "
                "recorded; no performance timings will be reported"
            )
        for event_name, event in (("start", start_event), ("end", end_event)):
            query = getattr(event, "query", None)
            if not callable(query):
                raise RuntimeError(
                    f"fixed-KV CUDA timing invalid: event pair {sample_index} "
                    f"{event_name} event cannot be validated; no performance timings "
                    "will be reported"
                )
            try:
                recorded_and_complete = query()
            except Exception as error:
                raise RuntimeError(
                    f"fixed-KV CUDA timing invalid: event pair {sample_index} "
                    f"{event_name} event was not recorded; no performance timings "
                    "will be reported"
                ) from error
            if not recorded_and_complete:
                raise RuntimeError(
                    f"fixed-KV CUDA timing invalid: event pair {sample_index} "
                    f"{event_name} event was not recorded/completed; no performance "
                    "timings will be reported"
                )
        try:
            duration_ms = float(start_event.elapsed_time(end_event))
        except Exception as error:
            raise RuntimeError(
                f"fixed-KV CUDA timing invalid: event pair {sample_index} could not be "
                "finalized after synchronization; no performance timings will be reported"
            ) from error
        if not math.isfinite(duration_ms) or duration_ms < 0:
            raise RuntimeError(
                f"fixed-KV CUDA timing invalid: event pair {sample_index} returned "
                f"duration {duration_ms!r}; no performance timings will be reported"
            )
        durations_ms.append(duration_ms)
    return durations_ms


def _generation_tensors_to_cpu(values) -> Tuple[torch.Tensor, ...]:
    if not values:
        return ()
    stacked = torch.stack(values).detach().cpu()
    return tuple(stacked[index] for index in range(stacked.shape[0]))


def _serialize_codebook(codebook: Dict[Any, Sequence[int]]) -> Tuple[List[Dict[str, Any]], str]:
    """Canonical JSON representation and digest for one predictor-selected codebook."""
    serialized = [
        {
            "hyper_id": int(hyper_id),
            "subtoken_ids": [int(token_id) for token_id in subtokens],
        }
        for hyper_id, subtokens in sorted(codebook.items(), key=lambda item: int(item[0]))
    ]
    canonical = json.dumps(serialized, sort_keys=True, separators=(",", ":"))
    return serialized, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _predictive_condition_codebooks(
    codebook: Dict[Any, Sequence[int]], selected_conditions: Sequence[str]
) -> List[Tuple[str, Dict[Any, Sequence[int]]]]:
    """Bind both predictive conditions to the exact one-time predictor result."""
    return [
        (condition, codebook)
        for condition in ("predictive_legacy_merged", "predictive_fast_merged")
        if condition in selected_conditions
    ]


def _select_codebook_once(policy: Any, original_prompt_ids: Sequence[int]):
    """Call the predictor exactly once for the shared predictive comparison."""
    started = time.perf_counter()
    codebook, predictor_metadata = policy.select_codebook(original_prompt_ids)
    return codebook, predictor_metadata, time.perf_counter() - started


def _cache_sequence_length(cache: Any) -> int:
    """Return the populated physical sequence length for HF Cache or tuple caches."""
    get_seq_length = getattr(cache, "get_seq_length", None)
    if callable(get_seq_length):
        return int(get_seq_length())
    if isinstance(cache, (tuple, list)) and cache:
        first_layer = cache[0]
        if isinstance(first_layer, (tuple, list)) and first_layer:
            key = first_layer[0]
            if torch.is_tensor(key) and key.ndim >= 3:
                return int(key.shape[-2])
    raise TypeError(
        "unsupported past_key_values representation; expected an HF Cache or "
        "legacy layer tuple with sequence at dimension -2"
    )


def _cache_tensors(cache: Any) -> List[torch.Tensor]:
    """Collect KV tensors used to ensure a cache copy has independent storage."""
    tensors: List[torch.Tensor] = []
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            for name in ("keys", "values"):
                value = getattr(layer, name, None)
                if torch.is_tensor(value):
                    tensors.append(value)
        return tensors

    def visit(value: Any) -> None:
        if torch.is_tensor(value):
            tensors.append(value)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)

    visit(cache)
    return tensors


def _cache_layer_count(cache: Any) -> int:
    if hasattr(cache, "layers"):
        return len(cache.layers)
    if isinstance(cache, (tuple, list)) and cache:
        return len(cache)
    raise TypeError(f"unsupported cache class for layer-count validation: {type(cache)!r}")


def _cache_tensor_metadata(cache: Any) -> List[Dict[str, Any]]:
    return [
        {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }
        for tensor in _cache_tensors(cache)
    ]


def _cache_copy_strategy(cache: Any) -> str:
    method = getattr(cache, "copy", None)
    if callable(method):
        return f"{type(cache).__name__}.copy() with post-copy validation"
    return "copy.deepcopy with post-copy validation"


def _cache_tensor_states_equal(left: Any, right: Any) -> bool:
    """Compare KV state without creating a second full-sized cache snapshot."""
    left_tensors = _cache_tensors(left)
    right_tensors = _cache_tensors(right)
    if len(left_tensors) != len(right_tensors):
        return False
    return all(
        left_tensor.shape == right_tensor.shape
        and left_tensor.dtype == right_tensor.dtype
        and torch.equal(left_tensor, right_tensor)
        for left_tensor, right_tensor in zip(left_tensors, right_tensors)
    )


def _clone_past_key_values(cache: Any) -> Any:
    """Clone a prepared cache outside timing and reject shallow/shared copies."""
    if not (hasattr(cache, "layers") or isinstance(cache, (tuple, list))):
        raise TypeError(
            f"unsupported cache class for fixed-KV timing: {type(cache)!r}"
        )
    try:
        official_copy = getattr(cache, "copy", None)
        cloned = official_copy() if callable(official_copy) else copy.deepcopy(cache)
    except Exception as error:
        raise RuntimeError(
            "the installed Transformers cache cannot be copied safely; "
            "refusing to time a cache that may grow between samples"
        ) from error
    if cloned is cache:
        raise RuntimeError("cache copy returned the original object")
    if type(cloned) is not type(cache):
        raise RuntimeError("cache copy changed the cache representation")
    if _cache_sequence_length(cloned) != _cache_sequence_length(cache):
        raise RuntimeError("cache copy changed the populated sequence length")
    if _cache_layer_count(cloned) != _cache_layer_count(cache):
        raise RuntimeError("cache copy changed the layer count")
    if _cache_tensor_metadata(cloned) != _cache_tensor_metadata(cache):
        raise RuntimeError("cache copy changed KV tensor shape, dtype, or device")
    source_ptrs = {
        (str(tensor.device), tensor.data_ptr())
        for tensor in _cache_tensors(cache)
        if tensor.numel()
    }
    copied_ptrs = {
        (str(tensor.device), tensor.data_ptr())
        for tensor in _cache_tensors(cloned)
        if tensor.numel()
    }
    if not source_ptrs or source_ptrs.intersection(copied_ptrs):
        raise RuntimeError("cache copy does not have independent KV tensor storage")
    return cloned


def _fixed_context_ids(
    prefix_ids: Sequence[int],
    source_prompt_ids: Sequence[int],
    *,
    context_length: int = FIXED_KV_CONTEXT_LENGTH,
    initial_vocab_size: int = benchmark.INITIAL_VOCAB,
    fallback_token_id: int = 0,
) -> List[int]:
    """Build a deterministic exact-length context, extending with a base ID."""
    if context_length <= 0:
        raise ValueError("fixed KV context length must be positive")
    context = [int(token_id) for token_id in prefix_ids[:context_length]]
    filler = next(
        (int(token_id) for token_id in source_prompt_ids if 0 <= int(token_id) < initial_vocab_size),
        int(fallback_token_id),
    )
    if filler < 0 or filler >= initial_vocab_size:
        raise ValueError("fixed-KV filler token must be a base-vocabulary ID")
    context.extend([filler] * (context_length - len(context)))
    if len(context) != context_length:
        raise AssertionError("fixed context builder did not produce the requested length")
    return context


def _fixed_kv_position_setup(
    model: torch.nn.Module,
    manager: Optional[StaticCodebookManager],
    context: torch.Tensor,
    attention_mask: torch.Tensor,
    context_ids: Sequence[int],
    device: torch.device,
) -> Tuple[torch.Tensor, int, str, str]:
    """Return prefill RoPE positions and the matching next semantic position."""
    if manager is None:
        position_mode = "vanilla_absolute"
        position_semantics = "ordinary absolute position after 0..255"
        prefill_positions = torch.arange(
            FIXED_KV_CONTEXT_LENGTH, dtype=torch.long, device=device
        ).unsqueeze(0)
        next_position_id = FIXED_KV_CONTEXT_LENGTH
    else:
        manager.reset()
        zip2zip_config = getattr(model, "zip2zip_config", None)
        position_mode = getattr(zip2zip_config, "position_mode", None)
        if position_mode == "base_token_end":
            position_semantics = "next base-token-end position after phrase-span expansion"
            prefill_positions = manager.prepare_input_ids(
                context, attention_mask=attention_mask
            )
            next_position_id = sum(
                len(manager.hyper_to_subtokens.get(token_id, (token_id,)))
                if manager.initial_vocab_size
                <= token_id
                < manager.initial_vocab_size + manager.max_codebook_size
                else 1
                for token_id in context_ids
            )
        elif position_mode == "compressed":
            position_semantics = "ordinary compressed-sequence position after 0..255"
            prefill_positions = torch.arange(
                FIXED_KV_CONTEXT_LENGTH, dtype=torch.long, device=device
            ).unsqueeze(0)
            next_position_id = FIXED_KV_CONTEXT_LENGTH
        else:
            raise RuntimeError(
                "fixed-KV benchmark requires a known predictive position_mode; "
                f"got {position_mode!r}"
            )
    return prefill_positions, next_position_id, str(position_mode), position_semantics


def _fixed_kv_microbenchmark(
    *,
    condition: str,
    model: torch.nn.Module,
    manager: Optional[StaticCodebookManager],
    prefix_ids: Sequence[int],
    source_prompt_ids: Sequence[int],
    device: torch.device,
    fallback_token_id: int = 0,
    warmup_iterations: int = FIXED_KV_WARMUP_ITERATIONS,
    measured_iterations: int = FIXED_KV_MEASURED_ITERATIONS,
) -> Dict[str, Any]:
    """Measure cached one-token forwards from a true, repeatedly reset 256-token KV state."""
    if warmup_iterations < FIXED_KV_WARMUP_ITERATIONS:
        raise ValueError(
            f"fixed-KV microbenchmark requires at least {FIXED_KV_WARMUP_ITERATIONS} warmup iterations"
        )
    if measured_iterations != FIXED_KV_MEASURED_ITERATIONS:
        raise ValueError(
            f"fixed-KV microbenchmark requires exactly {FIXED_KV_MEASURED_ITERATIONS} measured iterations"
        )
    context_ids = _fixed_context_ids(
        prefix_ids,
        source_prompt_ids,
        fallback_token_id=fallback_token_id,
    )
    context = torch.tensor([context_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(context)
    physical_cache_positions = torch.arange(
        FIXED_KV_CONTEXT_LENGTH, dtype=torch.long, device=device
    )

    position_ids, next_position_id, position_mode, position_semantics = (
        _fixed_kv_position_setup(
            model, manager, context, attention_mask, context_ids, device
        )
    )
    if tuple(position_ids.shape) != tuple(context.shape):
        raise RuntimeError(
            f"{condition} produced position_ids shape {tuple(position_ids.shape)} "
            f"for a {tuple(context.shape)} fixed-KV prefill; refusing invalid timing"
        )
    prefill_position_values = position_ids[0].detach().cpu().tolist()
    prefill_position_ids_sha256 = hashlib.sha256(
        json.dumps(prefill_position_values).encode("ascii")
    ).hexdigest()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prefill_started = time.perf_counter()
    with torch.inference_mode():
        prefill_output = model(
            input_ids=context,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=physical_cache_positions,
            use_cache=True,
            return_dict=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prefill_wall_ms = (time.perf_counter() - prefill_started) * 1000.0
    reference_cache = getattr(prefill_output, "past_key_values", None)
    if reference_cache is None:
        raise RuntimeError(f"{condition} prefill did not return past_key_values")
    reference_length_before = _cache_sequence_length(reference_cache)
    if reference_length_before != FIXED_KV_CONTEXT_LENGTH:
        raise RuntimeError(
            f"{condition} reference cache has {reference_length_before} positions; "
            f"expected exactly {FIXED_KV_CONTEXT_LENGTH}"
        )

    next_token_id = next(
        (int(token_id) for token_id in source_prompt_ids if 0 <= int(token_id) < benchmark.INITIAL_VOCAB),
        int(fallback_token_id),
    )
    one_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
    one_attention_mask = torch.ones(
        (1, FIXED_KV_CONTEXT_LENGTH + 1), dtype=torch.long, device=device
    )
    one_position = torch.tensor([[next_position_id]], dtype=torch.long, device=device)
    next_cache_position = torch.tensor(
        [FIXED_KV_CONTEXT_LENGTH], dtype=torch.long, device=device
    )

    # Probe only an independent copy. Mutable cache classes (including the
    # installed Transformers DynamicCache) append in place; this must not alter
    # the immutable reference that seeds every timed sample.
    probe_cache = _clone_past_key_values(reference_cache)
    probe_input_length_before = _cache_sequence_length(probe_cache)
    probe_kv_tensor_count = len(_cache_tensors(probe_cache))
    with torch.inference_mode():
        probe_output = model(
            input_ids=one_token,
            attention_mask=one_attention_mask,
            position_ids=one_position,
            cache_position=next_cache_position,
            past_key_values=probe_cache,
            use_cache=True,
            return_dict=True,
        )
    probe_input_length_after = _cache_sequence_length(probe_cache)
    probe_output_cache = getattr(probe_output, "past_key_values", None)
    probe_output_length = (
        _cache_sequence_length(probe_output_cache)
        if probe_output_cache is not None
        else None
    )
    probe_input_state_changed = not _cache_tensor_states_equal(
        reference_cache, probe_cache
    )
    cache_forward_mutates_input_in_place = (
        probe_input_length_after != probe_input_length_before
        or probe_input_state_changed
    )
    reference_length_after_probe = _cache_sequence_length(reference_cache)
    if reference_length_after_probe != FIXED_KV_CONTEXT_LENGTH:
        raise RuntimeError("cache-mutation probe changed the reference cache")
    del probe_cache, probe_output, probe_output_cache, prefill_output
    reference_length_before_timing = _cache_sequence_length(reference_cache)
    if reference_length_before_timing != FIXED_KV_CONTEXT_LENGTH:
        raise RuntimeError("reference cache changed before the timing loop")

    def run_forward(cache_copy: Any) -> Any:
        return model(
            input_ids=one_token,
            attention_mask=one_attention_mask,
            position_ids=one_position,
            cache_position=next_cache_position,
            past_key_values=cache_copy,
            use_cache=True,
            return_dict=True,
        )

    def fresh_sample_cache() -> Any:
        return (
            _clone_past_key_values(reference_cache)
            if cache_forward_mutates_input_in_place
            else reference_cache
        )

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            working_cache = fresh_sample_cache()
            if _cache_sequence_length(working_cache) != FIXED_KV_CONTEXT_LENGTH:
                raise AssertionError("warmup cache did not begin at exactly 256 positions")
            warmup_output = run_forward(working_cache)
            if _cache_sequence_length(reference_cache) != FIXED_KV_CONTEXT_LENGTH:
                raise AssertionError("warmup forward mutated the reference cache")
            del warmup_output, working_cache

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        measured_events = []
        timed_context_lengths = []
        with torch.inference_mode():
            for _ in range(measured_iterations):
                working_cache = fresh_sample_cache()
                working_length = _cache_sequence_length(working_cache)
                if working_length != FIXED_KV_CONTEXT_LENGTH:
                    raise AssertionError("measured cache did not begin at exactly 256 positions")
                timed_context_lengths.append(working_length)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_recorded = False
                end_recorded = False
                start_event.record()
                start_recorded = True
                measured_output = run_forward(working_cache)
                end_event.record()
                end_recorded = True
                measured_events.append(
                    {
                        "start_event": start_event,
                        "end_event": end_event,
                        "start_recorded": start_recorded,
                        "end_recorded": end_recorded,
                    }
                )
                if _cache_sequence_length(reference_cache) != FIXED_KV_CONTEXT_LENGTH:
                    raise AssertionError("measured forward mutated the reference cache")
                del measured_output, working_cache
        durations_ms = _fixed_kv_cuda_event_durations(
            measured_events,
            device=device,
            expected_iterations=measured_iterations,
        )
    else:
        durations_ms = []
        timed_context_lengths = []
        with torch.inference_mode():
            for _ in range(measured_iterations):
                working_cache = fresh_sample_cache()
                working_length = _cache_sequence_length(working_cache)
                if working_length != FIXED_KV_CONTEXT_LENGTH:
                    raise AssertionError("measured cache did not begin at exactly 256 positions")
                timed_context_lengths.append(working_length)
                started = time.perf_counter()
                measured_output = run_forward(working_cache)
                durations_ms.append((time.perf_counter() - started) * 1000.0)
                if _cache_sequence_length(reference_cache) != FIXED_KV_CONTEXT_LENGTH:
                    raise AssertionError("measured forward mutated the reference cache")
                del measured_output, working_cache

    reference_length_after = _cache_sequence_length(reference_cache)
    if reference_length_after != FIXED_KV_CONTEXT_LENGTH:
        raise AssertionError(
            f"reference cache grew to {reference_length_after} positions during timing"
        )
    ordered = sorted(durations_ms)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "measurement_type": "fixed_context_one_token_cached_forward",
        "condition": condition,
        "batch_size": 1,
        "active_kv_length": FIXED_KV_CONTEXT_LENGTH,
        "context_length": FIXED_KV_CONTEXT_LENGTH,
        "reference_context_input_ids": context_ids,
        "reference_context_input_ids_sha256": hashlib.sha256(
            json.dumps(context_ids).encode("ascii")
        ).hexdigest(),
        "position_mode": position_mode,
        "position_semantics": position_semantics,
        "prefill_position_ids": [int(value) for value in prefill_position_values],
        "prefill_position_ids_sha256": prefill_position_ids_sha256,
        "reference_cache_class": f"{type(reference_cache).__module__}.{type(reference_cache).__name__}",
        "reference_cache_layer_count": _cache_layer_count(reference_cache),
        "reference_cache_kv_tensor_metadata": _cache_tensor_metadata(reference_cache),
        "reference_cache_sequence_length_before_probe": reference_length_before,
        "reference_cache_sequence_length_before_timing": reference_length_before_timing,
        "probe_cache_input_length_before_forward": probe_input_length_before,
        "probe_cache_input_length_after_forward": probe_input_length_after,
        "probe_returned_cache_sequence_length": probe_output_length,
        "probe_cache_kv_tensor_count": probe_kv_tensor_count,
        "probe_cache_kv_state_changed": probe_input_state_changed,
        "cache_forward_mutates_input_in_place": cache_forward_mutates_input_in_place,
        "reference_cache_sequence_length_after_probe": reference_length_after_probe,
        "reference_cache_sequence_length_after_timing": reference_length_after,
        "timed_past_cache_sequence_lengths": sorted(set(timed_context_lengths)),
        "all_timed_past_cache_lengths_equal_256": set(timed_context_lengths)
        == {FIXED_KV_CONTEXT_LENGTH},
        "working_cache_copy": (
            "independent copy with independent KV tensor storage; created outside timed interval"
            if cache_forward_mutates_input_in_place
            else "reused immutable reference cache after mutation probe"
        ),
        "cache_copy_strategy": (
            _cache_copy_strategy(reference_cache)
            if cache_forward_mutates_input_in_place
            else "reference reuse; probe showed forward leaves cache unchanged"
        ),
        "context_prefill_wall_ms_excluded_from_forward_timings": prefill_wall_ms,
        "next_input_token_id": next_token_id,
        "next_position_id": next_position_id,
        "next_position_semantics": position_semantics,
        "next_cache_position": FIXED_KV_CONTEXT_LENGTH,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "mean_forward_ms": statistics.fmean(durations_ms),
        "median_forward_ms": statistics.median(durations_ms),
        "p95_forward_ms": p95,
        "stddev_forward_ms": statistics.stdev(durations_ms),
        "min_forward_ms": min(durations_ms),
        "max_forward_ms": max(durations_ms),
        "forward_steps_per_second": 1000.0 / statistics.fmean(durations_ms),
        "cache_reused_by_copy": cache_forward_mutates_input_in_place,
        "cache_copy_time_included": False,
    }


def _generation_result(
    *,
    condition: str,
    model: torch.nn.Module,
    tokenizer,
    prompt_ids: Sequence[int],
    input_ids: Sequence[int],
    manager: Optional[StaticCodebookManager],
    device: torch.device,
    args: argparse.Namespace,
    prompt_metadata: Dict[str, Any],
    setup_metrics: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Any]:
    if len(input_ids) + args.max_new_tokens > args.static_cache_capacity:
        raise ValueError(
            f"prompt ({len(input_ids)}) + requested output ({args.max_new_tokens}) "
            f"exceeds static cache capacity {args.static_cache_capacity}"
        )
    tensor_ids = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    warmup_report = _warmup_generation(
        model=model,
        input_ids=tensor_ids,
        tokenizer=tokenizer,
        manager=manager,
        device=device,
        static_cache_capacity=args.static_cache_capacity,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_started = time.perf_counter()
    timing_proc = benchmark.TimingLogitsProcessor(generation_started, static_mgr=manager)
    with torch.inference_mode():
        output = model.generate(
            input_ids=tensor_ids,
            max_new_tokens=args.max_new_tokens,
            max_cache_len=args.static_cache_capacity,
            cache_implementation="static",
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            min_new_tokens=0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList([timing_proc]),
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_wall_s = time.perf_counter() - generation_started

    raw_ids = output.sequences[0, tensor_ids.shape[1] :].detach().cpu().tolist()
    expanded_ids = manager.decode_sequence(raw_ids) if manager is not None else list(raw_ids)
    output_text = tokenizer.decode(expanded_ids, skip_special_tokens=True)
    events = []
    if manager is not None:
        for position, token_id in enumerate(raw_ids):
            phrase = manager.hyper_to_subtokens.get(int(token_id))
            if phrase is not None:
                events.append(
                    {
                        "position": position,
                        "token_id": int(token_id),
                        "phrase_token_ids": list(phrase),
                        "phrase_length": len(phrase),
                        "phrase_text": tokenizer.decode(list(phrase), skip_special_tokens=False),
                        "base_positions_saved": max(0, len(phrase) - 1),
                    }
                )
    eos_value = tokenizer.eos_token_id
    eos_ids = (
        [int(value) for value in eos_value]
        if isinstance(eos_value, (list, tuple))
        else [int(eos_value)] if eos_value is not None else []
    )
    eos_reached = any(int(token_id) in eos_ids for token_id in raw_ids)
    scores = _generation_tensors_to_cpu(getattr(output, "scores", ()) or ())
    logits = _generation_tensors_to_cpu(getattr(output, "logits", ()) or ())
    record: Dict[str, Any] = {
        "condition": condition,
        "prompt_id": prompt_metadata["id"],
        "prompt_domain": prompt_metadata.get("domain"),
        "source_prompt_sha256": hashlib.sha256(prompt_metadata["prompt_text"].encode("utf-8")).hexdigest(),
        "original_prompt_token_ids": list(prompt_ids),
        "original_prompt_token_ids_sha256": hashlib.sha256(json.dumps(list(prompt_ids)).encode("ascii")).hexdigest(),
        "input_token_ids_sha256": hashlib.sha256(json.dumps(list(input_ids)).encode("ascii")).hexdigest(),
        "tokenizer_id": benchmark.PHI_MODEL_ID,
        "tokenizer_revision": benchmark.DEFAULT_PHI_REVISION,
        "batch_size": 1,
        "measurement_type": "behavioral_smoke_generate",
        "static_cache_capacity": args.static_cache_capacity,
        "generation_context_grows_during_decode": True,
        "generation_warmup": warmup_report,
        "raw_decode_ids": [int(token_id) for token_id in raw_ids],
        "expanded_base_ids": [int(token_id) for token_id in expanded_ids],
        "output_text": output_text,
        "original_prompt_token_count": len(prompt_ids),
        "compressed_prompt_position_count": len(input_ids),
        "prompt_compression_pct": round(
            100.0 * (1.0 - len(input_ids) / max(1, len(prompt_ids))), 2
        ),
        "hypertoken_emissions": events,
        "hypertokens_emitted": len(events),
        "base_tokens_represented_by_hypertokens": sum(len(event["phrase_token_ids"]) for event in events),
        "transformer_decode_iterations": len(raw_ids),
        "expanded_output_tokens": len(expanded_ids),
        "net_decode_steps_saved": len(expanded_ids) - len(raw_ids),
        "raw_decode_reduction": (
            (len(expanded_ids) - len(raw_ids)) / len(expanded_ids)
            if expanded_ids else None
        ),
        "eos_emitted": eos_reached,
        "hit_max_new_tokens": len(raw_ids) >= args.max_new_tokens,
        "prompt_prefill_ttft_s": timing_proc.ttft,
        "generation_wall_time_s": generation_wall_s,
        "decode_wall_time_s": max(0.0, generation_wall_s - (timing_proc.ttft or 0.0)),
        "per_forward_cuda_timing_authoritative": False,
        "transformer_forward_timing": None,
        "per_forward_timing_note": (
            "Behavioral generate() does not install CUDA-event forward hooks; "
            "fixed_kv_microbenchmark is the authoritative per-step timing."
        ),
        "generation_score_tensors": scores,
        "generation_logit_tensors": logits,
        "setup": setup_metrics or {},
        **prompt_metadata.get("generation_settings", {}),
    }
    decode_wall_s = record["decode_wall_time_s"]
    record["transformer_steps_per_second"] = (
        len(raw_ids) / decode_wall_s if decode_wall_s > 0 else None
    )
    record["expanded_tokens_per_second"] = (
        len(expanded_ids) / decode_wall_s if decode_wall_s > 0 else None
    )
    record["wall_time_per_decode_step_s"] = (
        decode_wall_s / len(raw_ids) if raw_ids else None
    )
    record["wall_time_per_expanded_token_s"] = (
        decode_wall_s / len(expanded_ids) if expanded_ids else None
    )
    return record, output


def _compare_smoke_runs(legacy: Dict[str, Any], fast: Dict[str, Any]) -> Dict[str, Any]:
    if legacy.get("codebook_sha256") != fast.get("codebook_sha256"):
        raise RuntimeError("legacy/fast smoke comparison used different predictive codebooks")
    for shared_key in (
        "codebook_sha256",
        "source_prompt_sha256",
        "original_prompt_token_ids_sha256",
        "input_token_ids_sha256",
        "tokenizer_revision",
        "do_sample",
        "num_beams",
        "num_return_sequences",
        "min_new_tokens",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "use_cache",
        "max_new_tokens",
        "pad_token_id",
        "eos_token_id",
        "cache_implementation",
        "static_cache_capacity",
        "generation_context_grows_during_decode",
    ):
        if legacy.get(shared_key) != fast.get(shared_key):
            raise RuntimeError(f"legacy/fast smoke comparison differs at {shared_key}")
    for shared_key in (
        "reference_context_input_ids_sha256",
        "position_mode",
        "prefill_position_ids_sha256",
        "next_position_id",
        "next_input_token_id",
        "next_cache_position",
        "position_semantics",
    ):
        if legacy["fixed_kv_microbenchmark"].get(shared_key) != fast[
            "fixed_kv_microbenchmark"
        ].get(shared_key):
            raise RuntimeError(f"legacy/fast fixed-KV position/context differs at {shared_key}")
    legacy_ids = legacy["raw_decode_ids"]
    fast_ids = fast["raw_decode_ids"]
    shared = min(len(legacy_ids), len(fast_ids))
    divergent = next((i for i in range(shared) if legacy_ids[i] != fast_ids[i]), None)
    if divergent is None and len(legacy_ids) != len(fast_ids):
        divergent = shared
    available_score_steps = min(
        len(legacy["generation_score_tensors"]),
        len(fast["generation_score_tensors"]),
    )
    score_steps = min(
        available_score_steps,
        divergent + 1 if divergent is not None else available_score_steps,
    )
    if score_steps:
        left_scores = torch.stack(legacy["generation_score_tensors"][:score_steps]).squeeze(1).float().cpu()
        right_scores = torch.stack(fast["generation_score_tensors"][:score_steps]).squeeze(1).float().cpu()
        top1_matches = left_scores.argmax(-1).eq(right_scores.argmax(-1)).float()
        left_top5 = torch.topk(left_scores, 5, dim=-1).indices
        right_top5 = torch.topk(right_scores, 5, dim=-1).indices
        top5_overlaps = (
            left_top5.unsqueeze(-1).eq(right_top5.unsqueeze(-2)).any(dim=-1).float().sum(-1) / 5.0
        )
    else:
        left_scores = right_scores = None
        top1_matches = top5_overlaps = None
    logit_steps = min(
        len(legacy["generation_logit_tensors"]),
        len(fast["generation_logit_tensors"]),
    )
    logit_steps = min(logit_steps, score_steps)
    if logit_steps:
        left_logits = torch.stack(legacy["generation_logit_tensors"][:logit_steps]).squeeze(1).float().cpu()
        right_logits = torch.stack(fast["generation_logit_tensors"][:logit_steps]).squeeze(1).float().cpu()
        finite = torch.isfinite(left_logits) & torch.isfinite(right_logits)
        flat_diffs = (left_logits[finite] - right_logits[finite]).abs()
    else:
        flat_diffs = torch.empty(0)
    return {
        "measurement_type": "behavioral_smoke_generate",
        "smoke_pair": [legacy["condition"], fast["condition"]],
        "same_codebook_sha256": legacy.get("codebook_sha256") == fast.get("codebook_sha256"),
        "same_original_prompt_token_ids": legacy.get("original_prompt_token_ids_sha256")
        == fast.get("original_prompt_token_ids_sha256"),
        "same_compressed_predictive_input_ids": legacy.get("input_token_ids_sha256")
        == fast.get("input_token_ids_sha256"),
        "same_fixed_kv_position_setup": legacy["fixed_kv_microbenchmark"].get(
            "prefill_position_ids_sha256"
        )
        == fast["fixed_kv_microbenchmark"].get("prefill_position_ids_sha256"),
        "raw_decode_ids_identical": legacy_ids == fast_ids,
        "first_divergent_step": divergent,
        "aligned_score_steps_before_context_diverges": score_steps,
        "top1_agreement_fraction": float(top1_matches.mean()) if top1_matches is not None else None,
        "mean_top5_overlap_fraction": float(top5_overlaps.mean()) if top5_overlaps is not None else None,
        "max_abs_logit_difference_finite_values": float(flat_diffs.max()) if flat_diffs.numel() else None,
        "mean_abs_logit_difference_finite_values": float(flat_diffs.mean()) if flat_diffs.numel() else None,
        "interpretation": "FP16 comparison is diagnostic; bitwise logit identity is not required.",
    }


def _warmup_generation(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    tokenizer,
    manager: Optional[StaticCodebookManager],
    device: torch.device,
    static_cache_capacity: int,
) -> Dict[str, Any]:
    """Run a short unmeasured generation, then clear request-local position state."""
    warmup_tokens = 2
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    warmup_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": warmup_tokens,
        "min_new_tokens": warmup_tokens,
        "max_cache_len": static_cache_capacity,
        "cache_implementation": "static",
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if manager is not None:
        warmup_kwargs["logits_processor"] = LogitsProcessorList(
            [manager.get_logits_processor()]
        )
    try:
        with torch.inference_mode():
            warmup_output = model.generate(**warmup_kwargs)
        del warmup_output
    finally:
        if manager is not None:
            manager.reset()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return {
        "performed": True,
        "requested_new_tokens": warmup_tokens,
        "included_in_recorded_generation_timing": False,
        "request_position_state_reset_after": True,
    }


def _model_codebook_manager_bindings(model: torch.nn.Module) -> Tuple[Any, Any, Any]:
    base = getattr(model, "base_model", model)
    input_layer = base.get_input_embeddings() if hasattr(base, "get_input_embeddings") else None
    output_layer = base.get_output_embeddings() if hasattr(base, "get_output_embeddings") else None
    return (
        getattr(model, "codebook_manager", None),
        getattr(input_layer, "codebook_manager", None),
        getattr(output_layer, "codebook_manager", None),
    )


def _assert_codebook_manager_bindings(
    model: torch.nn.Module, expected: Tuple[Any, Any, Any]
) -> None:
    current = _model_codebook_manager_bindings(model)
    if any(actual is not prior for actual, prior in zip(current, expected)):
        raise RuntimeError("predictive condition did not restore its prior manager bindings")


def _run_predictive_pair(
    args: argparse.Namespace,
    prompt: Dict[str, Any],
    device: torch.device,
    on_condition_complete: Optional[
        Callable[[str, Dict[str, Any], Dict[str, Any]], None]
    ] = None,
):
    bundle = benchmark.load_predictive_model_bundle(
        args.checkpoint,
        str(device),
        base_revision=benchmark.DEFAULT_PHI_REVISION,
        model_revision=benchmark.DEFAULT_ZIP2ZIP_REVISION,
        expected_step=100,
    )
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    if getattr(policy, "max_subtokens", None) != PREDICTOR_MAX_SUBTOKENS:
        raise RuntimeError(
            "canonical predictor contract changed: expected max_subtokens=3, "
            f"got {getattr(policy, 'max_subtokens', None)!r}"
        )
    prepare_model_for_inference(model, merge_lora=True)
    original_ids = tokenizer.encode(prompt["prompt_text"], add_special_tokens=False)
    if any(token_id >= benchmark.INITIAL_VOCAB for token_id in original_ids):
        raise ValueError(
            "legacy comparison prompt contains a tokenizer tail ID at/above the "
            "hypertoken insertion point; the legacy input path cannot safely map it"
        )
    codebook, _predictor_meta, predictor_latency_s = _select_codebook_once(
        policy, original_ids
    )
    if len(codebook) != MAX_CODEBOOK_SIZE:
        raise RuntimeError(
            f"representative validation requires exactly K=32 selected entries; got {len(codebook)}"
        )
    serialized_codebook, codebook_sha256 = _serialize_codebook(codebook)
    generation_settings = {
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": 0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "cache_implementation": "static",
        "static_cache_capacity": args.static_cache_capacity,
        "generation_context_grows_during_decode": True,
    }
    prompt_metadata = {
        "id": prompt["id"],
        "domain": prompt.get("domain"),
        "prompt_text": prompt["prompt_text"],
        "generation_settings": generation_settings,
    }
    records = {}
    shared_predictive_input_ids = None
    for condition, condition_codebook in _predictive_condition_codebooks(
        codebook, args.conditions
    ):
        manager_setup_started = time.perf_counter()
        manager = StaticCodebookManager(
            initial_vocab_size=benchmark.INITIAL_VOCAB,
            max_codebook_size=MAX_CODEBOOK_SIZE,
            max_subtokens=MANAGER_MAX_SUBTOKENS,
            embedding_dim=bundle["embedding_dim"],
            pad_token_id=bundle["pad_id"],
            disabled_ids=bundle["disabled_ids"],
        )
        prior_bindings = _model_codebook_manager_bindings(model)
        manager.set_seeded_codebook(condition_codebook, batch_size=1, device=device)
        manager.attach_to_model(model)
        manager_setup_s = time.perf_counter() - manager_setup_started
        try:
            if condition == "predictive_legacy_merged":
                input_ids = benchmark.prepare_prompt_input_ids(
                    original_ids, manager, compress_prompt=True
                )
                if manager.decode_sequence(input_ids) != list(original_ids):
                    raise RuntimeError("legacy compressed prompt failed its round-trip")
                setup_metrics = {
                    "predictor_latency_s": predictor_latency_s,
                    "codebook_manager_setup_s": manager_setup_s,
                    "h_vector_synthesis_ms": None,
                    "effective_input_table_build_ms": None,
                    "effective_output_table_build_ms": None,
                    "legacy_tables_prepared": False,
                    "fast_tables_prepared": False,
                    "codebook_sha256": codebook_sha256,
                }
            else:
                base_model = model.base_model
                input_layer = base_model.get_input_embeddings()
                output_layer = base_model.get_output_embeddings()
                memory_estimate = estimate_effective_table_memory(
                    input_layer.weight.shape,
                    input_layer.weight.dtype,
                    output_layer.weight.shape,
                    output_layer.weight.dtype,
                    codebook_size=manager.max_codebook_size,
                    output_bias_shape=(
                        output_layer.bias.shape if output_layer.bias is not None else None
                    ),
                    output_bias_dtype=(
                        output_layer.bias.dtype if output_layer.bias is not None else None
                    ),
                )
                vram_before = int(torch.cuda.memory_allocated(device))
                torch.cuda.reset_peak_memory_stats(device)
                manager.prepare_inference_tables(model, batch_size=1)
                vram_after = int(torch.cuda.memory_allocated(device))
                peak_prepare = int(torch.cuda.max_memory_allocated(device))
                torch.cuda.reset_peak_memory_stats(device)
                input_ids = manager.prepare_input_sequence(original_ids, compress=True)
                if manager.decode_sequence(input_ids) != list(original_ids):
                    raise RuntimeError("fast compressed prompt failed its round-trip")
                setup_metrics = {
                    "predictor_latency_s": predictor_latency_s,
                    "codebook_manager_setup_s": manager_setup_s,
                    **manager.inference_timing_report,
                    "legacy_tables_prepared": False,
                    "fast_tables_prepared": manager.fast_inference_ready,
                    "effective_table_memory_estimate_before_build": memory_estimate,
                    "effective_table_memory": manager.inference_memory_report,
                    "vram_before_table_prepare_bytes": vram_before,
                    "vram_after_table_prepare_bytes": vram_after,
                    "peak_vram_during_table_prepare_bytes": peak_prepare,
                    "codebook_sha256": codebook_sha256,
                }
            if condition == "predictive_legacy_merged" and manager.fast_inference_ready:
                raise RuntimeError("legacy predictive condition unexpectedly prepared fast tables")
            if condition == "predictive_fast_merged" and not manager.fast_inference_ready:
                raise RuntimeError("fast predictive condition did not prepare effective tables")
            if shared_predictive_input_ids is None:
                shared_predictive_input_ids = list(input_ids)
            elif list(input_ids) != shared_predictive_input_ids:
                raise RuntimeError(
                    "legacy and fast prompt preparation produced different IDs "
                    "from the same source prompt and codebook"
                )
            if condition == "predictive_fast_merged":
                setup_metrics["codebook_sha256"] = codebook_sha256
            record, _ = _generation_result(
                condition=condition,
                model=model,
                tokenizer=tokenizer,
                prompt_ids=original_ids,
                input_ids=input_ids,
                manager=manager,
                device=device,
                args=args,
                prompt_metadata=prompt_metadata,
                setup_metrics=setup_metrics,
            )
            record["codebook_sha256"] = codebook_sha256
            record["condition_runtime_state"] = {
                "manager_attached": True,
                "fast_tables_prepared": manager.fast_inference_ready,
                "lora_merged_before_condition": True,
                "fresh_static_manager_for_condition": True,
                "same_merged_model_object_used_for_predictive_pair": True,
            }
            record["predictor_max_subtokens"] = PREDICTOR_MAX_SUBTOKENS
            record["manager_max_subtokens"] = MANAGER_MAX_SUBTOKENS
            record["original_prompt_token_count"] = len(original_ids)
            record["compressed_prompt_position_count"] = len(input_ids)
            record["prompt_compression_pct"] = round(
                100.0 * (1.0 - len(input_ids) / max(1, len(original_ids))), 2
            )
            record["total_request_wall_time_s"] = (
                predictor_latency_s + manager_setup_s
                + setup_metrics.get("total_table_preparation_ms", 0.0) / 1000.0
                + record["generation_wall_time_s"]
            )
            record["fixed_kv_microbenchmark"] = _fixed_kv_microbenchmark(
                condition=condition,
                model=model,
                manager=manager,
                prefix_ids=input_ids,
                source_prompt_ids=original_ids,
                device=device,
                fallback_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )
            records[condition] = record
            del _
        finally:
            manager.reset()
            manager.detach_from_model(model)
            _assert_codebook_manager_bindings(model, prior_bindings)
        records[condition]["condition_runtime_state"][
            "prior_manager_bindings_restored_after_detach"
        ] = True
        if on_condition_complete is not None:
            on_condition_complete(
                condition,
                records[condition],
                {
                    "original_prompt_token_ids": list(original_ids),
                    "serialized_codebook": serialized_codebook,
                    "codebook_sha256": codebook_sha256,
                },
            )
    smoke = None
    if "predictive_legacy_merged" in records and "predictive_fast_merged" in records:
        smoke = _compare_smoke_runs(
            records["predictive_legacy_merged"], records["predictive_fast_merged"]
        )
    for record in records.values():
        record.pop("generation_score_tensors", None)
        record.pop("generation_logit_tensors", None)
    del bundle, model
    gc.collect()
    torch.cuda.empty_cache()
    return records, smoke, original_ids, tokenizer, serialized_codebook, codebook_sha256


def _run_vanilla(args: argparse.Namespace, prompt: Dict[str, Any], device: torch.device):
    if "vanilla" not in args.conditions:
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(
        benchmark.PHI_MODEL_ID,
        revision=benchmark.DEFAULT_PHI_REVISION,
    )
    model = AutoModelForCausalLM.from_pretrained(
        benchmark.PHI_MODEL_ID,
        revision=benchmark.DEFAULT_PHI_REVISION,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    original_ids = tokenizer.encode(prompt["prompt_text"], add_special_tokens=False)
    metadata = {
        "id": prompt["id"],
        "domain": prompt.get("domain"),
        "prompt_text": prompt["prompt_text"],
        "generation_settings": {
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": 0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "use_cache": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "cache_implementation": "static",
            "static_cache_capacity": args.static_cache_capacity,
            "generation_context_grows_during_decode": True,
        },
    }
    record, _ = _generation_result(
        condition="vanilla",
        model=model,
        tokenizer=tokenizer,
        prompt_ids=original_ids,
        input_ids=original_ids,
        manager=None,
        device=device,
        args=args,
        prompt_metadata=metadata,
        setup_metrics={"legacy_tables_prepared": False},
    )
    record["fixed_kv_microbenchmark"] = _fixed_kv_microbenchmark(
        condition="vanilla",
        model=model,
        manager=None,
        prefix_ids=original_ids,
        source_prompt_ids=original_ids,
        device=device,
        fallback_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
    )
    record.pop("generation_score_tensors", None)
    record.pop("generation_logit_tensors", None)
    del _
    record["total_request_wall_time_s"] = record["generation_wall_time_s"]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return record, tokenizer


def _request_setup_summary(setup: Dict[str, Any]) -> Dict[str, Any]:
    memory = setup.get("effective_table_memory") or {}
    return {
        "predictor_latency_s": setup.get("predictor_latency_s"),
        "codebook_manager_setup_s": setup.get("codebook_manager_setup_s"),
        "h_vector_synthesis_ms": setup.get("h_vector_synthesis_ms"),
        "effective_input_table_build_ms": setup.get("effective_input_table_build_ms"),
        "effective_output_table_build_ms": setup.get("effective_output_table_build_ms"),
        "total_table_preparation_ms": setup.get("total_table_preparation_ms"),
        "effective_table_bytes": {
            key: value
            for key, value in memory.items()
            if key.endswith("_bytes")
        },
        "vram_before_setup_bytes": setup.get("vram_before_table_prepare_bytes"),
        "vram_after_setup_bytes": setup.get("vram_after_table_prepare_bytes"),
        "peak_vram_during_setup_bytes": setup.get("peak_vram_during_table_prepare_bytes"),
        "included_in_fixed_kv_forward_timing": False,
    }


def _derive_fixed_kv_comparisons(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    vanilla = results.get("vanilla", {}).get("median_forward_ms")
    legacy = results.get("predictive_legacy_merged", {}).get("median_forward_ms")
    fast = results.get("predictive_fast_merged", {}).get("median_forward_ms")

    def ratio_delta(numerator: Optional[float], denominator: Optional[float]):
        if numerator is None or denominator in (None, 0):
            return None
        return (numerator / denominator - 1.0) * 100.0

    return {
        "legacy_overhead_vs_vanilla_pct": ratio_delta(legacy, vanilla),
        "fast_overhead_vs_vanilla_pct": ratio_delta(fast, vanilla),
        "fast_speedup_vs_legacy_pct": (
            (legacy - fast) / legacy * 100.0
            if legacy is not None and fast is not None and legacy != 0
            else None
        ),
        "legacy_minus_vanilla_ms": legacy - vanilla if legacy is not None and vanilla is not None else None,
        "fast_minus_vanilla_ms": fast - vanilla if fast is not None and vanilla is not None else None,
        "legacy_minus_fast_ms": legacy - fast if legacy is not None and fast is not None else None,
        "fast_median_minus_vanilla_median_ms_residual_per_step_overhead": (
            fast - vanilla if fast is not None and vanilla is not None else None
        ),
        "interpretation": "Measured values only; historical V16 values are context and are not assertions.",
    }


def _render_human_readable_summary(payload: Dict[str, Any]) -> str:
    behavioral = payload["behavioral_generation_smoke"]
    fixed = payload["fixed_kv_microbenchmark"]
    lines = [
        "# Predictive Fast-Path Validation Summary",
        "",
        f"Prompt: `{payload['prompt_id']}`",
        f"Checkpoint step: {payload.get('checkpoint_step', 'n/a')}",
        f"Predictive codebook SHA-256: `{payload.get('predictive_codebook_sha256', 'n/a')}`",
        "",
        "## Behavioral Generation Smoke",
        "",
        f"Cache semantics: `{behavioral['cache_semantics']['implementation']}` cache, "
        f"capacity {behavioral['cache_semantics']['capacity']}; "
        f"{behavioral['cache_semantics']['interpretation']}.",
        "These generation timings are behavioral/end-to-end and are not fixed-KV measurements.",
        "Behavioral per-forward CUDA timing is disabled; fixed-KV CUDA Events are the authoritative per-step measurement.",
        "",
    ]
    for condition in CONDITIONS:
        record = behavioral["conditions"].get(condition)
        if record is None:
            continue
        lines.extend(
            [
                f"### {condition}",
                "",
                f"- EOS emitted: {record.get('eos_emitted')}",
                f"- TTFT: {record.get('prompt_prefill_ttft_s')} s",
                f"- Generation wall: {record.get('generation_wall_time_s')} s",
                f"- Raw decode IDs: `{json.dumps(record.get('raw_decode_ids', []))}`",
                f"- Expanded base IDs: `{json.dumps(record.get('expanded_base_ids', []))}`",
                f"- H emissions: `{json.dumps(record.get('hypertoken_emissions', []), ensure_ascii=False)}`",
                "- Output text:",
                "",
                "~~~text",
                str(record.get("output_text", "")),
                "~~~",
                "",
            ]
        )
    comparison = behavioral.get("legacy_vs_fast")
    if comparison is not None:
        lines.extend(
            [
                "### Legacy vs. Fast Smoke Comparison",
                "",
                f"- First divergent step: {comparison.get('first_divergent_step')}",
                f"- Top-1 agreement: {comparison.get('top1_agreement_fraction')}",
                f"- Mean top-5 overlap: {comparison.get('mean_top5_overlap_fraction')}",
                f"- Finite max/mean absolute logit difference: "
                f"{comparison.get('max_abs_logit_difference_finite_values')} / "
                f"{comparison.get('mean_abs_logit_difference_finite_values')}",
                f"- Same codebook / source IDs / compressed IDs: "
                f"{comparison.get('same_codebook_sha256')} / "
                f"{comparison.get('same_original_prompt_token_ids')} / "
                f"{comparison.get('same_compressed_predictive_input_ids')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Fixed-KV Microbenchmark",
            "",
            f"Active KV length: {fixed['protocol']['active_kv_length']}; "
            f"warmups: {fixed['protocol']['warmup_steps']}; "
            f"measured forwards: {fixed['protocol']['measured_steps']}.",
            "A mutation probe determines whether each sample needs an independent cache copy; "
            "any copy is made outside the timed forward, and the reference cache is asserted "
            "to remain at 256 positions.",
            "",
            "| Condition | Next position | Mean ms | Median ms | P95 ms | Stddev ms | Min ms | Max ms | Steps/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        result = fixed["conditions"].get(condition)
        if result is None:
            continue
        lines.append(
            "| {condition} | {position} | {mean} | {median} | {p95} | {stddev} | {minimum} | {maximum} | {rate} |".format(
                condition=condition,
                position=result.get("next_position_id"),
                mean=result.get("mean_forward_ms"),
                median=result.get("median_forward_ms"),
                p95=result.get("p95_forward_ms"),
                stddev=result.get("stddev_forward_ms"),
                minimum=result.get("min_forward_ms"),
                maximum=result.get("max_forward_ms"),
                rate=result.get("forward_steps_per_second"),
            )
        )
    lines.extend(["", "### Derived Comparisons", "", "```json"])
    lines.append(json.dumps(fixed.get("derived_comparisons", {}), indent=2))
    lines.extend(["```", "", "### Request Setup (Outside Decode Timing)", "", "```json"])
    lines.append(
        json.dumps(
            {condition: value.get("request_setup", {}) for condition, value in fixed["conditions"].items()},
            indent=2,
        )
    )
    lines.extend(["```", ""])
    return "\n".join(lines)


def _build_validation_payload(
    *,
    args: argparse.Namespace,
    prompt: Dict[str, Any],
    records: Dict[str, Dict[str, Any]],
    smoke: Optional[Dict[str, Any]],
    original_ids: Sequence[int],
    device: torch.device,
    gpu_name: str,
    serialized_codebook: Optional[List[Dict[str, Any]]],
    codebook_sha256: Optional[str],
) -> Dict[str, Any]:
    behavioral_conditions: Dict[str, Dict[str, Any]] = {}
    fixed_kv_conditions: Dict[str, Dict[str, Any]] = {}
    for condition in CONDITIONS:
        if condition not in records:
            continue
        record = records[condition]
        behavioral_record = {
            key: value
            for key, value in record.items()
            if key not in {"fixed_kv_microbenchmark", "setup"}
        }
        behavioral_conditions[condition] = behavioral_record
        fixed_result = dict(record["fixed_kv_microbenchmark"])
        fixed_result["active_kv_length"] = FIXED_KV_CONTEXT_LENGTH
        fixed_result["request_setup"] = _request_setup_summary(record.get("setup", {}))
        fixed_kv_conditions[condition] = fixed_result

    return {
        "schema": "predictive_fast_path_validation_v2",
        "prompt_id": prompt["id"],
        "condition_order": [condition for condition in CONDITIONS if condition in records],
        "condition_labels": list(CONDITIONS),
        "predictive_codebook": serialized_codebook,
        "predictive_codebook_sha256": codebook_sha256,
        "behavioral_generation_smoke": {
            "cache_semantics": {
                "implementation": "static",
                "capacity": args.static_cache_capacity,
                "interpretation": "growing active KV length during autoregressive generation",
            },
            "timings_are_fixed_kv_measurements": False,
            "per_forward_cuda_timing_authoritative": False,
            "transformer_forward_timing": None,
            "authoritative_per_step_timing": "fixed_kv_microbenchmark",
            "conditions": behavioral_conditions,
            "legacy_vs_fast": smoke,
        },
        "fixed_kv_microbenchmark": {
            "protocol": {
                "active_kv_length": FIXED_KV_CONTEXT_LENGTH,
                "warmup_steps": FIXED_KV_WARMUP_ITERATIONS,
                "measured_steps": FIXED_KV_MEASURED_ITERATIONS,
                "cache_strategy": "copy outside timing if the mutation probe detects changes; otherwise reuse only after proving the reference remains unchanged",
                "cache_reconstruction_in_timed_region": False,
                "reference_cache_must_remain_at": FIXED_KV_CONTEXT_LENGTH,
            },
            "conditions": fixed_kv_conditions,
            "derived_comparisons": _derive_fixed_kv_comparisons(fixed_kv_conditions),
        },
        "original_prompt_token_ids": list(original_ids),
        "checkpoint": args.checkpoint,
        "checkpoint_step": 100,
        "base_model_revision": benchmark.DEFAULT_PHI_REVISION,
        "zip2zip_revision": benchmark.DEFAULT_ZIP2ZIP_REVISION,
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_validation_explicitly_authorized_by_cli": bool(args.allow_gpu),
        "static_cache_capacity": args.static_cache_capacity,
        "batch_size": 1,
        "emission_gate": None,
    }


def _write_partial_condition_result(
    *,
    output_dir: str | os.PathLike[str],
    condition: str,
    payload: Dict[str, Any],
) -> Tuple[Path, Path]:
    """Durably publish a completed condition and the latest aggregate snapshot."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    partial = dict(payload)
    partial["partial_result"] = {
        "is_partial": True,
        "completed_condition": condition,
        "completed_conditions": list(partial.get("condition_order", [])),
        "remaining_conditions": [
            name for name in CONDITIONS if name not in partial.get("condition_order", [])
        ],
    }
    condition_path = directory / f"partial_{condition}.json"
    aggregate_path = directory / "partial_results.json"
    write_json_atomic(condition_path, partial)
    write_json_atomic(aggregate_path, partial)
    return condition_path, aggregate_path


def execute(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    if not args.execute:
        raise RuntimeError("model execution requires the explicit --execute flag")
    if device.type != "cuda":
        raise RuntimeError("model execution requires the explicit --device cuda[:N] flag")
    if not args.allow_gpu:
        raise RuntimeError("model execution requires the explicit --allow-gpu acknowledgement")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    gpu_name = torch.cuda.get_device_name(device)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"the reviewed first GPU protocol requires one T4; selected {gpu_name!r}")
    torch.cuda.set_device(device)
    prompt = _load_prompt(args.prompt_id)
    os.makedirs(args.output_dir, exist_ok=True)
    environment_provenance = None
    if args.environment_provenance:
        provenance_path = Path(args.environment_provenance)
        if not provenance_path.is_file():
            raise FileNotFoundError(
                f"requested environment provenance file is missing: {provenance_path}"
            )
        environment_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    records: Dict[str, Any] = {}
    smoke = None
    original_ids: List[int] = []
    tokenizer = None
    serialized_codebook = None
    codebook_sha256 = None

    def persist_partial(
        condition: str,
        *,
        current_smoke: Optional[Dict[str, Any]],
        current_original_ids: Sequence[int],
        current_serialized_codebook: Optional[List[Dict[str, Any]]],
        current_codebook_sha256: Optional[str],
    ) -> None:
        serializable_records = {
            name: {
                key: value
                for key, value in record.items()
                if key not in {"generation_score_tensors", "generation_logit_tensors"}
            }
            for name, record in records.items()
        }
        partial_payload = _build_validation_payload(
            args=args,
            prompt=prompt,
            records=serializable_records,
            smoke=current_smoke,
            original_ids=current_original_ids,
            device=device,
            gpu_name=gpu_name,
            serialized_codebook=current_serialized_codebook,
            codebook_sha256=current_codebook_sha256,
        )
        partial_payload["environment_provenance"] = environment_provenance
        _write_partial_condition_result(
            output_dir=args.output_dir,
            condition=condition,
            payload=partial_payload,
        )

    # Keep execution/report order aligned with the review protocol while
    # releasing Vanilla before loading the predictive bundle.
    if "vanilla" in args.conditions:
        vanilla_record, tokenizer = _run_vanilla(args, prompt, device)
        if vanilla_record is not None:
            records["vanilla"] = vanilla_record
            original_ids = list(vanilla_record["original_prompt_token_ids"])
            persist_partial(
                "vanilla",
                current_smoke=None,
                current_original_ids=original_ids,
                current_serialized_codebook=None,
                current_codebook_sha256=None,
            )
    if any(condition.startswith("predictive_") for condition in args.conditions):

        def on_predictive_condition_complete(
            condition: str, record: Dict[str, Any], context: Dict[str, Any]
        ) -> None:
            records[condition] = record
            current_ids = context["original_prompt_token_ids"]
            if original_ids and original_ids != current_ids:
                raise RuntimeError(
                    "Vanilla and predictive conditions tokenized different source prompt IDs"
                )
            current_smoke = None
            if (
                "predictive_legacy_merged" in records
                and "predictive_fast_merged" in records
            ):
                current_smoke = _compare_smoke_runs(
                    records["predictive_legacy_merged"],
                    records["predictive_fast_merged"],
                )
            persist_partial(
                condition,
                current_smoke=current_smoke,
                current_original_ids=current_ids,
                current_serialized_codebook=context["serialized_codebook"],
                current_codebook_sha256=context["codebook_sha256"],
            )

        (
            predictive_records,
            smoke,
            predictive_original_ids,
            predictive_tokenizer,
            serialized_codebook,
            codebook_sha256,
        ) = _run_predictive_pair(
            args,
            prompt,
            device,
            on_condition_complete=on_predictive_condition_complete,
        )
        if original_ids and original_ids != list(predictive_original_ids):
            raise RuntimeError("Vanilla and predictive conditions tokenized different source prompt IDs")
        original_ids = list(predictive_original_ids)
        tokenizer = predictive_tokenizer
        records.update(predictive_records)

    vanilla = records.get("vanilla")
    if vanilla is not None:
        for condition, record in records.items():
            if condition == "vanilla":
                continue
            if record["original_prompt_token_ids_sha256"] != vanilla["original_prompt_token_ids_sha256"]:
                raise RuntimeError("Vanilla and predictive conditions used different source token IDs")
            record["output_length_ratio_vs_vanilla"] = (
                record["expanded_output_tokens"] / vanilla["expanded_output_tokens"]
                if vanilla["expanded_output_tokens"] else None
            )
            record["latency_ratio_vs_vanilla"] = (
                record["total_request_wall_time_s"] / vanilla["total_request_wall_time_s"]
                if vanilla["total_request_wall_time_s"] else None
            )
    run_name = f"{prompt['id']}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_path = Path(args.output_dir) / f"{run_name}.json"
    payload = _build_validation_payload(
        args=args,
        prompt=prompt,
        records=records,
        smoke=smoke,
        original_ids=original_ids,
        device=device,
        gpu_name=gpu_name,
        serialized_codebook=serialized_codebook,
        codebook_sha256=codebook_sha256,
    )
    payload["environment_provenance"] = environment_provenance
    write_json_atomic(output_path, payload)
    summary_path = output_path.with_suffix(".md")
    summary_path.write_text(_render_human_readable_summary(payload), encoding="utf-8")
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    if not args.execute:
        print(json.dumps(dry_run_report(args), indent=2))
        return 0
    result_path = execute(args)
    print(f"Fast-inference JSON saved to {result_path}")
    print(f"Human-readable summary saved to {result_path.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
