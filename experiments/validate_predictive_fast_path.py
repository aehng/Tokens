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
import re
from collections.abc import Mapping
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
TIER1_12_PROMPTS_PATH = REPO_ROOT / "experiments" / "checkpoints" / "quality_benchmark" / "poc_12_prompt_ids.json"
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
    parser.add_argument(
        "--tier1-12",
        action="store_true",
        help="Run the canonical 12-prompt Tier-1 validation set.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size != 1:
        raise ValueError("the predictive fast validation harness requires --batch-size 1")
    if getattr(args, "tier1_12", False):
        if args.max_new_tokens == DEFAULT_MAX_NEW_TOKENS:
            args.max_new_tokens = 300
        if args.static_cache_capacity == STATIC_CACHE_CAPACITY:
            args.static_cache_capacity = 512
    if args.static_cache_capacity not in (256, 512):
        raise ValueError("the behavioral smoke protocol uses static cache capacity 256 or 512")
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


def _load_canonical_12_prompts() -> List[Dict[str, Any]]:
    tier1_path = TIER1_12_PROMPTS_PATH
    if not tier1_path.is_absolute():
        tier1_path = REPO_ROOT / tier1_path
    with tier1_path.open("r", encoding="utf-8") as f:
        meta_list = json.load(f)
    prompts = []
    for item in meta_list:
        p = _load_prompt(item["id"])
        prompts.append({**p, "poc_meta": item})
    return prompts


def extract_model_provenance(
    model: torch.nn.Module, tokenizer=None, bundle: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    base = getattr(model, "base_model", model)
    config = getattr(model, "config", getattr(base, "config", None))
    return {
        "model_class": type(model).__name__,
        "base_model_class": type(base).__name__,
        "config_model_type": getattr(config, "model_type", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "base_model_name_or_path": getattr(config, "_name_or_path", benchmark.PHI_MODEL_ID),
        "canonical_base_model_id": benchmark.PHI_MODEL_ID,
        "canonical_base_revision": benchmark.DEFAULT_PHI_REVISION,
        "canonical_zip2zip_revision": benchmark.DEFAULT_ZIP2ZIP_REVISION,
        "model_architecture_label": "Phi-3.5-mini-instruct",
        "provenance_note": (
            "Prior assistant mentions of 'Phi-1.5' were an assistant labeling error; "
            "the active model has always been Microsoft Phi-3.5-mini-instruct with Zip2Zip."
        ),
    }


def evaluate_quality_and_termination(
    record: Dict[str, Any],
    prompt_dict: Dict[str, Any],
) -> Dict[str, Any]:
    dom = prompt_dict.get("domain")
    output_text = record.get("output_text", "")
    eos_reached = record.get("eos_emitted", False)
    raw_ids = record.get("raw_decode_ids", [])

    eval_metrics: Dict[str, Any] = {}
    # Repetition check on all domains
    rep = benchmark.severe_repetition_metrics(output_text)
    eval_metrics.update(rep)

    first_answer_char_pos = None
    steps_to_answer = None
    post_answer_tail_steps = None

    if dom == "code":
        gt = prompt_dict.get("ground_truth_response", "")
        asserts = [line.strip() for line in gt.splitlines() if line.strip().startswith("assert")]
        code_eval = benchmark.evaluate_mbpp_code(output_text, asserts)
        eval_metrics.update(code_eval)
    elif dom == "reasoning":
        gt = prompt_dict.get("ground_truth_response", "")
        gsm_eval = benchmark.evaluate_gsm8k_reasoning(output_text, gt)
        eval_metrics.update(gsm_eval)
        extracted_answer = gsm_eval.get("extracted_answer")
        m = re.search(r"(?:####|\$\\boxed\{|\\boxed\{)", output_text)
        if m:
            first_answer_char_pos = m.start()
        elif extracted_answer and extracted_answer in output_text:
            first_answer_char_pos = output_text.find(extracted_answer)
        if first_answer_char_pos is not None:
            prefix_fraction = first_answer_char_pos / max(len(output_text), 1)
            steps_to_answer = int(round(prefix_fraction * len(raw_ids)))
            post_answer_tail_steps = max(0, len(raw_ids) - steps_to_answer)
    elif dom == "instruction":
        alp_eval = benchmark.evaluate_alpaca_instruction(output_text, eos_reached)
        eval_metrics.update(alp_eval)

    eval_metrics["first_answer_char_pos"] = first_answer_char_pos
    eval_metrics["steps_to_answer"] = steps_to_answer
    eval_metrics["post_answer_tail_steps"] = post_answer_tail_steps
    return eval_metrics


def compute_break_even_and_estimates(
    *,
    raw_iterations: int,
    expanded_base_tokens: int,
    vanilla_median_ms: float,
    legacy_median_ms: float,
    fast_median_ms: float,
    legacy_setup_ms: float = 0.0,
    fast_setup_ms: float = 0.0,
) -> Dict[str, Any]:
    comp_fraction = 1.0 - (raw_iterations / max(1, expanded_base_tokens))
    break_even_decode_ms = (
        vanilla_median_ms / (1.0 - comp_fraction) if (1.0 - comp_fraction) > 0 else None
    )
    vanilla_equiv_decode_ms = expanded_base_tokens * vanilla_median_ms
    legacy_decode_ms = raw_iterations * legacy_median_ms
    legacy_total_est_ms = legacy_decode_ms + legacy_setup_ms
    fast_decode_ms = raw_iterations * fast_median_ms
    fast_total_est_ms = fast_decode_ms + fast_setup_ms
    return {
        "compression_fraction": comp_fraction,
        "break_even_decode_ms": break_even_decode_ms,
        "vanilla_equivalent_decode_ms": vanilla_equiv_decode_ms,
        "legacy_decode_ms": legacy_decode_ms,
        "legacy_total_estimated_ms": legacy_total_est_ms,
        "fast_decode_ms": fast_decode_ms,
        "fast_total_estimated_ms": fast_total_est_ms,
        "fast_decode_speedup_vs_legacy_pct": (
            ((legacy_decode_ms - fast_decode_ms) / legacy_decode_ms * 100.0)
            if legacy_decode_ms > 0 else 0.0
        ),
        "fast_total_speedup_vs_legacy_pct": (
            ((legacy_total_est_ms - fast_total_est_ms) / legacy_total_est_ms * 100.0)
            if legacy_total_est_ms > 0 else 0.0
        ),
        "fast_decode_speedup_vs_vanilla_pct": (
            ((vanilla_equiv_decode_ms - fast_decode_ms) / vanilla_equiv_decode_ms * 100.0)
            if vanilla_equiv_decode_ms > 0 else 0.0
        ),
        "fast_total_speedup_vs_vanilla_pct": (
            ((vanilla_equiv_decode_ms - fast_total_est_ms) / vanilla_equiv_decode_ms * 100.0)
            if vanilla_equiv_decode_ms > 0 else 0.0
        ),
    }


def dry_run_report(args: argparse.Namespace) -> Dict[str, Any]:
    if getattr(args, "tier1_12", False):
        prompts = _load_canonical_12_prompts()
        prompt_info = {
            "tier1_12": True,
            "prompt_count": len(prompts),
            "prompts": [{"id": p["id"], "domain": p.get("domain")} for p in prompts],
            "prompt_id": "tier1_12_matrix",
            "source_prompt_characters": sum(len(p["prompt_text"]) for p in prompts),
            "source_prompt_sha256": hashlib.sha256(
                "".join(p["prompt_text"] for p in prompts).encode("utf-8")
            ).hexdigest(),
        }
    else:
        prompt = _load_prompt(args.prompt_id)
        prompt_info = {
            "tier1_12": False,
            "prompt_id": args.prompt_id,
            "source_prompt_sha256": hashlib.sha256(prompt["prompt_text"].encode("utf-8")).hexdigest(),
            "source_prompt_characters": len(prompt["prompt_text"]),
        }
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
        **prompt_info,
        "batch_size": args.batch_size,
        "device_requested": args.device,
        "checkpoint": args.checkpoint,
        "max_new_tokens": args.max_new_tokens,
        "protocols": {
            "behavioral_generation_smoke": {
                "cache_implementation": "static",
                "static_cache_capacity": args.static_cache_capacity,
                "disable_compile": True,
                "disable_compile_interpretation": "disables automatic JIT compilation in Transformers 5.17 static cache for identical eager baseline across all conditions",
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


def _validate_predictor_codebook(
    codebook: Mapping[Tuple[int, ...], int],
    *,
    initial_vocab_size: int,
    max_codebook_size: int,
    max_subtokens: int,
    disabled_ids: Sequence[int] = (),
) -> None:
    """Enforce CappedPredictorPolicy's phrase-tuple -> absolute-H-ID contract."""
    if not isinstance(codebook, dict):
        raise TypeError(
            "canonical predictor codebook must be the policy's dict mapping "
            "phrase tuples to absolute hypertoken IDs"
        )
    if len(codebook) != max_codebook_size:
        raise ValueError(
            f"predictor codebook must contain exactly K={max_codebook_size} entries; "
            f"got {len(codebook)}"
        )

    expected_hyper_ids = set(
        range(initial_vocab_size, initial_vocab_size + max_codebook_size)
    )
    observed_hyper_ids = set()
    observed_phrases = set()
    disabled = set(disabled_ids)
    for phrase, hyper_id in codebook.items():
        if not isinstance(phrase, tuple):
            raise ValueError(
                "canonical predictor codebook keys must be phrase tuples; expected "
                "Mapping[Tuple[int, ...], int] (phrase -> absolute H ID)"
            )
        if not 2 <= len(phrase) <= max_subtokens:
            raise ValueError(
                f"predictor phrase length must be between 2 and {max_subtokens}; "
                f"got {len(phrase)} for {phrase!r}"
            )
        if any(type(token_id) is not int for token_id in phrase):
            raise ValueError(f"predictor phrase IDs must be integers: {phrase!r}")
        if any(not 0 <= token_id < initial_vocab_size for token_id in phrase):
            raise ValueError(
                f"predictor phrase {phrase!r} contains an ID outside base vocabulary "
                f"[0, {initial_vocab_size})"
            )
        if any(token_id in disabled for token_id in phrase):
            raise ValueError(f"predictor phrase {phrase!r} contains a disabled token ID")
        if type(hyper_id) is not int:
            raise ValueError(
                f"predictor H ID for phrase {phrase!r} must be an integer; "
                f"got {hyper_id!r}"
            )
        if hyper_id not in expected_hyper_ids:
            raise ValueError(
                f"predictor H ID {hyper_id} is outside the required absolute range "
                f"[{initial_vocab_size}, {initial_vocab_size + max_codebook_size})"
            )
        if hyper_id in observed_hyper_ids:
            raise ValueError(f"predictor returned duplicate H ID {hyper_id}")
        if phrase in observed_phrases:
            raise ValueError(f"predictor returned duplicate phrase {phrase!r}")
        observed_hyper_ids.add(hyper_id)
        observed_phrases.add(phrase)

    if observed_hyper_ids != expected_hyper_ids:
        raise ValueError(
            "predictor H IDs must be unique and contiguous across the required "
            f"range [{initial_vocab_size}, {initial_vocab_size + max_codebook_size})"
        )


def _serialize_codebook(
    codebook: Mapping[Tuple[int, ...], int],
    *,
    disabled_ids: Sequence[int] = (),
) -> Tuple[List[Dict[str, Any]], str]:
    """Serialize the validated canonical phrase-tuple -> absolute-H-ID mapping."""
    _validate_predictor_codebook(
        codebook,
        initial_vocab_size=benchmark.INITIAL_VOCAB,
        max_codebook_size=MAX_CODEBOOK_SIZE,
        max_subtokens=PREDICTOR_MAX_SUBTOKENS,
        disabled_ids=disabled_ids,
    )
    serialized = [
        {
            "hyper_id": int(hyper_id),
            "subtoken_ids": [int(token_id) for token_id in phrase],
        }
        for phrase, hyper_id in sorted(codebook.items(), key=lambda item: int(item[1]))
    ]
    canonical = json.dumps(serialized, sort_keys=True, separators=(",", ":"))
    return serialized, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _predictive_condition_codebooks(
    codebook: Mapping[Tuple[int, ...], int], selected_conditions: Sequence[str]
) -> List[Tuple[str, Mapping[Tuple[int, ...], int]]]:
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
            disable_compile=True,
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
        "disable_compile": True,
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
        "behavioral_wall_time_is_authoritative": False,
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
        "disable_compile": True,
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
    bundle: Optional[Dict[str, Any]] = None,
):
    should_delete = False
    if bundle is None:
        bundle = benchmark.load_predictive_model_bundle(
            args.checkpoint,
            str(device),
            base_revision=benchmark.DEFAULT_PHI_REVISION,
            model_revision=benchmark.DEFAULT_ZIP2ZIP_REVISION,
            expected_step=100,
        )
        should_delete = True
        prepare_model_for_inference(bundle["model"], merge_lora=True)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    if getattr(policy, "max_subtokens", None) != PREDICTOR_MAX_SUBTOKENS:
        raise RuntimeError(
            "canonical predictor contract changed: expected max_subtokens=3, "
            f"got {getattr(policy, 'max_subtokens', None)!r}"
        )
    original_ids = tokenizer.encode(prompt["prompt_text"], add_special_tokens=False)
    if any(token_id >= benchmark.INITIAL_VOCAB for token_id in original_ids):
        raise ValueError(
            "legacy comparison prompt contains a tokenizer tail ID at/above the "
            "hypertoken insertion point; the legacy input path cannot safely map it"
        )
    codebook, _predictor_meta, predictor_latency_s = _select_codebook_once(
        policy, original_ids
    )
    codebook_disabled_ids = set(getattr(policy, "disabled_ids", ()))
    codebook_disabled_ids.update(bundle["disabled_ids"])
    serialized_codebook, codebook_sha256 = _serialize_codebook(
        codebook, disabled_ids=codebook_disabled_ids
    )
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
                vram_before = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                h_synth_start = time.perf_counter()
                manager.synthesize_hyper_vectors(model, batch_size=1)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                h_synth_ms = (time.perf_counter() - h_synth_start) * 1000.0
                vram_after = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
                peak_prepare = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                input_ids = benchmark.prepare_prompt_input_ids(
                    original_ids, manager, compress_prompt=True
                )
                if manager.decode_sequence(input_ids) != list(original_ids):
                    raise RuntimeError("legacy compressed prompt failed its round-trip")
                setup_metrics = {
                    "predictor_latency_s": predictor_latency_s,
                    "codebook_manager_setup_s": manager_setup_s,
                    "h_vector_synthesis_ms": h_synth_ms,
                    "effective_input_table_build_ms": None,
                    "effective_output_table_build_ms": None,
                    "total_table_preparation_ms": h_synth_ms,
                    "legacy_tables_prepared": False,
                    "fast_tables_prepared": False,
                    "vram_before_table_prepare_bytes": vram_before,
                    "vram_after_table_prepare_bytes": vram_after,
                    "peak_vram_during_table_prepare_bytes": peak_prepare,
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
                vram_before = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                manager.prepare_inference_tables(model, batch_size=1)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                vram_after = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
                peak_prepare = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                if device.type == "cuda":
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
    if should_delete:
        del bundle, model
        gc.collect()
        torch.cuda.empty_cache()
    return records, smoke, original_ids, tokenizer, serialized_codebook, codebook_sha256


def _run_vanilla(
    args: argparse.Namespace,
    prompt: Dict[str, Any],
    device: torch.device,
    model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[Any] = None,
):
    if "vanilla" not in args.conditions:
        return None, None
    should_delete = False
    if model is None or tokenizer is None:
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
        should_delete = True
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
    if should_delete:
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
        "model_provenance": {
            "canonical_base_model_id": benchmark.PHI_MODEL_ID,
            "canonical_base_revision": benchmark.DEFAULT_PHI_REVISION,
            "canonical_zip2zip_revision": benchmark.DEFAULT_ZIP2ZIP_REVISION,
            "model_architecture_label": "Phi-3.5-mini-instruct",
            "provenance_note": (
                "Prior assistant mentions of 'Phi-1.5' were an assistant labeling error; "
                "the active model has always been Microsoft Phi-3.5-mini-instruct with Zip2Zip."
            ),
        },
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


def _render_tier1_12_human_readable_summary(payload: Dict[str, Any]) -> str:
    agg = payload["aggregate_summary"]
    fixed = agg["fixed_kv_microbenchmark_summary"]
    q_sum = agg["quality_summary"]
    prompts = payload["prompts"]

    lines = [
        "# Tier-1 (12-Prompt) Authoritative Validation Report",
        "",
        f"- Device: `{payload['gpu_name']}` ({payload['device']})",
        f"- Checkpoint step: `{payload['checkpoint_step']}`",
        f"- Static cache capacity: `{payload['static_cache_capacity']}` (disable_compile=True)",
        f"- Fixed-KV Context Length: `{fixed['active_kv_length']}` (warmups: {fixed['warmup_iterations']}, measured: {fixed['measured_iterations']})",
        "",
        "## 1. Fixed-KV Microbenchmark (KV=256 Authoritative Decode Steps)",
        "",
        "| Condition | Median Forward (ms) | Overhead vs Vanilla (%) | Speedup vs Legacy (%) |",
        "|---|---:|---:|---:|",
        f"| Vanilla | {fixed['vanilla_median_ms']:.2f} ms | - | - |" if fixed.get("vanilla_median_ms") is not None else "| Vanilla | N/A | - | - |",
        f"| Predictive Legacy Merged | {fixed['legacy_median_ms']:.2f} ms | {fixed['legacy_overhead_vs_vanilla_pct']:+.2f}% | - |" if fixed.get("legacy_median_ms") is not None else "| Predictive Legacy Merged | N/A | - | - |",
        f"| Predictive Fast Merged | {fixed['fast_median_ms']:.2f} ms | {fixed['fast_overhead_vs_vanilla_pct']:+.2f}% | {fixed['fast_speedup_vs_legacy_pct']:+.2f}% |" if fixed.get("fast_median_ms") is not None else "| Predictive Fast Merged | N/A | - | - |",
        "",
        "## 2. 12-Prompt Aggregate Decode & Speedup Summary",
        "",
        f"- Total raw decode iterations: {agg['total_raw_decode_iterations']}",
        f"- Total expanded output tokens: {agg['total_expanded_output_tokens']}",
        f"- Net compression fraction: {agg['overall_compression_pct']}%",
        f"- Overall break-even decode latency: {agg['overall_break_even_decode_ms']:.2f} ms" if agg.get("overall_break_even_decode_ms") is not None else "- Overall break-even decode latency: N/A",
        f"- Total Vanilla equivalent decode time: {agg['total_vanilla_equivalent_decode_ms']:.1f} ms",
        f"- Total Legacy decode time: {agg['total_legacy_decode_ms']:.1f} ms (Total with setup: {agg['total_legacy_total_estimated_ms']:.1f} ms)",
        f"- Total Fast decode time: {agg['total_fast_decode_ms']:.1f} ms (Total with setup: {agg['total_fast_total_estimated_ms']:.1f} ms)",
        f"- Fast decode speedup vs Legacy: {agg['fast_decode_speedup_vs_legacy_pct']:+.2f}%",
        f"- Fast total speedup vs Legacy: {agg['fast_total_speedup_vs_legacy_pct']:+.2f}%",
        f"- Fast decode speedup vs Vanilla: {agg['fast_decode_speedup_vs_vanilla_pct']:+.2f}%",
        f"- Fast total speedup vs Vanilla: {agg['fast_total_speedup_vs_vanilla_pct']:+.2f}%",
        "",
        "## 3. Domain Quality & Termination Summary",
        "",
        "### MBPP Code Synthesis (4 prompts)",
        f"- Vanilla: {q_sum['mbpp']['vanilla_syntax_valid']}/{q_sum['mbpp']['count']} syntax valid, {q_sum['mbpp']['vanilla_problem_pass']}/{q_sum['mbpp']['count']} passed assertions",
        f"- Legacy: {q_sum['mbpp']['legacy_syntax_valid']}/{q_sum['mbpp']['count']} syntax valid, {q_sum['mbpp']['legacy_problem_pass']}/{q_sum['mbpp']['count']} passed assertions",
        f"- Fast: {q_sum['mbpp']['fast_syntax_valid']}/{q_sum['mbpp']['count']} syntax valid, {q_sum['mbpp']['fast_problem_pass']}/{q_sum['mbpp']['count']} passed assertions",
        "",
        "### GSM8K Math Reasoning (4 prompts)",
        f"- Vanilla: {q_sum['gsm8k']['vanilla_exact_correct']}/{q_sum['gsm8k']['count']} exact correct",
        f"- Legacy: {q_sum['gsm8k']['legacy_exact_correct']}/{q_sum['gsm8k']['count']} exact correct",
        f"- Fast: {q_sum['gsm8k']['fast_exact_correct']}/{q_sum['gsm8k']['count']} exact correct",
        "",
        "### Alpaca Instruction Following (4 prompts)",
        f"- Vanilla: {q_sum['alpaca']['vanilla_mechanical_pass']}/{q_sum['alpaca']['count']} mechanical pass",
        f"- Legacy: {q_sum['alpaca']['legacy_mechanical_pass']}/{q_sum['alpaca']['count']} mechanical pass",
        f"- Fast: {q_sum['alpaca']['fast_mechanical_pass']}/{q_sum['alpaca']['count']} mechanical pass",
        "",
        "## 4. Per-Prompt Matrix Details",
        "",
        "| Prompt ID | Domain | Raw / Exp Tokens | Compression | Vanilla ms | Fast ms | Speedup vs Vanilla |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for p in prompts:
        be = p.get("break_even_and_estimates", {})
        raw = (p.get("predictive_fast_merged") or p.get("predictive_legacy_merged") or {}).get("transformer_decode_iterations", "N/A")
        exp = (p.get("predictive_fast_merged") or p.get("predictive_legacy_merged") or {}).get("expanded_output_tokens", "N/A")
        comp = f"{be.get('compression_fraction', 0)*100.0:.1f}%" if be.get("compression_fraction") is not None else "N/A"
        v_ms = f"{be.get('vanilla_equivalent_decode_ms', 0):.1f}" if be.get("vanilla_equivalent_decode_ms") is not None else "N/A"
        f_ms = f"{be.get('fast_total_estimated_ms', 0):.1f}" if be.get("fast_total_estimated_ms") is not None else "N/A"
        spd = f"{be.get('fast_total_speedup_vs_vanilla_pct', 0):+.1f}%" if be.get("fast_total_speedup_vs_vanilla_pct") is not None else "N/A"
        lines.append(f"| {p['prompt_id']} | {p['domain']} | {raw} / {exp} | {comp} | {v_ms} | {f_ms} | {spd} |")

    lines.extend(["", "## 5. Model Provenance", "", "```json"])
    lines.append(json.dumps(payload.get("model_provenance", {}), indent=2))
    lines.extend(["```", ""])
    return "\n".join(lines)


def _execute_tier1_12(
    args: argparse.Namespace,
    device: torch.device,
    gpu_name: str,
    environment_provenance: Optional[Dict[str, Any]],
) -> Path:
    prompts = _load_canonical_12_prompts()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vanilla_results: Dict[str, Dict[str, Any]] = {}
    legacy_results: Dict[str, Dict[str, Any]] = {}
    fast_results: Dict[str, Dict[str, Any]] = {}
    smokes: Dict[str, Dict[str, Any]] = {}
    codebooks: Dict[str, Any] = {}
    original_ids_map: Dict[str, List[int]] = {}
    model_provenance: Optional[Dict[str, Any]] = None

    if "vanilla" in args.conditions:
        tokenizer = AutoTokenizer.from_pretrained(
            benchmark.PHI_MODEL_ID,
            revision=benchmark.DEFAULT_PHI_REVISION,
        )
        vanilla_model = AutoModelForCausalLM.from_pretrained(
            benchmark.PHI_MODEL_ID,
            revision=benchmark.DEFAULT_PHI_REVISION,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        model_provenance = extract_model_provenance(vanilla_model, tokenizer=tokenizer)
        for prompt in prompts:
            pid = prompt["id"]
            rec, _ = _run_vanilla(args, prompt, device, model=vanilla_model, tokenizer=tokenizer)
            rec["quality_and_termination"] = evaluate_quality_and_termination(rec, prompt)
            vanilla_results[pid] = rec
            original_ids_map[pid] = list(rec["original_prompt_token_ids"])
            partial_item = {
                "completed_prompt": pid,
                "completed_condition": "vanilla",
                "prompt_domain": prompt.get("domain"),
                "record": {k: v for k, v in rec.items() if k not in ("generation_score_tensors", "generation_logit_tensors")},
            }
            write_json_atomic(output_dir / f"partial_{pid}_vanilla.json", partial_item)
            write_json_atomic(output_dir / "partial_results.json", {
                "is_partial": True,
                "completed_prompts": list(vanilla_results.keys()),
                "last_condition": "vanilla",
                "environment_provenance": environment_provenance,
            })
        del vanilla_model
        gc.collect()
        torch.cuda.empty_cache()

    if any(c.startswith("predictive_") for c in args.conditions):
        bundle = benchmark.load_predictive_model_bundle(
            args.checkpoint,
            str(device),
            base_revision=benchmark.DEFAULT_PHI_REVISION,
            model_revision=benchmark.DEFAULT_ZIP2ZIP_REVISION,
            expected_step=100,
        )
        p_model = bundle["model"]
        p_tok = bundle["tokenizer"]
        prepare_model_for_inference(p_model, merge_lora=True)
        if model_provenance is None:
            model_provenance = extract_model_provenance(p_model, tokenizer=p_tok, bundle=bundle)

        for prompt in prompts:
            pid = prompt["id"]
            records, smoke, orig_ids, _, serialized_cb, cb_sha = _run_predictive_pair(
                args,
                prompt,
                device,
                bundle=bundle,
            )
            original_ids_map[pid] = list(orig_ids)
            codebooks[pid] = {"codebook": serialized_cb, "sha256": cb_sha}
            smokes[pid] = smoke
            for c_name, c_rec in records.items():
                c_rec["quality_and_termination"] = evaluate_quality_and_termination(c_rec, prompt)
                partial_item = {
                    "completed_prompt": pid,
                    "completed_condition": c_name,
                    "prompt_domain": prompt.get("domain"),
                    "record": {k: v for k, v in c_rec.items() if k not in ("generation_score_tensors", "generation_logit_tensors")},
                    "codebook_sha256": cb_sha,
                }
                write_json_atomic(output_dir / f"partial_{pid}_{c_name}.json", partial_item)
            if "predictive_legacy_merged" in records:
                legacy_results[pid] = records["predictive_legacy_merged"]
            if "predictive_fast_merged" in records:
                fast_results[pid] = records["predictive_fast_merged"]

            write_json_atomic(output_dir / "partial_results.json", {
                "is_partial": True,
                "completed_prompts": list(fast_results.keys() if fast_results else legacy_results.keys()),
                "last_condition": "predictive_fast_merged" if fast_results else "predictive_legacy_merged",
                "environment_provenance": environment_provenance,
            })
        del bundle, p_model
        gc.collect()
        torch.cuda.empty_cache()

    per_prompt_records = []
    for prompt in prompts:
        pid = prompt["id"]
        v_rec = vanilla_results.get(pid)
        l_rec = legacy_results.get(pid)
        f_rec = fast_results.get(pid)

        v_med = v_rec["fixed_kv_microbenchmark"]["median_forward_ms"] if v_rec else None
        l_med = l_rec["fixed_kv_microbenchmark"]["median_forward_ms"] if l_rec else None
        f_med = f_rec["fixed_kv_microbenchmark"]["median_forward_ms"] if f_rec else None
        raw_it = f_rec["transformer_decode_iterations"] if f_rec else (l_rec["transformer_decode_iterations"] if l_rec else None)
        exp_tok = f_rec["expanded_output_tokens"] if f_rec else (l_rec["expanded_output_tokens"] if l_rec else None)
        l_setup_ms = l_rec.get("setup", {}).get("total_table_preparation_ms", 0.0) if l_rec else 0.0
        f_setup_ms = f_rec.get("setup", {}).get("total_table_preparation_ms", 0.0) if f_rec else 0.0

        if v_med and l_med and f_med and raw_it and exp_tok:
            be_est = compute_break_even_and_estimates(
                raw_iterations=raw_it,
                expanded_base_tokens=exp_tok,
                vanilla_median_ms=v_med,
                legacy_median_ms=l_med,
                fast_median_ms=f_med,
                legacy_setup_ms=l_setup_ms,
                fast_setup_ms=f_setup_ms,
            )
        else:
            be_est = {}

        prompt_summary = {
            "prompt_id": pid,
            "domain": prompt.get("domain"),
            "poc_meta": prompt.get("poc_meta"),
            "original_prompt_token_count": len(original_ids_map.get(pid, [])),
            "codebook_sha256": codebooks.get(pid, {}).get("sha256"),
            "vanilla": {k: v for k, v in v_rec.items() if k not in ("generation_score_tensors", "generation_logit_tensors")} if v_rec else None,
            "predictive_legacy_merged": {k: v for k, v in l_rec.items() if k not in ("generation_score_tensors", "generation_logit_tensors")} if l_rec else None,
            "predictive_fast_merged": {k: v for k, v in f_rec.items() if k not in ("generation_score_tensors", "generation_logit_tensors")} if f_rec else None,
            "smoke_comparison": smokes.get(pid),
            "break_even_and_estimates": be_est,
        }
        per_prompt_records.append(prompt_summary)

    total_raw = sum(
        (fast_results.get(p["prompt_id"]) or legacy_results.get(p["prompt_id"]) or {}).get("transformer_decode_iterations", 0)
        for p in per_prompt_records
    )
    total_expanded = sum(
        (fast_results.get(p["prompt_id"]) or legacy_results.get(p["prompt_id"]) or {}).get("expanded_output_tokens", 0)
        for p in per_prompt_records
    )
    total_vanilla_equiv_ms = sum(p["break_even_and_estimates"].get("vanilla_equivalent_decode_ms", 0.0) for p in per_prompt_records)
    total_legacy_decode_ms = sum(p["break_even_and_estimates"].get("legacy_decode_ms", 0.0) for p in per_prompt_records)
    total_fast_decode_ms = sum(p["break_even_and_estimates"].get("fast_decode_ms", 0.0) for p in per_prompt_records)
    total_legacy_setup_ms = sum(p.get("predictive_legacy_merged", {}).get("setup", {}).get("total_table_preparation_ms", 0.0) or 0.0 for p in per_prompt_records)
    total_fast_setup_ms = sum(p.get("predictive_fast_merged", {}).get("setup", {}).get("total_table_preparation_ms", 0.0) or 0.0 for p in per_prompt_records)
    total_legacy_total_ms = total_legacy_decode_ms + total_legacy_setup_ms
    total_fast_total_ms = total_fast_decode_ms + total_fast_setup_ms

    overall_compression = 1.0 - (total_raw / max(1, total_expanded))
    overall_break_even_decode_ms = (total_vanilla_equiv_ms / max(1, total_raw)) if total_raw else None

    mbpp_prompts = [p for p in per_prompt_records if p["domain"] == "code"]
    gsm_prompts = [p for p in per_prompt_records if p["domain"] == "reasoning"]
    alpaca_prompts = [p for p in per_prompt_records if p["domain"] == "instruction"]

    quality_summary = {
        "mbpp": {
            "count": len(mbpp_prompts),
            "vanilla_syntax_valid": sum(1 for p in mbpp_prompts if p.get("vanilla", {}).get("quality_and_termination", {}).get("syntax_valid")),
            "vanilla_problem_pass": sum(1 for p in mbpp_prompts if p.get("vanilla", {}).get("quality_and_termination", {}).get("problem_pass")),
            "legacy_syntax_valid": sum(1 for p in mbpp_prompts if p.get("predictive_legacy_merged", {}).get("quality_and_termination", {}).get("syntax_valid")),
            "legacy_problem_pass": sum(1 for p in mbpp_prompts if p.get("predictive_legacy_merged", {}).get("quality_and_termination", {}).get("problem_pass")),
            "fast_syntax_valid": sum(1 for p in mbpp_prompts if p.get("predictive_fast_merged", {}).get("quality_and_termination", {}).get("syntax_valid")),
            "fast_problem_pass": sum(1 for p in mbpp_prompts if p.get("predictive_fast_merged", {}).get("quality_and_termination", {}).get("problem_pass")),
        },
        "gsm8k": {
            "count": len(gsm_prompts),
            "vanilla_exact_correct": sum(1 for p in gsm_prompts if p.get("vanilla", {}).get("quality_and_termination", {}).get("exact_correct")),
            "legacy_exact_correct": sum(1 for p in gsm_prompts if p.get("predictive_legacy_merged", {}).get("quality_and_termination", {}).get("exact_correct")),
            "fast_exact_correct": sum(1 for p in gsm_prompts if p.get("predictive_fast_merged", {}).get("quality_and_termination", {}).get("exact_correct")),
        },
        "alpaca": {
            "count": len(alpaca_prompts),
            "vanilla_mechanical_pass": sum(1 for p in alpaca_prompts if p.get("vanilla", {}).get("quality_and_termination", {}).get("mechanical_instruction_pass")),
            "legacy_mechanical_pass": sum(1 for p in alpaca_prompts if p.get("predictive_legacy_merged", {}).get("quality_and_termination", {}).get("mechanical_instruction_pass")),
            "fast_mechanical_pass": sum(1 for p in alpaca_prompts if p.get("predictive_fast_merged", {}).get("quality_and_termination", {}).get("mechanical_instruction_pass")),
        },
    }

    v_medians = [p.get("vanilla", {}).get("fixed_kv_microbenchmark", {}).get("median_forward_ms") for p in per_prompt_records if p.get("vanilla")]
    l_medians = [p.get("predictive_legacy_merged", {}).get("fixed_kv_microbenchmark", {}).get("median_forward_ms") for p in per_prompt_records if p.get("predictive_legacy_merged")]
    f_medians = [p.get("predictive_fast_merged", {}).get("fixed_kv_microbenchmark", {}).get("median_forward_ms") for p in per_prompt_records if p.get("predictive_fast_merged")]

    aggregate_summary = {
        "prompt_count": len(prompts),
        "total_raw_decode_iterations": total_raw,
        "total_expanded_output_tokens": total_expanded,
        "overall_compression_fraction": overall_compression,
        "overall_compression_pct": round(overall_compression * 100.0, 2),
        "total_vanilla_equivalent_decode_ms": total_vanilla_equiv_ms,
        "total_legacy_decode_ms": total_legacy_decode_ms,
        "total_fast_decode_ms": total_fast_decode_ms,
        "total_legacy_setup_ms": total_legacy_setup_ms,
        "total_fast_setup_ms": total_fast_setup_ms,
        "total_legacy_total_estimated_ms": total_legacy_total_ms,
        "total_fast_total_estimated_ms": total_fast_total_ms,
        "overall_break_even_decode_ms": overall_break_even_decode_ms,
        "fast_decode_speedup_vs_legacy_pct": ((total_legacy_decode_ms - total_fast_decode_ms) / total_legacy_decode_ms * 100.0) if total_legacy_decode_ms > 0 else 0.0,
        "fast_total_speedup_vs_legacy_pct": ((total_legacy_total_ms - total_fast_total_ms) / total_legacy_total_ms * 100.0) if total_legacy_total_ms > 0 else 0.0,
        "fast_decode_speedup_vs_vanilla_pct": ((total_vanilla_equiv_ms - total_fast_decode_ms) / total_vanilla_equiv_ms * 100.0) if total_vanilla_equiv_ms > 0 else 0.0,
        "fast_total_speedup_vs_vanilla_pct": ((total_vanilla_equiv_ms - total_fast_total_ms) / total_vanilla_equiv_ms * 100.0) if total_vanilla_equiv_ms > 0 else 0.0,
        "fixed_kv_microbenchmark_summary": {
            "active_kv_length": FIXED_KV_CONTEXT_LENGTH,
            "warmup_iterations": FIXED_KV_WARMUP_ITERATIONS,
            "measured_iterations": FIXED_KV_MEASURED_ITERATIONS,
            "vanilla_median_ms": statistics.median(v_medians) if v_medians else None,
            "legacy_median_ms": statistics.median(l_medians) if l_medians else None,
            "fast_median_ms": statistics.median(f_medians) if f_medians else None,
            "fast_overhead_vs_vanilla_pct": ((statistics.median(f_medians) / statistics.median(v_medians) - 1.0) * 100.0) if v_medians and f_medians else None,
            "legacy_overhead_vs_vanilla_pct": ((statistics.median(l_medians) / statistics.median(v_medians) - 1.0) * 100.0) if v_medians and l_medians else None,
            "fast_speedup_vs_legacy_pct": ((statistics.median(l_medians) - statistics.median(f_medians)) / statistics.median(l_medians) * 100.0) if l_medians and f_medians else None,
        },
        "quality_summary": quality_summary,
    }

    tier1_payload = {
        "schema": "predictive_fast_path_tier1_12_v1",
        "benchmark_type": "tier1_12_validation",
        "checkpoint": args.checkpoint,
        "checkpoint_step": 100,
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_validation_explicitly_authorized_by_cli": bool(args.allow_gpu),
        "static_cache_capacity": args.static_cache_capacity,
        "batch_size": 1,
        "conditions": list(args.conditions),
        "model_provenance": model_provenance,
        "environment_provenance": environment_provenance,
        "aggregate_summary": aggregate_summary,
        "prompts": per_prompt_records,
    }

    run_name = f"tier1_12_{time.strftime('%Y%m%d_%H%M%S')}"
    output_path = output_dir / f"{run_name}.json"
    write_json_atomic(output_path, tier1_payload)
    summary_path = output_path.with_suffix(".md")
    summary_path.write_text(_render_tier1_12_human_readable_summary(tier1_payload), encoding="utf-8")
    return output_path


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
    os.makedirs(args.output_dir, exist_ok=True)
    environment_provenance = None
    if args.environment_provenance:
        provenance_path = Path(args.environment_provenance)
        if not provenance_path.is_file():
            raise FileNotFoundError(
                f"requested environment provenance file is missing: {provenance_path}"
            )
        environment_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if getattr(args, "tier1_12", False):
        return _execute_tier1_12(args, device, gpu_name, environment_provenance)
    prompt = _load_prompt(args.prompt_id)
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
