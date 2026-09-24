"""Explicitly gated real-Phi comparison for legacy and prepared fast inference.

The default is a local, non-loading dry run. Real model execution requires all
three flags: ``--execute --device cuda --allow-gpu``. This harness is separate
from ``run_quality_benchmark.py`` so that benchmark remains the legacy control.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from zip2zip import StaticCodebookManager, prepare_model_for_inference
from zip2zip.static_codebook import estimate_effective_table_memory
from experiments import run_quality_benchmark as benchmark


CONDITIONS = (
    "vanilla",
    "predictive_legacy_merged",
    "predictive_fast_merged",
)
DEFAULT_PROMPT_ID = "gsm_2956"
DEFAULT_MAX_NEW_TOKENS = 101
FIXED_KV_CACHE_LENGTH = 256
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
    parser.add_argument("--kv-cache-length", type=int, default=FIXED_KV_CACHE_LENGTH)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "experiments" / "checkpoints" / "fast_inference_validation"),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size != 1:
        raise ValueError("the predictive fast validation harness requires --batch-size 1")
    if args.kv_cache_length != FIXED_KV_CACHE_LENGTH:
        raise ValueError("the reviewed first GPU protocol fixes KV cache length at 256")
    if args.max_new_tokens < 2 or args.max_new_tokens > args.kv_cache_length:
        raise ValueError("max-new-tokens must be between 2 and the fixed KV cache length")
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
        "load pinned Step-100 predictive bundle and canonical predictor policy",
        "tokenize one source prompt once with the pinned Phi tokenizer",
        "select one K=32 codebook once and reuse it unchanged for legacy and fast",
        "merge LoRA explicitly; preserve predictor max_subtokens=3 and manager max_subtokens=4",
        "legacy mode: old prepare_prompt_input_ids -> model.generate -> manager.decode_sequence",
        "fast mode: prepare_inference_tables -> prepare_input_sequence -> generate -> decode_sequence",
        "detach the static manager after each request",
        "run Vanilla Phi separately to avoid holding two Phi models in T4 memory",
    ]
    assert lifecycle.index("legacy mode: old prepare_prompt_input_ids -> model.generate -> manager.decode_sequence") < lifecycle.index("fast mode: prepare_inference_tables -> prepare_input_sequence -> generate -> decode_sequence")
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
        "kv_cache_length": args.kv_cache_length,
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


def _install_forward_timers(model: torch.nn.Module, device: torch.device):
    calls: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []

    def before_forward(module, args, kwargs):
        call: Dict[str, Any] = {"started": time.perf_counter()}
        if device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            call["start_event"] = event
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args and torch.is_tensor(args[0]):
            input_ids = args[0]
        call["input_tokens"] = int(input_ids.shape[-1]) if torch.is_tensor(input_ids) else None
        pending.append(call)

    def after_forward(module, args, kwargs, output):
        call = pending.pop()
        if device.type == "cuda":
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            call["end_event"] = end_event
        else:
            call["duration_s"] = time.perf_counter() - call["started"]
        calls.append(call)

    pre_hook = model.register_forward_pre_hook(before_forward, with_kwargs=True)
    post_hook = model.register_forward_hook(after_forward, with_kwargs=True)
    return calls, (pre_hook, post_hook)


def _timer_summary(calls: Sequence[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        durations_ms = [
            float(call["start_event"].elapsed_time(call["end_event"]))
            for call in calls
            if "start_event" in call and "end_event" in call
        ]
    else:
        durations_ms = [float(call["duration_s"] * 1000) for call in calls]
    decode_ms = durations_ms[1:]
    if not decode_ms:
        return {
            "transformer_forward_calls": len(durations_ms),
            "decode_forward_calls": 0,
            "mean_decode_forward_ms": None,
            "median_decode_forward_ms": None,
            "p95_decode_forward_ms": None,
            "transformer_steps_per_second": None,
        }
    ordered = sorted(decode_ms)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "transformer_forward_calls": len(durations_ms),
        "decode_forward_calls": len(decode_ms),
        "mean_decode_forward_ms": statistics.fmean(decode_ms),
        "median_decode_forward_ms": statistics.median(decode_ms),
        "p95_decode_forward_ms": p95,
        "transformer_steps_per_second": 1000.0 / statistics.mean(decode_ms),
    }


def _generation_tensors_to_cpu(values) -> Tuple[torch.Tensor, ...]:
    if not values:
        return ()
    stacked = torch.stack(values).detach().cpu()
    return tuple(stacked[index] for index in range(stacked.shape[0]))


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
    if len(input_ids) + args.max_new_tokens > args.kv_cache_length:
        raise ValueError(
            f"prompt ({len(input_ids)}) + requested output ({args.max_new_tokens}) "
            f"exceeds fixed KV cache length {args.kv_cache_length}"
        )
    tensor_ids = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    timer_model = model.base_model if hasattr(model, "base_model") else model
    calls, hooks = _install_forward_timers(timer_model, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_started = time.perf_counter()
    timing_proc = benchmark.TimingLogitsProcessor(generation_started, static_mgr=manager)
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=tensor_ids,
                max_new_tokens=args.max_new_tokens,
                max_cache_len=args.kv_cache_length,
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
    finally:
        hooks[0].remove()
        hooks[1].remove()
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
    timing = _timer_summary(calls, device)
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
        "fixed_kv_cache_length": args.kv_cache_length,
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
        "cached_decode_forward_calls": timing["decode_forward_calls"],
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
        "runtime_step_timing": timing,
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
        "fixed_kv_cache_length",
    ):
        if legacy.get(shared_key) != fast.get(shared_key):
            raise RuntimeError(f"legacy/fast smoke comparison differs at {shared_key}")
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
        "smoke_pair": [legacy["condition"], fast["condition"]],
        "same_codebook_sha256": legacy.get("codebook_sha256") == fast.get("codebook_sha256"),
        "raw_decode_ids_identical": legacy_ids == fast_ids,
        "first_divergent_step": divergent,
        "aligned_score_steps_before_context_diverges": score_steps,
        "top1_agreement_fraction": float(top1_matches.mean()) if top1_matches is not None else None,
        "mean_top5_overlap_fraction": float(top5_overlaps.mean()) if top5_overlaps is not None else None,
        "max_abs_logit_difference_finite_values": float(flat_diffs.max()) if flat_diffs.numel() else None,
        "mean_abs_logit_difference_finite_values": float(flat_diffs.mean()) if flat_diffs.numel() else None,
        "interpretation": "FP16 comparison is diagnostic; bitwise logit identity is not required.",
    }


def _run_predictive_pair(args: argparse.Namespace, prompt: Dict[str, Any], device: torch.device):
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
    predictor_started = time.perf_counter()
    codebook, _predictor_meta = policy.select_codebook(original_ids)
    predictor_latency_s = time.perf_counter() - predictor_started
    if len(codebook) != MAX_CODEBOOK_SIZE:
        raise RuntimeError(
            f"representative validation requires exactly K=32 selected entries; got {len(codebook)}"
        )
    codebook_sha256 = hashlib.sha256(repr(sorted(codebook.items())).encode("utf-8")).hexdigest()
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
        "fixed_kv_cache_length": args.kv_cache_length,
    }
    prompt_metadata = {
        "id": prompt["id"],
        "domain": prompt.get("domain"),
        "prompt_text": prompt["prompt_text"],
        "generation_settings": generation_settings,
    }
    records = {}
    shared_predictive_input_ids = None
    for condition in ("predictive_legacy_merged", "predictive_fast_merged"):
        if condition not in args.conditions:
            continue
        manager_setup_started = time.perf_counter()
        manager = StaticCodebookManager(
            initial_vocab_size=benchmark.INITIAL_VOCAB,
            max_codebook_size=MAX_CODEBOOK_SIZE,
            max_subtokens=MANAGER_MAX_SUBTOKENS,
            embedding_dim=bundle["embedding_dim"],
            pad_token_id=bundle["pad_id"],
            disabled_ids=bundle["disabled_ids"],
        )
        manager.set_seeded_codebook(codebook, batch_size=1, device=device)
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
            records[condition] = record
            del _
        finally:
            manager.detach_from_model(model)
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
    return records, smoke, original_ids, tokenizer


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
            "fixed_kv_cache_length": args.kv_cache_length,
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
    record.pop("generation_score_tensors", None)
    record.pop("generation_logit_tensors", None)
    del _
    record["total_request_wall_time_s"] = record["generation_wall_time_s"]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return record, tokenizer


def execute(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    gpu_name = torch.cuda.get_device_name(device)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"the reviewed first GPU protocol requires one T4; selected {gpu_name!r}")
    torch.cuda.set_device(device)
    prompt = _load_prompt(args.prompt_id)
    if any(condition.startswith("predictive_") for condition in args.conditions):
        predictive_records, smoke, original_ids, tokenizer = _run_predictive_pair(args, prompt, device)
    else:
        predictive_records, smoke, original_ids, tokenizer = {}, None, None, None
    records: Dict[str, Any] = dict(predictive_records)
    if "vanilla" in args.conditions:
        vanilla_record, vanilla_tokenizer = _run_vanilla(args, prompt, device)
        if vanilla_record is not None:
            records["vanilla"] = vanilla_record
        if tokenizer is None:
            tokenizer = vanilla_tokenizer
            original_ids = vanilla_record["original_prompt_token_ids"] if vanilla_record else []

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
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = f"{prompt['id']}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_path = Path(args.output_dir) / f"{run_name}.json"
    payload = {
        "schema": "predictive_fast_path_validation_v1",
        "prompt_id": prompt["id"],
        "condition_order": [condition for condition in CONDITIONS if condition in records],
        "records": records,
        "legacy_vs_fast_smoke_comparison": smoke,
        "original_prompt_token_ids": list(original_ids),
        "checkpoint": args.checkpoint,
        "checkpoint_step": 100,
        "base_model_revision": benchmark.DEFAULT_PHI_REVISION,
        "zip2zip_revision": benchmark.DEFAULT_ZIP2ZIP_REVISION,
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_validation_explicitly_authorized_by_cli": bool(args.allow_gpu),
        "kv_cache_length": args.kv_cache_length,
        "batch_size": 1,
        "emission_gate": None,
    }
    with output_path.open("w", encoding="utf-8") as target:
        json.dump(payload, target, indent=2, ensure_ascii=False)
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
    print(f"Fast-inference validation saved to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
