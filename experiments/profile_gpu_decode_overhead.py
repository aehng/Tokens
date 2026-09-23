"""Controlled GPU runtime decomposition & matched 12-prompt benchmark harness.

Profiles and isolates the per-decode-step overhead of Predictive Zip2Zip:
1. Active PEFT/LoRA matmuls (Predictive unmerged vs Predictive merged)
2. Custom position handling (StaticCodebookManager direct position_ids forward vs prepared fallback)
3. HyperEmbedding overhead (direct nn.Embedding forward with no CUDA data branching)
4. HyperLinear / hypertoken-logit overhead (H-logit scoring bypass)
5. Logits processors / masking overhead (short fixed-trajectory generate-loop benchmark)
6. H-token decode step vs ordinary base-token decode step cost
7. Short torch.profiler captures looking for aten::any, aten::_local_scalar_dense, cudaDeviceSynchronize
8. Matched 12-prompt Vanilla vs Predictive unmerged vs Predictive merged benchmark
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments import run_quality_benchmark as benchmark
from experiments.benchmark_provenance import (
    build_generation_cache_key,
    canonical_sha256,
    file_sha256,
    resolve_tested_commit,
    select_prompt_subset,
    write_json_atomic,
)
from experiments.load_joint_checkpoint import load_joint_checkpoint
from experiments.load_oracle_predictor import load_oracle_predictor
from experiments.mbpp_prompt import build_mbpp_prompt
from experiments.runtime_diagnostics import (
    add_runtime_derived_metrics,
    first_deterministic_answer_position,
    first_repeated_trigram_position,
)
from zip2zip import ContextualEmissionGate, StaticCodebookManager, Zip2ZipModel
from zip2zip.nn.embedding import HyperEmbedding
from zip2zip.nn.linear import HyperLinear
from zip2zip.predictor_policy import CappedPredictorPolicy
from zip2zip.static_codebook import StaticCodebookLogitsWarper


def _cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _compute_stats(samples_ms: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(float(v) for v in samples_ms)
    n = len(ordered)
    if n == 0:
        return {"mean": 0.0, "median": 0.0, "p50": 0.0, "p95": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0, "samples_count": 0}
    mean_val = sum(ordered) / n
    std_val = statistics.stdev(ordered) if n > 1 else 0.0
    p50_idx = int(0.50 * (n - 1))
    p95_idx = int(0.95 * (n - 1))
    return {
        "mean": round(mean_val, 4),
        "median": round(ordered[p50_idx], 4),
        "p50": round(ordered[p50_idx], 4),
        "p95": round(ordered[p95_idx], 4),
        "stddev": round(std_val, 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
        "samples_count": n,
    }


def _clone_kv_cache(kv: Any) -> Any:
    """Deep clone past_key_values cache to enforce immutable reference KV across iterations."""
    if kv is None:
        return None
    if isinstance(kv, tuple):
        return tuple(
            tuple(t.clone() for t in layer) if isinstance(layer, tuple) else [t.clone() for t in layer]
            for layer in kv
        )
    if isinstance(kv, list):
        return [
            [t.clone() for t in layer] if isinstance(layer, list) else tuple(t.clone() for t in layer)
            for layer in kv
        ]
    if hasattr(kv, "clone"):
        return kv.clone()
    raise TypeError(f"Unsupported past_key_values cache type for cloning: {type(kv)}")


def _get_kv_cache_seq_len(kv: Any) -> int:
    """Inspect sequence length from KV cache."""
    if kv is None:
        return 0
    try:
        if isinstance(kv, (tuple, list)) and len(kv) > 0:
            first_layer = kv[0]
            if isinstance(first_layer, (tuple, list)) and len(first_layer) > 0:
                return int(first_layer[0].shape[-2])
    except Exception:
        pass
    return 0


def _measure_fixed_kv_decode_steps_cuda_events(
    model: nn.Module,
    input_ids: torch.Tensor,
    reference_past_key_values: Any,
    device: torch.device,
    warmup_steps: int = 20,
    measured_steps: int = 100,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[List[float], int, int]:
    """Measure single-token decode steps consuming the exact same reference KV cache every step."""
    _cuda_sync_if_needed(device)
    initial_seq_len = _get_kv_cache_seq_len(reference_past_key_values)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_steps):
            curr_kv = _clone_kv_cache(reference_past_key_values)
            kwargs: Dict[str, Any] = {"use_cache": True}
            if position_ids is not None:
                kwargs["position_ids"] = position_ids
            model(input_ids, past_key_values=curr_kv, **kwargs)

    _cuda_sync_if_needed(device)

    # Measured
    samples_ms: List[float] = []
    final_seq_len = initial_seq_len
    with torch.no_grad():
        for _ in range(measured_steps):
            curr_kv = _clone_kv_cache(reference_past_key_values)
            kwargs = {"use_cache": True}
            if position_ids is not None:
                kwargs["position_ids"] = position_ids

            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                t0 = time.perf_counter()

            out = model(input_ids, past_key_values=curr_kv, **kwargs)

            if device.type == "cuda":
                end_event.record()
                _cuda_sync_if_needed(device)
                samples_ms.append(start_event.elapsed_time(end_event))
            else:
                samples_ms.append((time.perf_counter() - t0) * 1000.0)

            final_seq_len = _get_kv_cache_seq_len(out.past_key_values)

    return samples_ms, initial_seq_len, final_seq_len


@contextlib.contextmanager
def hyper_embedding_fast_path_context(model: Zip2ZipModel):
    """Diagnostic bypass: calls underlying nn.Embedding forward with NO CUDA data branching.

    Intended strictly for base-token input streams during microbenchmarks.
    """
    embedding_layer = model.base_model.get_input_embeddings()
    if not isinstance(embedding_layer, HyperEmbedding):
        yield
        return

    original_forward = embedding_layer.forward

    def direct_embedding_forward(self, input: torch.Tensor) -> torch.Tensor:
        # Call nn.Embedding.forward directly on base embedding weights without checking is_hyper on GPU
        return nn.Embedding.forward(self, input)

    embedding_layer.forward = MethodType(direct_embedding_forward, embedding_layer)
    try:
        yield
    finally:
        embedding_layer.forward = original_forward


@contextlib.contextmanager
def hyper_linear_bypassed_context(model: Zip2ZipModel):
    """Diagnostic bypass: skips hyper_logits calculation and cat, returning base_logits directly."""
    linear_layer = model.base_model.get_output_embeddings()
    if not isinstance(linear_layer, HyperLinear):
        yield
        return

    original_forward = linear_layer.forward

    def base_only_forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.detach() if getattr(self, "detach_input", False) else x
        return super(HyperLinear, self).forward(h)

    linear_layer.forward = MethodType(base_only_forward, linear_layer)
    try:
        yield
    finally:
        linear_layer.forward = original_forward


class PrepareInputIdsCallCounter:
    """Context manager to instrument and count calls to StaticCodebookManager.prepare_input_ids."""
    def __init__(self, manager: StaticCodebookManager) -> None:
        self.manager = manager
        self.call_count = 0
        self.original_prepare = manager.prepare_input_ids

    def __enter__(self) -> "PrepareInputIdsCallCounter":
        self.call_count = 0
        counter = self

        def counting_prepare(mgr_self, *args, **kwargs):
            counter.call_count += 1
            return counter.original_prepare(*args, **kwargs)

        self.manager.prepare_input_ids = MethodType(counting_prepare, self.manager)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.manager.prepare_input_ids = self.original_prepare


def run_single_step_microbenchmarks(
    model: nn.Module,
    device: torch.device,
    condition_name: str,
    kv_lengths: Sequence[int] = (64, 256, 512),
    warmup_steps: int = 20,
    measured_steps: int = 100,
    seed_token_id: int = 1000,
    bypass_position_ids: bool = False,
    instrumented_static_manager: Optional[StaticCodebookManager] = None,
) -> Dict[str, Any]:
    """Run controlled single-token fixed-KV cached decode microbenchmark across KV lengths."""
    results: Dict[str, Any] = {}
    model.eval()

    for kv_len in kv_lengths:
        dummy_prompt = torch.full((1, kv_len), seed_token_id, dtype=torch.long, device=device)
        with torch.no_grad():
            prefill_out = model(dummy_prompt, use_cache=True)
            ref_past_kv = prefill_out.past_key_values

        decode_input = torch.tensor([[seed_token_id]], dtype=torch.long, device=device)
        explicit_pos_ids = None
        if bypass_position_ids:
            # Explicitly supply sequential position_id = kv_len so Zip2ZipModel.forward bypasses prepare_input_ids
            explicit_pos_ids = torch.tensor([[kv_len]], dtype=torch.long, device=device)

        call_counter = None
        if instrumented_static_manager is not None:
            call_counter = PrepareInputIdsCallCounter(instrumented_static_manager)
            call_counter.__enter__()

        try:
            samples_ms, init_seq_len, post_seq_len = _measure_fixed_kv_decode_steps_cuda_events(
                model=model,
                input_ids=decode_input,
                reference_past_key_values=ref_past_kv,
                device=device,
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
                position_ids=explicit_pos_ids,
            )
        finally:
            if call_counter is not None:
                call_counter.__exit__(None, None, None)

        stats = _compute_stats(samples_ms)
        stats["kv_cache_initial_seq_len"] = init_seq_len
        stats["kv_cache_post_step_seq_len"] = post_seq_len
        if call_counter is not None:
            stats["prepare_input_ids_call_count"] = call_counter.call_count

        results[str(kv_len)] = {
            "kv_length": kv_len,
            "condition": condition_name,
            **stats,
        }
        prep_str = f" | prepare_calls={call_counter.call_count}" if call_counter is not None else ""
        print(f"  [Microbench] {condition_name:32s} | KV={kv_len:4d} | Median: {stats['median']:6.2f} ms | p95: {stats['p95']:6.2f} ms{prep_str}", flush=True)

    return results


def run_h_token_vs_base_token_microbenchmark(
    predictive_model: Zip2ZipModel,
    static_manager: StaticCodebookManager,
    device: torch.device,
    kv_lengths: Sequence[int] = (64, 256, 512),
    warmup_steps: int = 20,
    measured_steps: int = 100,
    base_token_id: int = 1000,
) -> Dict[str, Any]:
    """Compare per-step cost when input token is base token vs seeded H-token under exact same fixed KV."""
    results: Dict[str, Any] = {}
    predictive_model.eval()

    seeded_h_id = next(iter(static_manager.hyper_to_subtokens.keys()))

    for kv_len in kv_lengths:
        dummy_prompt = torch.full((1, kv_len), base_token_id, dtype=torch.long, device=device)
        with torch.no_grad():
            prefill_out = predictive_model(dummy_prompt, use_cache=True)
            ref_past_kv = prefill_out.past_key_values

        base_input = torch.tensor([[base_token_id]], dtype=torch.long, device=device)
        base_samples, _, _ = _measure_fixed_kv_decode_steps_cuda_events(
            model=predictive_model,
            input_ids=base_input,
            reference_past_key_values=ref_past_kv,
            device=device,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
        )
        base_stats = _compute_stats(base_samples)

        h_input = torch.tensor([[seeded_h_id]], dtype=torch.long, device=device)
        h_samples, _, _ = _measure_fixed_kv_decode_steps_cuda_events(
            model=predictive_model,
            input_ids=h_input,
            reference_past_key_values=ref_past_kv,
            device=device,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
        )
        h_stats = _compute_stats(h_samples)

        delta_ms = round(h_stats["median"] - base_stats["median"], 4)
        results[str(kv_len)] = {
            "kv_length": kv_len,
            "base_token_step": base_stats,
            "h_token_step": h_stats,
            "h_minus_base_median_ms": delta_ms,
        }
        print(f"  [H vs Base] KV={kv_len:4d} | Base: {base_stats['median']:6.2f} ms | H-step: {h_stats['median']:6.2f} ms | Delta: {delta_ms:+6.2f} ms", flush=True)

    return results


def run_logits_mask_generate_benchmark(
    model: Zip2ZipModel,
    static_manager: StaticCodebookManager,
    tok: AutoTokenizer,
    device: torch.device,
    prompt_text: str = "Solve this arithmetic: 25 * 14 = ",
    gen_steps: int = 30,
) -> Dict[str, Any]:
    """Microbenchmark comparing generate loop with StaticCodebookLogitsWarper vs masking bypassed."""
    model.eval()
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # 1. Normal masked generate
    mask_proc = static_manager.get_logits_processor()
    _cuda_sync_if_needed(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_masked = model.generate(
            input_ids=input_tensor,
            max_new_tokens=gen_steps,
            min_new_tokens=gen_steps,
            logits_processor=LogitsProcessorList([mask_proc]),
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    _cuda_sync_if_needed(device)
    masked_time_s = time.perf_counter() - t0

    # 2. Masking bypassed generate (no logits warper)
    _cuda_sync_if_needed(device)
    t1 = time.perf_counter()
    with torch.no_grad():
        out_bypassed = model.generate(
            input_ids=input_tensor,
            max_new_tokens=gen_steps,
            min_new_tokens=gen_steps,
            logits_processor=LogitsProcessorList([]),
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    _cuda_sync_if_needed(device)
    bypassed_time_s = time.perf_counter() - t1

    masked_ids = out_masked[0, len(prompt_ids):].tolist()
    bypassed_ids = out_bypassed[0, len(prompt_ids):].tolist()
    trajectory_match = (masked_ids == bypassed_ids)

    ms_per_step_masked = (masked_time_s / gen_steps) * 1000.0
    ms_per_step_bypassed = (bypassed_time_s / gen_steps) * 1000.0
    masking_delta_ms = ms_per_step_masked - ms_per_step_bypassed

    res = {
        "gen_steps": gen_steps,
        "masked_total_s": round(masked_time_s, 4),
        "bypassed_total_s": round(bypassed_time_s, 4),
        "masked_ms_per_step": round(ms_per_step_masked, 4),
        "bypassed_ms_per_step": round(ms_per_step_bypassed, 4),
        "masking_overhead_ms_per_step": round(masking_delta_ms, 4),
        "trajectory_match": trajectory_match,
    }
    print(f"  [Logits Mask Generate] Masked: {ms_per_step_masked:.2f} ms/step | Bypassed: {ms_per_step_bypassed:.2f} ms/step | Delta: {masking_delta_ms:+.2f} ms/step (match={trajectory_match})", flush=True)
    return res


def run_calibration_check(
    control_model: nn.Module,
    device: torch.device,
    label: str,
    kv_len: int = 128,
    warmup_steps: int = 10,
    measured_steps: int = 30,
) -> Dict[str, Any]:
    """Short fixed cached-decode microbenchmark on cuda:1 Vanilla control."""
    control_model.eval()
    dummy_prompt = torch.full((1, kv_len), 1000, dtype=torch.long, device=device)
    with torch.no_grad():
        prefill_out = control_model(dummy_prompt, use_cache=True)
        past_kv = prefill_out.past_key_values

    decode_input = torch.tensor([[1000]], dtype=torch.long, device=device)
    samples_ms, _, _ = _measure_fixed_kv_decode_steps_cuda_events(
        model=control_model,
        input_ids=decode_input,
        reference_past_key_values=past_kv,
        device=device,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
    )
    stats = _compute_stats(samples_ms)
    stats["label"] = label
    stats["timestamp_utc"] = time.time()
    print(f"  [Control Calibration: {label}] Median: {stats['median']:.3f} ms | p95: {stats['p95']:.3f} ms", flush=True)
    return stats


def run_torch_profiler_capture(
    model: nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any,
    device: torch.device,
    profile_steps: int = 15,
) -> Dict[str, Any]:
    """Capture a short torch.profiler trace of warmed decode steps."""
    from torch.profiler import ProfilerActivity, profile, record_function

    _cuda_sync_if_needed(device)

    # Warmup
    curr_kv = _clone_kv_cache(past_key_values)
    with torch.no_grad():
        for _ in range(5):
            out = model(input_ids, past_key_values=curr_kv, use_cache=True)
            curr_kv = out.past_key_values
    _cuda_sync_if_needed(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, profile_memory=False) as prof:
        with torch.no_grad():
            for step_i in range(profile_steps):
                curr_kv = _clone_kv_cache(past_key_values)
                with record_function(f"decode_step_{step_i}"):
                    model(input_ids, past_key_values=curr_kv, use_cache=True)
        _cuda_sync_if_needed(device)

    key_averages = prof.key_averages()
    table_str = key_averages.table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=30)

    suspect_names = [
        "aten::any",
        "aten::_local_scalar_dense",
        "cudaDeviceSynchronize",
        "aten::cat",
        "aten::where",
        "aten::cumsum",
        "aten::bmm",
        "aten::embedding",
    ]
    suspect_summary: Dict[str, Any] = {}
    for evt in key_averages:
        if any(s in evt.key for s in suspect_names):
            suspect_summary[evt.key] = {
                "count": evt.count,
                "cpu_time_ms": round(evt.cpu_time_total / 1000.0, 3),
                "cuda_time_ms": round(evt.cuda_time_total / 1000.0, 3) if hasattr(evt, "cuda_time_total") else 0.0,
            }

    cuda_kernel_count = sum(evt.count for evt in key_averages if getattr(evt, "is_async", False) or getattr(evt, "cuda_time_total", 0) > 0)
    return {
        "profile_steps": profile_steps,
        "table_summary": table_str,
        "suspect_events": suspect_summary,
        "cuda_kernel_count": cuda_kernel_count,
        "any_cuda_sync_detected": any("Synchronize" in k or "_local_scalar_dense" in k for k in suspect_summary),
    }


def evaluate_3prompt_equivalence(
    model: Zip2ZipModel,
    tok: AutoTokenizer,
    device: torch.device,
    samples: List[Dict[str, Any]],
    saved_references: Dict[str, Dict[str, Any]],
    dim: int = 3072,
    pad_id: int = 32000,
    disabled_ids: Optional[Sequence[int]] = None,
    max_new_tokens: int = 300,
) -> Tuple[Dict[str, Any], bool]:
    """Run exact 3-prompt merge-equivalence test using fresh StaticCodebookManager and H-expansion."""
    model.eval()
    reports: Dict[str, Any] = {}
    all_equivalent = True

    for sample in samples:
        prompt_id = sample["id"]
        ref = saved_references.get(prompt_id)
        if not ref:
            continue

        saved_codebook = ref["codebook"]
        compressed_ids = ref["compressed_prompt_ids"]
        saved_raw_gen_ids = ref["generated_decode_ids"]
        saved_expanded_tokens = ref["expanded_tokens"]
        saved_text = ref["output_text"]
        saved_first_step_logits = ref.get("first_step_logits")

        # 1. Instantiate fresh StaticCodebookManager
        static_mgr = StaticCodebookManager(
            initial_vocab_size=benchmark.INITIAL_VOCAB,
            max_codebook_size=32,
            max_subtokens=4,
            embedding_dim=dim,
            pad_token_id=pad_id,
            disabled_ids=disabled_ids,
        )
        static_mgr.set_seeded_codebook(saved_codebook, batch_size=1, device=device)
        static_mgr.synthesize_hyper_vectors(model, batch_size=1)
        static_mgr.attach_to_model(model)

        input_tensor = torch.tensor([compressed_ids], dtype=torch.long, device=device)

        # Measure merged first-step logits for base + seeded H
        logits_comp_report: Dict[str, Any] = {}
        with torch.no_grad():
            first_out = model(input_tensor, use_cache=True)
            merged_logits_step0 = first_out.logits[0, -1, :].detach().cpu()

        if saved_first_step_logits is not None:
            unmerged_logits_step0 = torch.tensor(saved_first_step_logits["full_step0_logits"])
            diff = torch.abs(merged_logits_step0 - unmerged_logits_step0)
            max_abs_diff = float(diff.max().item())
            mean_abs_diff = float(diff.mean().item())

            top1_unmerged = int(torch.argmax(unmerged_logits_step0).item())
            top1_merged = int(torch.argmax(merged_logits_step0).item())
            top5_unmerged = set(torch.topk(unmerged_logits_step0, 5).indices.tolist())
            top5_merged = set(torch.topk(merged_logits_step0, 5).indices.tolist())

            logits_comp_report = {
                "max_abs_diff": round(max_abs_diff, 6),
                "mean_abs_diff": round(mean_abs_diff, 6),
                "top1_agreement": (top1_unmerged == top1_merged),
                "top1_unmerged": top1_unmerged,
                "top1_merged": top1_merged,
                "top5_overlap_count": len(top5_unmerged.intersection(top5_merged)),
            }

        # Full generation
        proc_list = LogitsProcessorList([static_mgr.get_logits_processor()])
        with torch.no_grad():
            out = model.generate(
                input_ids=input_tensor,
                max_new_tokens=max_new_tokens,
                logits_processor=proc_list,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

        static_mgr.detach_from_model(model)
        model.codebook_manager.reset()

        merged_raw_gen_ids = out[0, len(compressed_ids):].tolist()
        hyper_to_tokens = {int(v): list(k) for k, v in saved_codebook.items()}
        merged_expanded_tokens = []
        for tid in merged_raw_gen_ids:
            if tid in hyper_to_tokens:
                merged_expanded_tokens.extend(hyper_to_tokens[tid])
            else:
                merged_expanded_tokens.append(tid)

        merged_text = tok.decode(merged_expanded_tokens, skip_special_tokens=True)

        raw_ids_match = (merged_raw_gen_ids == saved_raw_gen_ids)
        expanded_ids_match = (merged_expanded_tokens == saved_expanded_tokens)
        text_equal = (merged_text == saved_text)

        first_divergent_step = None
        if not raw_ids_match:
            for step_idx, (g1, g2) in enumerate(zip(saved_raw_gen_ids, merged_raw_gen_ids)):
                if g1 != g2:
                    first_divergent_step = step_idx
                    break
            if first_divergent_step is None:
                first_divergent_step = min(len(saved_raw_gen_ids), len(merged_raw_gen_ids))

        if not raw_ids_match or not expanded_ids_match or not text_equal:
            all_equivalent = False

        reports[prompt_id] = {
            "prompt_id": prompt_id,
            "raw_ids_match": raw_ids_match,
            "expanded_ids_match": expanded_ids_match,
            "text_equal": text_equal,
            "first_divergent_step": first_divergent_step,
            "unmerged_raw_steps": len(saved_raw_gen_ids),
            "merged_raw_steps": len(merged_raw_gen_ids),
            "unmerged_expanded_tokens": len(saved_expanded_tokens),
            "merged_expanded_tokens": len(merged_expanded_tokens),
            "logits_comparison": logits_comp_report,
            "unmerged_text_preview": saved_text[:100],
            "merged_text_preview": merged_text[:100],
        }

    return reports, all_equivalent


def run_benchmark_12prompts(
    condition_name: str,
    model: nn.Module,
    tok: AutoTokenizer,
    device: torch.device,
    samples: List[Dict[str, Any]],
    policy: Optional[CappedPredictorPolicy] = None,
    dim: Optional[int] = None,
    pad_id: Optional[int] = None,
    disabled_ids: Optional[Sequence[int]] = None,
    max_new_tokens: int = 300,
    compress_prompt: bool = True,
    save_first_step_logits_for_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Execute fixed 12-prompt evaluation for Vanilla or Predictive condition."""
    model.eval()
    results: List[Dict[str, Any]] = []
    saved_references: Dict[str, Dict[str, Any]] = {}
    is_predictive = (policy is not None)
    save_logits_set = set(save_first_step_logits_for_ids or [])

    for idx, s in enumerate(samples, 1):
        prompt_id = s["id"]
        prompt_text = build_mbpp_prompt(s) if s["domain"] == "code" else s["prompt"]
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        base_prompt_len = len(prompt_ids)

        if not is_predictive:
            # Vanilla Phi path
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            _cuda_sync_if_needed(device)
            t_start = time.perf_counter()
            timing_proc = benchmark.TimingLogitsProcessor(t_start)
            proc_list = LogitsProcessorList([timing_proc])

            with torch.no_grad():
                out = model.generate(
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                    logits_processor=proc_list,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            _cuda_sync_if_needed(device)
            t_total = time.perf_counter() - t_start

            gen_ids = out[0, base_prompt_len:].tolist()
            decode_steps = len(gen_ids)
            expanded_tokens = gen_ids
            expanded_output_tokens = len(expanded_tokens)
            output_text = tok.decode(gen_ids, skip_special_tokens=True)
            codebook_dict = {}
            codebook_sha256 = ""
            predictor_time_s = 0.0
            codebook_time_s = 0.0
            tokens_saved = 0
            decode_reduction_pct = 0.0
            hypertokens_emitted = []
            first_hyper_pos = -1
            ttft = timing_proc.ttft or 0.0
            decode_time = max(0.0, t_total - ttft)
            answer_decode_pos, answer_expanded_pos = benchmark._answer_trace_positions(
                gen_ids, tok, s["domain"]
            )
        else:
            # Predictive Zip2Zip path
            t_pred_start = time.perf_counter()
            codebook_dict, _ = policy.select_codebook(prompt_ids)
            predictor_time_s = time.perf_counter() - t_pred_start

            t_setup_start = time.perf_counter()
            static_mgr = StaticCodebookManager(
                initial_vocab_size=benchmark.INITIAL_VOCAB,
                max_codebook_size=32,
                max_subtokens=4,
                embedding_dim=dim or 3072,
                pad_token_id=pad_id or tok.pad_token_id or 32000,
                disabled_ids=disabled_ids,
            )
            static_mgr.set_seeded_codebook(codebook_dict, batch_size=1, device=device)
            static_mgr.synthesize_hyper_vectors(model, batch_size=1)
            static_mgr.attach_to_model(model)
            hyper_setup_time_s = time.perf_counter() - t_setup_start
            codebook_time_s = predictor_time_s + hyper_setup_time_s
            codebook_sha256 = hashlib.sha256(
                repr(sorted(codebook_dict.items())).encode("utf-8")
            ).hexdigest()

            model_prompt_ids = benchmark.prepare_prompt_input_ids(
                prompt_ids, static_mgr, compress_prompt=compress_prompt
            )
            model_prompt_len = len(model_prompt_ids)
            input_tensor = torch.tensor([model_prompt_ids], dtype=torch.long, device=device)

            first_step_logits_info = None
            if prompt_id in save_logits_set:
                with torch.no_grad():
                    first_out = model(input_tensor, use_cache=True)
                    logits_step0 = first_out.logits[0, -1, :].detach().cpu()
                    top5 = torch.topk(logits_step0, 5)
                    top1_val = float(top5.values[0].item())
                    top2_val = float(top5.values[1].item())
                    first_step_logits_info = {
                        "full_step0_logits": logits_step0.tolist(),
                        "top1_id": int(top5.indices[0].item()),
                        "top1_val": top1_val,
                        "top2_val": top2_val,
                        "margin": top1_val - top2_val,
                        "top5_ids": top5.indices.tolist(),
                    }

            _cuda_sync_if_needed(device)
            t_gen_start = time.perf_counter()
            timing_proc = benchmark.TimingLogitsProcessor(t_gen_start, static_mgr=static_mgr)
            proc_list = LogitsProcessorList([timing_proc, static_mgr.get_logits_processor()])

            with torch.no_grad():
                out = model.generate(
                    input_ids=input_tensor,
                    max_new_tokens=max_new_tokens,
                    logits_processor=proc_list,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            _cuda_sync_if_needed(device)
            t_gen = time.perf_counter() - t_gen_start
            t_total = codebook_time_s + t_gen
            ttft = timing_proc.ttft or 0.0
            decode_time = max(0.0, t_gen - ttft)

            gen_ids = out[0, model_prompt_len:].tolist()
            decode_steps = len(gen_ids)

            hyper_to_tokens = {v: list(k) for k, v in codebook_dict.items()}
            expanded_tokens = []
            hypertokens_emitted = []
            first_hyper_pos = -1

            for pos, tid in enumerate(gen_ids):
                if tid in hyper_to_tokens:
                    if first_hyper_pos == -1:
                        first_hyper_pos = pos
                    phrase_str = tok.decode(hyper_to_tokens[tid])
                    hypertokens_emitted.append({"pos": pos, "id": tid, "phrase": phrase_str, "subtokens": hyper_to_tokens[tid]})
                    expanded_tokens.extend(hyper_to_tokens[tid])
                else:
                    expanded_tokens.append(tid)

            answer_decode_pos, answer_expanded_pos = benchmark._answer_trace_positions(
                gen_ids, tok, s["domain"], expansion_map=hyper_to_tokens
            )
            output_text = tok.decode(expanded_tokens, skip_special_tokens=True)
            expanded_output_tokens = len(expanded_tokens)
            tokens_saved = max(0, expanded_output_tokens - decode_steps)
            decode_reduction_pct = round((1.0 - decode_steps / max(expanded_output_tokens, 1)) * 100, 2) if expanded_output_tokens > decode_steps else 0.0

            static_mgr.detach_from_model(model)
            model.codebook_manager.reset()

            saved_references[prompt_id] = {
                "prompt_id": prompt_id,
                "codebook": {tuple(k): v for k, v in codebook_dict.items()},
                "codebook_sha256": codebook_sha256,
                "compressed_prompt_ids": model_prompt_ids,
                "generated_decode_ids": gen_ids,
                "expanded_tokens": expanded_tokens,
                "output_text": output_text,
                "hypertokens_emitted": hypertokens_emitted,
                "first_step_logits": first_step_logits_info,
            }

        eos_reached = benchmark.sequence_reached_eos(gen_ids, tok.eos_token_id)
        hit_max_length = decode_steps >= max_new_tokens

        rec: Dict[str, Any] = {
            "id": prompt_id,
            "domain": s["domain"],
            "condition": condition_name,
            "base_prompt_tokens": base_prompt_len,
            "compressed_prompt_tokens": base_prompt_len if not is_predictive else len(model_prompt_ids),
            "codebook_size": len(codebook_dict),
            "codebook_sha256": codebook_sha256,
            "decode_steps": decode_steps,
            "expanded_output_tokens": expanded_output_tokens,
            "tokens_saved": tokens_saved,
            "decode_reduction_pct": decode_reduction_pct,
            "wall_time_s": round(t_total, 3),
            "ttft_s": round(ttft, 3),
            "prefill_time_s": round(ttft, 3),
            "decode_time_s": round(decode_time, 3),
            "generation_wall_time_s": round(t_total, 3),
            "decode_step_intervals_s": timing_proc.decode_step_intervals_s,
            "first_answer_decode_position": answer_decode_pos,
            "first_answer_expanded_position": answer_expanded_pos,
            "first_repeated_trigram_position": first_repeated_trigram_position(gen_ids),
            "predictor_time_s": round(predictor_time_s, 4),
            "codebook_time_s": round(codebook_time_s, 4),
            "throughput_tok_per_s": round(expanded_output_tokens / max(t_total, 0.001), 2),
            "eos_reached": eos_reached,
            "hit_max_length": hit_max_length,
            "hypertokens_count": len(hypertokens_emitted),
            "hypertokens_emitted": hypertokens_emitted,
            "first_hypertoken_pos": first_hyper_pos,
            "output_text": output_text,
        }
        rec.update(
            benchmark.generation_health_fields(
                output_text, gen_ids, tok.eos_token_id, max_new_tokens, expanded_output_tokens
            )
        )

        dom = s["domain"]
        if dom == "code":
            asserts = [line.strip() for line in s["ground_truth_response"].splitlines() if line.strip().startswith("assert")]
            code_eval = benchmark.evaluate_mbpp_code(output_text, asserts)
            rec.update(code_eval)
        elif dom == "reasoning":
            gsm_eval = benchmark.evaluate_gsm8k_reasoning(output_text, s["ground_truth_response"])
            rec.update(gsm_eval)
        elif dom == "instruction":
            alp_eval = benchmark.evaluate_alpaca_instruction(output_text, eos_reached)
            rec.update(alp_eval)

        rec = add_runtime_derived_metrics(rec)
        results.append(rec)
        print(
            f"[{idx:2d}/{len(samples)}] {prompt_id:10s} | {dom:11s} | Dec: {decode_steps:3d} -> Exp: {expanded_output_tokens:3d} "
            f"| Time: {rec['wall_time_s']:5.2f}s | Pass: {rec.get('problem_pass', rec.get('exact_correct', rec.get('mechanical_instruction_pass')))}",
            flush=True,
        )

    return results, saved_references


def run_orchestrator(
    output_dir: Path,
    checkpoint_path: Path,
    predictor_path: Path,
    prompt_ids_path: Path,
    validation_data_path: Path,
    base_revision: str,
    zip2zip_revision: str,
    tested_commit: str,
) -> None:
    """Execute the full controlled GPU runtime decomposition protocol (V14)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device1 = torch.device("cuda:1" if torch.cuda.device_count() >= 2 else "cpu")

    print(f"\n========================================================")
    print(f"STARTING RUNTIME DECOMPOSITION EXPERIMENT (V14)")
    print(f"Primary GPU: {device0} | Control GPU: {device1}")
    print(f"========================================================\n")

    # Load 12 prompts
    with open(validation_data_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)
    samples_12 = select_prompt_subset(
        all_samples, prompt_ids_path, {dom: 4 for dom in ("code", "reasoning", "instruction")}
    )
    samples_smoke = [s for s in samples_12 if s["id"] in ("mbpp_542", "gsm_2956", "alpaca_183")]
    sample_gsm2956 = next(s for s in samples_12 if s["id"] == "gsm_2956")

    # 1. Load pristine Vanilla models
    tokenizer = AutoTokenizer.from_pretrained(
        benchmark.PHI_MODEL_ID, **benchmark._revision_kwargs(base_revision)
    )

    print("\n[Step 1] Loading pristine Vanilla Phi on cuda:0 (primary)...", flush=True)
    phi_primary = AutoModelForCausalLM.from_pretrained(
        benchmark.PHI_MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        **benchmark._revision_kwargs(base_revision),
    )
    phi_primary.to(device0)
    phi_primary.eval()

    print("[Step 1] Loading pristine Vanilla Phi on cuda:1 (control)...", flush=True)
    phi_control = AutoModelForCausalLM.from_pretrained(
        benchmark.PHI_MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        **benchmark._revision_kwargs(base_revision),
    )
    phi_control.to(device1)
    phi_control.eval()

    # Save representative base tensors before wrapping for strict in-place reuse verification
    named_params_before = dict(phi_primary.named_parameters())
    probe_tensor_names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.15.mlp.gate_up_proj.weight",
        "model.layers.31.self_attn.o_proj.weight",
        "lm_head.weight",
    ]
    probe_before: Dict[str, Dict[str, Any]] = {}
    for name in probe_tensor_names:
        if name in named_params_before:
            p = named_params_before[name]
            h = hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
            probe_before[name] = {"data_ptr": int(p.data_ptr()), "sha256": h, "shape": list(p.shape)}
    vram_vanilla = benchmark.cuda_memory_snapshot(device0) if device0.type == "cuda" else {}

    # Control calibration 0 (baseline)
    print("\n[Control] Running calibration 0 on cuda:1 (cuda:0 idle)...", flush=True)
    calib_0 = run_calibration_check(phi_control, device1, "0_baseline")
    calibrations = [calib_0]

    # Vanilla fixed-KV microbenchmark on cuda:0
    print("\n[Vanilla] Running fixed-KV per-step microbenchmark on cuda:0...", flush=True)
    microbench_vanilla = run_single_step_microbenchmarks(
        phi_primary, device0, condition_name="vanilla_phi"
    )

    # Vanilla torch.profiler capture on cuda:0 BEFORE mutating phi_primary
    print("\n[Vanilla] Running torch.profiler capture on pristine Vanilla on cuda:0...", flush=True)
    dummy_input = torch.tensor([[1000]], dtype=torch.long, device=device0)
    with torch.no_grad():
        out_v = phi_primary(dummy_input, use_cache=True)
        kv_v = out_v.past_key_values
    profiler_vanilla = run_torch_profiler_capture(phi_primary, dummy_input, kv_v, device0)

    # 12-prompt Vanilla on cuda:0
    print("\n[Vanilla] Running fixed 12 prompts on cuda:0...", flush=True)
    vanilla_12_records, _ = run_benchmark_12prompts(
        condition_name="original_phi",
        model=phi_primary,
        tok=tokenizer,
        device=device0,
        samples=samples_12,
    )

    # Control calibration 1 (post-vanilla)
    print("\n[Control] Running calibration 1 on cuda:1...", flush=True)
    calib_1 = run_calibration_check(phi_control, device1, "1_post_vanilla")
    calibrations.append(calib_1)

    # 2. Transform resident phi_primary into Zip2ZipModel
    print("\n[Step 2] Wrapping resident phi_primary into Zip2ZipModel on cuda:0...", flush=True)
    predictive_model = Zip2ZipModel.from_pretrained(
        benchmark.ZIP2ZIP_MODEL_ID,
        base_model=phi_primary,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        **benchmark._revision_kwargs(zip2zip_revision),
    )
    vram_post_wrap = benchmark.cuda_memory_snapshot(device0) if device0.type == "cuda" else {}
    print(f"Post-wrap VRAM on cuda:0: allocated={vram_post_wrap.get('allocated_bytes', 0) / (1024**2):.1f} MiB", flush=True)

    # Verify resident base reuse
    wrapped_named_params = dict(predictive_model.named_parameters())
    probe_matches = 0
    probe_details: Dict[str, Any] = {}
    for name, b_info in probe_before.items():
        cand = None
        target_name = None
        # Look for tensor by matching suffix in wrapped_named_params
        for k, v in wrapped_named_params.items():
            if k.endswith(name) or k.endswith(f"base_layer.{name.split('.')[-1]}") or name in k:
                if tuple(v.shape) == tuple(b_info["shape"]):
                    cand = v
                    target_name = k
                    break
        if cand is not None:
            cand_ptr = int(cand.data_ptr())
            cand_h = hashlib.sha256(cand.detach().cpu().numpy().tobytes()).hexdigest()
            ptr_match = (cand_ptr == b_info["data_ptr"])
            hash_match = (cand_h == b_info["sha256"])
            if hash_match or ptr_match:
                probe_matches += 1
            probe_details[name] = {
                "wrapped_name": target_name,
                "ptr_match": ptr_match,
                "hash_match": hash_match,
            }

    vram_delta_mb = (vram_post_wrap.get("allocated_bytes", 0) - vram_vanilla.get("allocated_bytes", 0)) / (1024**2)
    # Full Phi-3.5 allocation is ~7,600 MiB. A second copy would increase allocated VRAM by >7,000 MiB.
    vram_reuse_consistent = (vram_delta_mb < 2000.0)
    resident_reuse_success = (probe_matches >= 3) and vram_reuse_consistent
    print(f"Resident base reuse verification: success={resident_reuse_success} (probe_matches={probe_matches}/{len(probe_before)}, vram_delta={vram_delta_mb:.1f} MiB)")
    if not resident_reuse_success:
        raise RuntimeError(f"Resident base reuse failed: probe_details={probe_details}, vram_delta_mb={vram_delta_mb}")

    # Apply Step-100 via load_joint_checkpoint
    print("\n[Step 3] Applying verified Step-100 joint checkpoint...", flush=True)
    load_report = load_joint_checkpoint(
        predictive_model,
        checkpoint_path,
        expected_step=100,
        expected_model_id=benchmark.ZIP2ZIP_MODEL_ID,
    )
    predictive_model.to(device0)
    predictive_model.eval()

    # Enforce checkpoint contract
    assert load_report["changed_tensor_count"] == 298, f"Expected 298 changed tensors, got {load_report['changed_tensor_count']}"
    assert load_report["lora_tensors"] == 256, f"Expected 256 LoRA tensors, got {load_report['lora_tensors']}"
    assert load_report["input_encoder_tensors"] == 21, f"Expected 21 input encoder tensors, got {load_report['input_encoder_tensors']}"
    assert load_report["output_encoder_tensors"] == 21, f"Expected 21 output encoder tensors, got {load_report['output_encoder_tensors']}"
    assert load_report["base_hash_status"] == "missing", f"Expected base_hash_status='missing', got {load_report['base_hash_status']}"
    for comp_name, m_keys in load_report["missing_keys"].items():
        assert len(m_keys) == 0, f"Unexpected missing keys in {comp_name}: {m_keys}"
    for comp_name, u_keys in load_report["unexpected_keys"].items():
        assert len(u_keys) == 0, f"Unexpected unexpected keys in {comp_name}: {u_keys}"
    print(f"Checkpoint contract passed: 298 tensors verified, zero missing/unexpected keys.")

    vram_unmerged = benchmark.cuda_memory_snapshot(device0) if device0.type == "cuda" else {}

    # Load canonical predictor policy
    raw_predictor = load_oracle_predictor(predictor_path)
    predictor_index = getattr(raw_predictor, "index", raw_predictor)
    policy = CappedPredictorPolicy(
        predictor_index,
        tokenizer,
        budget=32,
        max_structural_slots=0,
        allow_numeric=True,
        filter_bare_punctuation=True,
    )
    dim = predictive_model.zip2zip_config.encoder.hidden_size
    pad_id = tokenizer.pad_token_id or 32000
    disabled_ids = list(predictive_model.zip2zip_config.compression.disabled_ids)

    # Build canonical fixed representative K=32 codebook from gsm_2956 for microbenchmarking
    gsm_prompt_text = sample_gsm2956["prompt"]
    gsm_prompt_ids = tokenizer.encode(gsm_prompt_text, add_special_tokens=False)
    canonical_codebook_dict, _ = policy.select_codebook(gsm_prompt_ids)
    print(f"Synthesized canonical representative K={len(canonical_codebook_dict)} codebook from gsm_2956 for microbenchmarks.")

    # Create canonical static manager for microbenchmarks
    microbench_static_mgr = StaticCodebookManager(
        initial_vocab_size=benchmark.INITIAL_VOCAB,
        max_codebook_size=32,
        max_subtokens=4,
        embedding_dim=dim,
        pad_token_id=pad_id,
        disabled_ids=disabled_ids,
    )
    microbench_static_mgr.set_seeded_codebook(canonical_codebook_dict, batch_size=1, device=device0)
    microbench_static_mgr.synthesize_hyper_vectors(predictive_model, batch_size=1)
    microbench_static_mgr.attach_to_model(predictive_model)

    # Control calibration 2 (pre-predictive-unmerged)
    print("\n[Control] Running calibration 2 on cuda:1...", flush=True)
    calib_2 = run_calibration_check(phi_control, device1, "2_pre_predictive_unmerged")
    calibrations.append(calib_2)

    # Fixed-KV microbenchmark Predictive Unmerged with canonical static manager attached
    print("\n[Predictive Unmerged] Running fixed-KV per-step microbenchmark on cuda:0...", flush=True)
    microbench_unmerged = run_single_step_microbenchmarks(
        predictive_model, device0, condition_name="predictive_unmerged"
    )

    # Predictive Unmerged torch.profiler capture
    print("\n[Predictive Unmerged] Running torch.profiler capture on cuda:0...", flush=True)
    with torch.no_grad():
        out_u = predictive_model(dummy_input, use_cache=True)
        kv_u = out_u.past_key_values
    profiler_unmerged = run_torch_profiler_capture(predictive_model, dummy_input, kv_u, device0)

    # Detach static manager before running 12 prompts (run_benchmark_12prompts attaches per-prompt manager)
    microbench_static_mgr.detach_from_model(predictive_model)
    predictive_model.codebook_manager.reset()

    # 12-prompt Predictive Unmerged (save first-step logits for mbpp_542, gsm_2956, alpaca_183)
    print("\n[Predictive Unmerged] Running fixed 12 prompts on cuda:0...", flush=True)
    smoke_ids = [s["id"] for s in samples_smoke]
    unmerged_12_records, saved_references = run_benchmark_12prompts(
        condition_name="predictive_step_100_unmerged",
        model=predictive_model,
        tok=tokenizer,
        device=device0,
        samples=samples_12,
        policy=policy,
        dim=dim,
        pad_id=pad_id,
        disabled_ids=disabled_ids,
        compress_prompt=True,
        save_first_step_logits_for_ids=smoke_ids,
    )

    # 3. Merge LoRA in place
    print("\n[Step 4] Merging LoRA weights in place...", flush=True)
    old_peft_model = predictive_model.base_model
    lora_params_before = sum(1 for n, _ in old_peft_model.named_parameters() if "lora" in n.lower())

    if hasattr(old_peft_model, "merge_and_unload"):
        try:
            merged_base = old_peft_model.merge_and_unload(safe_merge=True)
            merge_method = "merge_and_unload(safe_merge=True)"
        except TypeError:
            merged_base = old_peft_model.merge_and_unload()
            merge_method = "merge_and_unload()"
    elif hasattr(old_peft_model, "base_model") and hasattr(old_peft_model.base_model, "merge_and_unload"):
        merged_base = old_peft_model.base_model.merge_and_unload(safe_merge=True)
        merge_method = "base_model.merge_and_unload(safe_merge=True)"
    else:
        raise RuntimeError(f"Cannot find merge_and_unload on PEFT model: {type(old_peft_model)}")

    predictive_model.base_model = merged_base

    # Enforce merge contract
    lora_params_after = sum(1 for n, _ in predictive_model.base_model.named_parameters() if "lora" in n.lower())
    input_embed_after = predictive_model.base_model.get_input_embeddings()
    output_embed_after = predictive_model.base_model.get_output_embeddings()
    hyper_embed_survived = isinstance(input_embed_after, HyperEmbedding)
    hyper_linear_survived = isinstance(output_embed_after, HyperLinear)
    all_on_device0 = all(p.device == device0 for p in predictive_model.parameters())

    assert lora_params_after == 0, f"Expected 0 LoRA parameters after merge, got {lora_params_after}"
    assert hyper_embed_survived, "HyperEmbedding did not survive merge"
    assert hyper_linear_survived, "HyperLinear did not survive merge"
    assert all_on_device0, "Not all parameters on cuda:0 after merge"

    # Reinstall generation position hook exactly once
    predictive_model._install_base_position_generation_hook()
    vram_merged = benchmark.cuda_memory_snapshot(device0) if device0.type == "cuda" else {}

    print(f"LoRA merge complete ({merge_method}):")
    print(f"  LoRA params: {lora_params_before} -> {lora_params_after}")
    print(f"  HyperEmbedding survived: {hyper_embed_survived}")
    print(f"  HyperLinear survived: {hyper_linear_survived}")
    print(f"  All parameters on cuda:0: {all_on_device0}")
    print(f"  Position hook reinstalled: True")

    # 4. 3-prompt merge-equivalence test
    print("\n[Step 5] Running 3-prompt merge-equivalence verification...", flush=True)
    equiv_reports, all_equivalent = evaluate_3prompt_equivalence(
        model=predictive_model,
        tok=tokenizer,
        device=device0,
        samples=samples_smoke,
        saved_references=saved_references,
        dim=dim,
        pad_id=pad_id,
        disabled_ids=disabled_ids,
    )
    print(f"Merge equivalence outcome: all_equivalent={all_equivalent}")
    for pid, r in equiv_reports.items():
        logits_info = r.get("logits_comparison", {})
        print(f"  {pid}: raw_match={r['raw_ids_match']}, expanded_match={r['expanded_ids_match']}, text_equal={r['text_equal']} | logits_max_diff={logits_info.get('max_abs_diff')}, top1_agree={logits_info.get('top1_agreement')}")

    # Control calibration 3 (post-merge)
    print("\n[Control] Running calibration 3 on cuda:1...", flush=True)
    calib_3 = run_calibration_check(phi_control, device1, "3_post_merge")
    calibrations.append(calib_3)

    # Attach canonical static manager back onto merged predictive model for all merged microbenchmarks & ablations
    microbench_static_mgr.attach_to_model(predictive_model)

    # 5. Fixed-KV microbenchmark Predictive Merged
    print("\n[Predictive Merged] Running fixed-KV per-step microbenchmark on cuda:0...", flush=True)
    microbench_merged = run_single_step_microbenchmarks(
        predictive_model, device0, condition_name="predictive_merged",
        instrumented_static_manager=microbench_static_mgr,
    )

    # Predictive Merged torch.profiler capture
    print("\n[Predictive Merged] Running torch.profiler capture on cuda:0...", flush=True)
    with torch.no_grad():
        out_m = predictive_model(dummy_input, use_cache=True)
        kv_m = out_m.past_key_values
    profiler_merged = run_torch_profiler_capture(predictive_model, dummy_input, kv_m, device0)

    # Diagnostic ablations on Merged model
    # A. Position logic bypass (direct position_ids supplied to forward)
    print("\n[Ablations] Running position logic bypassed microbenchmark...", flush=True)
    microbench_pos_bypass = run_single_step_microbenchmarks(
        predictive_model, device0, condition_name="merged_pos_bypass",
        bypass_position_ids=True,
        instrumented_static_manager=microbench_static_mgr,
    )

    # Verify position ablation validity via prepare_input_ids call counts
    normal_calls = microbench_merged.get("256", {}).get("prepare_input_ids_call_count", 0)
    bypass_calls = microbench_pos_bypass.get("256", {}).get("prepare_input_ids_call_count", 0)
    position_ablation_valid = (normal_calls > 0 and bypass_calls == 0)
    print(f"Position ablation validity: valid={position_ablation_valid} (normal_calls={normal_calls}, bypass_calls={bypass_calls})")

    # B. HyperEmbedding fast path (verify numerical equality against current output before timing)
    print("\n[Ablations] Verifying HyperEmbedding fast-path numerical equality before timing...", flush=True)
    test_input = torch.tensor([[1000, 2000, 3000]], dtype=torch.long, device=device0)
    emb_module = predictive_model.base_model.get_input_embeddings()
    with torch.no_grad():
        orig_emb_out = emb_module(test_input)
        fast_emb_out = nn.Embedding.forward(emb_module, test_input)
        emb_diff = torch.abs(orig_emb_out - fast_emb_out).max().item()
    print(f"HyperEmbedding base-token verification: max abs diff = {emb_diff:.8f}")
    assert emb_diff < 1e-5, f"HyperEmbedding fast-path numerical mismatch: max diff = {emb_diff}"

    print("[Ablations] Running HyperEmbedding fast-path microbenchmark...", flush=True)
    with hyper_embedding_fast_path_context(predictive_model):
        microbench_embed_fastpath = run_single_step_microbenchmarks(
            predictive_model, device0, condition_name="merged_embed_fastpath"
        )

    # C. HyperLinear bypass
    print("\n[Ablations] Running HyperLinear bypassed microbenchmark...", flush=True)
    with hyper_linear_bypassed_context(predictive_model):
        microbench_linear_bypass = run_single_step_microbenchmarks(
            predictive_model, device0, condition_name="merged_linear_bypass"
        )

    # D. Logits processor separate generate benchmark
    print("\n[Ablations] Running logits-mask separate generate benchmark...", flush=True)
    logits_mask_benchmark_res = run_logits_mask_generate_benchmark(
        predictive_model, microbench_static_mgr, tokenizer, device0, gen_steps=30
    )

    # E. Base token vs seeded H-token decode step comparison
    print("\n[Ablations] Running base token vs H-token decode step comparison...", flush=True)
    h_vs_base_results = run_h_token_vs_base_token_microbenchmark(
        predictive_model, microbench_static_mgr, device0
    )

    # Detach static manager before running 12 prompts
    microbench_static_mgr.detach_from_model(predictive_model)
    predictive_model.codebook_manager.reset()

    # 6. 12-prompt Predictive Merged benchmark
    print("\n[Predictive Merged] Running fixed 12 prompts on cuda:0...", flush=True)
    merged_12_records, _ = run_benchmark_12prompts(
        condition_name="predictive_step_100_merged",
        model=predictive_model,
        tok=tokenizer,
        device=device0,
        samples=samples_12,
        policy=policy,
        dim=dim,
        pad_id=pad_id,
        disabled_ids=disabled_ids,
        compress_prompt=True,
    )

    # Control calibration 4 (final)
    print("\n[Control] Running final calibration 4 on cuda:1...", flush=True)
    calib_4 = run_calibration_check(phi_control, device1, "4_final")
    calibrations.append(calib_4)

    # Check drift across calibrations
    calib_medians = [c["median"] for c in calibrations]
    base_calib = calib_medians[0]
    max_drift_pct = max(abs(m - base_calib) / max(base_calib, 1e-9) * 100 for m in calib_medians)
    environment_timing_unstable = (max_drift_pct > 5.0)
    print(f"\nControl GPU drift: max drift = {max_drift_pct:.2f}% (unstable={environment_timing_unstable})")

    # 7. Build output artifacts
    print("\n[Step 7] Writing summary artifacts...", flush=True)

    # Model reuse report
    reuse_report = {
        "probe_details": probe_details,
        "vram_vanilla": vram_vanilla,
        "vram_post_wrap": vram_post_wrap,
        "vram_unmerged": vram_unmerged,
        "vram_merged": vram_merged,
        "vram_delta_mb": round(vram_delta_mb, 2),
        "probe_matches": probe_matches,
        "resident_reuse_success": resident_reuse_success,
    }
    write_json_atomic(output_dir / "model_reuse_report.json", reuse_report)

    # Merge equivalence report
    equiv_summary = {
        "all_equivalent": all_equivalent,
        "merge_method": merge_method,
        "lora_params_before": lora_params_before,
        "lora_params_after": lora_params_after,
        "hyper_embedding_survived": hyper_embed_survived,
        "hyper_linear_survived": hyper_linear_survived,
        "position_hook_reinstalled": True,
        "prompt_reports": equiv_reports,
    }
    write_json_atomic(output_dir / "merge_equivalence.json", equiv_summary)

    # Per-step microbenchmark report
    microbench_summary = {
        "vanilla": microbench_vanilla,
        "predictive_unmerged": microbench_unmerged,
        "predictive_merged": microbench_merged,
        "merged_pos_bypass": microbench_pos_bypass,
        "merged_embed_fastpath": microbench_embed_fastpath,
        "merged_linear_bypass": microbench_linear_bypass,
        "position_ablation_valid": position_ablation_valid,
        "logits_mask_generate": logits_mask_benchmark_res,
        "h_vs_base": h_vs_base_results,
    }
    write_json_atomic(output_dir / "per_step_microbenchmark.json", microbench_summary)

    # Profiler summary report
    prof_summary = {
        "vanilla_profiler": profiler_vanilla,
        "unmerged_profiler": profiler_unmerged,
        "merged_profiler": profiler_merged,
    }
    write_json_atomic(output_dir / "profiler_summary.json", prof_summary)

    # Tier-1 12-prompt records
    tier1_all = {
        "vanilla_records": vanilla_12_records,
        "unmerged_records": unmerged_12_records,
        "merged_records": merged_12_records,
        "calibrations": calibrations,
        "control_max_drift_pct": max_drift_pct,
        "environment_timing_unstable": environment_timing_unstable,
    }
    write_json_atomic(output_dir / "tier1_gpu_runtime.json", tier1_all)

    # Runtime decomposition calculation
    decomp: Dict[str, Any] = {"per_kv": {}}
    for kv in ("64", "256", "512"):
        v_med = microbench_vanilla.get(kv, {}).get("median", 0.0)
        u_med = microbench_unmerged.get(kv, {}).get("median", 0.0)
        m_med = microbench_merged.get(kv, {}).get("median", 0.0)
        pos_med = microbench_pos_bypass.get(kv, {}).get("median", 0.0)
        emb_med = microbench_embed_fastpath.get(kv, {}).get("median", 0.0)
        lin_med = microbench_linear_bypass.get(kv, {}).get("median", 0.0)

        total_overhead = max(0.0, u_med - v_med)
        lora_removable = max(0.0, u_med - m_med)
        remaining_after_merge = max(0.0, m_med - v_med)
        pos_delta = (m_med - pos_med) if position_ablation_valid else None
        emb_delta = m_med - emb_med
        lin_delta = m_med - lin_med

        decomp["per_kv"][kv] = {
            "vanilla_median_ms": v_med,
            "unmerged_median_ms": u_med,
            "merged_median_ms": m_med,
            "total_predictive_overhead_ms": round(total_overhead, 4),
            "lora_removable_overhead_ms": round(lora_removable, 4),
            "remaining_overhead_after_merge_ms": round(remaining_after_merge, 4),
            "position_associated_delta_ms": round(pos_delta, 4) if pos_delta is not None else None,
            "embedding_associated_delta_ms": round(emb_delta, 4),
            "linear_associated_delta_ms": round(lin_delta, 4),
        }
    decomp["logits_mask_generate"] = logits_mask_benchmark_res
    write_json_atomic(output_dir / "runtime_decomposition.json", decomp)

    # Markdown table generation
    md_lines = [
        "# Controlled GPU Runtime Decomposition Summary (V14)",
        "",
        "## Fixed-KV Per-Step Cached Decode Microbenchmark (ms/step)",
        "",
        "| Condition | KV=64 | KV=256 | KV=512 |",
        "|---|---:|---:|---:|",
    ]
    cond_rows = [
        ("Vanilla Phi", microbench_vanilla),
        ("Predictive Unmerged", microbench_unmerged),
        ("Predictive Merged", microbench_merged),
        ("Merged Position Bypass", microbench_pos_bypass if position_ablation_valid else {}),
        ("Merged Embedding Fast Path", microbench_embed_fastpath),
        ("Merged HyperLinear Bypass", microbench_linear_bypass),
    ]
    for cond_name, d_dict in cond_rows:
        if not d_dict:
            md_lines.append(f"| {cond_name} | INVALID | INVALID | INVALID |")
            continue
        row_str = f"| {cond_name} | {d_dict.get('64', {}).get('median', 0.0):.2f} | {d_dict.get('256', {}).get('median', 0.0):.2f} | {d_dict.get('512', {}).get('median', 0.0):.2f} |"
        md_lines.append(row_str)

    md_lines.extend([
        "",
        "## Overhead Decomposition (KV=256)",
        "",
        f"- Total Predictive Overhead (Unmerged - Vanilla): {decomp['per_kv'].get('256', {}).get('total_predictive_overhead_ms', 0.0):.2f} ms",
        f"- LoRA Removable Overhead (Unmerged - Merged): {decomp['per_kv'].get('256', {}).get('lora_removable_overhead_ms', 0.0):.2f} ms",
        f"- Remaining Overhead After Merge (Merged - Vanilla): {decomp['per_kv'].get('256', {}).get('remaining_overhead_after_merge_ms', 0.0):.2f} ms",
        f"- Position-Associated Delta: {decomp['per_kv'].get('256', {}).get('position_associated_delta_ms') or 'N/A (invalid)'} ms",
        f"- Embedding-Associated Delta: {decomp['per_kv'].get('256', {}).get('embedding_associated_delta_ms', 0.0):.2f} ms",
        f"- Linear-Associated Delta: {decomp['per_kv'].get('256', {}).get('linear_associated_delta_ms', 0.0):.2f} ms",
        f"- Logits Masking Generate Delta: {logits_mask_benchmark_res['masking_overhead_ms_per_step']:.2f} ms/step",
        "",
        "## Merge Equivalence Outcome",
        f"- All 3 smoke prompts equivalent: **{all_equivalent}**",
        "",
        "## cuda:1 Control Stability",
        f"- Control Max Drift: {max_drift_pct:.2f}% (unstable={environment_timing_unstable})",
    ])
    (output_dir / "runtime_decomposition.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    status_payload = {
        "status": "complete",
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "all_equivalent": all_equivalent,
        "resident_reuse_success": resident_reuse_success,
        "environment_timing_unstable": environment_timing_unstable,
    }
    write_json_atomic(output_dir / "run_status.json", status_payload)
    print("\n[Done] All summary artifacts written successfully!", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled GPU runtime decomposition harness.")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs")
    parser.add_argument("--checkpoint", default="experiments/checkpoints/predictive_joint_pilot/checkpoint_step_100.pt")
    parser.add_argument("--predictor", default="experiments/checkpoints/oracle_guided_predictor.pkl")
    parser.add_argument("--prompt-ids-file", default="experiments/checkpoints/quality_benchmark/poc_12_prompt_ids.json")
    parser.add_argument("--validation-data", default="data/cached_pure_pred_val_60.json")
    parser.add_argument("--base-revision", default=benchmark.DEFAULT_PHI_REVISION)
    parser.add_argument("--zip2zip-revision", default=benchmark.DEFAULT_ZIP2ZIP_REVISION)
    parser.add_argument("--tested-commit", default="df1d85e43b3e9e2a9a83c5e5c5f51fa7d4f1b2c9")
    args = parser.parse_args()

    run_orchestrator(
        output_dir=Path(args.output_dir).resolve(),
        checkpoint_path=Path(args.checkpoint).resolve(),
        predictor_path=Path(args.predictor).resolve(),
        prompt_ids_path=Path(args.prompt_ids_file).resolve(),
        validation_data_path=Path(args.validation_data).resolve(),
        base_revision=args.base_revision,
        zip2zip_revision=args.zip2zip_revision,
        tested_commit=args.tested_commit,
    )


if __name__ == "__main__":
    main()
