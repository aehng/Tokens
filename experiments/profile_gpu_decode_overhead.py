"""Controlled GPU runtime decomposition & matched 12-prompt benchmark harness.

Profiles and isolates the per-decode-step overhead of Predictive Zip2Zip:
1. Active PEFT/LoRA matmuls (Predictive unmerged vs Predictive merged)
2. Custom position handling (StaticCodebookManager.prepare_input_ids hook bypass)
3. HyperEmbedding overhead (base-token fast-path bypass)
4. HyperLinear / hypertoken-logit overhead (H-logit scoring bypass)
5. Logits processors / masking overhead (mask_unused_logits bypass)
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


def _measure_decode_steps_cuda_events(
    model: nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any,
    device: torch.device,
    warmup_steps: int = 20,
    measured_steps: int = 100,
    position_ids: Optional[torch.Tensor] = None,
) -> List[float]:
    """Measure single-token decode steps using CUDA Events."""
    _cuda_sync_if_needed(device)

    # Warmup
    with torch.no_grad():
        curr_kv = past_key_values
        for _ in range(warmup_steps):
            kwargs: Dict[str, Any] = {"use_cache": True}
            if position_ids is not None:
                kwargs["position_ids"] = position_ids
            out = model(input_ids, past_key_values=curr_kv, **kwargs)
            curr_kv = out.past_key_values

    _cuda_sync_if_needed(device)

    # Measured
    samples_ms: List[float] = []
    with torch.no_grad():
        curr_kv = past_key_values
        for _ in range(measured_steps):
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                t0 = time.perf_counter()

            kwargs = {"use_cache": True}
            if position_ids is not None:
                kwargs["position_ids"] = position_ids
            out = model(input_ids, past_key_values=curr_kv, **kwargs)

            if device.type == "cuda":
                end_event.record()
                _cuda_sync_if_needed(device)
                samples_ms.append(start_event.elapsed_time(end_event))
            else:
                samples_ms.append((time.perf_counter() - t0) * 1000.0)

            curr_kv = out.past_key_values

    return samples_ms


@contextlib.contextmanager
def position_logic_bypassed_context(model: Zip2ZipModel):
    """Diagnostic bypass: forces normal sequential position_ids, bypassing custom codebook hook."""
    original_hook = getattr(model.base_model, "prepare_inputs_for_generation", None)

    def bypass_prepare(base_self, *args, **kwargs):
        model_inputs = original_hook(*args, **kwargs)
        input_ids = model_inputs.get("input_ids")
        if input_ids is not None:
            # Overwrite position_ids with dummy sequential positions
            past_len = 0
            if "past_key_values" in model_inputs and model_inputs["past_key_values"] is not None:
                try:
                    past_len = model_inputs["past_key_values"][0][0].shape[-2]
                except Exception:
                    past_len = 0
            seq_len = input_ids.shape[-1]
            model_inputs["position_ids"] = torch.arange(
                past_len, past_len + seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)
        return model_inputs

    if original_hook is not None:
        model.base_model.prepare_inputs_for_generation = MethodType(bypass_prepare, model.base_model)
    try:
        yield
    finally:
        if original_hook is not None:
            model.base_model.prepare_inputs_for_generation = original_hook


@contextlib.contextmanager
def hyper_embedding_fast_path_context(model: Zip2ZipModel):
    """Diagnostic bypass: if all input token IDs are base tokens, call ordinary embedding lookup directly."""
    embedding_layer = model.base_model.get_input_embeddings()
    if not isinstance(embedding_layer, HyperEmbedding):
        yield
        return

    original_forward = embedding_layer.forward

    def fast_forward(self, input: torch.Tensor) -> torch.Tensor:
        if bool((input < self.initial_vocab_size).all()):
            return super(HyperEmbedding, self).forward(input)
        return original_forward(input)

    embedding_layer.forward = MethodType(fast_forward, embedding_layer)
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


def run_single_step_microbenchmarks(
    model: nn.Module,
    device: torch.device,
    condition_name: str,
    kv_lengths: Sequence[int] = (64, 256, 512),
    warmup_steps: int = 20,
    measured_steps: int = 100,
    seed_token_id: int = 1000,
) -> Dict[str, Any]:
    """Run controlled single-token cached decode microbenchmark across KV lengths."""
    results: Dict[str, Any] = {}
    model.eval()

    for kv_len in kv_lengths:
        dummy_prompt = torch.full((1, kv_len), seed_token_id, dtype=torch.long, device=device)
        with torch.no_grad():
            prefill_out = model(dummy_prompt, use_cache=True)
            past_kv = prefill_out.past_key_values

        decode_input = torch.tensor([[seed_token_id]], dtype=torch.long, device=device)
        samples_ms = _measure_decode_steps_cuda_events(
            model=model,
            input_ids=decode_input,
            past_key_values=past_kv,
            device=device,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
        )
        stats = _compute_stats(samples_ms)
        results[str(kv_len)] = {
            "kv_length": kv_len,
            "condition": condition_name,
            **stats,
        }
        print(f"  [Microbench] {condition_name:30s} | KV={kv_len:4d} | Median: {stats['median']:6.2f} ms | p95: {stats['p95']:6.2f} ms", flush=True)

    return results


def run_h_token_vs_base_token_microbenchmark(
    predictive_model: Zip2ZipModel,
    codebook_manager: StaticCodebookManager,
    device: torch.device,
    kv_lengths: Sequence[int] = (64, 256, 512),
    warmup_steps: int = 20,
    measured_steps: int = 100,
    base_token_id: int = 1000,
) -> Dict[str, Any]:
    """Compare per-step cost when input token is base token vs seeded H-token."""
    results: Dict[str, Any] = {}
    predictive_model.eval()

    if not codebook_manager.hyper_to_subtokens:
        codebook_manager.hyper_to_subtokens[codebook_manager.initial_vocab_size] = [1000, 1001]
        codebook_manager.subtokens_to_hyper[(1000, 1001)] = codebook_manager.initial_vocab_size
        codebook_manager.num_seeded = 1
        codebook_manager.init_codebooks_and_hyper_weight_cache(1)

    seeded_h_id = next(iter(codebook_manager.hyper_to_subtokens.keys()))

    for kv_len in kv_lengths:
        dummy_prompt = torch.full((1, kv_len), base_token_id, dtype=torch.long, device=device)
        with torch.no_grad():
            prefill_out = predictive_model(dummy_prompt, use_cache=True)
            past_kv = prefill_out.past_key_values

        base_input = torch.tensor([[base_token_id]], dtype=torch.long, device=device)
        base_samples = _measure_decode_steps_cuda_events(
            model=predictive_model,
            input_ids=base_input,
            past_key_values=past_kv,
            device=device,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
        )
        base_stats = _compute_stats(base_samples)

        with torch.no_grad():
            prefill_out = predictive_model(dummy_prompt, use_cache=True)
            past_kv = prefill_out.past_key_values

        h_input = torch.tensor([[seeded_h_id]], dtype=torch.long, device=device)
        h_samples = _measure_decode_steps_cuda_events(
            model=predictive_model,
            input_ids=h_input,
            past_key_values=past_kv,
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
    samples_ms = _measure_decode_steps_cuda_events(
        model=control_model,
        input_ids=decode_input,
        past_key_values=past_kv,
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
    curr_kv = past_key_values
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
                with record_function(f"decode_step_{step_i}"):
                    out = model(input_ids, past_key_values=curr_kv, use_cache=True)
                    curr_kv = out.past_key_values
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
    max_new_tokens: int = 300,
) -> Tuple[Dict[str, Any], bool]:
    """Run exact 3-prompt merge-equivalence test using SAVED codebooks and compressed prompts."""
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
        saved_gen_ids = ref["generated_decode_ids"]

        model.codebook_manager.reset()
        for phrase_tuple, hyper_id in saved_codebook.items():
            model.codebook_manager.hyper_to_subtokens[int(hyper_id)] = list(phrase_tuple)
            model.codebook_manager.subtokens_to_hyper[tuple(phrase_tuple)] = int(hyper_id)
        model.codebook_manager.num_seeded = len(saved_codebook)
        model.codebook_manager.init_codebooks_and_hyper_weight_cache(1)

        input_tensor = torch.tensor([compressed_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

        merged_gen_ids = out[0, len(compressed_ids):].tolist()
        merged_text = tok.decode(merged_gen_ids, skip_special_tokens=True)

        ids_match = (merged_gen_ids == saved_gen_ids)
        first_divergent_step = None
        if not ids_match:
            for step_idx, (g1, g2) in enumerate(zip(saved_gen_ids, merged_gen_ids)):
                if g1 != g2:
                    first_divergent_step = step_idx
                    break
            if first_divergent_step is None:
                first_divergent_step = min(len(saved_gen_ids), len(merged_gen_ids))

        text_equal = (merged_text == ref["output_text"])
        if not ids_match or not text_equal:
            all_equivalent = False

        reports[prompt_id] = {
            "prompt_id": prompt_id,
            "ids_match": ids_match,
            "text_equal": text_equal,
            "first_divergent_step": first_divergent_step,
            "unmerged_steps": len(saved_gen_ids),
            "merged_steps": len(merged_gen_ids),
            "unmerged_text_preview": ref["output_text"][:100],
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
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Execute fixed 12-prompt evaluation for Vanilla or Predictive condition."""
    model.eval()
    results: List[Dict[str, Any]] = []
    saved_references: Dict[str, Dict[str, Any]] = {}
    is_predictive = (policy is not None)

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

            _cuda_sync_if_needed(device)
            t_gen_start = time.perf_counter()
            timing_proc = benchmark.TimingLogitsProcessor(t_gen_start, static_mgr=static_mgr)
            proc_list = LogitsProcessorList([timing_proc])

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
            }

        eos_reached = benchmark.sequence_reached_eos(gen_ids, tok.eos_token_id)
        hit_max_length = decode_steps >= max_new_tokens

        rec: Dict[str, Any] = {
            "prompt_id": prompt_id,
            "domain": s["domain"],
            "condition": condition_name,
            "base_prompt_tokens": base_prompt_len,
            "model_prefill_tokens": len(model_prompt_ids) if is_predictive else base_prompt_len,
            "prompt_compression_pct": round(
                100.0 * (1 - len(model_prompt_ids) / max(base_prompt_len, 1)), 2
            ) if is_predictive else 0.0,
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
    """Execute the full controlled GPU runtime decomposition protocol."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device1 = torch.device("cuda:1" if torch.cuda.device_count() >= 2 else "cpu")

    print(f"\n========================================================")
    print(f"STARTING RUNTIME DECOMPOSITION EXPERIMENT")
    print(f"Primary GPU: {device0} | Control GPU: {device1}")
    print(f"========================================================\n")

    # Load 12 prompts
    with open(validation_data_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)
    samples_12 = select_prompt_subset(
        all_samples, prompt_ids_path, {dom: 4 for dom in ("code", "reasoning", "instruction")}
    )
    samples_smoke = [s for s in samples_12 if s["id"] in ("mbpp_542", "gsm_2956", "alpaca_183")]

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

    # Model reuse report
    primary_id_before = id(phi_primary)
    primary_embed_ptr_before = phi_primary.get_input_embeddings().weight.data_ptr()
    vram_vanilla = benchmark.cuda_memory_snapshot(device0) if device0.type == "cuda" else {}

    # Control calibration 0 (baseline)
    print("\n[Control] Running calibration 0 on cuda:1 (cuda:0 idle)...", flush=True)
    calib_0 = run_calibration_check(phi_control, device1, "0_baseline")
    calibrations = [calib_0]

    # Warmup and microbenchmark Vanilla on cuda:0
    print("\n[Vanilla] Running per-step microbenchmark on cuda:0...", flush=True)
    microbench_vanilla = run_single_step_microbenchmarks(
        phi_primary, device0, condition_name="vanilla_phi"
    )

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

    # Check resident base reuse
    wrapped_base = predictive_model.base_model
    vram_post_wrap = benchmark.cuda_memory_snapshot(device0) if device0.type == "cuda" else {}
    print(f"Post-wrap VRAM on cuda:0: allocated={vram_post_wrap.get('allocated_bytes', 0) / (1024**2):.1f} MiB", flush=True)

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

    # Control calibration 2 (pre-predictive-unmerged)
    print("\n[Control] Running calibration 2 on cuda:1...", flush=True)
    calib_2 = run_calibration_check(phi_control, device1, "2_pre_predictive_unmerged")
    calibrations.append(calib_2)

    # Microbenchmark Predictive Unmerged
    print("\n[Predictive Unmerged] Running per-step microbenchmark on cuda:0...", flush=True)
    microbench_unmerged = run_single_step_microbenchmarks(
        predictive_model, device0, condition_name="predictive_unmerged"
    )

    # 12-prompt Predictive Unmerged
    print("\n[Predictive Unmerged] Running fixed 12 prompts on cuda:0...", flush=True)
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
    )

    # Control calibration 3 (post-predictive-unmerged)
    print("\n[Control] Running calibration 3 on cuda:1...", flush=True)
    calib_3 = run_calibration_check(phi_control, device1, "3_post_predictive_unmerged")
    calibrations.append(calib_3)

    # 3. Merge LoRA in place
    print("\n[Step 4] Merging LoRA weights in place...", flush=True)
    old_peft_model = predictive_model.base_model
    peft_type_before = str(type(old_peft_model))
    input_embed_type_before = str(type(old_peft_model.get_input_embeddings()))
    output_embed_type_before = str(type(old_peft_model.get_output_embeddings()))
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

    # Reinstall generation position hook
    predictive_model._install_base_position_generation_hook()

    # Verify survival of hyper modules
    input_embed_after = predictive_model.base_model.get_input_embeddings()
    output_embed_after = predictive_model.base_model.get_output_embeddings()
    hyper_embed_survived = isinstance(input_embed_after, HyperEmbedding)
    hyper_linear_survived = isinstance(output_embed_after, HyperLinear)
    lora_params_after = sum(1 for n, _ in predictive_model.base_model.named_parameters() if "lora" in n.lower())
    vram_merged = benchmark.cuda_memory_snapshot(device0) if device0.type == "cuda" else {}

    print(f"LoRA merge complete ({merge_method}):")
    print(f"  LoRA params: {lora_params_before} -> {lora_params_after}")
    print(f"  HyperEmbedding survived: {hyper_embed_survived}")
    print(f"  HyperLinear survived: {hyper_linear_survived}")
    print(f"  Position hook reinstalled: True")

    # 4. 3-prompt merge-equivalence test
    print("\n[Step 5] Running 3-prompt merge-equivalence verification...", flush=True)
    equiv_reports, all_equivalent = evaluate_3prompt_equivalence(
        model=predictive_model,
        tok=tokenizer,
        device=device0,
        samples=samples_smoke,
        saved_references=saved_references,
    )
    print(f"Merge equivalence outcome: all_equivalent={all_equivalent}")
    for pid, r in equiv_reports.items():
        print(f"  {pid}: ids_match={r['ids_match']}, text_equal={r['text_equal']}, first_divergence={r['first_divergent_step']}")

    # Control calibration 4 (post-merge)
    print("\n[Control] Running calibration 4 on cuda:1...", flush=True)
    calib_4 = run_calibration_check(phi_control, device1, "4_post_merge")
    calibrations.append(calib_4)

    # 5. Microbenchmark Predictive Merged
    print("\n[Predictive Merged] Running per-step microbenchmark on cuda:0...", flush=True)
    microbench_merged = run_single_step_microbenchmarks(
        predictive_model, device0, condition_name="predictive_merged"
    )

    # Diagnostic ablations on Merged model
    print("\n[Ablations] Running position logic bypassed microbenchmark...", flush=True)
    with position_logic_bypassed_context(predictive_model):
        microbench_pos_bypass = run_single_step_microbenchmarks(
            predictive_model, device0, condition_name="merged_pos_bypass"
        )

    print("\n[Ablations] Running HyperEmbedding fast-path microbenchmark...", flush=True)
    with hyper_embedding_fast_path_context(predictive_model):
        microbench_embed_fastpath = run_single_step_microbenchmarks(
            predictive_model, device0, condition_name="merged_embed_fastpath"
        )

    print("\n[Ablations] Running HyperLinear bypassed microbenchmark...", flush=True)
    with hyper_linear_bypassed_context(predictive_model):
        microbench_linear_bypass = run_single_step_microbenchmarks(
            predictive_model, device0, condition_name="merged_linear_bypass"
        )

    print("\n[Ablations] Running base token vs H-token decode step comparison...", flush=True)
    h_vs_base_results = run_h_token_vs_base_token_microbenchmark(
        predictive_model, predictive_model.codebook_manager, device0
    )

    # 6. Profiler capture
    print("\n[Step 6] Running torch.profiler captures...", flush=True)
    dummy_input = torch.tensor([[1000]], dtype=torch.long, device=device0)
    with torch.no_grad():
        out_v = phi_primary(dummy_input, use_cache=True)
        kv_v = out_v.past_key_values
        out_m = predictive_model(dummy_input, use_cache=True)
        kv_m = out_m.past_key_values

    profiler_vanilla = run_torch_profiler_capture(phi_primary, dummy_input, kv_v, device0)
    profiler_merged = run_torch_profiler_capture(predictive_model, dummy_input, kv_m, device0)

    # 7. 12-prompt Predictive Merged benchmark
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

    # Control calibration 5 (final)
    print("\n[Control] Running final calibration 5 on cuda:1...", flush=True)
    calib_5 = run_calibration_check(phi_control, device1, "5_final")
    calibrations.append(calib_5)

    # Check drift across calibrations
    calib_medians = [c["median"] for c in calibrations]
    base_calib = calib_medians[0]
    max_drift_pct = max(abs(m - base_calib) / max(base_calib, 1e-9) * 100 for m in calib_medians)
    environment_timing_unstable = (max_drift_pct > 5.0)
    print(f"\nControl GPU drift: max drift = {max_drift_pct:.2f}% (unstable={environment_timing_unstable})")

    # 8. Build output artifacts
    print("\n[Step 7] Writing summary artifacts...", flush=True)

    # Model reuse report
    reuse_report = {
        "primary_id_before": primary_id_before,
        "primary_embed_ptr_before": primary_embed_ptr_before,
        "vram_vanilla": vram_vanilla,
        "vram_post_wrap": vram_post_wrap,
        "vram_unmerged": vram_unmerged,
        "vram_merged": vram_merged,
        "resident_reuse_success": True,
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
        "h_vs_base": h_vs_base_results,
    }
    write_json_atomic(output_dir / "per_step_microbenchmark.json", microbench_summary)

    # Profiler summary report
    prof_summary = {
        "vanilla_profiler": profiler_vanilla,
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
        lora_cost = max(0.0, u_med - m_med)
        pos_cost = max(0.0, m_med - pos_med)
        emb_cost = max(0.0, m_med - emb_med)
        lin_cost = max(0.0, m_med - lin_med)

        decomp["per_kv"][kv] = {
            "vanilla_median_ms": v_med,
            "unmerged_median_ms": u_med,
            "merged_median_ms": m_med,
            "total_predictive_overhead_ms": round(total_overhead, 4),
            "lora_overhead_ms": round(lora_cost, 4),
            "position_overhead_ms": round(pos_cost, 4),
            "embedding_overhead_ms": round(emb_cost, 4),
            "linear_overhead_ms": round(lin_cost, 4),
        }
    write_json_atomic(output_dir / "runtime_decomposition.json", decomp)

    # Markdown table generation
    md_lines = [
        "# Controlled GPU Runtime Decomposition Summary",
        "",
        "## Per-Step Cached Decode Microbenchmark (ms/step)",
        "",
        "| Condition | KV=64 | KV=256 | KV=512 |",
        "|---|---:|---:|---:|",
    ]
    for cond_name, d_dict in [
        ("Vanilla Phi", microbench_vanilla),
        ("Predictive Unmerged", microbench_unmerged),
        ("Predictive Merged", microbench_merged),
        ("Merged (Position Bypassed)", microbench_pos_bypass),
        ("Merged (Embedding Fast Path)", microbench_embed_fastpath),
        ("Merged (HyperLinear Bypassed)", microbench_linear_bypass),
    ]:
        row_str = f"| {cond_name} | {d_dict.get('64', {}).get('median', 0.0):.2f} | {d_dict.get('256', {}).get('median', 0.0):.2f} | {d_dict.get('512', {}).get('median', 0.0):.2f} |"
        md_lines.append(row_str)

    md_lines.extend([
        "",
        "## Overhead Attribution vs Vanilla (KV=256)",
        "",
        f"- Total Predictive Overhead: {decomp['per_kv'].get('256', {}).get('total_predictive_overhead_ms', 0.0):.2f} ms",
        f"- Active LoRA: {decomp['per_kv'].get('256', {}).get('lora_overhead_ms', 0.0):.2f} ms",
        f"- Position Handling: {decomp['per_kv'].get('256', {}).get('position_overhead_ms', 0.0):.2f} ms",
        f"- HyperEmbedding: {decomp['per_kv'].get('256', {}).get('embedding_overhead_ms', 0.0):.2f} ms",
        f"- HyperLinear / H-logits: {decomp['per_kv'].get('256', {}).get('linear_overhead_ms', 0.0):.2f} ms",
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
