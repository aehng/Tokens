"""Exact Expanded-Cache / Block Decoding Control Engine (Arm B).

Tests whether evaluating an N-token phrase [A, B, ...] as a single causal block forward:
  1. Produces identical next-token logits and KV cache entries as N serial forwards.
  2. Speeds up decode time by collapsing N serial decode rounds into 1 block forward.
  3. Evaluates block sizes 2, 3, and 4.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def clone_cache(cache: Any) -> Any:
    """Safely clone past_key_values supporting DynamicCache and legacy tuples."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        # Transformers DynamicCache
        try:
            from transformers import DynamicCache
            new_cache = DynamicCache()
            new_cache.key_cache = [k.clone() for k in cache.key_cache]
            new_cache.value_cache = [v.clone() for v in cache.value_cache]
            if hasattr(cache, "_seen_tokens"):
                new_cache._seen_tokens = cache._seen_tokens
            return new_cache
        except Exception:
            return copy.deepcopy(cache)
    elif isinstance(cache, (list, tuple)):
        return tuple(tuple(t.clone() for t in layer) for layer in cache)
    else:
        return copy.deepcopy(cache)


def get_cache_seq_len(cache: Any) -> int:
    """Extract sequence length stored in cache."""
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    elif hasattr(cache, "key_cache") and cache.key_cache:
        return int(cache.key_cache[0].shape[-2])
    elif isinstance(cache, (list, tuple)) and cache and cache[0]:
        return int(cache[0][0].shape[-2])
    return 0


def get_layer_kv(cache: Any, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Retrieve (key, value) tensors for specified layer index."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    elif isinstance(cache, (list, tuple)):
        return cache[layer_idx][0], cache[layer_idx][1]
    raise ValueError(f"Unrecognized cache type: {type(cache)}")


def compute_kl_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    temperature: float = 1.0,
) -> float:
    """Compute KL(P || Q) in nats between two logit vectors."""
    p_log = F.log_softmax(p_logits / temperature, dim=-1)
    q_log = F.log_softmax(q_logits / temperature, dim=-1)
    p_probs = F.softmax(p_logits / temperature, dim=-1)
    kl = F.kl_div(q_log, p_probs, reduction="sum", log_target=False)
    return float(kl.detach().cpu().item())


@dataclass
class BlockCorrectnessReport:
    block_size: int
    phrase_tokens: List[int]
    top1_matches: bool
    top1_serial_id: int
    top1_block_id: int
    kl_serial_to_block_nats: float
    max_abs_logit_diff: float
    mean_abs_logit_diff: float
    serial_cache_seq_len: int
    block_cache_seq_len: int
    cache_lengths_match: bool
    per_layer_kv_max_diff: List[float]
    per_layer_kv_mean_diff: List[float]
    max_overall_kv_diff: float
    mean_overall_kv_diff: float
    behaviorally_equivalent: bool


@dataclass
class BlockTimingReport:
    block_size: int
    repetitions: int
    serial_median_ms: float
    serial_p90_ms: float
    block_median_ms: float
    block_p90_ms: float
    speedup_ratio_median: float
    speedup_ratio_p90: float
    latency_reduction_pct_median: float


def evaluate_block_cache_correctness(
    model: Any,
    prefix_ids: Sequence[int],
    phrase_tokens: Sequence[int],
    device: torch.device,
) -> BlockCorrectnessReport:
    """Compare serial 1-token decode vs 1-call block forward from identical prefix cache."""
    N = len(phrase_tokens)
    C = len(prefix_ids)
    if N < 2:
        raise ValueError("Phrase must contain at least 2 tokens for block testing")

    # 1. Warm prefix cache
    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    positions_prefix = torch.arange(0, C, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        prefix_out = model(
            input_ids=prefix_tensor,
            position_ids=positions_prefix,
            use_cache=True,
        )
    prefix_cache = prefix_out.past_key_values

    # 2. SERIAL PATH (N consecutive single-token forwards)
    serial_cache = clone_cache(prefix_cache)
    curr_pos = C
    out_serial = None
    with torch.no_grad():
        for token_id in phrase_tokens:
            inp = torch.tensor([[token_id]], dtype=torch.long, device=device)
            pos = torch.tensor([[curr_pos]], dtype=torch.long, device=device)
            out_serial = model(
                input_ids=inp,
                position_ids=pos,
                past_key_values=serial_cache,
                use_cache=True,
            )
            serial_cache = out_serial.past_key_values
            curr_pos += 1

    serial_logits = out_serial.logits[0, -1]  # Logits after final token in phrase

    # 3. BLOCK PATH (1 forward of length N)
    block_cache = clone_cache(prefix_cache)
    block_input = torch.tensor([phrase_tokens], dtype=torch.long, device=device)
    block_pos = torch.arange(C, C + N, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        out_block = model(
            input_ids=block_input,
            position_ids=block_pos,
            past_key_values=block_cache,
            use_cache=True,
        )
        final_block_cache = out_block.past_key_values

    block_logits = out_block.logits[0, -1]  # Logits after final token in phrase

    # 4. Compare Logits
    top1_serial = int(serial_logits.argmax().item())
    top1_block = int(block_logits.argmax().item())
    top1_match = bool(top1_serial == top1_block)
    kl_nats = compute_kl_divergence(serial_logits, block_logits)
    diff = (serial_logits - block_logits).abs()
    max_logit_diff = float(diff.max().item())
    mean_logit_diff = float(diff.mean().item())

    # 5. Compare KV Caches
    serial_len = get_cache_seq_len(serial_cache)
    block_len = get_cache_seq_len(final_block_cache)
    lengths_match = bool(serial_len == block_len == (C + N))

    num_layers = getattr(model.config, "num_hidden_layers", 32)
    per_layer_max: List[float] = []
    per_layer_mean: List[float] = []

    for l in range(num_layers):
        k_s, v_s = get_layer_kv(serial_cache, l)
        k_b, v_b = get_layer_kv(final_block_cache, l)
        k_diff = (k_s - k_b).abs()
        v_diff = (v_s - v_b).abs()
        l_max = float(max(k_diff.max().item(), v_diff.max().item()))
        l_mean = float(0.5 * (k_diff.mean().item() + v_diff.mean().item()))
        per_layer_max.append(l_max)
        per_layer_mean.append(l_mean)

    max_overall_kv = max(per_layer_max) if per_layer_max else 0.0
    mean_overall_kv = float(np.mean(per_layer_mean)) if per_layer_mean else 0.0

    # Behavioral equivalence: top-1 match and KL on numerical noise scale (< 1e-3 nats)
    equiv = bool(top1_match and kl_nats < 1e-3 and lengths_match)

    return BlockCorrectnessReport(
        block_size=N,
        phrase_tokens=list(phrase_tokens),
        top1_matches=top1_match,
        top1_serial_id=top1_serial,
        top1_block_id=top1_block,
        kl_serial_to_block_nats=kl_nats,
        max_abs_logit_diff=max_logit_diff,
        mean_abs_logit_diff=mean_logit_diff,
        serial_cache_seq_len=serial_len,
        block_cache_seq_len=block_len,
        cache_lengths_match=lengths_match,
        per_layer_kv_max_diff=per_layer_max,
        per_layer_kv_mean_diff=per_layer_mean,
        max_overall_kv_diff=max_overall_kv,
        mean_overall_kv_diff=mean_overall_kv,
        behaviorally_equivalent=equiv,
    )


def benchmark_block_cache_timing(
    model: Any,
    prefix_ids: Sequence[int],
    phrase_tokens: Sequence[int],
    device: torch.device,
    *,
    warmup_reps: int = 10,
    timed_reps: int = 50,
) -> BlockTimingReport:
    """Benchmark GPU execution latency of N serial forwards vs 1 block forward."""
    N = len(phrase_tokens)
    C = len(prefix_ids)
    use_cuda_events = (device.type == "cuda")

    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    pos_prefix = torch.arange(0, C, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        prefix_out = model(
            input_ids=prefix_tensor,
            position_ids=pos_prefix,
            use_cache=True,
        )
    prefix_cache = prefix_out.past_key_values

    block_input = torch.tensor([phrase_tokens], dtype=torch.long, device=device)
    block_pos = torch.arange(C, C + N, dtype=torch.long, device=device).unsqueeze(0)

    # Prepare single-token inputs
    single_inputs = [
        (torch.tensor([[t]], dtype=torch.long, device=device), torch.tensor([[C + i]], dtype=torch.long, device=device))
        for i, t in enumerate(phrase_tokens)
    ]

    # Warmup
    for _ in range(warmup_reps):
        c_s = clone_cache(prefix_cache)
        with torch.no_grad():
            for inp, pos in single_inputs:
                c_s = model(input_ids=inp, position_ids=pos, past_key_values=c_s, use_cache=True).past_key_values
        c_b = clone_cache(prefix_cache)
        with torch.no_grad():
            _ = model(input_ids=block_input, position_ids=block_pos, past_key_values=c_b, use_cache=True)
    if use_cuda_events:
        torch.cuda.synchronize(device)

    # Benchmark Serial
    serial_times_ms: List[float] = []
    for _ in range(timed_reps):
        c_s = clone_cache(prefix_cache)
        if use_cuda_events:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            with torch.no_grad():
                for inp, pos in single_inputs:
                    c_s = model(input_ids=inp, position_ids=pos, past_key_values=c_s, use_cache=True).past_key_values
            end_ev.record()
            end_ev.synchronize()
            serial_times_ms.append(float(start_ev.elapsed_time(end_ev)))
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                for inp, pos in single_inputs:
                    c_s = model(input_ids=inp, position_ids=pos, past_key_values=c_s, use_cache=True).past_key_values
            serial_times_ms.append((time.perf_counter() - t0) * 1000.0)

    # Benchmark Block
    block_times_ms: List[float] = []
    for _ in range(timed_reps):
        c_b = clone_cache(prefix_cache)
        if use_cuda_events:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            with torch.no_grad():
                _ = model(input_ids=block_input, position_ids=block_pos, past_key_values=c_b, use_cache=True)
            end_ev.record()
            end_ev.synchronize()
            block_times_ms.append(float(start_ev.elapsed_time(end_ev)))
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(input_ids=block_input, position_ids=block_pos, past_key_values=c_b, use_cache=True)
            block_times_ms.append((time.perf_counter() - t0) * 1000.0)

    s_med = float(np.median(serial_times_ms))
    s_p90 = float(np.percentile(serial_times_ms, 90))
    b_med = float(np.median(block_times_ms))
    b_p90 = float(np.percentile(block_times_ms, 90))

    speedup_med = s_med / max(b_med, 1e-6)
    speedup_p90 = s_p90 / max(b_p90, 1e-6)
    lat_red_pct = ((s_med - b_med) / max(s_med, 1e-6)) * 100.0

    return BlockTimingReport(
        block_size=N,
        repetitions=timed_reps,
        serial_median_ms=s_med,
        serial_p90_ms=s_p90,
        block_median_ms=b_med,
        block_p90_ms=b_p90,
        speedup_ratio_median=speedup_med,
        speedup_ratio_p90=speedup_p90,
        latency_reduction_pct_median=lat_red_pct,
    )
