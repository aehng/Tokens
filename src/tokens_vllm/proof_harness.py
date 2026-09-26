"""CPU checks for the vLLM proof harness.

No vLLM import. The GPU runner uses these helpers to size the preemption
pool and to stop a manual engine loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .contract import (
    BASE_VOCAB_SIZE,
    CODEBOOK_SIZE,
    INITIAL_VOCAB_SIZE,
    LOGICAL_VOCAB_SIZE,
)

NULL_KV_BLOCKS = 1
DEFAULT_BLOCK_SIZE = 16
MAX_ENGINE_STEPS = 500

BASE_GPU_MEMORY_UTILIZATION = 0.90
PREDICTIVE_GPU_MEMORY_UTILIZATION = 0.75
DEFAULT_HEADROOM_SAFETY_MARGIN_BYTES = 512 * 1024 * 1024  # 512 MiB


def inspect_parameter_footprint(
    target: Mapping[str, Any] | torch.nn.Module,
) -> dict[str, Any]:
    """Calculate exact parameter count, bytes, and dtypes using numel * element_size."""
    if isinstance(target, torch.nn.Module):
        tensors = [p for p in target.parameters() if isinstance(p, torch.Tensor)]
    elif isinstance(target, Mapping):
        tensors = [p for p in target.values() if isinstance(p, torch.Tensor)]
    else:
        raise TypeError(f"expected Mapping or nn.Module, got {type(target)}")

    count = sum(p.numel() for p in tensors)
    total_bytes = sum(p.numel() * p.element_size() for p in tensors)
    dtypes = sorted(list({str(p.dtype) for p in tensors}))
    return {
        "count": count,
        "bytes": total_bytes,
        "dtypes": dtypes,
    }


def build_encoder_memory_plan(
    encoder_source: str | Path | dict[str, Any],
    *,
    base_utilization: float = BASE_GPU_MEMORY_UTILIZATION,
    predictive_utilization: float = PREDICTIVE_GPU_MEMORY_UTILIZATION,
) -> dict[str, Any]:
    """Calculate the hyperencoder footprint plan from saved blob or dict."""
    if isinstance(encoder_source, (str, Path)):
        blob = torch.load(str(encoder_source), map_location="cpu", weights_only=False)
    elif isinstance(encoder_source, dict):
        blob = encoder_source
    else:
        raise TypeError(f"expected path or dict, got {type(encoder_source)}")

    in_state = blob.get("input_state") or blob.get("input_encoder_state_dict") or blob.get("input_encoder")
    out_state = blob.get("output_state") or blob.get("output_encoder_state_dict") or blob.get("output_encoder")
    if in_state is None or out_state is None:
        raise ValueError("encoder source is missing input or output encoder state")

    in_fp = inspect_parameter_footprint(in_state)
    out_fp = inspect_parameter_footprint(out_state)
    total_bytes = in_fp["bytes"] + out_fp["bytes"]
    dtypes = sorted(list(set(in_fp["dtypes"] + out_fp["dtypes"])))

    return {
        "input_parameter_count": in_fp["count"],
        "output_parameter_count": out_fp["count"],
        "input_parameter_bytes": in_fp["bytes"],
        "output_parameter_bytes": out_fp["bytes"],
        "total_parameter_bytes": total_bytes,
        "dtypes": dtypes,
        "base_gpu_memory_utilization": base_utilization,
        "predictive_gpu_memory_utilization": predictive_utilization,
    }


def check_encoder_headroom(
    free_gpu_bytes: int,
    total_encoder_bytes: int,
    safety_margin_bytes: int = DEFAULT_HEADROOM_SAFETY_MARGIN_BYTES,
) -> None:
    """Fail early if free GPU memory cannot accommodate encoders plus safety margin."""
    required = total_encoder_bytes + safety_margin_bytes
    if free_gpu_bytes < required:
        raise RuntimeError(
            f"insufficient reserved GPU headroom for predictive encoders: "
            f"free={free_gpu_bytes} bytes ({free_gpu_bytes / (1024**2):.2f} MiB) < "
            f"required={required} bytes ({required / (1024**2):.2f} MiB) "
            f"(encoders={total_encoder_bytes} bytes, safety_margin={safety_margin_bytes} bytes)"
        )


def blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    if num_tokens < 0 or block_size <= 0:
        raise ValueError(
            f"num_tokens={num_tokens} block_size={block_size} is not usable"
        )
    if num_tokens == 0:
        return 0
    return (num_tokens + block_size - 1) // block_size


def new_tokens_to_cross_block(prompt_len: int, block_size: int) -> int:
    """Smallest generation length that allocates one block past the prompt."""
    if prompt_len <= 0 or block_size <= 0:
        raise ValueError(
            f"prompt_len={prompt_len} block_size={block_size} is not usable"
        )
    remainder = prompt_len % block_size
    if remainder == 0:
        return 1
    return block_size - remainder + 1


def preemption_block_budget(
    prompt_a: int,
    prompt_b: int,
    block_size: int,
    *,
    null_blocks: int = NULL_KV_BLOCKS,
) -> dict[str, int]:
    """Size a KV pool so each prompt fits and the pair cannot stay resident.

    vLLM 0.30.0 full attention with prefix caching off:

    - the block pool holds one unusable null block
    - a waiting request is admitted only when its prompt fits
    - a later token that crosses a block boundary allocates one more block
    - a running request that cannot allocate preempts another running request

    Usable blocks equal the sum of the two prompt block counts. Generation
    is long enough that each sequence needs one extra block, so the pair
    does not fit and one request is preempted.
    """
    if null_blocks < 1:
        raise ValueError("the null block must be reserved")
    a_prompt_blocks = blocks_for_tokens(prompt_a, block_size)
    b_prompt_blocks = blocks_for_tokens(prompt_b, block_size)
    if a_prompt_blocks < 1 or b_prompt_blocks < 1:
        raise ValueError("both prompts must occupy at least one KV block")
    max_new = max(
        new_tokens_to_cross_block(prompt_a, block_size),
        new_tokens_to_cross_block(prompt_b, block_size),
    )
    a_full = blocks_for_tokens(prompt_a + max_new, block_size)
    b_full = blocks_for_tokens(prompt_b + max_new, block_size)
    usable = a_prompt_blocks + b_prompt_blocks
    max_model_len = max(prompt_a, prompt_b) + max_new
    if blocks_for_tokens(max_model_len, block_size) > usable:
        raise ValueError("max_model_len does not fit in the usable block pool")
    if a_full > usable or b_full > usable:
        raise ValueError("a request alone does not fit in the usable block pool")
    if a_full + b_full <= usable:
        raise ValueError("both full sequences fit; preemption is not forced")
    return {
        "block_size": block_size,
        "null_blocks": null_blocks,
        "usable_blocks": usable,
        "num_gpu_blocks": usable + null_blocks,
        "prompt_a_tokens": prompt_a,
        "prompt_b_tokens": prompt_b,
        "a_prompt_blocks": a_prompt_blocks,
        "b_prompt_blocks": b_prompt_blocks,
        "max_new_tokens": max_new,
        "a_full_blocks": a_full,
        "b_full_blocks": b_full,
        "combined_full_blocks": a_full + b_full,
        "max_model_len": max_model_len,
    }


def engine_step_decision(step_index: int, max_steps: int, unfinished: bool) -> str:
    """Decide whether a manual ``engine.step`` loop may continue.

    ``step_index`` is the number of steps already taken. ``max_steps``
    unfinished iterations return ``budget_exceeded``.
    """
    if step_index < 0 or max_steps < 1:
        raise ValueError(
            f"step_index={step_index} max_steps={max_steps} is not usable"
        )
    if not unfinished:
        return "stop"
    if step_index >= max_steps:
        return "budget_exceeded"
    return "continue"


def request_prefill_trace(
    position_trace: list[dict[str, Any]],
    request_id: str,
    prefill_token_count: int,
) -> list[dict[str, Any]]:
    """Select only prefill batches for one request, excluding prior and decode rows."""
    selected = []
    for event in position_trace:
        if event.get("target_request_id") != request_id:
            continue
        target_rows = [
            row
            for row in event.get("requests", [])
            if row.get("request_id") == request_id
        ]
        if not target_rows:
            continue
        if all(
            int(row.get("num_computed_tokens", prefill_token_count))
            >= prefill_token_count
            for row in target_rows
        ):
            continue
        selected.append(event)
    return selected


def position_values_for_stage(
    position_trace: list[dict[str, Any]], stage: str
) -> list[int]:
    """Flatten the target-request values captured at one handoff stage."""
    values: list[int] = []
    source_keys = {
        "A": "stock_input_batch_positions",
        "B": "calculated_semantic_positions",
        "C": "returned_positions",
    }
    for event in position_trace:
        if stage in source_keys:
            snapshot = event.get(source_keys[stage]) or {}
            values.extend(int(value) for value in snapshot.get("target_values") or [])
            continue
        calls = (event.get("handoff") or {}).get(stage) or []
        for call in calls:
            values.extend(int(value) for value in call.get("target_values") or [])
    return values


def preemption_cycle(events: list[dict], request_id: str) -> bool:
    """True when ``request_id`` was added, removed, then added again."""
    seen_add = False
    seen_remove = False
    for event in events:
        if event.get("req_id") != request_id:
            continue
        kind = event.get("event")
        if kind == "add" and not seen_add:
            seen_add = True
        elif kind == "remove" and seen_add:
            seen_remove = True
        elif kind == "add" and seen_remove:
            return True
    return False


def preempted_request_ids(events: list[dict]) -> list[str]:
    """Request ids whose admission log shows add, remove, re-add."""
    ordered: list[str] = []
    for event in events:
        req_id = event.get("req_id")
        if req_id is not None and req_id not in ordered:
            ordered.append(req_id)
    return [req_id for req_id in ordered if preemption_cycle(events, req_id)]


import contextlib


def assert_lora_merged_and_unloaded(model: Any) -> None:
    """Assert that LoRA adapters have been merged and unloaded from the base model."""
    base = getattr(model, "base_model", model)
    base_cls_name = type(base).__name__
    if "PeftModel" in base_cls_name:
        raise AssertionError(f"base_model is still a PEFT model: {base_cls_name}")
    if hasattr(base, "peft_config") and getattr(base, "peft_config"):
        raise AssertionError("base_model still has peft_config")

    for name, module in base.named_modules():
        mod_type = type(module).__name__
        if "LoraLayer" in mod_type or "LoraLinear" in mod_type:
            raise AssertionError(f"module {name} is still a LoRA layer: {mod_type}")
        if hasattr(module, "lora_A") and getattr(module, "lora_A") is not None:
            lora_a = getattr(module, "lora_A")
            if isinstance(lora_a, (torch.nn.Parameter, torch.Tensor)):
                raise AssertionError(f"module {name} still has active lora_A weights")
            if isinstance(lora_a, torch.nn.Module) and list(lora_a.parameters()):
                raise AssertionError(f"module {name} still has active lora_A weights")


@contextlib.contextmanager
def disable_hyper_modules(model: Any):
    """Temporarily replace HyperEmbedding and HyperLinear with standard PyTorch modules.

    This ensures that forward passes and generation execute standard HuggingFace
    base architecture without any hypertoken input embedding or output projection logic.
    """
    base_model = getattr(model, "base_model", model)
    embed_parent = getattr(base_model, "model", base_model)
    orig_embed = (
        getattr(embed_parent, "embed_tokens", None)
        or base_model.get_input_embeddings()
    )
    orig_lm_head = (
        getattr(base_model, "lm_head", None)
        or base_model.get_output_embeddings()
    )

    plain_embed = torch.nn.Embedding(
        orig_embed.num_embeddings,
        orig_embed.embedding_dim,
        padding_idx=orig_embed.padding_idx,
        _weight=orig_embed.weight,
    )
    plain_lm_head = torch.nn.Linear(
        orig_lm_head.in_features,
        orig_lm_head.out_features,
        bias=(orig_lm_head.bias is not None),
        device=orig_lm_head.weight.device,
        dtype=orig_lm_head.weight.dtype,
    )
    plain_lm_head.weight = orig_lm_head.weight
    if orig_lm_head.bias is not None:
        plain_lm_head.bias = orig_lm_head.bias

    if hasattr(embed_parent, "embed_tokens"):
        embed_parent.embed_tokens = plain_embed
    if hasattr(base_model, "set_input_embeddings"):
        base_model.set_input_embeddings(plain_embed)

    if hasattr(base_model, "lm_head"):
        base_model.lm_head = plain_lm_head
    if hasattr(base_model, "set_output_embeddings"):
        base_model.set_output_embeddings(plain_lm_head)

    try:
        yield base_model
    finally:
        if hasattr(embed_parent, "embed_tokens"):
            embed_parent.embed_tokens = orig_embed
        if hasattr(base_model, "set_input_embeddings"):
            base_model.set_input_embeddings(orig_embed)

        if hasattr(base_model, "lm_head"):
            base_model.lm_head = orig_lm_head
        if hasattr(base_model, "set_output_embeddings"):
            base_model.set_output_embeddings(orig_lm_head)


def compute_step_parity_metric(
    *,
    step: int,
    prefix_length: int,
    hf_position: int,
    vllm_position: int,
    hf_logits: torch.Tensor,
    vllm_logits: torch.Tensor,
    is_predictive: bool = True,
) -> dict[str, Any]:
    """Compute exact parity metrics between HF and vLLM next-token logits for one step."""
    ref = hf_logits.float()
    act = vllm_logits.float()

    if ref.ndim > 1:
        ref = ref.squeeze(0)
    if act.ndim > 1:
        act = act.squeeze(0)

    if not is_predictive:
        # Baseline must compare strictly physical/base vocabulary (32064)
        if ref.shape[-1] != BASE_VOCAB_SIZE or act.shape[-1] != BASE_VOCAB_SIZE:
            raise ValueError(
                f"baseline requires vocab width {BASE_VOCAB_SIZE}, got HF={ref.shape[-1]} vLLM={act.shape[-1]}"
            )
        delta = (ref - act).abs()
        max_abs = float(delta.max().item())
        mean_abs = float(delta.mean().item())
        base_max_abs = max_abs
        h_max_abs = None
    else:
        # Predictive experiments compare strictly logical vocabulary (32096)
        if ref.shape[-1] != LOGICAL_VOCAB_SIZE or act.shape[-1] != LOGICAL_VOCAB_SIZE:
            raise ValueError(
                f"predictive requires vocab width {LOGICAL_VOCAB_SIZE}, got HF={ref.shape[-1]} vLLM={act.shape[-1]}"
            )
        delta = (ref - act).abs()
        max_abs = float(delta.max().item())
        mean_abs = float(delta.mean().item())
        base_delta = torch.cat(
            (
                delta[:INITIAL_VOCAB_SIZE],
                delta[INITIAL_VOCAB_SIZE + CODEBOOK_SIZE :],
            )
        )
        base_max_abs = float(base_delta.max().item()) if base_delta.numel() else 0.0
        h_delta = delta[INITIAL_VOCAB_SIZE : INITIAL_VOCAB_SIZE + CODEBOOK_SIZE]
        h_max_abs = float(h_delta.max().item()) if h_delta.numel() else 0.0

    ref_top2 = ref.topk(2)
    act_top2 = act.topk(2)
    hf_top1_id = int(ref_top2.indices[0].item())
    hf_top2_id = int(ref_top2.indices[1].item())
    hf_top1_logit = float(ref_top2.values[0].item())
    hf_top2_logit = float(ref_top2.values[1].item())
    hf_margin = float(hf_top1_logit - hf_top2_logit)

    vllm_top1_id = int(act_top2.indices[0].item())
    vllm_top2_id = int(act_top2.indices[1].item())
    vllm_top1_logit = float(act_top2.values[0].item())
    vllm_top2_logit = float(act_top2.values[1].item())
    vllm_margin = float(vllm_top1_logit - vllm_top2_logit)

    top1_match = bool(hf_top1_id == vllm_top1_id)
    position_match = bool(hf_position == vllm_position)

    hf_top5_ids = ref.topk(5).indices.tolist()
    vllm_top5_ids = act.topk(5).indices.tolist()
    top5_overlap = len(set(hf_top5_ids).intersection(vllm_top5_ids))

    return {
        "step": step,
        "reference_input_prefix_length": prefix_length,
        "hf_position": hf_position,
        "vllm_position": vllm_position,
        "position_match": position_match,
        "hf_top1_id": hf_top1_id,
        "vllm_top1_id": vllm_top1_id,
        "top1_match": top1_match,
        "hf_top1_logit": hf_top1_logit,
        "vllm_top1_logit": vllm_top1_logit,
        "hf_top2_id": hf_top2_id,
        "vllm_top2_id": vllm_top2_id,
        "hf_margin": hf_margin,
        "vllm_margin": vllm_margin,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "base_vocab_max_abs": base_max_abs,
        "h_vocab_max_abs": h_max_abs,
        "hf_top5_ids": hf_top5_ids,
        "vllm_top5_ids": vllm_top5_ids,
        "top5_overlap": top5_overlap,
        "compared_vocab_width": ref.shape[-1],
    }
