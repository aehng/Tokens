"""CPU checks for the vLLM proof harness.

No vLLM import. The GPU runner uses these helpers to size the preemption
pool and to stop a manual engine loop.
"""

from __future__ import annotations

import hashlib
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
    """Smallest output length whose cached prefix crosses a KV block boundary.

    The final sampled output token is not fed through the model and therefore
    is not part of the KV cache. A request that emits ``n`` tokens caches at
    most ``prompt_len + n - 1`` tokens.
    """
    if prompt_len <= 0 or block_size <= 0:
        raise ValueError(
            f"prompt_len={prompt_len} block_size={block_size} is not usable"
        )
    remainder = prompt_len % block_size
    if remainder == 0:
        return 2
    return block_size - remainder + 2


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
    - the final sampled output token is not fed back into the KV cache
    - generation crosses the next block boundary before the final output
    - a running request that cannot allocate preempts another running request

    Usable blocks equal the sum of the two prompt block counts. Generation
    is long enough that each cached prefix needs one extra block, so the pair
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
    a_peak_tokens = prompt_a + max_new - 1
    b_peak_tokens = prompt_b + max_new - 1
    a_peak_blocks = blocks_for_tokens(a_peak_tokens, block_size)
    b_peak_blocks = blocks_for_tokens(b_peak_tokens, block_size)
    usable = a_prompt_blocks + b_prompt_blocks
    max_model_len = max(prompt_a, prompt_b) + max_new
    if blocks_for_tokens(max_model_len, block_size) > usable:
        raise ValueError("max_model_len does not fit in the usable block pool")
    if a_peak_blocks > usable or b_peak_blocks > usable:
        raise ValueError("a request alone does not fit in the usable block pool")
    if a_peak_blocks + b_peak_blocks <= usable:
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
        "a_peak_kv_tokens": a_peak_tokens,
        "b_peak_kv_tokens": b_peak_tokens,
        "a_peak_kv_blocks": a_peak_blocks,
        "b_peak_kv_blocks": b_peak_blocks,
        "combined_peak_kv_blocks": a_peak_blocks + b_peak_blocks,
        "max_model_len": max_model_len,
    }


def max_scheduler_preemptions(
    observations: list[Mapping[str, Any]], request_ids: tuple[str, ...]
) -> dict[str, int]:
    """Return the largest scheduler preemption count observed per request."""
    counts = {request_id: 0 for request_id in request_ids}
    for observation in observations:
        for queue_name in ("running", "waiting"):
            for request in observation.get(queue_name, ()):
                request_id = request.get("request_id")
                if request_id in counts:
                    counts[request_id] = max(
                        counts[request_id], int(request.get("num_preemptions", 0))
                    )
    return counts


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


def two_codebook_isolation_passes(
    *,
    a_match: bool,
    b_match: bool,
    row0_match: bool,
    row1_match: bool,
    rows_differ: bool,
) -> bool:
    """Require concurrent codebook rows to stay distinct and match their owners."""
    return bool(a_match and b_match and row0_match and row1_match and rows_differ)


def _int_list(values: Any) -> list[int]:
    if not isinstance(values, (list, tuple)):
        return []
    return [int(value) for value in values]


def _token_ids_sha256(token_ids: list[int]) -> str:
    packed = b"".join(int(token).to_bytes(8, "little", signed=True) for token in token_ids)
    return hashlib.sha256(packed).hexdigest()


def resume_position_contract_report(
    position_trace: list[dict[str, Any]],
    request_id: str,
    *,
    admission_generation: int | None,
    prefill_token_count: int,
    position_mode: str,
    expected_readmission_count: int | None = None,
    expected_prefill_token_ids_sha256: str | None = None,
) -> dict[str, Any]:
    """Prove prefill positions follow the exact scheduler row after readmission.

    This Phase 9 contract is for compressed positions. Its expected model and
    RoPE positions are the scheduler's physical prefill interval. A future
    cached-prefix admission can start at a positive computed-token count.
    """
    prompt_count = int(prefill_token_count)
    failures: list[str] = []
    if admission_generation is None:
        failures.append("resumed admission generation is missing")
    req_events = [
        event
        for event in position_trace
        if event.get("target_request_id") == request_id
        and event.get("admission_generation") == admission_generation
    ]
    batches: list[dict[str, Any]] = []
    for event in req_events:
        rows = [
            row for row in event.get("requests", [])
            if row.get("request_id") == request_id
        ]
        if len(rows) != 1:
            failures.append("trace row is not uniquely bound to request_id")
            continue
        row = rows[0]
        computed = int(row.get("num_computed_tokens", -1))
        scheduled = int(row.get("num_scheduled_tokens", -1))
        if computed < 0 or scheduled < 1:
            failures.append("scheduler computed/scheduled counts are missing or invalid")
            continue
        if computed >= prompt_count:
            # Decode records are not part of prompt recomputation.
            continue
        expected = list(range(computed, computed + scheduled))
        query_slice = _int_list(row.get("query_token_slice"))
        token_ids = _int_list(event.get("target_token_ids"))
        checks = {
            "request_id_bound": event.get("request_id") == request_id,
            "generation_bound": event.get("admission_generation") == admission_generation,
            "slot_bound": event.get("slot") == row.get("slot"),
            "row_generation_bound": row.get("admission_generation") == admission_generation,
            "scheduler_count_matches_batch_snapshot": (
                event.get("num_computed_tokens_on_batch") == computed
            ),
            "scheduler_source_bound": (
                row.get("num_computed_tokens_source")
                == "input_batch.num_computed_tokens_np[row]"
                and event.get("num_computed_tokens_source")
                == "input_batch.num_computed_tokens_np[row]"
            ),
            "query_slice_matches_scheduled": (
                len(query_slice) == 2
                and query_slice[1] - query_slice[0] == scheduled
                and event.get("target_query_slice") == query_slice
            ),
            "target_token_count_matches_scheduled": len(token_ids) == scheduled,
            "trace_position_mode_matches_admission": event.get("position_mode") == position_mode,
            "scheduled_interval_stays_inside_prefill": computed + scheduled <= prompt_count,
        }
        if not all(checks.values()):
            failures.extend(name for name, ok in checks.items() if not ok)

        stage_values: dict[str, list[int]] = {}
        for stage, key in (
            ("A", "stock_input_batch_positions"),
            ("B", "calculated_semantic_positions"),
            ("C", "returned_positions"),
        ):
            stage_values[stage] = _int_list((event.get(key) or {}).get("target_values"))
        for stage in ("D", "E", "F"):
            calls = (event.get("handoff") or {}).get(stage) or []
            stage_values[stage] = [
                int(value)
                for call in calls
                for value in (call.get("target_values") or [])
            ]
        stage_matches = {
            stage: values == expected for stage, values in stage_values.items()
        }
        if position_mode != "compressed":
            failures.append("Phase 9 scheduler-position proof requires compressed mode")
        if not all(stage_matches.values()):
            failures.extend(f"stage_{stage}_positions_do_not_match_scheduler" for stage, ok in stage_matches.items() if not ok)
        batches.append(
            {
                "num_computed_tokens": computed,
                "num_scheduled_tokens": scheduled,
                "expected_positions": expected,
                "stage_positions": stage_values,
                "stage_matches": stage_matches,
                "checks": checks,
            }
        )

    readmission_count = (
        int(batches[0]["num_computed_tokens"]) if batches else None
    )
    resume_strategy = (
        "recompute_from_zero"
        if readmission_count == 0
        else "cached_prefix"
        if readmission_count is not None and readmission_count > 0
        else None
    )
    trace_complete = False
    trace_token_ids: list[int] = []
    cursor = readmission_count
    if batches and cursor is not None:
        for batch in batches:
            if batch["num_computed_tokens"] != cursor:
                failures.append("scheduler prefill chunks are not contiguous")
            cursor = batch["num_computed_tokens"] + batch["num_scheduled_tokens"]
        trace_complete = cursor == prompt_count
        if not trace_complete:
            failures.append("resumed prefill trace does not reach full prompt length")
        for event in req_events:
            rows = [row for row in event.get("requests", []) if row.get("request_id") == request_id]
            if rows and int(rows[0].get("num_computed_tokens", -1)) < prompt_count:
                trace_token_ids.extend(_int_list(event.get("target_token_ids")))
    if expected_readmission_count is not None and readmission_count != int(expected_readmission_count):
        failures.append("readmission scheduler count differs from configured expectation")
    if readmission_count is None:
        failures.append("no resumed prefill position trace was captured")
    if (
        expected_prefill_token_ids_sha256
        and readmission_count == 0
        and _token_ids_sha256(trace_token_ids) != expected_prefill_token_ids_sha256
    ):
        failures.append("zero-count recompute trace does not cover the admitted token history")

    stage_names = ("A", "B", "C", "D", "E", "F")
    all_positions_match = bool(batches) and all(
        all(batch["stage_matches"].get(stage) is True for stage in stage_names)
        for batch in batches
    )
    return {
        "resume_strategy": resume_strategy,
        "num_computed_tokens_on_readmission": readmission_count,
        "source": "scheduler num_computed_tokens",
        "trace_complete": trace_complete,
        "all_model_positions_match_expected": all_positions_match,
        "prefill_token_history_hash_match": (
            expected_prefill_token_ids_sha256 is None
            or readmission_count != 0
            or _token_ids_sha256(trace_token_ids) == expected_prefill_token_ids_sha256
        ),
        "batches": batches,
        "failures": sorted(set(failures)),
        "valid": bool(
            trace_complete
            and all_positions_match
            and not failures
            and (expected_readmission_count is None or readmission_count == int(expected_readmission_count))
        ),
    }


def preemption_rebuild_report(
    events: list[dict],
    request_id: str,
    *,
    scheduler_preemption_count: int = 0,
    position_resume: dict[str, Any] | None = None,
    trajectory_match: bool = False,
    expected_readmission_count: int | None = None,
) -> dict[str, Any]:
    """Prove a preempted request rebuilt predictive state and replayed positions."""
    request_events = [event for event in events if event.get("req_id") == request_id]
    additions = [event for event in request_events if event.get("event") == "add"]
    cycle = preemption_cycle(events, request_id)
    initial = additions[0] if additions else None
    resumed = additions[1] if cycle and len(additions) >= 2 else None
    removal = None
    if initial is not None and resumed is not None:
        initial_index = events.index(initial)
        resumed_index = events.index(resumed)
        removal = next(
            (
                event for event in events[initial_index + 1 : resumed_index]
                if event.get("req_id") == request_id and event.get("event") == "remove"
            ),
            None,
        )
    identity_fields = (
        "h_enabled",
        "position_mode",
        "sha256",
        "h_input_hash",
        "h_output_hash",
        "spans",
        "h_active_after_synthesis",
    )
    identity_recorded = bool(
        initial
        and resumed
        and all(
            initial.get(field) is not None and resumed.get(field) is not None
            for field in identity_fields
        )
    )
    predictive_checks = {
        "codebook_sha_match": bool(identity_recorded and initial["sha256"] == resumed["sha256"]),
        "spans_match": bool(identity_recorded and initial["spans"] == resumed["spans"]),
        "h_input_match": bool(identity_recorded and initial["h_input_hash"] == resumed["h_input_hash"]),
        "h_output_match": bool(identity_recorded and initial["h_output_hash"] == resumed["h_output_hash"]),
        "position_mode_match": bool(identity_recorded and initial["position_mode"] == resumed["position_mode"]),
        "h_enabled": bool(
            identity_recorded
            and initial["h_enabled"] is True
            and resumed["h_enabled"] is True
        ),
        "h_active": bool(
            identity_recorded
            and initial["h_active_after_synthesis"] is True
            and resumed["h_active_after_synthesis"] is True
        ),
    }
    predictive_rebuild_valid = all(predictive_checks.values())
    initial_clear = bool(
        initial
        and initial.get("slot_clear_before_synthesis", {}).get("valid") is True
    )
    removal_clear = bool(
        removal
        and removal.get("slot_clear_after_remove", {}).get("valid") is True
    )
    resumed_clear = bool(
        resumed
        and resumed.get("slot_clear_before_synthesis", {}).get("valid") is True
    )
    slot_clear = {
        "valid": bool(initial_clear and removal_clear and resumed_clear),
        "initial_admission_clear": initial_clear,
        "clear_after_remove": removal_clear,
        "clear_before_resynthesis": resumed_clear,
        "resumed_slot_was_clear": bool(resumed and resumed.get("was_clear") is True),
    }
    already = int(resumed.get("already", -1)) if resumed else None
    resume_strategy = (
        "recompute_from_zero"
        if already == 0
        else "cached_prefix"
        if already is not None and already > 0
        else None
    )
    readmission_count_valid = bool(
        already is not None
        and already >= 0
        and (expected_readmission_count is None or already == int(expected_readmission_count))
        and position_resume is not None
        and position_resume.get("num_computed_tokens_on_readmission") == already
    )
    scheduler_confirmed = int(scheduler_preemption_count) > 0
    position_valid = bool(position_resume and position_resume.get("valid") is True)
    state_rebuilt = bool(
        scheduler_confirmed
        and cycle
        and slot_clear["valid"]
        and predictive_rebuild_valid
        and readmission_count_valid
        and position_valid
        and trajectory_match
    )
    return {
        "request_id": request_id,
        "resume_strategy": resume_strategy,
        "num_computed_tokens_on_readmission": already,
        "expected_readmission_count": expected_readmission_count,
        "scheduler_preemption_count": int(scheduler_preemption_count),
        "scheduler_preemption_confirmed": scheduler_confirmed,
        "add_remove_add_cycle": cycle,
        "admission_events": request_events,
        "initial_add": initial,
        "resumed_add": resumed,
        "remove_event": removal,
        "codebook_identity_fields": list(identity_fields),
        "codebook_identity_recorded": identity_recorded,
        "predictive_rebuild": {**predictive_checks, "valid": predictive_rebuild_valid},
        "slot_clear": slot_clear,
        "resumed_slot_was_clear": slot_clear["resumed_slot_was_clear"],
        "position_resume": position_resume,
        "readmission_count_valid": readmission_count_valid,
        "trajectory_match": bool(trajectory_match),
        "state_rebuilt": state_rebuilt,
    }


def semantic_kv_rope_contract_report(
    *,
    physical_positions: list[int],
    semantic_positions: list[int],
    expected_semantic_positions: list[int],
    physical_token_count: int,
    max_new_tokens: int,
    max_model_len: int,
    max_rope_position: int,
) -> dict[str, Any]:
    """Check physical KV bounds separately from extended semantic RoPE positions."""
    physical = [int(position) for position in physical_positions]
    semantic = [int(position) for position in semantic_positions]
    expected_semantic = [int(position) for position in expected_semantic_positions]
    physical_positions_match = physical == list(range(physical_token_count))
    physical_positions_within_model_len = bool(physical) and all(
        0 <= position < max_model_len for position in physical
    )
    physical_sequence_fits = physical_token_count + max_new_tokens <= max_model_len
    semantic_positions_match = semantic == expected_semantic
    semantic_positions_exceed_model_len = bool(semantic) and max(semantic) > max_model_len
    semantic_positions_within_rope_limit = bool(semantic) and all(
        0 <= position < max_rope_position for position in semantic
    )
    checks = {
        "physical_positions_match_zero_based_kv_slots": physical_positions_match,
        "physical_positions_within_max_model_len": physical_positions_within_model_len,
        "physical_prompt_plus_generation_fits_max_model_len": physical_sequence_fits,
        "semantic_positions_match_expected_bte_positions": semantic_positions_match,
        "semantic_positions_exceed_max_model_len": semantic_positions_exceed_model_len,
        "semantic_positions_within_rope_limit": semantic_positions_within_rope_limit,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "physical_kv": {
            "positions": physical,
            "token_count": physical_token_count,
            "max_model_len": max_model_len,
            "prompt_plus_generation_tokens": physical_token_count + max_new_tokens,
        },
        "semantic_rope": {
            "positions": semantic,
            "expected_positions": expected_semantic,
            "max_position": max(semantic) if semantic else None,
            "max_model_len": max_model_len,
            "max_rope_position_exclusive": max_rope_position,
        },
        "checks": checks,
    }


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
