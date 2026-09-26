"""vLLM 0.30.0 kernel-warmup admission.

``warmup.py`` builds request ids ``_warmup_<integer>_`` with
``SamplingParams.for_sampler_warmup()``. Those requests have no codebook.
A missing codebook on any other id is still an error.
"""

from __future__ import annotations

import re

import torch

_WARMUP_REQUEST = re.compile(r"^_warmup_[0-9]+_$")


def is_vllm_internal_warmup_request(req_id: str) -> bool:
    """True only for the pinned ``_warmup_<integer>_`` ids."""
    return bool(_WARMUP_REQUEST.fullmatch(req_id))


def admission_kind(req_id: str, extra: dict | None) -> str:
    """Choose codebook synthesis, internal warmup, or a hard failure.

    A real payload wins even if the id looks like warmup. Anything else
    without that exact warmup id raises.
    """
    payload = (extra or {}).get("predictive_codebook")
    if isinstance(payload, dict):
        return "codebook"
    if is_vllm_internal_warmup_request(req_id):
        return "warmup"
    raise RuntimeError(
        f"request {req_id} is missing extra_args['predictive_codebook']"
    )


def prepare_warmup_slot(state, req_index: int, num_computed_tokens: int) -> None:
    """Base-token slot: no H vectors, span 1, predictive H inactive."""
    state.h_input[req_index].zero_()
    state.h_output[req_index].zero_()
    state.h_spans[req_index].fill_(1)
    state.semantic_offset[req_index] = num_computed_tokens
    state.physical_accounted[req_index] = num_computed_tokens
    state.h_active[req_index] = False


def activate_codebook_slot(state, req_index: int) -> None:
    state.h_active[req_index] = True


def clear_predictive_slot(state, req_index: int) -> None:
    state.h_input[req_index].zero_()
    state.h_output[req_index].zero_()
    state.h_spans[req_index].zero_()
    state.semantic_offset[req_index] = 0
    state.physical_accounted[req_index] = 0
    state.h_active[req_index] = False
    if hasattr(state, "pending_semantic_advance"):
        state.pending_semantic_advance[req_index] = 0
    if hasattr(state, "pending_physical_advance"):
        state.pending_physical_advance[req_index] = 0


def predictive_slot_clear_report(state, req_index: int) -> dict:
    """Audit that every predictive tensor and activity flag is clear.

    This performs device reductions and is intended for explicitly enabled proof
    instrumentation only; ordinary request admission should not call it.
    """
    checks = {
        "h_spans_clear": bool(torch.count_nonzero(state.h_spans[req_index]).item() == 0),
        "h_input_clear": bool(torch.count_nonzero(state.h_input[req_index]).item() == 0),
        "h_output_clear": bool(torch.count_nonzero(state.h_output[req_index]).item() == 0),
        "h_active_false": not bool(state.h_active[req_index].item()),
    }
    for name in (
        "semantic_offset",
        "physical_accounted",
        "pending_semantic_advance",
        "pending_physical_advance",
    ):
        tensor = getattr(state, name, None)
        if tensor is not None:
            checks[f"{name}_zero"] = bool(torch.count_nonzero(tensor[req_index]).item() == 0)
    return {"valid": all(checks.values()), "checks": checks}


def mask_inactive_h_logits(h_logits: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Keep H logits for active rows and set inactive rows to -inf."""
    gate = active.to(dtype=torch.bool).unsqueeze(-1)
    fill = torch.full_like(h_logits, float("-inf"))
    return torch.where(gate, h_logits, fill)
