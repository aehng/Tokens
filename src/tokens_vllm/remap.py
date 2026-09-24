"""Tensor remaps for the logical predictive vocabulary.

These ops stay on device. They do not read tensor values back to Python.
"""

from __future__ import annotations

import torch

from tokens_vllm.contract import (
    BASE_VOCAB_SIZE,
    CODEBOOK_SIZE,
    H_END,
    H_START,
    INITIAL_VOCAB_SIZE,
)


def remap_logical_ids(
    logical_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map logical ids to a physical Phi index, an H mask, and an H slot.

    Hypertoken rows are sent to physical id 0 and then overwritten. No logical
    id is passed through unchanged when it would land at or above 32064.
    """
    logical_ids = logical_ids.long()
    is_h = (logical_ids >= H_START) & (logical_ids < H_END)
    is_tail = logical_ids >= H_END
    physical = torch.where(is_tail, logical_ids - CODEBOOK_SIZE, logical_ids)
    physical = torch.where(is_h, torch.zeros_like(physical), physical)
    slots = (logical_ids - H_START).clamp(min=0, max=CODEBOOK_SIZE - 1)
    return physical, is_h, slots


def insert_h_logits(base_logits: torch.Tensor, h_logits: torch.Tensor) -> torch.Tensor:
    """Concatenate base and H logits in the trained insertion order."""
    if base_logits.shape[-1] != BASE_VOCAB_SIZE:
        raise ValueError(
            f"base logits width {base_logits.shape[-1]} != {BASE_VOCAB_SIZE}"
        )
    return torch.cat(
        (
            base_logits[..., :INITIAL_VOCAB_SIZE],
            h_logits,
            base_logits[..., INITIAL_VOCAB_SIZE:],
        ),
        dim=-1,
    )
