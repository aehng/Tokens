"""Logical/physical vocabulary and semantic-position contract.

These functions reproduce the prepared Hugging Face fast-inference layout
without allocating a 32096-row embedding table. They do not import vLLM.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch

INITIAL_VOCAB_SIZE = 32011
CODEBOOK_SIZE = 32
BASE_VOCAB_SIZE = 32064
LOGICAL_VOCAB_SIZE = BASE_VOCAB_SIZE + CODEBOOK_SIZE  # 32096
MAX_SUBTOKENS = 3
MAX_POSITION_EMBEDDINGS = 131072
SAFE_PHYSICAL_ID = 0

H_START = INITIAL_VOCAB_SIZE
H_END = INITIAL_VOCAB_SIZE + CODEBOOK_SIZE  # 32043, exclusive


def logical_kind(logical_id: int) -> str:
    """Return ``base``, ``hypertoken``, or ``shifted_tail``."""
    if logical_id < 0 or logical_id >= LOGICAL_VOCAB_SIZE:
        raise ValueError(
            f"logical id {logical_id} is outside [0, {LOGICAL_VOCAB_SIZE})"
        )
    if logical_id < H_START:
        return "base"
    if logical_id < H_END:
        return "hypertoken"
    return "shifted_tail"


def physical_embed_id(logical_id: int) -> int:
    """Physical Phi row to read before any hypertoken overwrite.

    Hypertoken ids are replaced with ``SAFE_PHYSICAL_ID`` so the physical
    embedding never receives an id at or above ``BASE_VOCAB_SIZE``.
    """
    kind = logical_kind(logical_id)
    if kind == "base":
        physical = logical_id
    elif kind == "hypertoken":
        physical = SAFE_PHYSICAL_ID
    else:
        physical = logical_id - CODEBOOK_SIZE
    if physical < 0 or physical >= BASE_VOCAB_SIZE:
        raise ValueError(
            f"logical id {logical_id} mapped to physical id {physical}, "
            f"outside [0, {BASE_VOCAB_SIZE})"
        )
    return physical


def hypertoken_slot(logical_id: int) -> int | None:
    if H_START <= logical_id < H_END:
        return logical_id - H_START
    return None


def expand_logical_id(logical_id: int, phrases: Sequence[Sequence[int]]) -> list[int]:
    """Expand one logical id to base tokenizer ids."""
    kind = logical_kind(logical_id)
    if kind == "hypertoken":
        slot = logical_id - H_START
        if slot >= len(phrases):
            raise ValueError(f"hypertoken slot {slot} is missing from the codebook")
        return [int(token_id) for token_id in phrases[slot]]
    if kind == "shifted_tail":
        return [logical_id - CODEBOOK_SIZE]
    return [logical_id]


def expand_logical_ids(
    logical_ids: Sequence[int], phrases: Sequence[Sequence[int]]
) -> list[int]:
    expanded: list[int] = []
    for logical_id in logical_ids:
        expanded.extend(expand_logical_id(int(logical_id), phrases))
    return expanded


def insert_hypertoken_logits(
    base_logits: torch.Tensor, h_logits: torch.Tensor
) -> torch.Tensor:
    """Insert H logits at the trained position.

    ``base_logits`` is ``[..., 32064]`` and ``h_logits`` is ``[..., 32]``.
    The result is ``[..., 32096]``, matching ``HyperLinear`` concatenation.
    """
    if base_logits.shape[-1] != BASE_VOCAB_SIZE:
        raise ValueError(
            f"base logits width {base_logits.shape[-1]} != {BASE_VOCAB_SIZE}"
        )
    if h_logits.shape[-1] != CODEBOOK_SIZE:
        raise ValueError(
            f"hypertoken logits width {h_logits.shape[-1]} != {CODEBOOK_SIZE}"
        )
    if base_logits.shape[:-1] != h_logits.shape[:-1]:
        raise ValueError(
            f"logit batch shapes differ: {tuple(base_logits.shape)} vs "
            f"{tuple(h_logits.shape)}"
        )
    return torch.cat(
        (
            base_logits[..., :INITIAL_VOCAB_SIZE],
            h_logits,
            base_logits[..., INITIAL_VOCAB_SIZE:],
        ),
        dim=-1,
    )


def token_request_indices(
    idx_mapping: Sequence[int], query_start_loc: Sequence[int]
) -> list[int]:
    """Map each model-input token row to a stable request slot.

    ``idx_mapping[batch_row]`` is the request-state index. ``query_start_loc``
    has one more entry than ``idx_mapping`` and is an exclusive prefix sum.
    """
    if len(query_start_loc) != len(idx_mapping) + 1:
        raise ValueError(
            "query_start_loc must have one more entry than idx_mapping, "
            f"got {len(query_start_loc)} and {len(idx_mapping)}"
        )
    owners: list[int] = []
    for batch_row, req_index in enumerate(idx_mapping):
        start = int(query_start_loc[batch_row])
        end = int(query_start_loc[batch_row + 1])
        if end < start:
            raise ValueError(
                f"query_start_loc decreased at batch row {batch_row}: {start} -> {end}"
            )
        owners.extend([int(req_index)] * (end - start))
    if owners and int(query_start_loc[-1]) != len(owners):
        raise ValueError(
            f"query_start_loc ends at {query_start_loc[-1]}, not {len(owners)}"
        )
    return owners


def spans_for_logical_ids(
    logical_ids: Sequence[int],
    req_indices: Sequence[int],
    h_spans: torch.Tensor,
) -> list[int]:
    """Span of 1 for base and shifted-tail ids, codebook span for H ids."""
    if len(logical_ids) != len(req_indices):
        raise ValueError("logical ids and request indices must have the same length")
    spans: list[int] = []
    for logical_id, req_index in zip(logical_ids, req_indices):
        logical_id = int(logical_id)
        # Reject ids the sampler is not allowed to emit. This helper is for
        # CPU reconstruction and tests; the decode hot path stays on tensors.
        logical_kind(logical_id)
        slot = hypertoken_slot(logical_id)
        if slot is None:
            spans.append(1)
            continue
        span = int(h_spans[int(req_index), slot].item())
        if span not in (2, 3):
            raise ValueError(
                f"request {req_index} slot {slot} has span {span}, expected 2 or 3"
            )
        spans.append(span)
    return spans


def semantic_positions(
    spans: Sequence[int],
    req_indices: Sequence[int],
    semantic_offset: Sequence[int],
) -> list[int]:
    """Positions for one scheduled chunk, without committing offsets.

    Within a request, position is ``offset + cumsum(span) - 1``. Requests are
    interleaved by ``req_indices``; each request has its own running sum.
    """
    if len(spans) != len(req_indices):
        raise ValueError("spans and request indices must have the same length")
    running: dict[int, int] = {}
    positions: list[int] = []
    for span, req_index in zip(spans, req_indices):
        req_index = int(req_index)
        cursor = running.get(req_index, int(semantic_offset[req_index]))
        position = cursor + int(span) - 1
        if position < 0 or position >= MAX_POSITION_EMBEDDINGS:
            raise ValueError(
                f"semantic position {position} is outside "
                f"[0, {MAX_POSITION_EMBEDDINGS})"
            )
        positions.append(position)
        running[req_index] = cursor + int(span)
    return positions


def pending_advances(
    spans: Sequence[int], req_indices: Sequence[int], num_reqs: int
) -> tuple[list[int], list[int]]:
    """Semantic and physical advances for this chunk. Physical advance is 1 per token."""
    semantic = [0] * num_reqs
    physical = [0] * num_reqs
    for span, req_index in zip(spans, req_indices):
        req_index = int(req_index)
        if req_index < 0 or req_index >= num_reqs:
            raise ValueError(f"request index {req_index} is outside [0, {num_reqs})")
        semantic[req_index] += int(span)
        physical[req_index] += 1
    return semantic, physical


def reconstruct_semantic_offset(
    logical_ids: Sequence[int], h_spans_row: Sequence[int]
) -> int:
    """Sum of spans over already-accounted logical history."""
    total = 0
    for logical_id in logical_ids:
        slot = hypertoken_slot(int(logical_id))
        if slot is None:
            total += 1
        else:
            span = int(h_spans_row[slot])
            if span not in (2, 3):
                raise ValueError(f"slot {slot} has span {span}, expected 2 or 3")
            total += span
    return total


@dataclass(frozen=True)
class PredictiveCodebook:
    phrases: tuple[tuple[int, ...], ...]
    sha256: str

    @property
    def spans(self) -> tuple[int, ...]:
        return tuple(len(phrase) for phrase in self.phrases)


def _canonical_payload(phrases: Sequence[Sequence[int]]) -> bytes:
    return json.dumps(phrases, separators=(",", ":")).encode("utf-8")


def codebook_sha256(phrases: Sequence[Sequence[int]]) -> str:
    return hashlib.sha256(_canonical_payload(phrases)).hexdigest()


def validate_codebook(
    payload: Mapping[str, object],
    *,
    disabled_ids: Iterable[int] = (),
) -> PredictiveCodebook:
    """Validate the offline ``extra_args['predictive_codebook']`` payload."""
    if int(payload.get("version", 0)) != 1:
        raise ValueError("predictive_codebook.version must be 1")
    if int(payload.get("k", 0)) != CODEBOOK_SIZE:
        raise ValueError(f"predictive_codebook.k must be {CODEBOOK_SIZE}")
    raw_phrases = payload.get("phrases")
    if not isinstance(raw_phrases, list) or len(raw_phrases) != CODEBOOK_SIZE:
        raise ValueError(f"predictive_codebook.phrases must contain {CODEBOOK_SIZE} phrases")

    disabled = {int(token_id) for token_id in disabled_ids}
    phrases: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for slot, phrase in enumerate(raw_phrases):
        if not isinstance(phrase, (list, tuple)):
            raise ValueError(f"phrase {slot} must be a list of token ids")
        if len(phrase) not in (2, 3):
            raise ValueError(f"phrase {slot} length {len(phrase)} must be 2 or 3")
        tokens: list[int] = []
        for token_id in phrase:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError(f"phrase {slot} contains non-integer token {token_id!r}")
            if not 0 <= token_id < INITIAL_VOCAB_SIZE:
                raise ValueError(
                    f"phrase {slot} token {token_id} is outside [0, {INITIAL_VOCAB_SIZE})"
                )
            if token_id in disabled:
                raise ValueError(f"phrase {slot} contains disabled token {token_id}")
            tokens.append(token_id)
        key = tuple(tokens)
        if key in seen:
            raise ValueError(f"duplicate phrase at slot {slot}: {key}")
        seen.add(key)
        phrases.append(key)

    digest = codebook_sha256(phrases)
    claimed = payload.get("sha256")
    if claimed is not None and str(claimed) != digest:
        raise ValueError("predictive_codebook.sha256 does not match the phrases")
    return PredictiveCodebook(phrases=tuple(phrases), sha256=digest)
