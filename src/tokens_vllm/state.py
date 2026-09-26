"""Request-local predictive state for vLLM Model Runner V2."""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
import torch

from tokens_vllm.contract import (
    CODEBOOK_SIZE,
    H_END,
    H_START,
    MAX_POSITION_EMBEDDINGS,
    chunk_positions_from_computed,
    validate_codebook,
    validate_position_mode,
)
from tokens_vllm.warmup import (
    activate_codebook_slot,
    admission_kind,
    clear_predictive_slot,
    prepare_warmup_slot,
)


class RequestScopedPositionTrace:
    """Keep position diagnostics limited to one explicitly selected request."""

    def __init__(self) -> None:
        self.target_request_id: str | None = None
        self.records: list[dict[str, Any]] = []
        self.active_record: dict[str, Any] | None = None
        self._sequence = 0

    def begin(self, request_id: str) -> None:
        self.target_request_id = str(request_id)
        self.records.clear()
        self.active_record = None
        self._sequence = 0

    def clear_active(self) -> None:
        self.active_record = None

    def capture(self, record: dict[str, Any]) -> dict[str, Any] | None:
        self.active_record = None
        target = self.target_request_id
        request_ids = [str(req_id) for req_id in record.get("req_ids", [])]
        if target is None or target not in request_ids:
            return None
        self._sequence += 1
        record["target_request_id"] = target
        record["trace_id"] = f"{target}:{self._sequence}"
        self.records.append(record)
        self.active_record = record
        return record


def position_tensor_snapshot(
    tensor: torch.Tensor,
    *,
    target_slice: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Copy a position tensor immediately and retain its storage metadata."""
    flat = tensor.detach().reshape(-1)
    values = flat.to(device="cpu").tolist()
    start, end = target_slice if target_slice is not None else (0, len(values))
    return {
        "values": values,
        "target_values": values[start:end],
        "target_slice": [int(start), int(end)],
        "shape": [int(size) for size in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "data_ptr": int(tensor.data_ptr()),
    }


def write_semantic_positions(
    buffer: torch.Tensor,
    semantic_positions: torch.Tensor,
    *,
    num_tokens: int,
    num_tokens_after_padding: int,
) -> torch.Tensor:
    """Return the model-position buffer without changing vLLM's physical positions."""
    if not 0 <= num_tokens <= num_tokens_after_padding <= buffer.numel():
        raise ValueError("position buffer lengths are inconsistent")
    semantic = semantic_positions.reshape(-1)
    if semantic.numel() != num_tokens:
        raise ValueError(
            f"semantic position count {semantic.numel()} != num_tokens {num_tokens}"
        )
    buffer[:num_tokens].copy_(semantic)
    if num_tokens_after_padding > num_tokens:
        buffer[num_tokens:num_tokens_after_padding].zero_()
    return buffer[:num_tokens_after_padding]


class PredictiveModelState:
    """Filled in by ``_build_state`` once vLLM can be imported."""


def _build_state() -> None:
    global PredictiveModelState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    class _PredictiveModelState(DefaultModelState):
        supports_prompt_embeds = False

        def __init__(self, vllm_config, model, encoder_cache, device):
            super().__init__(vllm_config, model, encoder_cache, device)
            hidden = int(model.config.hidden_size)
            self.h_input = torch.zeros(
                self.max_num_reqs, CODEBOOK_SIZE, hidden, dtype=self.dtype, device=device
            )
            self.h_output = torch.zeros(
                self.max_num_reqs, CODEBOOK_SIZE, hidden, dtype=self.dtype, device=device
            )
            self.h_spans = torch.zeros(
                self.max_num_reqs, CODEBOOK_SIZE, dtype=torch.int64, device=device
            )
            self.h_active = torch.zeros(self.max_num_reqs, dtype=torch.bool, device=device)
            self.semantic_offset = torch.zeros(
                self.max_num_reqs, dtype=torch.int64, device=device
            )
            self.physical_accounted = torch.zeros(
                self.max_num_reqs, dtype=torch.int64, device=device
            )
            self.token_req_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int64, device=device
            )
            self.logit_req_indices = torch.zeros(
                max(self.max_num_reqs, self.max_num_tokens),
                dtype=torch.int64,
                device=device,
            )
            self.pending_semantic_advance = torch.zeros(
                self.max_num_reqs, dtype=torch.int64, device=device
            )
            self.pending_physical_advance = torch.zeros(
                self.max_num_reqs, dtype=torch.int64, device=device
            )
            self.positions_buffer = torch.zeros(
                self.max_num_tokens, dtype=torch.int64, device=device
            )
            self.req_id_to_index: dict[str, int] = {}
            self.codebook_sha: dict[int, str] = {}
            self.setup_ms: dict[int, float] = {}
            self.admission_log: list[dict[str, Any]] = []
            self.last_positions: torch.Tensor | None = None
            self.position_trace = RequestScopedPositionTrace()
            self.position_mode = validate_position_mode(
                getattr(model, "position_mode", "compressed")
            )
            model.predictive_state = self
            if hasattr(model, "model"):
                model.model.predictive_state = self
                model.model.h_enabled = model.h_enabled
                model.model.position_mode = self.position_mode

        def add_request(self, req_index: int, new_req_data) -> None:
            super().add_request(req_index, new_req_data)
            req_id = new_req_data.req_id
            self.req_id_to_index[req_id] = req_index
            sampling = new_req_data.sampling_params
            extra = dict(getattr(sampling, "extra_args", None) or {})
            history = list(new_req_data.prefill_token_ids or [])
            already = int(new_req_data.num_computed_tokens or 0)
            # Setup check only. The decode loop does not read this flag.
            was_clear = bool(torch.count_nonzero(self.h_spans[req_index]).item() == 0)
            if not self.model.h_enabled:
                prepare_warmup_slot(self, req_index, already)
                self.admission_log.append(
                    {
                        "event": "add",
                        "req_id": req_id,
                        "req_index": req_index,
                        "h_enabled": False,
                        "was_clear": was_clear,
                        "already": already,
                    }
                )
                return

            kind = admission_kind(req_id, extra)
            if kind == "warmup":
                prepare_warmup_slot(self, req_index, already)
                self.admission_log.append(
                    {
                        "event": "add",
                        "req_id": req_id,
                        "req_index": int(req_index),
                        "mode": "vllm_warmup",
                        "h_enabled": False,
                        "was_clear": was_clear,
                        "already": already,
                        "spans": [1] * CODEBOOK_SIZE,
                    }
                )
                return

            payload = extra.get("predictive_codebook")
            codebook = validate_codebook(
                payload,
                disabled_ids=payload.get("disabled_ids") or (),
            )
            started = time.perf_counter()
            # TTFT instrumentation around one-time H synthesis, not the decode loop.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._synthesize(req_index, codebook)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.setup_ms[req_index] = elapsed_ms
            self.codebook_sha[req_index] = codebook.sha256
            spans = [len(phrase) for phrase in codebook.phrases]
            offset = _semantic_offset_from_history(
                history[:already], spans, position_mode=self.position_mode
            )
            if offset >= MAX_POSITION_EMBEDDINGS:
                raise RuntimeError(
                    f"reconstructed semantic offset {offset} exceeds Phi RoPE"
                )
            self.semantic_offset[req_index] = offset
            self.physical_accounted[req_index] = already
            activate_codebook_slot(self, req_index)
            digest = _tensor_hash(self.h_input[req_index])
            self.admission_log.append(
                {
                    "event": "add",
                    "req_id": req_id,
                    "req_index": int(req_index),
                    "h_enabled": True,
                    "position_mode": self.position_mode,
                    "was_clear": was_clear,
                    "already": already,
                    "semantic_offset": int(offset),
                    "sha256": codebook.sha256,
                    "h_input_hash": digest,
                    "setup_ms": elapsed_ms,
                    "spans": spans,
                }
            )

        def remove_request(self, req_id: str) -> None:
            super().remove_request(req_id)
            req_index = self.req_id_to_index.pop(req_id, None)
            if req_index is None:
                return
            clear_predictive_slot(self, req_index)
            self.codebook_sha.pop(req_index, None)
            self.admission_log.append(
                {"event": "remove", "req_id": req_id, "req_index": int(req_index)}
            )

        def prepare_inputs(self, input_batch, req_states) -> dict[str, Any]:
            self.position_trace.clear_active()
            req_ids = list(input_batch.req_ids)
            known = [req_id in self.req_id_to_index for req_id in req_ids]
            if not known or not all(known):
                # Dummy / capture batches use unregistered request ids.
                self.token_req_indices[: input_batch.num_tokens_after_padding].zero_()
                return {}

            n = int(input_batch.num_tokens)
            n_pad = int(input_batch.num_tokens_after_padding)
            num_reqs = int(input_batch.num_reqs)
            # Proof path: ownership is built on CPU and copied to GPU.
            owners = np.zeros(n_pad, dtype=np.int64)
            query = input_batch.query_start_loc_np
            mapping = input_batch.idx_mapping_np
            for row in range(num_reqs):
                start = int(query[row])
                end = int(query[row + 1])
                start_n = min(max(start, 0), n)
                end_n = min(max(end, 0), n)
                owners[start_n:end_n] = int(mapping[row])
            self.token_req_indices[:n_pad].copy_(torch.from_numpy(owners))
            expanded = input_batch.expanded_idx_mapping
            self.logit_req_indices[: expanded.shape[0]].copy_(
                expanded.to(dtype=torch.int64)
            )

            if n == 0:
                return {"positions": self.positions_buffer[:n_pad]}

            logical = input_batch.input_ids[:n].to(dtype=torch.int64)
            computed_np = np.asarray(input_batch.num_computed_tokens_np)
            if len(computed_np) < num_reqs:
                raise RuntimeError(
                    f"num_computed_tokens has {len(computed_np)} rows for {num_reqs} requests"
                )
            # Scheduler count of tokens already computed before this chunk.
            # This is the batch snapshot, not semantic_offset.
            num_computed = {
                int(mapping[row]): int(computed_np[row]) for row in range(num_reqs)
            }
            use_spans = self.position_mode == "base_token_end" and bool(self.model.h_enabled)
            h_spans_by_req: dict[int, list[int]] = {}
            history_by_req: dict[int, list[int]] = {}
            if use_spans:
                # TODO: replace this proof-path GPU-to-CPU history replay with
                # an optimized request-local span prefix structure for decode.
                token_table = req_states.all_token_ids.gpu
                for slot, count in num_computed.items():
                    h_spans_by_req[slot] = [
                        int(span) for span in self.h_spans[slot].detach().cpu().tolist()
                    ]
                    if count <= 0:
                        history_by_req[slot] = []
                        continue
                    history_by_req[slot] = [
                        int(token)
                        for token in token_table[slot, :count].detach().cpu().tolist()
                    ]
            positions_list = chunk_positions_from_computed(
                [int(token) for token in logical.detach().cpu().tolist()],
                [int(slot) for slot in owners[:n].tolist()],
                num_computed,
                h_spans_by_req if use_spans else None,
                history_by_req if use_spans else None,
                position_mode=self.position_mode,
                use_hypertoken_spans=use_spans,
            )
            positions = torch.tensor(positions_list, dtype=torch.int64, device=self.device)
            returned_positions = write_semantic_positions(
                self.positions_buffer,
                positions,
                num_tokens=n,
                num_tokens_after_padding=n_pad,
            )
            self.last_positions = returned_positions
            query = input_batch.query_start_loc_np
            scheduled = input_batch.num_scheduled_tokens
            request_rows = []
            target_slice = None
            for row, req_id in enumerate(req_ids):
                slot = int(mapping[row])
                start = int(query[row])
                end = int(query[row + 1])
                request_rows.append(
                    {
                        "request_id": str(req_id),
                        "slot": slot,
                        "num_computed_tokens": int(computed_np[row]),
                        "num_scheduled_tokens": int(scheduled[row]),
                        "query_token_slice": [start, end],
                    }
                )
                if req_id == self.position_trace.target_request_id:
                    target_slice = (start, end)

            if target_slice is not None:
                trace_entry = {
                    "req_ids": [str(req_id) for req_id in req_ids],
                    "slots": [int(mapping[row]) for row in range(num_reqs)],
                    "requests": request_rows,
                    "target_query_slice": [int(target_slice[0]), int(target_slice[1])],
                    "input_batch_num_tokens": n,
                    "input_batch_num_tokens_after_padding": n_pad,
                    "target_token_ids": [
                        int(token)
                        for token in (
                            logical[target_slice[0] : target_slice[1]]
                            .detach()
                            .cpu()
                            .tolist()
                        )
                    ],
                    "stock_input_batch_positions": position_tensor_snapshot(
                        input_batch.positions[:n], target_slice=target_slice
                    ),
                    "calculated_semantic_positions": position_tensor_snapshot(
                        positions, target_slice=target_slice
                    ),
                    "returned_positions": position_tensor_snapshot(
                        returned_positions, target_slice=target_slice
                    ),
                    "position_mode": self.position_mode,
                }
                self.position_trace.capture(trace_entry)
            return {"positions": returned_positions}

        def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
            del num_reqs, num_tokens
            return {}

        def postprocess_state(
            self,
            idx_mapping: torch.Tensor,
            num_sampled: torch.Tensor,
            num_computed_tokens: torch.Tensor | None = None,
        ) -> None:
            # Positions are rebuilt from the scheduler's computed-token count
            # on the next prepare. There is no offset to commit here.
            del idx_mapping, num_sampled, num_computed_tokens

        def _synthesize(self, req_index: int, codebook) -> None:
            model = self.model
            if model.input_encoder is None or model.output_encoder is None:
                raise RuntimeError("predictive hyperencoders are not attached")
            max_subtokens = int(
                model.input_encoder.compression_config.max_subtokens
            )
            pad_id = int(model.pad_token_id)
            rows = []
            spans = []
            for phrase in codebook.phrases:
                spans.append(len(phrase))
                padded = list(phrase) + [pad_id] * (max_subtokens - len(phrase))
                rows.append(padded[:max_subtokens])
            device = self.h_input.device
            updates = torch.tensor(rows, dtype=torch.long, device=device).view(
                1, CODEBOOK_SIZE, max_subtokens
            )
            embed_weight = model.model.embed_tokens.weight
            head_weight = model.lm_head.weight
            with torch.no_grad():
                h_in = model.input_encoder(updates, embed_weight, pad_id)[0]
                h_out = model.output_encoder(updates, head_weight, pad_id)[0]
            self.h_input[req_index].copy_(h_in.to(dtype=self.h_input.dtype))
            self.h_output[req_index].copy_(h_out.to(dtype=self.h_output.dtype))
            self.h_spans[req_index].copy_(
                torch.tensor(spans, dtype=torch.int64, device=device)
            )

    PredictiveModelState = _PredictiveModelState


def _semantic_offset_from_history(
    history: list[int], spans: list[int], position_mode: str = "compressed"
) -> int:
    if position_mode == "compressed":
        return len(history)
    total = 0
    for logical_id in history:
        if H_START <= int(logical_id) < H_END:
            total += int(spans[int(logical_id) - H_START])
        else:
            total += 1
    return total


def _tensor_hash(tensor: torch.Tensor) -> str:
    raw = tensor.detach().float().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


try:
    _build_state()
except ImportError:
    pass
