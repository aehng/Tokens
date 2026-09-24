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
    validate_codebook,
)
from tokens_vllm.warmup import (
    activate_codebook_slot,
    admission_kind,
    clear_predictive_slot,
    prepare_warmup_slot,
)


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
            self._commit_pending = False
            self.last_positions: torch.Tensor | None = None
            model.predictive_state = self
            if hasattr(model, "model"):
                model.model.predictive_state = self
                model.model.h_enabled = model.h_enabled

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
            offset = _semantic_offset_from_history(history[:already], spans)
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
            del req_states
            req_ids = list(input_batch.req_ids)
            known = [req_id in self.req_id_to_index for req_id in req_ids]
            if not known or not all(known):
                # Dummy / capture batches use unregistered request ids.
                self._commit_pending = False
                self.pending_semantic_advance.zero_()
                self.pending_physical_advance.zero_()
                self.token_req_indices[: input_batch.num_tokens_after_padding].zero_()
                return {}

            n = int(input_batch.num_tokens)
            n_pad = int(input_batch.num_tokens_after_padding)
            num_reqs = int(input_batch.num_reqs)
            # Proof path: ownership is built on CPU and copied to GPU.
            owners = np.zeros(n_pad, dtype=np.int64)
            query = input_batch.query_start_loc_np
            mapping = input_batch.idx_mapping_np
            batch_rows = np.zeros(n, dtype=np.int64)
            for row in range(num_reqs):
                start = int(query[row])
                end = int(query[row + 1])
                start_n = min(max(start, 0), n)
                end_n = min(max(end, 0), n)
                owners[start_n:end_n] = int(mapping[row])
                batch_rows[start_n:end_n] = row
            self.token_req_indices[:n_pad].copy_(torch.from_numpy(owners))
            expanded = input_batch.expanded_idx_mapping
            self.logit_req_indices[: expanded.shape[0]].copy_(
                expanded.to(dtype=torch.int64)
            )

            logical = input_batch.input_ids[:n].to(dtype=torch.int64)
            req = self.token_req_indices[:n]
            is_h = (logical >= H_START) & (logical < H_END)
            slots = (logical - H_START).clamp(0, CODEBOOK_SIZE - 1)
            span = torch.ones(n, dtype=torch.int64, device=self.device)
            if self.model.h_enabled and n > 0:
                active = self.h_active[req]
                span = torch.where(is_h & active, self.h_spans[req, slots], span)
            if n == 0:
                self._commit_pending = True
                return {"positions": self.positions_buffer[:n_pad]}

            csum = torch.cumsum(span, dim=0)
            prev = torch.zeros_like(csum)
            prev[1:] = csum[:-1]
            starts = input_batch.query_start_loc[:num_reqs].to(dtype=torch.int64)
            starts = starts.clamp(0, n - 1)
            origin = prev[starts][torch.from_numpy(batch_rows).to(self.device)]
            local = csum - origin
            positions = self.semantic_offset[req] + local - 1
            self.positions_buffer[:n].copy_(positions)
            if n_pad > n:
                self.positions_buffer[n:n_pad].zero_()

            self.pending_semantic_advance.zero_()
            self.pending_physical_advance.zero_()
            ends = input_batch.query_start_loc[1 : num_reqs + 1].to(dtype=torch.int64) - 1
            ends = ends.clamp(0, n - 1)
            slot_idx = input_batch.idx_mapping[:num_reqs].to(dtype=torch.int64)
            self.pending_semantic_advance[slot_idx] = local[ends]
            scheduled = torch.from_numpy(
                np.ascontiguousarray(input_batch.num_scheduled_tokens[:num_reqs])
            ).to(device=self.device, dtype=torch.int64)
            self.pending_physical_advance[slot_idx] = scheduled
            self._commit_pending = True
            self.last_positions = self.positions_buffer[:n_pad]
            return {"positions": self.positions_buffer[:n_pad]}

        def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
            self._commit_pending = False
            return {}

        def postprocess_state(
            self,
            idx_mapping: torch.Tensor,
            num_sampled: torch.Tensor,
            num_computed_tokens: torch.Tensor | None = None,
        ) -> None:
            del idx_mapping, num_sampled, num_computed_tokens
            if not self._commit_pending:
                self.pending_semantic_advance.zero_()
                self.pending_physical_advance.zero_()
                return
            self.semantic_offset.add_(self.pending_semantic_advance)
            self.physical_accounted.add_(self.pending_physical_advance)
            self.pending_semantic_advance.zero_()
            self.pending_physical_advance.zero_()
            self._commit_pending = False

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


def _semantic_offset_from_history(history: list[int], spans: list[int]) -> int:
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
