from __future__ import annotations

import logging
from numbers import Integral
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import torch
from transformers import AutoTokenizer, LogitsProcessor

from zip2zip.config import Zip2ZipConfig
from zip2zip.nn.encoders.base import EncoderFn

logger = logging.getLogger(__name__)


if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
    _compiler_disable = torch.compiler.disable
else:
    def _compiler_disable(fn):
        return fn



def _dtype_element_size(dtype: torch.dtype) -> int:
    """Return dtype storage size using a tiny CPU scalar, never model weights."""
    return torch.empty((), dtype=dtype).element_size()


def _shape_numel(shape: Sequence[int]) -> int:
    count = 1
    for size in shape:
        if size < 0:
            raise ValueError(f"Tensor shape dimensions must be non-negative: {shape}")
        count *= int(size)
    return count


def estimate_effective_table_memory(
    input_shape: Sequence[int],
    input_dtype: torch.dtype,
    output_shape: Sequence[int],
    output_dtype: torch.dtype,
    *,
    codebook_size: int,
    output_bias_shape: Optional[Sequence[int]] = None,
    output_bias_dtype: Optional[torch.dtype] = None,
) -> Dict[str, int]:
    """Estimate prepared-table storage from model tensor shapes and dtypes only.

    Input and output weights must be matrix shapes ``(vocab_rows, hidden)``.
    The estimate includes each full effective table, because the request cache
    holds those copies alongside the model's original parameters.
    """
    if len(input_shape) != 2 or len(output_shape) != 2:
        raise ValueError("input and output weight shapes must both be rank 2")
    if codebook_size < 0:
        raise ValueError("codebook_size must be non-negative")
    input_rows, input_width = map(int, input_shape)
    output_rows, output_width = map(int, output_shape)
    input_el_size = _dtype_element_size(input_dtype)
    output_el_size = _dtype_element_size(output_dtype)

    base_input_bytes = _shape_numel(input_shape) * input_el_size
    effective_input_bytes = (input_rows + codebook_size) * input_width * input_el_size
    base_output_weight_bytes = _shape_numel(output_shape) * output_el_size
    effective_output_weight_bytes = (
        (output_rows + codebook_size) * output_width * output_el_size
    )

    base_bias_bytes = 0
    effective_bias_bytes = 0
    if output_bias_shape is not None:
        if output_bias_dtype is None:
            raise ValueError("output_bias_dtype is required when output_bias_shape is set")
        if len(output_bias_shape) != 1 or int(output_bias_shape[0]) != output_rows:
            raise ValueError("output bias shape must match the output row count")
        bias_el_size = _dtype_element_size(output_bias_dtype)
        base_bias_bytes = _shape_numel(output_bias_shape) * bias_el_size
        effective_bias_bytes = (output_rows + codebook_size) * bias_el_size

    base_output_bytes = base_output_weight_bytes + base_bias_bytes
    effective_output_bytes = effective_output_weight_bytes + effective_bias_bytes
    additional_input_bytes = effective_input_bytes - base_input_bytes
    additional_output_bytes = effective_output_bytes - base_output_bytes
    additional_bytes = effective_input_bytes + effective_output_bytes

    return {
        "base_input_embedding_bytes": base_input_bytes,
        "effective_input_embedding_bytes": effective_input_bytes,
        "additional_input_embedding_bytes": additional_input_bytes,
        "base_output_head_bytes": base_output_bytes,
        "effective_output_head_bytes": effective_output_bytes,
        "additional_output_head_bytes": additional_output_bytes,
        "additional_bytes": additional_bytes,
        "additional_effective_table_bytes": additional_bytes,
    }


class StaticCodebookManager:
    """Request-specific static codebook manager for seeded hypertokens.

    Replaces dynamic LZW codebook discovery with a fixed dictionary of 2-3 token
    phrases seeded before inference starts. Hypertokens occupy the dynamic
    zip2zip codebook index range: [initial_vocab_size, initial_vocab_size + max_codebook_size).
    """

    def __init__(
        self,
        initial_vocab_size: int,
        max_codebook_size: int,
        max_subtokens: int,
        embedding_dim: int,
        pad_token_id: int,
        disabled_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.initial_vocab_size = initial_vocab_size
        self.max_codebook_size = max_codebook_size
        self.max_subtokens = max_subtokens
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        self.disabled_ids = set(disabled_ids) if disabled_ids else set()

        # Seeded codebook definitions
        # Maps hypertoken_id (absolute) -> list of base token IDs
        self.hyper_to_subtokens: Dict[int, List[int]] = {}
        # Maps tuple of base token IDs -> hypertoken_id (absolute)
        self.subtokens_to_hyper: Dict[Tuple[int, ...], int] = {}
        self.num_seeded: int = 0

        # Model weight caches & updates
        self.updates: Optional[torch.Tensor] = None
        self.updates_indices: Optional[List[List[int]]] = None
        self.hyper_embedding_weight_cache: Optional[torch.Tensor] = None
        self.hyper_linear_weight_cache: Optional[torch.Tensor] = None
        self.fast_inference_ready = False
        self.effective_embedding_weight_cache: Optional[torch.Tensor] = None
        self.effective_linear_weight_cache: Optional[torch.Tensor] = None
        self.effective_linear_bias_cache: Optional[torch.Tensor] = None
        self.inference_tables_build_count = 0
        self.inference_tables_version = 0
        self.inference_memory_report: Dict[str, int] = {}
        self.inference_timing_report: Dict[str, float] = {}

        # Position tracking (zip2zip++ base token positions)
        self.runtime_batch_size: Optional[int] = None
        self.hyper_token_spans: Optional[torch.Tensor] = None
        self.base_position_offset: Optional[torch.Tensor] = None
        self.position_ids: Optional[torch.Tensor] = None
        self._prepared_for_embedding: bool = False

        # Instrumentation
        self.input_encoder_calls: int = 0
        self.output_encoder_calls: int = 0

    def set_seeded_codebook(
        self,
        dictionary: Union[Dict[int, List[int]], List[List[int]]],
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        """Seed the codebook with a fixed dictionary of hypertokens.

        Args:
            dictionary: Either a dict mapping hypertoken ID (0-indexed or absolute)
                        to component base tokens, or a list of component base token lists.
            batch_size: Batch size for generation / prefill.
            device: Target device for tensor allocation.
        """
        if isinstance(dictionary, dict):
            if not dictionary:
                items = []
            else:
                first_k, first_v = next(iter(dictionary.items()))
                if isinstance(first_k, (tuple, list)):
                    # Dict maps subtokens -> hyper_id
                    items = [(v, k) for k, v in dictionary.items()]
                else:
                    items = list(dictionary.items())
        else:
            items = list(enumerate(dictionary))

        if len(items) > self.max_codebook_size:
            raise ValueError(
                f"Dictionary contains {len(items)} items, exceeding "
                f"max_codebook_size={self.max_codebook_size}"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        new_hyper_to_subtokens: Dict[int, List[int]] = {}
        new_subtokens_to_hyper: Dict[Tuple[int, ...], int] = {}
        used_slots = set()

        for raw_id, subtokens in items:
            if isinstance(raw_id, bool) or not isinstance(raw_id, Integral):
                raise ValueError(f"Hypertoken ID must be an integer, got {raw_id!r}")
            raw_id = int(raw_id)

            # Support relative slots [0, K) and absolute IDs [V, V + K).
            if raw_id < self.initial_vocab_size:
                entry_idx = raw_id
                hyper_id = self.initial_vocab_size + raw_id
            else:
                entry_idx = raw_id - self.initial_vocab_size
                hyper_id = raw_id

            if entry_idx < 0 or entry_idx >= self.max_codebook_size:
                raise ValueError(
                    f"Hypertoken ID {raw_id} is outside relative [0, {self.max_codebook_size}) "
                    f"or absolute [{self.initial_vocab_size}, "
                    f"{self.initial_vocab_size + self.max_codebook_size}) bounds"
                )
            if entry_idx in used_slots:
                raise ValueError(f"Duplicate hypertoken slot {entry_idx}")
            used_slots.add(entry_idx)

            try:
                subtokens_list = list(subtokens)
            except TypeError as exc:
                raise ValueError("Hypertoken definition must be a sequence of token IDs") from exc

            if not (2 <= len(subtokens_list) <= self.max_subtokens):
                raise ValueError(
                    f"Subtoken length {len(subtokens_list)} must be between 2 and "
                    f"max_subtokens={self.max_subtokens}"
                )
            normalized_subtokens = []
            for token_id in subtokens_list:
                if isinstance(token_id, bool) or not isinstance(token_id, Integral):
                    raise ValueError(
                        f"Base token ID must be an integer, got {token_id!r}"
                    )
                token_id = int(token_id)
                if not 0 <= token_id < self.initial_vocab_size:
                    raise ValueError(
                        f"Subtoken ID {token_id} is outside base vocabulary "
                        f"[0, {self.initial_vocab_size})"
                    )
                if token_id in self.disabled_ids:
                    raise ValueError("Subtokens contain a disabled token ID")
                normalized_subtokens.append(token_id)

            phrase = tuple(normalized_subtokens)
            if phrase in new_subtokens_to_hyper:
                raise ValueError(f"Duplicate hypertoken phrase {phrase}")
            new_hyper_to_subtokens[hyper_id] = normalized_subtokens
            new_subtokens_to_hyper[phrase] = hyper_id

        if used_slots != set(range(len(used_slots))):
            raise ValueError(
                "Seeded hypertoken slots must be contiguous from zero because "
                "unused-logit masking assumes a packed codebook"
            )

        # Commit the new dictionary only after all validation succeeds.
        self.hyper_to_subtokens = new_hyper_to_subtokens
        self.subtokens_to_hyper = new_subtokens_to_hyper
        self.num_seeded = len(new_hyper_to_subtokens)
        self.runtime_batch_size = batch_size

        # Invalidate weight caches so new embeddings will be computed
        self.hyper_embedding_weight_cache = None
        self.hyper_linear_weight_cache = None
        self._invalidate_inference_tables()
        self.base_position_offset = None
        self.position_ids = None
        self._prepared_for_embedding = False
        self._build_updates_tensor(batch_size, device=device)

    def _build_updates_tensor(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> None:
        """Construct the updates and hyper_token_spans tensors from seeded dictionary."""
        if self.num_seeded == 0:
            self.updates = torch.full(
                (batch_size, 0, self.max_subtokens),
                self.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            self.updates_indices = [[] for _ in range(batch_size)]
            self.hyper_token_spans = torch.zeros(
                (batch_size, self.max_codebook_size), dtype=torch.long, device=device
            )
            return

        sorted_items = sorted(
            self.hyper_to_subtokens.items(),
            key=lambda item: item[0] - self.initial_vocab_size,
        )
        updates_list = []
        indices_list = []

        for hyper_id, subtokens in sorted_items:
            entry_idx = hyper_id - self.initial_vocab_size
            indices_list.append(entry_idx)
            padded = subtokens + [self.pad_token_id] * (self.max_subtokens - len(subtokens))
            updates_list.append(padded)

        # Shape: (batch_size, num_seeded, max_subtokens)
        base_updates = torch.tensor(updates_list, dtype=torch.long, device=device)
        self.updates = base_updates.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        self.updates_indices = [list(indices_list) for _ in range(batch_size)]

        # Shape: (batch_size, max_codebook_size)
        spans = torch.zeros(
            (batch_size, self.max_codebook_size), dtype=torch.long, device=device
        )
        for hyper_id, subtokens in sorted_items:
            entry_idx = hyper_id - self.initial_vocab_size
            spans[:, entry_idx] = len(subtokens)
        self.hyper_token_spans = spans

    @_compiler_disable
    def prepare_input_ids(
        self,
        ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.LongTensor:
        """Calculate zip2zip++ RoPE positions for prefill and generation steps."""
        if ids.ndim != 2:
            raise ValueError(f"input_ids must be rank 2, got shape {tuple(ids.shape)}")

        batch_size, _ = ids.shape
        if self.fast_inference_ready and batch_size != 1:
            raise NotImplementedError(
                "prepared predictive fast inference currently supports batch_size=1"
            )
        if self.runtime_batch_size is not None and self.runtime_batch_size != batch_size:
            # Batch size changed: re-expand updates and spans
            self._build_updates_tensor(batch_size, device=ids.device)
            self.runtime_batch_size = batch_size
        elif self.hyper_token_spans is None:
            self._build_updates_tensor(batch_size, device=ids.device)

        device = ids.device
        if self.hyper_token_spans is not None and self.hyper_token_spans.device != device:
            self.hyper_token_spans = self.hyper_token_spans.to(device)
        if self.updates is not None and self.updates.device != device:
            self.updates = self.updates.to(device)

        is_hyper = (ids >= self.initial_vocab_size) & (
            ids < self.initial_vocab_size + self.max_codebook_size
        )
        spans = torch.ones_like(ids)

        if self.fast_inference_ready:
            entry_ids = (ids - self.initial_vocab_size).clamp(0, self.max_codebook_size - 1)
            hyper_spans = self.hyper_token_spans.gather(1, entry_ids)
            spans = torch.where(is_hyper, hyper_spans, spans)
        elif is_hyper.any():
            entry_ids = (ids - self.initial_vocab_size).clamp(0, self.max_codebook_size - 1)
            hyper_spans = self.hyper_token_spans.gather(1, entry_ids)
            # Verify no unseeded hypertoken is referenced
            if (hyper_spans[is_hyper] == 0).any():
                raise ValueError("input_ids reference an unseeded or invalid hypertoken ID")
            spans = torch.where(is_hyper, hyper_spans, spans)

        if attention_mask is not None:
            if attention_mask.shape != ids.shape:
                raise ValueError(
                    f"attention_mask shape {tuple(attention_mask.shape)} must match "
                    f"input_ids shape {tuple(ids.shape)}"
                )
            valid = attention_mask.to(device=ids.device, dtype=torch.bool)
            spans = torch.where(valid, spans, torch.zeros_like(spans))
        else:
            valid = torch.ones_like(ids, dtype=torch.bool)

        if self.base_position_offset is None:
            self.base_position_offset = torch.zeros(
                batch_size, 1, device=device, dtype=torch.long
            )
        else:
            self.base_position_offset = self.base_position_offset.to(device)

        positions = self.base_position_offset + spans.cumsum(dim=-1) - 1
        positions = torch.where(valid, positions, torch.zeros_like(positions))
        self.base_position_offset = (
            self.base_position_offset + spans.sum(dim=-1, keepdim=True)
        ).detach().clone()
        self.position_ids = positions
        self._prepared_for_embedding = True
        return positions

    def init_codebooks_and_hyper_weight_cache(
        self, batch_size: int, codebooks: Optional[List] = None
    ) -> None:
        """Called by Zip2ZipModel.generate() before decoding loop."""
        self.runtime_batch_size = batch_size
        if self.updates is None or self.updates.shape[0] != batch_size:
            self._build_updates_tensor(batch_size)

    def get_hyper_embedding_weights(
        self,
        ids: torch.LongTensor,
        base_weight: torch.Tensor,
        encoder_fn: EncoderFn,
    ) -> torch.Tensor:
        """Synthesize and cache input embeddings for seeded hypertokens."""
        curr_device = base_weight.device
        dtype = base_weight.dtype
        batch_size = ids.shape[0]

        if (
            self.hyper_embedding_weight_cache is not None
            and self.hyper_embedding_weight_cache.shape[0] == batch_size
            and self.hyper_embedding_weight_cache.device == curr_device
            and self.hyper_embedding_weight_cache.dtype == dtype
        ):
            return self.hyper_embedding_weight_cache

        self.runtime_batch_size = batch_size
        self.hyper_embedding_weight_cache = torch.zeros(
            batch_size,
            self.max_codebook_size,
            self.embedding_dim,
            dtype=dtype,
            device=curr_device,
        )

        if not self._prepared_for_embedding:
            self.prepare_input_ids(ids)

        if self.updates is not None and self.updates_indices is not None:
            self.updates = self.updates.to(curr_device)
            if any(len(ui) > 0 for ui in self.updates_indices):
                new_weights = encoder_fn(self.updates, base_weight, self.pad_token_id)
                self.input_encoder_calls += 1
                if new_weights.dtype != dtype:
                    new_weights = new_weights.to(dtype=dtype)
                for i, ui in enumerate(self.updates_indices):
                    self.hyper_embedding_weight_cache[i, ui] = new_weights[i, : len(ui)]

        self._prepared_for_embedding = True
        return self.hyper_embedding_weight_cache

    def get_hyper_linear_weights(
        self, base_weight: torch.Tensor, encoder_fn: EncoderFn
    ) -> torch.Tensor:
        """Synthesize and cache output projection vectors for seeded hypertokens."""
        curr_device = base_weight.device
        dtype = base_weight.dtype
        batch_size = self.runtime_batch_size or 1

        if (
            self.hyper_linear_weight_cache is not None
            and self.hyper_linear_weight_cache.shape[0] == batch_size
            and self.hyper_linear_weight_cache.device == curr_device
            and self.hyper_linear_weight_cache.dtype == dtype
        ):
            return self.hyper_linear_weight_cache

        self.hyper_linear_weight_cache = torch.zeros(
            batch_size,
            self.max_codebook_size,
            self.embedding_dim,
            dtype=dtype,
            device=curr_device,
        )

        if self.updates is not None and self.updates_indices is not None:
            self.updates = self.updates.to(curr_device)
            if any(len(ui) > 0 for ui in self.updates_indices):
                new_weights = encoder_fn(self.updates, base_weight, self.pad_token_id)
                self.output_encoder_calls += 1
                if new_weights.dtype != dtype:
                    new_weights = new_weights.to(dtype=dtype)
                for i, ui in enumerate(self.updates_indices):
                    self.hyper_linear_weight_cache[i, ui] = new_weights[i, : len(ui)]

        return self.hyper_linear_weight_cache

    def synthesize_hyper_vectors(
        self, model: torch.nn.Module, batch_size: int = 1, dummy_input_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precompute both input embeddings and output linear weights ONCE before generation."""
        self.runtime_batch_size = batch_size
        base = getattr(model, "base_model", model)
        inp_emb = base.get_input_embeddings()
        out_emb = base.get_output_embeddings()

        inp_enc_fn = model.input_encoder.get_encoder_fn()
        out_enc_fn = (
            model.output_encoder.get_encoder_fn()
            if getattr(model, "output_encoder", None) is not None
            else inp_enc_fn
        )

        device = inp_emb.weight.device
        if dummy_input_ids is None:
            dummy_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

        with torch.no_grad():
            w_emb = self.get_hyper_embedding_weights(dummy_input_ids, inp_emb.weight, inp_enc_fn)
            w_lin = self.get_hyper_linear_weights(out_emb.weight, out_enc_fn)
        return w_emb, w_lin

    def _invalidate_inference_tables(self) -> None:
        self.fast_inference_ready = False
        self.effective_embedding_weight_cache = None
        self.effective_linear_weight_cache = None
        self.effective_linear_bias_cache = None
        self.inference_memory_report = {}
        self.inference_timing_report: Dict[str, float] = {}

    @staticmethod
    def _token_ids_from_generation_value(value: object, field_name: str) -> List[int]:
        if value is None:
            return []
        if isinstance(value, bool):
            raise ValueError(f"Unsupported boolean generation token ID in {field_name}")
        if isinstance(value, Integral):
            return [int(value)]
        if isinstance(value, (list, tuple)):
            token_ids: List[int] = []
            for item in value:
                token_ids.extend(
                    StaticCodebookManager._token_ids_from_generation_value(
                        item, field_name
                    )
                )
            return token_ids
        raise ValueError(
            f"Unsupported generation token ID value for {field_name}: "
            f"{type(value).__name__}"
        )

    def validate_generation_token_space(
        self,
        model: torch.nn.Module,
        input_padding_idx: Optional[int] = None,
        generation_overrides: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Reject generation or padding IDs that collide with inserted H rows."""
        base = getattr(model, "base_model", model)
        configs = [
            ("base_model.generation_config", getattr(base, "generation_config", None)),
            ("base_model.config", getattr(base, "config", None)),
        ]
        if generation_overrides is not None:
            configs.append(("generation arguments", generation_overrides))
        special_fields = (
            "eos_token_id",
            "pad_token_id",
            "bos_token_id",
            "decoder_start_token_id",
            "forced_bos_token_id",
            "forced_eos_token_id",
            "suppress_tokens",
            "begin_suppress_tokens",
            "bad_words_ids",
            "force_words_ids",
        )
        for config_name, config in configs:
            if config is None:
                continue
            for field_name in special_fields:
                value = (
                    config.get(field_name)
                    if isinstance(config, Mapping)
                    else getattr(config, field_name, None)
                )
                for token_id in self._token_ids_from_generation_value(
                    value, f"{config_name}.{field_name}"
                ):
                    if token_id >= self.initial_vocab_size:
                        raise ValueError(
                            "Fast predictive inference currently requires generation "
                            "special token IDs to remain below the hypertoken insertion "
                            f"point; {config_name}.{field_name} contains {token_id} "
                            f"(V={self.initial_vocab_size})."
                        )

            forced_decoder_ids = (
                config.get("forced_decoder_ids")
                if isinstance(config, Mapping)
                else getattr(config, "forced_decoder_ids", None)
            )
            if forced_decoder_ids is not None:
                if not isinstance(forced_decoder_ids, (list, tuple)):
                    raise ValueError(
                        f"Unsupported generation token ID value for "
                        f"{config_name}.forced_decoder_ids"
                    )
                for entry in forced_decoder_ids:
                    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                        raise ValueError(
                            f"Malformed {config_name}.forced_decoder_ids entry: {entry!r}"
                        )
                    token_id = self._token_ids_from_generation_value(
                        entry[1], f"{config_name}.forced_decoder_ids"
                    )
                    if any(value >= self.initial_vocab_size for value in token_id):
                        raise ValueError(
                            "Fast predictive inference currently requires generation "
                            "special token IDs to remain below the hypertoken insertion "
                            f"point; {config_name}.forced_decoder_ids contains a tail ID."
                        )

            sequence_bias = (
                config.get("sequence_bias")
                if isinstance(config, Mapping)
                else getattr(config, "sequence_bias", None)
            )
            if sequence_bias is not None:
                if not isinstance(sequence_bias, dict):
                    raise ValueError(
                        f"Unsupported generation token ID value for "
                        f"{config_name}.sequence_bias"
                    )
                for sequence in sequence_bias:
                    token_ids = self._token_ids_from_generation_value(
                        sequence, f"{config_name}.sequence_bias"
                    )
                    if any(value >= self.initial_vocab_size for value in token_ids):
                        raise ValueError(
                            "Fast predictive inference currently requires generation "
                            "special token IDs to remain below the hypertoken insertion "
                            f"point; {config_name}.sequence_bias contains a tail ID."
                        )

        if input_padding_idx is not None:
            if isinstance(input_padding_idx, bool) or not isinstance(
                input_padding_idx, Integral
            ):
                raise ValueError(f"Unsupported embedding padding_idx: {input_padding_idx!r}")
            if int(input_padding_idx) >= self.initial_vocab_size:
                raise ValueError(
                    "Fast predictive inference currently requires HyperEmbedding.padding_idx "
                    "to remain below the hypertoken insertion point."
                )

    def prepare_inference_tables(
        self,
        model: torch.nn.Module,
        batch_size: int = 1,
        dummy_input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, int]:
        """Synthesize request H vectors and build effective tables once.

        Prepared predictive inference currently supports batch size one only.
        H rows are request-shared, but table lookup/position state is not yet
        implemented for multi-sequence batches.
        """
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, Integral)
            or batch_size != 1
        ):
            raise NotImplementedError(
                "prepared predictive fast inference currently supports batch_size=1"
            )
        if dummy_input_ids is not None and (
            dummy_input_ids.ndim != 2 or dummy_input_ids.shape[0] != 1
        ):
            raise NotImplementedError(
                "prepared predictive fast inference currently supports batch_size=1"
            )
        if model.training:
            raise RuntimeError("prepare_inference_tables() requires model.eval()")
        if self.num_seeded != len(self.hyper_to_subtokens):
            raise RuntimeError("seeded codebook state is inconsistent")
        if self.updates is None or self.hyper_token_spans is None:
            raise RuntimeError("set_seeded_codebook() must run before table preparation")

        base = getattr(model, "base_model", model)
        input_layer = base.get_input_embeddings()
        output_layer = base.get_output_embeddings()
        if input_layer is None or output_layer is None:
            raise ValueError("Fast inference requires input embeddings and an output head")
        if (
            getattr(input_layer, "codebook_manager", None) is not self
            or getattr(output_layer, "codebook_manager", None) is not self
        ):
            raise RuntimeError(
                "attach_to_model() must install this manager on both hyper modules "
                "before preparing inference tables"
            )
        if input_layer.weight.shape[0] < self.initial_vocab_size:
            raise ValueError("input embedding rows are smaller than initial_vocab_size")
        if output_layer.weight.shape[0] < self.initial_vocab_size:
            raise ValueError("output head rows are smaller than initial_vocab_size")
        if input_layer.weight.shape[1] != self.embedding_dim:
            raise ValueError("input embedding width does not match the encoder width")
        if output_layer.weight.shape[1] != self.embedding_dim:
            raise ValueError("output head width does not match the encoder width")
        self.validate_generation_token_space(
            model, input_padding_idx=getattr(input_layer, "padding_idx", None)
        )

        if (
            self.fast_inference_ready
            and self.effective_embedding_weight_cache is not None
            and self.effective_linear_weight_cache is not None
        ):
            return dict(self.inference_memory_report)

        self.clear_weight_caches()
        self.runtime_batch_size = 1
        device = input_layer.weight.device
        cuda_stage_events = {}
        cpu_stage_times = {}

        def start_stage(name: str):
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                cuda_stage_events[name] = (start_event, end_event)
                return end_event
            cpu_stage_times[name] = time.perf_counter()
            return None

        def end_stage(name: str, end_event) -> None:
            if device.type == "cuda":
                end_event.record()
            else:
                cpu_stage_times[name] = time.perf_counter() - cpu_stage_times[name]

        if dummy_input_ids is None:
            dummy_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        elif dummy_input_ids.device != device:
            dummy_input_ids = dummy_input_ids.to(device)

        try:
            total_end = start_stage("total_table_preparation")
            synthesis_end = start_stage("h_vector_synthesis")
            input_h, output_h = self.synthesize_hyper_vectors(
                model, batch_size=1, dummy_input_ids=dummy_input_ids
            )
            end_stage("h_vector_synthesis", synthesis_end)
            with torch.no_grad():
                input_h = input_h[0].to(
                    device=input_layer.weight.device, dtype=input_layer.weight.dtype
                )
                output_h = output_h[0].to(
                    device=output_layer.weight.device, dtype=output_layer.weight.dtype
                )

                input_table_end = start_stage("effective_input_table_build")
                effective_input = torch.cat(
                    (
                        input_layer.weight[: self.initial_vocab_size],
                        input_h,
                        input_layer.weight[self.initial_vocab_size :],
                    ),
                    dim=0,
                )
                end_stage("effective_input_table_build", input_table_end)
                output_table_end = start_stage("effective_output_table_build")
                effective_output = torch.cat(
                    (
                        output_layer.weight[: self.initial_vocab_size],
                        output_h,
                        output_layer.weight[self.initial_vocab_size :],
                    ),
                    dim=0,
                )

                effective_bias = None
                if output_layer.bias is not None:
                    h_bias = torch.zeros(
                        self.max_codebook_size,
                        device=output_layer.bias.device,
                        dtype=output_layer.bias.dtype,
                    )
                    effective_bias = torch.cat(
                        (
                            output_layer.bias[: self.initial_vocab_size],
                            h_bias,
                            output_layer.bias[self.initial_vocab_size :],
                        ),
                        dim=0,
                    )
                end_stage("effective_output_table_build", output_table_end)

            self.effective_embedding_weight_cache = effective_input.detach()
            self.effective_linear_weight_cache = effective_output.detach()
            self.effective_linear_bias_cache = (
                effective_bias.detach() if effective_bias is not None else None
            )
            self.hyper_embedding_weight_cache = None
            self.hyper_linear_weight_cache = None

            base_input_bytes = input_layer.weight.numel() * input_layer.weight.element_size()
            effective_input_bytes = (
                self.effective_embedding_weight_cache.numel()
                * self.effective_embedding_weight_cache.element_size()
            )
            base_output_bytes = output_layer.weight.numel() * output_layer.weight.element_size()
            effective_output_bytes = (
                self.effective_linear_weight_cache.numel()
                * self.effective_linear_weight_cache.element_size()
            )
            base_bias_bytes = (
                output_layer.bias.numel() * output_layer.bias.element_size()
                if output_layer.bias is not None
                else 0
            )
            effective_bias_bytes = (
                self.effective_linear_bias_cache.numel()
                * self.effective_linear_bias_cache.element_size()
                if self.effective_linear_bias_cache is not None
                else 0
            )
            cpu_table_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    self.effective_embedding_weight_cache,
                    self.effective_linear_weight_cache,
                    self.effective_linear_bias_cache,
                )
                if tensor is not None and tensor.device.type == "cpu"
            )
            self.inference_memory_report = {
                "base_input_embedding_bytes": base_input_bytes,
                "effective_input_embedding_bytes": effective_input_bytes,
                "additional_input_embedding_bytes": effective_input_bytes - base_input_bytes,
                "base_output_head_bytes": base_output_bytes + base_bias_bytes,
                "effective_output_head_bytes": effective_output_bytes + effective_bias_bytes,
                "additional_output_head_bytes": (
                    effective_output_bytes + effective_bias_bytes
                    - base_output_bytes - base_bias_bytes
                ),
                "additional_cpu_ram_bytes": cpu_table_bytes,
            }
            self.inference_memory_report.update(
                estimate_effective_table_memory(
                    input_layer.weight.shape,
                    input_layer.weight.dtype,
                    output_layer.weight.shape,
                    output_layer.weight.dtype,
                    codebook_size=self.max_codebook_size,
                    output_bias_shape=(
                        output_layer.bias.shape if output_layer.bias is not None else None
                    ),
                    output_bias_dtype=(
                        output_layer.bias.dtype if output_layer.bias is not None else None
                    ),
                )
            )
            end_stage("total_table_preparation", total_end)
            if device.type == "cuda":
                # Synchronize only at this setup boundary, never inside decode.
                torch.cuda.synchronize(device)
                self.inference_timing_report = {
                    f"{name}_ms": float(start.elapsed_time(end))
                    for name, (start, end) in cuda_stage_events.items()
                }
            else:
                self.inference_timing_report = {
                    f"{name}_ms": float(elapsed * 1000)
                    for name, elapsed in cpu_stage_times.items()
                }
            self.inference_tables_build_count += 1
            self.inference_tables_version += 1
            # The encoder cache is seeded with dummy IDs only to trigger the
            # existing encoder API. Do not let that synthetic setup input
            # advance the semantic position state for the real request.
            self.base_position_offset = None
            self.position_ids = None
            self._prepared_for_embedding = False
            self.fast_inference_ready = True
            return dict(self.inference_memory_report)
        except Exception:
            self.clear_weight_caches()
            self.base_position_offset = None
            self.position_ids = None
            self._prepared_for_embedding = False
            self._invalidate_inference_tables()
            raise

    def attach_to_model(self, model: torch.nn.Module) -> None:
        """Attach this static codebook manager to a Zip2ZipModel."""
        if hasattr(model, "codebook_manager"):
            self._previous_manager = model.codebook_manager
            model.codebook_manager = self
        base = getattr(model, "base_model", model)
        if hasattr(base, "get_input_embeddings"):
            input_emb = base.get_input_embeddings()
            if hasattr(input_emb, "codebook_manager"):
                input_emb.codebook_manager = self
        if hasattr(base, "get_output_embeddings"):
            output_emb = base.get_output_embeddings()
            if hasattr(output_emb, "codebook_manager"):
                output_emb.codebook_manager = self

    def detach_from_model(self, model: torch.nn.Module) -> None:
        """Detach this static codebook manager and restore the previous manager."""
        prev = getattr(self, "_previous_manager", None)
        if prev is not None:
            if hasattr(model, "codebook_manager"):
                model.codebook_manager = prev
            base = getattr(model, "base_model", model)
            if hasattr(base, "get_input_embeddings"):
                input_emb = base.get_input_embeddings()
                if hasattr(input_emb, "codebook_manager"):
                    input_emb.codebook_manager = prev
            if hasattr(base, "get_output_embeddings"):
                output_emb = base.get_output_embeddings()
                if hasattr(output_emb, "codebook_manager"):
                    output_emb.codebook_manager = prev

    def mask_unused_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask unused hypertoken logits to -inf so they cannot be generated.

        Args:
            logits: Logits tensor of shape (..., total_vocab_size)
        Returns:
            Logits with unused hypertoken slots masked to -inf.
        """
        if self.num_seeded < self.max_codebook_size:
            start = self.initial_vocab_size + self.num_seeded
            end = self.initial_vocab_size + self.max_codebook_size
            logits[..., start:end] = float("-inf")
        return logits

    def get_logits_processor(self) -> LogitsProcessor:
        """Return a HuggingFace LogitsProcessor that masks unseeded hypertoken slots."""
        return StaticCodebookLogitsWarper(
            initial_vocab_size=self.initial_vocab_size,
            num_seeded=self.num_seeded,
            max_codebook_size=self.max_codebook_size,
        )

    def segment_sequence(self, token_ids: Sequence[int]) -> List[int]:
        """Segment a sequence of base tokens using the current seeded codebook."""
        from zip2zip.segmenter import DynamicSegmenter

        segmenter = DynamicSegmenter(
            subtokens_to_hyper=self.subtokens_to_hyper,
            disabled_ids=self.disabled_ids,
            max_subtokens=self.max_subtokens,
        )
        return segmenter.segment(token_ids)

    def segment_batch(
        self, batch_ids: Sequence[Sequence[int]]
    ) -> List[List[int]]:
        """Segment a batch of base token sequences using the current seeded codebook."""
        return [self.segment_sequence(seq) for seq in batch_ids]

    def prepare_input_sequence(
        self, token_ids: Sequence[int], *, compress: bool = True
    ) -> List[int]:
        """Map original tokenizer IDs into the prepared fast-inference space.

        Original vocabulary-tail IDs (IDs at or above ``initial_vocab_size``)
        move by ``max_codebook_size`` because the effective tables reserve all
        H slots at the insertion point. This must happen before segmentation:
        otherwise a raw tail ID can collide numerically with an H ID.

        The legacy/training path intentionally keeps its historical behavior;
        this mapping is only valid after effective inference tables are ready.
        """
        if not self.fast_inference_ready:
            raise RuntimeError(
                "prepare_input_sequence() requires prepared fast-inference tables"
            )

        expanded_ids: List[int] = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, Integral):
                raise ValueError(f"Token ID must be an integer, got {token_id!r}")
            token_id = int(token_id)
            if token_id < 0:
                raise ValueError(f"Token ID must be non-negative, got {token_id}")
            if token_id >= self.initial_vocab_size:
                token_id += self.max_codebook_size
            expanded_ids.append(token_id)
        if not compress:
            return expanded_ids

        from zip2zip.segmenter import DynamicSegmenter

        segmenter = DynamicSegmenter(
            subtokens_to_hyper=self.subtokens_to_hyper,
            disabled_ids=self.disabled_ids,
            max_subtokens=self.max_subtokens,
        )
        return segmenter.segment(expanded_ids)

    def decode_hypertoken(self, token_id: int) -> List[int]:
        """Expand H IDs and restore shifted original-tail IDs to base IDs."""
        if token_id in self.hyper_to_subtokens:
            return list(self.hyper_to_subtokens[token_id])
        if token_id >= self.initial_vocab_size + self.max_codebook_size:
            return [token_id - self.max_codebook_size]
        return [token_id]

    def decode_sequence(self, token_ids: Sequence[int]) -> List[int]:
        """Expand H IDs and restore original tail IDs from expanded output space."""
        result: List[int] = []
        for tid in token_ids:
            result.extend(self.decode_hypertoken(tid))
        return result

    def clear_weight_caches(self) -> None:
        """Clear autograd weight caches between training steps."""
        self.hyper_embedding_weight_cache = None
        self.hyper_linear_weight_cache = None
        self._invalidate_inference_tables()
        self._prepared_for_embedding = False

    def reset(self, clear_dictionary: bool = False, clear_caches: bool = False) -> None:
        """Reset runtime generation state between requests.

        Args:
            clear_dictionary: If True, also clears the seeded codebook dictionary and weight caches.
                              Defaults to False so seeded codebook and synthesized weights persist for generation.
            clear_caches: If True, clears synthesized weight caches (essential for training backprop).
        """
        self.base_position_offset = None
        self.position_ids = None
        self._prepared_for_embedding = False

        if clear_caches or clear_dictionary:
            self.hyper_embedding_weight_cache = None
            self.hyper_linear_weight_cache = None
            self._invalidate_inference_tables()

        if clear_dictionary:
            self.hyper_to_subtokens.clear()
            self.subtokens_to_hyper.clear()
            self.num_seeded = 0
            self.updates = None
            self.updates_indices = None
            self.hyper_token_spans = None
            self.runtime_batch_size = None
            self.input_encoder_calls = 0
            self.output_encoder_calls = 0

    @classmethod
    def from_config(cls, config: Zip2ZipConfig) -> StaticCodebookManager:
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        return cls(
            initial_vocab_size=config.compression.initial_vocab_size,
            max_codebook_size=config.compression.max_codebook_size,
            max_subtokens=config.compression.max_subtokens,
            embedding_dim=getattr(config.encoder, "model_hidden_size", None)
            or config.encoder.hidden_size,
            pad_token_id=pad_token_id,
            disabled_ids=config.compression.disabled_ids,
        )


class StaticCodebookLogitsWarper(LogitsProcessor):
    """Masks unseeded hypertoken vocabulary slots during generation."""

    def __init__(
        self,
        initial_vocab_size: int,
        num_seeded: int,
        max_codebook_size: int,
    ) -> None:
        self.initial_vocab_size = initial_vocab_size
        self.num_seeded = num_seeded
        self.max_codebook_size = max_codebook_size

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self.num_seeded < self.max_codebook_size:
            start = self.initial_vocab_size + self.num_seeded
            end = self.initial_vocab_size + self.max_codebook_size
            scores[:, start:end] = float("-inf")
        return scores
