from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from transformers import AutoTokenizer, LogitsProcessor

from zip2zip.config import Zip2ZipConfig
from zip2zip.nn.encoders.base import EncoderFn

logger = logging.getLogger(__name__)


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

        # Position tracking (zip2zip++ base token positions)
        self.runtime_batch_size: Optional[int] = None
        self.hyper_token_spans: Optional[torch.Tensor] = None
        self.base_position_offset: Optional[torch.Tensor] = None
        self.position_ids: Optional[torch.Tensor] = None
        self._prepared_for_embedding: bool = False

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
        self.hyper_to_subtokens.clear()
        self.subtokens_to_hyper.clear()

        if isinstance(dictionary, dict):
            items = list(dictionary.items())
        else:
            items = list(enumerate(dictionary))

        if len(items) > self.max_codebook_size:
            raise ValueError(
                f"Dictionary contains {len(items)} items, exceeding "
                f"max_codebook_size={self.max_codebook_size}"
            )

        for raw_id, subtokens in items:
            if not (2 <= len(subtokens) <= self.max_subtokens):
                raise ValueError(
                    f"Subtoken length {len(subtokens)} must be between 2 and "
                    f"max_subtokens={self.max_subtokens}"
                )
            if any(t >= self.initial_vocab_size for t in subtokens):
                raise ValueError("Hypertokens cannot contain other hypertokens")
            if any(t in self.disabled_ids for t in subtokens):
                raise ValueError("Subtokens contain a disabled token ID")

            # Support relative index (0..K-1) or absolute ID (initial_vocab_size..)
            if raw_id < self.initial_vocab_size:
                hyper_id = raw_id + self.initial_vocab_size
                entry_idx = raw_id
            else:
                hyper_id = raw_id
                entry_idx = raw_id - self.initial_vocab_size

            if entry_idx >= self.max_codebook_size:
                raise ValueError(
                    f"Hypertoken entry index {entry_idx} >= max_codebook_size={self.max_codebook_size}"
                )

            subtokens_list = list(subtokens)
            self.hyper_to_subtokens[hyper_id] = subtokens_list
            self.subtokens_to_hyper[tuple(subtokens_list)] = hyper_id

        self.num_seeded = len(self.hyper_to_subtokens)
        self.runtime_batch_size = batch_size

        # Invalidate weight caches so new embeddings will be computed
        self.hyper_embedding_weight_cache = None
        self.hyper_linear_weight_cache = None
        self.base_position_offset = None
        self.position_ids = None
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

    def prepare_input_ids(
        self,
        ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.LongTensor:
        """Calculate zip2zip++ RoPE positions for prefill and generation steps."""
        if ids.ndim != 2:
            raise ValueError(f"input_ids must be rank 2, got shape {tuple(ids.shape)}")

        batch_size, _ = ids.shape
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

        if is_hyper.any():
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
        self.base_position_offset = self.base_position_offset + spans.sum(
            dim=-1, keepdim=True
        )
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
            self.hyper_embedding_weight_cache is None
            or self.hyper_embedding_weight_cache.shape[0] != batch_size
        ):
            self.runtime_batch_size = batch_size
            self.hyper_embedding_weight_cache = torch.zeros(
                batch_size,
                self.max_codebook_size,
                self.embedding_dim,
                dtype=dtype,
                device=curr_device,
            )
        else:
            self.hyper_embedding_weight_cache = self.hyper_embedding_weight_cache.to(
                curr_device
            ).to(dtype)

        if not self._prepared_for_embedding:
            self.prepare_input_ids(ids)

        if self.updates is not None and self.updates_indices is not None:
            self.updates = self.updates.to(curr_device)
            if any(len(ui) > 0 for ui in self.updates_indices):
                new_weights = encoder_fn(self.updates, base_weight, self.pad_token_id)
                for i, ui in enumerate(self.updates_indices):
                    self.hyper_embedding_weight_cache[i, ui] = new_weights[i, : len(ui)]

        self._prepared_for_embedding = False
        return self.hyper_embedding_weight_cache

    def get_hyper_linear_weights(
        self, base_weight: torch.Tensor, encoder_fn: EncoderFn
    ) -> torch.Tensor:
        """Synthesize and cache output projection vectors for seeded hypertokens."""
        curr_device = base_weight.device
        dtype = base_weight.dtype
        batch_size = self.runtime_batch_size or 1

        if (
            self.hyper_linear_weight_cache is None
            or self.hyper_linear_weight_cache.shape[0] != batch_size
        ):
            self.hyper_linear_weight_cache = torch.zeros(
                batch_size,
                self.max_codebook_size,
                self.embedding_dim,
                dtype=dtype,
                device=curr_device,
            )
        else:
            self.hyper_linear_weight_cache = self.hyper_linear_weight_cache.to(
                curr_device
            ).to(dtype)

        if self.updates is not None and self.updates_indices is not None:
            self.updates = self.updates.to(curr_device)
            if any(len(ui) > 0 for ui in self.updates_indices):
                new_weights = encoder_fn(self.updates, base_weight, self.pad_token_id)
                for i, ui in enumerate(self.updates_indices):
                    self.hyper_linear_weight_cache[i, ui] = new_weights[i, : len(ui)]

        return self.hyper_linear_weight_cache

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

    def decode_hypertoken(self, token_id: int) -> List[int]:
        """Expand a single hypertoken into its component base tokens."""
        if token_id in self.hyper_to_subtokens:
            return list(self.hyper_to_subtokens[token_id])
        return [token_id]

    def decode_sequence(self, token_ids: Sequence[int]) -> List[int]:
        """Expand all hypertokens in a sequence back into base tokens."""
        result: List[int] = []
        for tid in token_ids:
            if tid in self.hyper_to_subtokens:
                result.extend(self.hyper_to_subtokens[tid])
            else:
                result.append(tid)
        return result

    def reset(self, clear_dictionary: bool = False) -> None:
        """Reset runtime generation state between requests.

        Args:
            clear_dictionary: If True, also clears the seeded codebook dictionary.
                              Defaults to False so seeded codebook persists for generation.
        """
        self.hyper_embedding_weight_cache = None
        self.hyper_linear_weight_cache = None
        self.base_position_offset = None
        self.position_ids = None
        self._prepared_for_embedding = False

        if clear_dictionary:
            self.hyper_to_subtokens.clear()
            self.subtokens_to_hyper.clear()
            self.num_seeded = 0
            self.updates = None
            self.updates_indices = None
            self.hyper_token_spans = None
            self.runtime_batch_size = None

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
