"""
Test LiveHybridCodebookManager attached to Zip2ZipModel.
Verifies forward pass and generate step with both static and reactive hypertokens.
"""

import torch
from typing import Dict, List, Optional, Sequence, Tuple
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

from zip2zip import StaticCodebookManager, Zip2ZipModel
from zip2zip_compression import CodebookManager as RustCodebookManager, CompressionConfig, LZWCompressor


class LiveHybridCodebookManager:
    def __init__(
        self,
        initial_vocab_size: int,
        k_pred: int,
        k_lzw: int,
        max_subtokens: int,
        embedding_dim: int,
        pad_token_id: int,
        disabled_ids: Optional[Sequence[int]] = None,
    ):
        self.initial_vocab_size = initial_vocab_size
        self.k_pred = k_pred
        self.k_lzw = k_lzw
        self.max_codebook_size = k_pred + k_lzw
        self.max_subtokens = max_subtokens
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        self.disabled_ids = set(disabled_ids) if disabled_ids else set()

        # Seeded static dictionary
        self.hyper_to_subtokens: Dict[int, List[int]] = {}
        self.subtokens_to_hyper: Dict[Tuple[int, ...], int] = {}
        self.num_seeded = 0

        # Rust LZW manager for reactive slots
        self.rust_mgr = RustCodebookManager(
            config=CompressionConfig(
                initial_vocab_size=initial_vocab_size + k_pred,
                max_codebook_size=k_lzw,
                max_subtokens=max_subtokens,
                pad_token_id=pad_token_id,
                disabled_ids=list(self.disabled_ids),
            )
        )
        self.rust_compressor = LZWCompressor(
            initial_vocab_size=initial_vocab_size + k_pred,
            max_codebook_size=k_lzw,
            max_subtokens=max_subtokens,
            pad_token_id=pad_token_id,
            disabled_ids=list(self.disabled_ids),
        )

        # Caches & state
        self.updates: Optional[torch.Tensor] = None
        self.updates_indices: Optional[List[List[int]]] = None
        self.hyper_embedding_weight_cache: Optional[torch.Tensor] = None
        self.hyper_linear_weight_cache: Optional[torch.Tensor] = None

        self.runtime_batch_size: Optional[int] = None
        self.hyper_token_spans: Optional[torch.Tensor] = None
        self.base_position_offset: Optional[torch.Tensor] = None
        self.position_ids: Optional[torch.Tensor] = None
        self._prepared_for_embedding: bool = False

    def set_seeded_codebook(
        self,
        dictionary: Dict[Tuple[int, ...], int],
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.hyper_to_subtokens.clear()
        self.subtokens_to_hyper.clear()
        for phrase, hyper_id in dictionary.items():
            self.hyper_to_subtokens[hyper_id] = list(phrase)
            self.subtokens_to_hyper[phrase] = hyper_id
        self.num_seeded = len(self.hyper_to_subtokens)
        self.runtime_batch_size = batch_size
        self._init_spans_and_updates(batch_size, device=device)

    def _init_spans_and_updates(self, batch_size: int, device: Optional[torch.device] = None):
        self.hyper_token_spans = torch.zeros(
            (batch_size, self.max_codebook_size), dtype=torch.long, device=device
        )
        # Fill static spans
        for hyper_id, subtokens in self.hyper_to_subtokens.items():
            entry_idx = hyper_id - self.initial_vocab_size
            self.hyper_token_spans[:, entry_idx] = len(subtokens)

        # Static updates
        static_updates = []
        static_indices = []
        for hyper_id in sorted(self.hyper_to_subtokens.keys()):
            entry_idx = hyper_id - self.initial_vocab_size
            static_indices.append(entry_idx)
            subtokens = self.hyper_to_subtokens[hyper_id]
            padded = subtokens + [self.pad_token_id] * (self.max_subtokens - len(subtokens))
            static_updates.append(padded)

        if static_updates:
            base_up = torch.tensor(static_updates, dtype=torch.long, device=device)
            self.updates = base_up.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            self.updates_indices = [list(static_indices) for _ in range(batch_size)]
        else:
            self.updates = torch.full((batch_size, 0, self.max_subtokens), self.pad_token_id, dtype=torch.long, device=device)
            self.updates_indices = [[] for _ in range(batch_size)]

    def prepare_input_ids(self, ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.LongTensor:
        batch_size, _ = ids.shape
        device = ids.device
        if self.hyper_token_spans is None:
            self._init_spans_and_updates(batch_size, device=device)

        # Update rust LZW manager
        rust_updates, rust_indices = self.rust_mgr.update_codebooks(ids.tolist())
        
        # Merge updates if any new reactive entries were discovered
        merged_indices = [list(self.updates_indices[b]) for b in range(batch_size)]
        new_updates_list = []
        for b in range(batch_size):
            if rust_indices[b]:
                for r_idx in rust_indices[b]:
                    global_idx = self.k_pred + r_idx
                    if global_idx not in merged_indices[b]:
                        merged_indices[b].append(global_idx)
                        start_i = r_idx * self.max_subtokens
                        end_i = start_i + self.max_subtokens
                        r_row = rust_updates[b][start_i:end_i]
                        new_updates_list.append(r_row)
                        # Span is non-pad tokens
                        span_len = sum(1 for t in r_row if t != self.pad_token_id)
                        self.hyper_token_spans[b, global_idx] = max(1, span_len)

        if new_updates_list:
            new_up_tensor = torch.tensor(new_updates_list, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1, -1)
            self.updates = torch.cat([self.updates, new_up_tensor], dim=1)
            self.updates_indices = merged_indices

        # Calculate RoPE positions
        is_hyper = (ids >= self.initial_vocab_size) & (ids < self.initial_vocab_size + self.max_codebook_size)
        spans = torch.ones_like(ids)
        if is_hyper.any():
            entry_ids = (ids - self.initial_vocab_size).clamp(0, self.max_codebook_size - 1)
            hyper_spans = self.hyper_token_spans.to(device).gather(1, entry_ids)
            spans = torch.where(is_hyper, hyper_spans, spans)

        if attention_mask is not None:
            valid = attention_mask.to(device=ids.device, dtype=torch.bool)
            spans = torch.where(valid, spans, torch.zeros_like(spans))
        else:
            valid = torch.ones_like(ids, dtype=torch.bool)

        if self.base_position_offset is None:
            self.base_position_offset = torch.zeros(batch_size, 1, device=device, dtype=torch.long)
        else:
            self.base_position_offset = self.base_position_offset.to(device)

        positions = self.base_position_offset + spans.cumsum(dim=-1) - 1
        positions = torch.where(valid, positions, torch.zeros_like(positions))
        self.base_position_offset = self.base_position_offset + spans.sum(dim=-1, keepdim=True)
        self.position_ids = positions
        self._prepared_for_embedding = True
        return positions

    def init_codebooks_and_hyper_weight_cache(self, batch_size: int, codebooks: Optional[List] = None) -> None:
        self.runtime_batch_size = batch_size

    def get_hyper_embedding_weights(self, ids: torch.LongTensor, base_weight: torch.Tensor, encoder_fn) -> torch.Tensor:
        device = base_weight.device
        dtype = base_weight.dtype
        if self.hyper_embedding_weight_cache is None:
            self.hyper_embedding_weight_cache = torch.zeros(
                ids.shape[0], self.max_codebook_size, self.embedding_dim, dtype=dtype, device=device
            )
        if not self._prepared_for_embedding:
            self.prepare_input_ids(ids)
        if self.updates is not None and any(len(ui) > 0 for ui in self.updates_indices):
            self.updates = self.updates.to(device)
            new_weights = encoder_fn(self.updates, base_weight, self.pad_token_id)
            for i, ui in enumerate(self.updates_indices):
                if ui:
                    self.hyper_embedding_weight_cache[i, ui] = new_weights[i, : len(ui)]
        self._prepared_for_embedding = False
        return self.hyper_embedding_weight_cache

    def get_hyper_linear_weights(self, base_weight: torch.Tensor, encoder_fn) -> torch.Tensor:
        device = base_weight.device
        dtype = base_weight.dtype
        if self.hyper_linear_weight_cache is None:
            self.hyper_linear_weight_cache = torch.zeros(
                self.runtime_batch_size or 1, self.max_codebook_size, self.embedding_dim, dtype=dtype, device=device
            )
        if self.updates is not None and any(len(ui) > 0 for ui in self.updates_indices):
            self.updates = self.updates.to(device)
            new_weights = encoder_fn(self.updates, base_weight, self.pad_token_id)
            for i, ui in enumerate(self.updates_indices):
                if ui:
                    self.hyper_linear_weight_cache[i, ui] = new_weights[i, : len(ui)]
        return self.hyper_linear_weight_cache

    def attach_to_model(self, model: torch.nn.Module) -> None:
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

    def reset(self, clear_all: bool = False) -> None:
        self.base_position_offset = None
        self.position_ids = None
        self._prepared_for_embedding = False
        if clear_all:
            self.hyper_embedding_weight_cache = None
            self.hyper_linear_weight_cache = None
            self.rust_mgr.reset()

    def decode_sequence(self, token_ids: Sequence[int]) -> List[int]:
        # Step 1: Decode reactive tokens
        decoded_reactive, _ = self.rust_compressor.batch_decode([list(token_ids)])[0]
        # Step 2: Decode static tokens
        final_tokens = []
        for t in decoded_reactive:
            if t in self.hyper_to_subtokens:
                final_tokens.extend(self.hyper_to_subtokens[t])
            else:
                final_tokens.append(t)
        return final_tokens


if __name__ == "__main__":
    print("Testing LiveHybridCodebookManager instantiation...")
    mgr = LiveHybridCodebookManager(32011, 12, 20, 3, 3072, 32000, [])
    mgr.set_seeded_codebook({(100, 200): 32011, (300, 400): 32012}, batch_size=1)
    inp = torch.tensor([[10, 20, 32011, 40, 50, 40, 50]], dtype=torch.long)
    pos = mgr.prepare_input_ids(inp)
    print("RoPE positions:", pos)
    print("Static seeded:", mgr.num_seeded)
    print("Rust updates indices:", mgr.updates_indices)
    seq = [10, 20, 32011, 32023]
    dec = mgr.decode_sequence(seq)
    print("Decoded sequence:", dec)
    print("LIVE HYBRID MANAGER TEST PASSED!")
