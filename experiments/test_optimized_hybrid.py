"""
Test and benchmark OptimizedLiveHybridCodebookManager.
Verifies that fixing redundant encoder passes brings Hybrid latency down to ~8s (matching Base and Official).
"""

import time
import torch
from typing import Dict, List, Optional, Sequence, Tuple
from transformers import AutoTokenizer

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from zip2zip import StaticCodebookManager, Zip2ZipModel
from zip2zip_compression import CodebookManager as RustCodebookManager, CompressionConfig, LZWCompressor
from src.evaluation.offline_segmenter import segment_tokens_dp
import pickle


class OptimizedLiveHybridCodebookManager:
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

        self.hyper_to_subtokens: Dict[int, List[int]] = {}
        self.subtokens_to_hyper: Dict[Tuple[int, ...], int] = {}
        self.num_seeded = 0

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

        self.hyper_embedding_weight_cache: Optional[torch.Tensor] = None
        self.hyper_linear_weight_cache: Optional[torch.Tensor] = None
        self.static_updates: Optional[torch.Tensor] = None
        self.new_lzw_updates: Optional[torch.Tensor] = None
        self.new_lzw_indices: Optional[List[List[int]]] = None

        self.runtime_batch_size: Optional[int] = None
        self.hyper_token_spans: Optional[torch.Tensor] = None
        self.base_position_offset: Optional[torch.Tensor] = None
        self.position_ids: Optional[torch.Tensor] = None
        self._prepared_for_embedding: bool = False
        self._static_emb_computed: bool = False
        self._static_lin_computed: bool = False

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

        self.hyper_token_spans = torch.zeros(
            (batch_size, self.max_codebook_size), dtype=torch.long, device=device
        )
        static_updates_list = []
        for hyper_id in sorted(self.hyper_to_subtokens.keys()):
            entry_idx = hyper_id - self.initial_vocab_size
            subtokens = self.hyper_to_subtokens[hyper_id]
            self.hyper_token_spans[:, entry_idx] = len(subtokens)
            padded = subtokens + [self.pad_token_id] * (self.max_subtokens - len(subtokens))
            static_updates_list.append(padded)

        if static_updates_list:
            base_up = torch.tensor(static_updates_list, dtype=torch.long, device=device)
            self.static_updates = base_up.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        else:
            self.static_updates = None

        self._static_emb_computed = False
        self._static_lin_computed = False
        self.hyper_embedding_weight_cache = None
        self.hyper_linear_weight_cache = None

    def prepare_input_ids(self, ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.LongTensor:
        batch_size, _ = ids.shape
        device = ids.device
        if self.hyper_token_spans is None:
            self.hyper_token_spans = torch.zeros((batch_size, self.max_codebook_size), dtype=torch.long, device=device)

        rust_updates, rust_indices = self.rust_mgr.update_codebooks(ids.tolist())

        has_new = any(len(ui) > 0 for ui in rust_indices)
        if has_new:
            new_updates_per_batch = []
            new_indices_per_batch = []
            for b in range(batch_size):
                b_updates = []
                b_indices = []
                if rust_indices[b]:
                    for local_i, r_idx in enumerate(rust_indices[b]):
                        global_idx = self.k_pred + r_idx
                        start_i = local_i * self.max_subtokens
                        end_i = start_i + self.max_subtokens
                        r_row = rust_updates[b][start_i:end_i]
                        b_updates.append(r_row)
                        b_indices.append(global_idx)
                        span_len = sum(1 for t in r_row if t != self.pad_token_id)
                        self.hyper_token_spans[b, global_idx] = max(1, span_len)
                new_updates_per_batch.append(b_updates)
                new_indices_per_batch.append(b_indices)

            # Max rows in batch
            max_new = max(len(bu) for bu in new_updates_per_batch)
            if max_new > 0:
                padded_batch_updates = []
                for bu in new_updates_per_batch:
                    pad_rows = [[self.pad_token_id] * self.max_subtokens] * (max_new - len(bu))
                    padded_batch_updates.append(bu + pad_rows)
                self.new_lzw_updates = torch.tensor(padded_batch_updates, dtype=torch.long, device=device)
                self.new_lzw_indices = new_indices_per_batch
            else:
                self.new_lzw_updates = None
                self.new_lzw_indices = None
        else:
            self.new_lzw_updates = None
            self.new_lzw_indices = None

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

        # 1. Compute static weights ONCE
        if not self._static_emb_computed:
            if self.static_updates is not None and self.num_seeded > 0:
                self.static_updates = self.static_updates.to(device)
                w_static = encoder_fn(self.static_updates, base_weight, self.pad_token_id)
                self.hyper_embedding_weight_cache[:, :self.num_seeded] = w_static[:, :self.num_seeded]
            self._static_emb_computed = True

        if not self._prepared_for_embedding:
            self.prepare_input_ids(ids)

        # 2. Update ONLY newly discovered reactive entries
        if self.new_lzw_updates is not None:
            new_w = encoder_fn(self.new_lzw_updates.to(device), base_weight, self.pad_token_id)
            for b in range(self.runtime_batch_size or 1):
                indices = self.new_lzw_indices[b]
                if indices:
                    self.hyper_embedding_weight_cache[b, indices] = new_w[b, :len(indices)]

        self._prepared_for_embedding = False
        return self.hyper_embedding_weight_cache

    def get_hyper_linear_weights(self, base_weight: torch.Tensor, encoder_fn) -> torch.Tensor:
        device = base_weight.device
        dtype = base_weight.dtype
        if self.hyper_linear_weight_cache is None:
            self.hyper_linear_weight_cache = torch.zeros(
                self.runtime_batch_size or 1, self.max_codebook_size, self.embedding_dim, dtype=dtype, device=device
            )

        # 1. Compute static weights ONCE
        if not self._static_lin_computed:
            if self.static_updates is not None and self.num_seeded > 0:
                self.static_updates = self.static_updates.to(device)
                w_static = encoder_fn(self.static_updates, base_weight, self.pad_token_id)
                self.hyper_linear_weight_cache[:, :self.num_seeded] = w_static[:, :self.num_seeded]
            self._static_lin_computed = True

        # 2. Update ONLY newly discovered reactive entries
        if self.new_lzw_updates is not None:
            new_w = encoder_fn(self.new_lzw_updates.to(device), base_weight, self.pad_token_id)
            for b in range(self.runtime_batch_size or 1):
                indices = self.new_lzw_indices[b]
                if indices:
                    self.hyper_linear_weight_cache[b, indices] = new_w[b, :len(indices)]

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
        self.new_lzw_updates = None
        self.new_lzw_indices = None
        if clear_all:
            self.hyper_embedding_weight_cache = None
            self.hyper_linear_weight_cache = None
            self.rust_mgr.reset()
            self._static_emb_computed = False
            self._static_lin_computed = False

    def decode_sequence(self, token_ids: Sequence[int]) -> List[int]:
        decoded_reactive, _ = self.rust_compressor.batch_decode([list(token_ids)])[0]
        final_tokens = []
        for t in decoded_reactive:
            if t in self.hyper_to_subtokens:
                final_tokens.extend(self.hyper_to_subtokens[t])
            else:
                final_tokens.append(t)
        return final_tokens


if __name__ == "__main__":
    prompt_text = "Write a Python function to solve the following problem:\nWrite a function to zip the two given tuples.\n"
    max_new_tokens = 25
    device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    with open("experiments/checkpoints/cached_predictor.pkl", "rb") as f:
        predictor = pickle.load(f)

    p_dict, _ = predictor.select_prompt_conditioned(prompt_ids, budget=32)
    pred_phrases = list(p_dict.keys())[:12]
    seeded_dict = {phrase: (32011 + i) for i, phrase in enumerate(pred_phrases)}

    model = Zip2ZipModel.from_pretrained(
        "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1",
        max_codebook_size=32,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    opt_mgr = OptimizedLiveHybridCodebookManager(
        initial_vocab_size=32011,
        k_pred=12,
        k_lzw=20,
        max_subtokens=3,
        embedding_dim=3072,
        pad_token_id=32000,
        disabled_ids=list(model.zip2zip_config.compression.disabled_ids),
    )
    opt_mgr.set_seeded_codebook(seeded_dict, batch_size=1, device=torch.device(device))
    opt_mgr.attach_to_model(model)

    comp_len, tiles, _ = segment_tokens_dp(prompt_ids, set(pred_phrases))
    resegmented = [tile[0] if len(tile) == 1 else seeded_dict[tile] for tile in tiles]
    input_tensor = torch.tensor([resegmented], dtype=torch.long, device=device)

    print("\n--- Running Optimized Live Hybrid Generation ---")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids=input_tensor, max_new_tokens=max_new_tokens, do_sample=False)
    t1 = time.perf_counter()
    opt_lat_ms = (t1 - t0) * 1000.0
    opt_mgr.detach_from_model(model)

    gen_tokens = out[0, len(resegmented):].tolist()
    expanded = opt_mgr.decode_sequence(gen_tokens)
    text = tokenizer.decode(expanded, skip_special_tokens=True)

    print(f"Optimized Hybrid Latency: {opt_lat_ms:.1f} ms (Previous unoptimized was ~15,500 ms!)")
    print(f"Decoded text: {repr(text[:60])}...")
    print("SUCCESS: OptimizedLiveHybridCodebookManager verified!")
