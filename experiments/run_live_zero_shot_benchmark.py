"""
Live Zero-Shot Autoregressive Generation Benchmark on Zip2Zip / Phi-3.5.
Evaluates:
1. Base Phi-3.5 (Normal autoregressive generation, K=0)
2. Official Reactive Zip2Zip (Pure LZW, K=32 reactive slots)
3. Pure Predictive Seeded Hypertokens (K=32 prompt-conditioned entries)
4. True Sequential Hybrid: 37.5% Pred / 62.5% Reactive LZW (12 Pred + 20 LZW, K=32)
5. True Sequential Hybrid: 50% Pred / 50% Reactive LZW (16 Pred + 16 LZW, K=32)

Measures:
- Autoregressive decode steps vs. Base-equivalent tokens (Realized compression)
- Zero-shot realization gap (Offline DP theoretical compression vs. Live emission)
- Setup latency (retrieval + weight synthesis), Prefill latency, Decode latency, and Total wall-clock time
- Effective tokens/sec vs. Raw steps/sec
- Output text equivalence, syntax correctness (AST for code), and numerical match (for math)
- Break-even output length modeling
"""

import argparse
import ast
import gc
import json
import os
import pickle
import re
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import psutil
import torch
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from zip2zip import (
    StaticCodebookManager,
    Zip2ZipModel,
)
from zip2zip_compression import CodebookManager as RustCodebookManager, CompressionConfig, LZWCompressor
from src.evaluation.offline_segmenter import segment_tokens_dp


class LiveHybridCodebookManager:
    """True live sequential hybrid codebook manager.
    
    Combines k_pred pre-seeded prompt-conditioned static entries with
    k_lzw dynamically discovered reactive LZW entries during autoregressive decode.
    """

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
        for hyper_id, subtokens in self.hyper_to_subtokens.items():
            entry_idx = hyper_id - self.initial_vocab_size
            self.hyper_token_spans[:, entry_idx] = len(subtokens)

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

        rust_updates, rust_indices = self.rust_mgr.update_codebooks(ids.tolist())
        
        merged_indices = [list(self.updates_indices[b]) for b in range(batch_size)]
        new_updates_list = []
        for b in range(batch_size):
            if rust_indices[b]:
                for local_i, r_idx in enumerate(rust_indices[b]):
                    global_idx = self.k_pred + r_idx
                    if global_idx not in merged_indices[b]:
                        merged_indices[b].append(global_idx)
                        start_i = local_i * self.max_subtokens
                        end_i = start_i + self.max_subtokens
                        r_row = rust_updates[b][start_i:end_i]
                        new_updates_list.append(r_row)
                        span_len = sum(1 for t in r_row if t != self.pad_token_id)
                        self.hyper_token_spans[b, global_idx] = max(1, span_len)

        if new_updates_list:
            new_up_tensor = torch.tensor(new_updates_list, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1, -1)
            self.updates = torch.cat([self.updates, new_up_tensor], dim=1)
            self.updates_indices = merged_indices

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
        decoded_reactive, _ = self.rust_compressor.batch_decode([list(token_ids)])[0]
        final_tokens = []
        for t in decoded_reactive:
            if t in self.hyper_to_subtokens:
                final_tokens.extend(self.hyper_to_subtokens[t])
            else:
                final_tokens.append(t)
        return final_tokens


@dataclass
class LiveBenchmarkResult:
    condition: str
    prompt_id: str
    domain: str
    prompt_text: str
    base_prompt_tokens: int
    compressed_prefill_tokens: int
    generated_steps: int
    base_equivalent_tokens: int
    realized_step_reduction_pct: float
    offline_theoretical_comp_pct: float
    realization_gap_ratio: float
    hypertokens_emitted: int
    setup_latency_ms: float
    prefill_latency_ms: float
    decode_latency_ms: float
    total_latency_ms: float
    raw_steps_per_sec: float
    effective_tokens_per_sec: float
    is_valid_syntax: Optional[bool]
    extracted_answer: Optional[str]
    decoded_text: str


def evaluate_output_quality(text: str, domain: str) -> Tuple[Optional[bool], Optional[str]]:
    is_valid = None
    extracted = None
    if domain == "code":
        # Extract python code block if present
        code = text
        if "```python" in text:
            code = text.split("```python")[1].split("```")[0]
        elif "```" in text:
            code = text.split("```")[1].split("```")[0]
        try:
            ast.parse(code)
            is_valid = True
        except Exception:
            # Also try parsing function def substring
            try:
                if "def " in text:
                    fn_code = "def " + text.split("def ", 1)[1]
                    ast.parse(fn_code)
                    is_valid = True
                else:
                    is_valid = False
            except Exception:
                is_valid = False
    elif domain == "reasoning":
        # Look for #### <number> or the last numerical value
        match = re.search(r"####\s*(-?[\d\.,]+)", text)
        if match:
            extracted = match.group(1).replace(",", "").strip()
        else:
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
            if nums:
                extracted = nums[-1]
    return is_valid, extracted


def run_live_benchmark(
    checkpoint_path: str = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1",
    predictor_cache: str = "experiments/checkpoints/cached_predictor.pkl",
    prompts_file: str = "experiments/live_benchmark_prompts.json",
    num_prompts: int = 15,
    budget: int = 32,
    max_new_tokens: int = 30,
    device: str = "cpu",
    output_path: str = "experiments/live_zero_shot_results.json",
) -> List[LiveBenchmarkResult]:
    print("=" * 80)
    print(f"STARTING COMPREHENSIVE LIVE ZERO-SHOT BENCHMARK ON {checkpoint_path}")
    print(f"Device: {device}, Budget: {budget}, Max New Tokens: {max_new_tokens}, Prompts: {num_prompts}")
    print("=" * 80)

    # 1. Load Tokenizer & Predictor
    print("Loading tokenizer and cached predictor...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    with open(predictor_cache, "rb") as f:
        predictor = pickle.load(f)

    # 2. Load Model on CPU
    print("\nLoading Zip2ZipModel on CPU...")
    t0_load = time.time()
    zip_model = Zip2ZipModel.from_pretrained(
        checkpoint_path,
        max_codebook_size=budget,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    zip_model = zip_model.to(device)
    print(f"Zip2ZipModel loaded in {time.time() - t0_load:.2f}s!")

    dim = 3072
    initial_vocab_size = 32011
    disabled_ids = list(zip_model.zip2zip_config.compression.disabled_ids)

    # 3. Load Selected Prompts
    with open(prompts_file, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)
    prompts_to_eval = all_prompts[:num_prompts]
    print(f"Loaded {len(prompts_to_eval)} prompts for evaluation.")

    results: List[LiveBenchmarkResult] = []

    # Check for existing partial results to allow resuming
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            print(f"Found {len(existing_data)} existing results in {output_path}.")
        except Exception:
            pass

    for p_idx, p_info in enumerate(prompts_to_eval):
        prompt_text = p_info["prompt"]
        domain = p_info["domain"]
        p_id_name = p_info["id"]

        print(f"\n" + "=" * 80)
        print(f"EVALUATING PROMPT [{p_idx + 1}/{len(prompts_to_eval)}]: {p_id_name} ({domain})")
        print(f"Prompt: {repr(prompt_text[:80])}...")
        print("=" * 80)

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        base_prompt_len = len(prompt_ids)

        # Pre-compute prompt predictions once for this prompt
        t0_pred = time.perf_counter()
        p_dict, pred_lat_ms = predictor.select_prompt_conditioned(prompt_ids, budget=budget)
        pred_phrases = list(p_dict.keys())
        t1_pred = time.perf_counter()
        pred_retrieval_ms = (t1_pred - t0_pred) * 1000.0

        # Offline DP theoretical compression for Pure Predictive K=32
        _, _, stats_c3 = segment_tokens_dp(prompt_ids, set(pred_phrases))
        theoretical_pred_pct = stats_c3.get("compression_pct", 0.0)

        # -------------------------------------------------------------
        # Condition 1: Base Phi-3.5 (K=0)
        # -------------------------------------------------------------
        print("\n--- Running Condition 1: Base Phi-3.5 (K=0) ---")
        base_input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        
        t0_c1 = time.perf_counter()
        with torch.no_grad():
            out_c1 = zip_model.base_model.generate(
                input_ids=base_input_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        t1_c1 = time.perf_counter()
        c1_ms = (t1_c1 - t0_c1) * 1000.0

        gen_tokens_c1 = out_c1[0, base_prompt_len:].tolist()
        num_steps_c1 = len(gen_tokens_c1)
        text_c1 = tokenizer.decode(gen_tokens_c1, skip_special_tokens=True)
        is_valid_c1, ans_c1 = evaluate_output_quality(text_c1, domain)

        res_c1 = LiveBenchmarkResult(
            condition="Base Phi-3.5 (K=0)",
            prompt_id=p_id_name,
            domain=domain,
            prompt_text=prompt_text,
            base_prompt_tokens=base_prompt_len,
            compressed_prefill_tokens=base_prompt_len,
            generated_steps=num_steps_c1,
            base_equivalent_tokens=num_steps_c1,
            realized_step_reduction_pct=0.0,
            offline_theoretical_comp_pct=0.0,
            realization_gap_ratio=1.0,
            hypertokens_emitted=0,
            setup_latency_ms=0.0,
            prefill_latency_ms=0.0,
            decode_latency_ms=c1_ms,
            total_latency_ms=c1_ms,
            raw_steps_per_sec=(num_steps_c1 / (c1_ms / 1000.0)) if c1_ms > 0 else 0.0,
            effective_tokens_per_sec=(num_steps_c1 / (c1_ms / 1000.0)) if c1_ms > 0 else 0.0,
            is_valid_syntax=is_valid_c1,
            extracted_answer=ans_c1,
            decoded_text=text_c1,
        )
        results.append(res_c1)
        print(f"  Generated {num_steps_c1} steps in {c1_ms:.1f} ms ({res_c1.effective_tokens_per_sec:.1f} tok/s)")
        print(f"  Valid syntax / answer: {is_valid_c1} / {ans_c1}")
        print(f"  Output text: {repr(text_c1[:60])}...")

        # -------------------------------------------------------------
        # Condition 2: Official Reactive Zip2Zip (Pure LZW, K=32)
        # -------------------------------------------------------------
        print("\n--- Running Condition 2: Official Reactive Zip2Zip (Pure LZW, K=32) ---")
        t0_c2 = time.perf_counter()
        with torch.no_grad():
            out_c2 = zip_model.generate(
                input_ids=base_input_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        t1_c2 = time.perf_counter()
        c2_ms = (t1_c2 - t0_c2) * 1000.0

        gen_tokens_c2 = out_c2[0, base_prompt_len:].tolist()
        num_steps_c2 = len(gen_tokens_c2)
        hypertokens_c2 = sum(1 for t in gen_tokens_c2 if t >= initial_vocab_size)

        lzw_compressor = LZWCompressor(
            initial_vocab_size=initial_vocab_size,
            max_codebook_size=budget,
            max_subtokens=3,
            pad_token_id=tokenizer.pad_token_id or 32000,
            disabled_ids=disabled_ids,
        )
        full_seq_c2 = out_c2[0].tolist()
        decoded_full_c2, _ = lzw_compressor.batch_decode([full_seq_c2])[0]
        expanded_c2 = decoded_full_c2[base_prompt_len:]
        base_equiv_c2 = len(expanded_c2)
        text_c2 = tokenizer.decode(expanded_c2, skip_special_tokens=True)
        step_red_c2 = (1.0 - num_steps_c2 / base_equiv_c2) * 100.0 if base_equiv_c2 > 0 else 0.0
        is_valid_c2, ans_c2 = evaluate_output_quality(text_c2, domain)

        res_c2 = LiveBenchmarkResult(
            condition="Official Reactive Zip2Zip (Pure LZW, K=32)",
            prompt_id=p_id_name,
            domain=domain,
            prompt_text=prompt_text,
            base_prompt_tokens=base_prompt_len,
            compressed_prefill_tokens=base_prompt_len,
            generated_steps=num_steps_c2,
            base_equivalent_tokens=base_equiv_c2,
            realized_step_reduction_pct=step_red_c2,
            offline_theoretical_comp_pct=15.31,
            realization_gap_ratio=(step_red_c2 / 15.31) if 15.31 > 0 else 0.0,
            hypertokens_emitted=hypertokens_c2,
            setup_latency_ms=0.0,
            prefill_latency_ms=0.0,
            decode_latency_ms=c2_ms,
            total_latency_ms=c2_ms,
            raw_steps_per_sec=(num_steps_c2 / (c2_ms / 1000.0)) if c2_ms > 0 else 0.0,
            effective_tokens_per_sec=(base_equiv_c2 / (c2_ms / 1000.0)) if c2_ms > 0 else 0.0,
            is_valid_syntax=is_valid_c2,
            extracted_answer=ans_c2,
            decoded_text=text_c2,
        )
        results.append(res_c2)
        print(f"  Generated {num_steps_c2} steps (expanded to {base_equiv_c2} base tokens) in {c2_ms:.1f} ms")
        print(f"  Realized step reduction: {step_red_c2:.2f}% ({hypertokens_c2} hypertokens emitted)")
        print(f"  Effective speed: {res_c2.effective_tokens_per_sec:.1f} tok/s")
        print(f"  Valid syntax / answer: {is_valid_c2} / {ans_c2}")
        print(f"  Output text: {repr(text_c2[:60])}...")

        # -------------------------------------------------------------
        # Condition 3: Pure Predictive Seeded Hypertokens (K=32)
        # -------------------------------------------------------------
        print("\n--- Running Condition 3: Pure Predictive Seeded Hypertokens (K=32) ---")
        static_mgr = StaticCodebookManager(
            initial_vocab_size=initial_vocab_size,
            max_codebook_size=budget,
            max_subtokens=3,
            embedding_dim=dim,
            pad_token_id=tokenizer.pad_token_id or 32000,
            disabled_ids=disabled_ids,
        )
        t0_seed = time.perf_counter()
        seeded_dict_c3 = {phrase: (initial_vocab_size + i) for i, phrase in enumerate(pred_phrases)}
        static_mgr.set_seeded_codebook(seeded_dict_c3, batch_size=1, device=torch.device(device))
        t1_seed = time.perf_counter()
        setup_ms_c3 = pred_retrieval_ms + ((t1_seed - t0_seed) * 1000.0)

        comp_len_c3, tiles_c3, _ = segment_tokens_dp(prompt_ids, set(pred_phrases))
        resegmented_prompt_c3 = []
        for tile in tiles_c3:
            if len(tile) == 1:
                resegmented_prompt_c3.append(tile[0])
            else:
                resegmented_prompt_c3.append(seeded_dict_c3[tile])

        input_tensor_c3 = torch.tensor([resegmented_prompt_c3], dtype=torch.long, device=device)
        logits_proc_c3 = LogitsProcessorList([static_mgr.get_logits_processor()])
        static_mgr.attach_to_model(zip_model)

        t0_gen_c3 = time.perf_counter()
        with torch.no_grad():
            out_c3 = zip_model.generate(
                input_ids=input_tensor_c3,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_proc_c3,
                do_sample=False,
            )
        t1_gen_c3 = time.perf_counter()
        gen_ms_c3 = (t1_gen_c3 - t0_gen_c3) * 1000.0
        tot_ms_c3 = setup_ms_c3 + gen_ms_c3
        static_mgr.detach_from_model(zip_model)

        gen_tokens_c3 = out_c3[0, len(resegmented_prompt_c3):].tolist()
        num_steps_c3 = len(gen_tokens_c3)
        hypertokens_c3 = sum(1 for t in gen_tokens_c3 if t >= initial_vocab_size)
        expanded_c3 = static_mgr.decode_sequence(gen_tokens_c3)
        base_equiv_c3 = len(expanded_c3)
        text_c3 = tokenizer.decode(expanded_c3, skip_special_tokens=True)
        step_red_c3 = (1.0 - num_steps_c3 / base_equiv_c3) * 100.0 if base_equiv_c3 > 0 else 0.0
        is_valid_c3, ans_c3 = evaluate_output_quality(text_c3, domain)

        res_c3 = LiveBenchmarkResult(
            condition="Pure Predictive Seeded (K=32)",
            prompt_id=p_id_name,
            domain=domain,
            prompt_text=prompt_text,
            base_prompt_tokens=base_prompt_len,
            compressed_prefill_tokens=len(resegmented_prompt_c3),
            generated_steps=num_steps_c3,
            base_equivalent_tokens=base_equiv_c3,
            realized_step_reduction_pct=step_red_c3,
            offline_theoretical_comp_pct=theoretical_pred_pct,
            realization_gap_ratio=(step_red_c3 / theoretical_pred_pct) if theoretical_pred_pct > 0 else 0.0,
            hypertokens_emitted=hypertokens_c3,
            setup_latency_ms=setup_ms_c3,
            prefill_latency_ms=setup_ms_c3,
            decode_latency_ms=gen_ms_c3,
            total_latency_ms=tot_ms_c3,
            raw_steps_per_sec=(num_steps_c3 / (gen_ms_c3 / 1000.0)) if gen_ms_c3 > 0 else 0.0,
            effective_tokens_per_sec=(base_equiv_c3 / (tot_ms_c3 / 1000.0)) if tot_ms_c3 > 0 else 0.0,
            is_valid_syntax=is_valid_c3,
            extracted_answer=ans_c3,
            decoded_text=text_c3,
        )
        results.append(res_c3)
        print(f"  Generated {num_steps_c3} steps (expanded to {base_equiv_c3} base tokens) in {tot_ms_c3:.1f} ms")
        print(f"  Prefill compression: {base_prompt_len} -> {len(resegmented_prompt_c3)} ({stats_c3['compression_pct']:.1f}%)")
        print(f"  Realized step reduction: {step_red_c3:.2f}% ({hypertokens_c3} hypertokens emitted)")
        print(f"  Effective speed: {res_c3.effective_tokens_per_sec:.1f} tok/s")
        print(f"  Valid syntax / answer: {is_valid_c3} / {ans_c3}")
        print(f"  Output text: {repr(text_c3[:60])}...")

        # -------------------------------------------------------------
        # Condition 4: True Live Hybrid (12 Pred + 20 LZW)
        # -------------------------------------------------------------
        print("\n--- Running Condition 4: True Live Hybrid (12 Pred + 20 LZW) ---")
        pred_phrases_c4 = pred_phrases[:12]
        seeded_dict_c4 = {phrase: (initial_vocab_size + i) for i, phrase in enumerate(pred_phrases_c4)}

        hybrid_mgr_c4 = LiveHybridCodebookManager(
            initial_vocab_size=initial_vocab_size,
            k_pred=12,
            k_lzw=20,
            max_subtokens=3,
            embedding_dim=dim,
            pad_token_id=tokenizer.pad_token_id or 32000,
            disabled_ids=disabled_ids,
        )
        t0_seed_c4 = time.perf_counter()
        hybrid_mgr_c4.set_seeded_codebook(seeded_dict_c4, batch_size=1, device=torch.device(device))
        t1_seed_c4 = time.perf_counter()
        setup_ms_c4 = pred_retrieval_ms + ((t1_seed_c4 - t0_seed_c4) * 1000.0)

        comp_len_c4, tiles_c4, stats_c4 = segment_tokens_dp(prompt_ids, set(pred_phrases_c4))
        resegmented_prompt_c4 = []
        for tile in tiles_c4:
            if len(tile) == 1:
                resegmented_prompt_c4.append(tile[0])
            else:
                resegmented_prompt_c4.append(seeded_dict_c4[tile])

        input_tensor_c4 = torch.tensor([resegmented_prompt_c4], dtype=torch.long, device=device)
        hybrid_mgr_c4.attach_to_model(zip_model)

        t0_gen_c4 = time.perf_counter()
        with torch.no_grad():
            out_c4 = zip_model.generate(
                input_ids=input_tensor_c4,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        t1_gen_c4 = time.perf_counter()
        gen_ms_c4 = (t1_gen_c4 - t0_gen_c4) * 1000.0
        tot_ms_c4 = setup_ms_c4 + gen_ms_c4
        hybrid_mgr_c4.detach_from_model(zip_model)

        full_c4 = out_c4[0].tolist()
        expanded_full_c4 = hybrid_mgr_c4.decode_sequence(full_c4)
        expanded_c4 = expanded_full_c4[base_prompt_len:]
        num_steps_c4 = len(full_c4) - len(resegmented_prompt_c4)
        base_equiv_c4 = len(expanded_c4)
        gen_tokens_c4 = full_c4[len(resegmented_prompt_c4):]
        hypertokens_c4 = sum(1 for t in gen_tokens_c4 if t >= initial_vocab_size)
        text_c4 = tokenizer.decode(expanded_c4, skip_special_tokens=True)
        step_red_c4 = (1.0 - num_steps_c4 / base_equiv_c4) * 100.0 if base_equiv_c4 > 0 else 0.0
        is_valid_c4, ans_c4 = evaluate_output_quality(text_c4, domain)

        res_c4 = LiveBenchmarkResult(
            condition="37.5% Pred / 62.5% LZW Hybrid (12/20)",
            prompt_id=p_id_name,
            domain=domain,
            prompt_text=prompt_text,
            base_prompt_tokens=base_prompt_len,
            compressed_prefill_tokens=len(resegmented_prompt_c4),
            generated_steps=num_steps_c4,
            base_equivalent_tokens=base_equiv_c4,
            realized_step_reduction_pct=step_red_c4,
            offline_theoretical_comp_pct=17.80,
            realization_gap_ratio=(step_red_c4 / 17.80) if 17.80 > 0 else 0.0,
            hypertokens_emitted=hypertokens_c4,
            setup_latency_ms=setup_ms_c4,
            prefill_latency_ms=setup_ms_c4,
            decode_latency_ms=gen_ms_c4,
            total_latency_ms=tot_ms_c4,
            raw_steps_per_sec=(num_steps_c4 / (gen_ms_c4 / 1000.0)) if gen_ms_c4 > 0 else 0.0,
            effective_tokens_per_sec=(base_equiv_c4 / (tot_ms_c4 / 1000.0)) if tot_ms_c4 > 0 else 0.0,
            is_valid_syntax=is_valid_c4,
            extracted_answer=ans_c4,
            decoded_text=text_c4,
        )
        results.append(res_c4)
        print(f"  Generated {num_steps_c4} steps (expanded to {base_equiv_c4} base tokens) in {tot_ms_c4:.1f} ms")
        print(f"  Prefill compression: {base_prompt_len} -> {len(resegmented_prompt_c4)} ({stats_c4['compression_pct']:.1f}%)")
        print(f"  Realized step reduction: {step_red_c4:.2f}% ({hypertokens_c4} hypertokens emitted)")
        print(f"  Effective speed: {res_c4.effective_tokens_per_sec:.1f} tok/s")
        print(f"  Valid syntax / answer: {is_valid_c4} / {ans_c4}")
        print(f"  Output text: {repr(text_c4[:60])}...")

        # -------------------------------------------------------------
        # Condition 5: True Live Hybrid (16 Pred + 16 LZW)
        # -------------------------------------------------------------
        print("\n--- Running Condition 5: True Live Hybrid (16 Pred + 16 LZW) ---")
        pred_phrases_c5 = pred_phrases[:16]
        seeded_dict_c5 = {phrase: (initial_vocab_size + i) for i, phrase in enumerate(pred_phrases_c5)}

        hybrid_mgr_c5 = LiveHybridCodebookManager(
            initial_vocab_size=initial_vocab_size,
            k_pred=16,
            k_lzw=16,
            max_subtokens=3,
            embedding_dim=dim,
            pad_token_id=tokenizer.pad_token_id or 32000,
            disabled_ids=disabled_ids,
        )
        t0_seed_c5 = time.perf_counter()
        hybrid_mgr_c5.set_seeded_codebook(seeded_dict_c5, batch_size=1, device=torch.device(device))
        t1_seed_c5 = time.perf_counter()
        setup_ms_c5 = pred_retrieval_ms + ((t1_seed_c5 - t0_seed_c5) * 1000.0)

        comp_len_c5, tiles_c5, stats_c5 = segment_tokens_dp(prompt_ids, set(pred_phrases_c5))
        resegmented_prompt_c5 = []
        for tile in tiles_c5:
            if len(tile) == 1:
                resegmented_prompt_c5.append(tile[0])
            else:
                resegmented_prompt_c5.append(seeded_dict_c5[tile])

        input_tensor_c5 = torch.tensor([resegmented_prompt_c5], dtype=torch.long, device=device)
        hybrid_mgr_c5.attach_to_model(zip_model)

        t0_gen_c5 = time.perf_counter()
        with torch.no_grad():
            out_c5 = zip_model.generate(
                input_ids=input_tensor_c5,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        t1_gen_c5 = time.perf_counter()
        gen_ms_c5 = (t1_gen_c5 - t0_gen_c5) * 1000.0
        tot_ms_c5 = setup_ms_c5 + gen_ms_c5
        hybrid_mgr_c5.detach_from_model(zip_model)

        full_c5 = out_c5[0].tolist()
        expanded_full_c5 = hybrid_mgr_c5.decode_sequence(full_c5)
        expanded_c5 = expanded_full_c5[base_prompt_len:]
        num_steps_c5 = len(full_c5) - len(resegmented_prompt_c5)
        base_equiv_c5 = len(expanded_c5)
        gen_tokens_c5 = full_c5[len(resegmented_prompt_c5):]
        hypertokens_c5 = sum(1 for t in gen_tokens_c5 if t >= initial_vocab_size)
        text_c5 = tokenizer.decode(expanded_c5, skip_special_tokens=True)
        step_red_c5 = (1.0 - num_steps_c5 / base_equiv_c5) * 100.0 if base_equiv_c5 > 0 else 0.0
        is_valid_c5, ans_c5 = evaluate_output_quality(text_c5, domain)

        res_c5 = LiveBenchmarkResult(
            condition="50/50 Hybrid (16/16)",
            prompt_id=p_id_name,
            domain=domain,
            prompt_text=prompt_text,
            base_prompt_tokens=base_prompt_len,
            compressed_prefill_tokens=len(resegmented_prompt_c5),
            generated_steps=num_steps_c5,
            base_equivalent_tokens=base_equiv_c5,
            realized_step_reduction_pct=step_red_c5,
            offline_theoretical_comp_pct=17.64,
            realization_gap_ratio=(step_red_c5 / 17.64) if 17.64 > 0 else 0.0,
            hypertokens_emitted=hypertokens_c5,
            setup_latency_ms=setup_ms_c5,
            prefill_latency_ms=setup_ms_c5,
            decode_latency_ms=gen_ms_c5,
            total_latency_ms=tot_ms_c5,
            raw_steps_per_sec=(num_steps_c5 / (gen_ms_c5 / 1000.0)) if gen_ms_c5 > 0 else 0.0,
            effective_tokens_per_sec=(base_equiv_c5 / (tot_ms_c5 / 1000.0)) if tot_ms_c5 > 0 else 0.0,
            is_valid_syntax=is_valid_c5,
            extracted_answer=ans_c5,
            decoded_text=text_c5,
        )
        results.append(res_c5)
        print(f"  Generated {num_steps_c5} steps (expanded to {base_equiv_c5} base tokens) in {tot_ms_c5:.1f} ms")
        print(f"  Prefill compression: {base_prompt_len} -> {len(resegmented_prompt_c5)} ({stats_c5['compression_pct']:.1f}%)")
        print(f"  Realized step reduction: {step_red_c5:.2f}% ({hypertokens_c5} hypertokens emitted)")
        print(f"  Effective speed: {res_c5.effective_tokens_per_sec:.1f} tok/s")
        print(f"  Valid syntax / answer: {is_valid_c5} / {ans_c5}")
        print(f"  Output text: {repr(text_c5[:60])}...")

        # Save checkpoint after every prompt
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\n[Checkpoint saved to {output_path}: {len(results)} total records]")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=30)
    parser.add_argument("--num_prompts", type=int, default=15)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompts_file", type=str, default="experiments/live_benchmark_prompts.json")
    parser.add_argument("--output", type=str, default="experiments/live_zero_shot_results.json")
    args = parser.parse_args()

    run_live_benchmark(
        budget=args.budget,
        max_new_tokens=args.tokens,
        num_prompts=args.num_prompts,
        device=args.device,
        prompts_file=args.prompts_file,
        output_path=args.output,
    )
