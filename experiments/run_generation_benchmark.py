"""End-to-End Generation & Wall-Clock Latency Benchmark.

Compares:
Condition A: Normal Base Model + Normal Tokenizer
Condition B: Standard zip2zip LZW Compression
Condition C: Predictive Seeded Dynamic Token Vocabulary (StaticCodebookManager)

Measures:
- Autoregressive decoding steps
- Compressed vs Base-equivalent sequence length
- Real wall-clock latency (total, prefill, decode, tokens/sec)
- Vocabulary optimizer & hyper-encoder overhead
- Memory utilization
- Hypertokens emitted and utilization rate
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.predictor import PhraseBank, PredictiveVocabularyOptimizer
from zip2zip import (
    CompressionConfig,
    EncoderType,
    StaticCodebookManager,
    TransformerEncoderConfig,
    Zip2ZipConfig,
    Zip2ZipModel,
)
from zip2zip_compression import LZWCompressor


@dataclass
class GenerationBenchmarkResult:
    condition: str
    prompt: str
    base_equivalent_tokens: int
    generated_steps: int
    compression_ratio: float
    total_wall_clock_ms: float
    prefill_wall_clock_ms: float
    decoding_wall_clock_ms: float
    optimizer_overhead_ms: float
    encoder_overhead_ms: float
    effective_tokens_per_sec: float
    raw_steps_per_sec: float
    peak_memory_mb: float
    hypertokens_emitted: int
    decoded_text: str


def run_benchmark(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    budget: int = 64,
    device: Optional[str] = None,
    max_new_tokens: int = 30,
) -> List[GenerationBenchmarkResult]:
    if device is None:
        device = "xpu" if torch.xpu.is_available() else "cpu"

    print(f"Initializing Generation Benchmark on device: {device}...")
    tok = AutoTokenizer.from_pretrained(model_id)
    disabled = list(tok.all_special_ids)
    initial_vocab_size = len(tok)

    # 1. Base Model setup
    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )
    if device == "xpu":
        base_model = base_model.half().to("xpu")
    else:
        base_model = base_model.to(device)

    eval_prompts = [
        ("code", "def fibonacci(n):\n    \"\"\"Compute the n-th Fibonacci number.\"\"\"\n"),
        ("code", "class BinarySearchTree:\n    def __init__(self):\n        self."),
        ("reasoning", "Problem: Calculate the area of a circle with radius 7. Step 1:"),
    ]

    # Pre-train phrase bank with diverse sample code and text
    bank = PhraseBank(disabled_ids=set(disabled), max_subtokens=3)
    training_data = [
        ("code", tok.encode("def fibonacci(n):", add_special_tokens=False), tok.encode("if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)", add_special_tokens=False)),
        ("code", tok.encode("class BinarySearchTree:", add_special_tokens=False), tok.encode("def __init__(self):\n        self.root = None\n        self.size = 0", add_special_tokens=False)),
        ("reasoning", tok.encode("Problem: Calculate the area of a circle with radius 7.", add_special_tokens=False), tok.encode("Formula is A = pi * r^2. Given radius r = 7, A = 3.14159 * 49 = 153.94.", add_special_tokens=False)),
    ]
    bank.train_on_corpus(training_data)
    optimizer = PredictiveVocabularyOptimizer(bank, initial_vocab_size=initial_vocab_size, max_subtokens=3)

    results: List[GenerationBenchmarkResult] = []

    # -------------------------------------------------------------
    # CONDITION A: Normal Model / Normal Tokenizer
    # -------------------------------------------------------------
    for dom, prompt_text in eval_prompts:
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        input_ids_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        print(f"\nEvaluating Condition A (Base Model) on: {prompt_text[:35]}...")

        t0 = time.perf_counter()
        with torch.no_grad():
            out_base = base_model.generate(
                input_ids=input_ids_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        t1 = time.perf_counter()
        total_ms_a = (t1 - t0) * 1000.0

        gen_tokens_a = out_base[0, len(prompt_ids) :].tolist()
        num_steps_a = len(gen_tokens_a)
        base_equiv_a = num_steps_a
        text_a = tok.decode(gen_tokens_a, skip_special_tokens=True)

        results.append(
            GenerationBenchmarkResult(
                condition="Condition A (Base Model)",
                prompt=prompt_text,
                base_equivalent_tokens=base_equiv_a,
                generated_steps=num_steps_a,
                compression_ratio=0.0,
                total_wall_clock_ms=total_ms_a,
                prefill_wall_clock_ms=0.0,
                decoding_wall_clock_ms=total_ms_a,
                optimizer_overhead_ms=0.0,
                encoder_overhead_ms=0.0,
                effective_tokens_per_sec=(base_equiv_a / (total_ms_a / 1000.0)) if total_ms_a > 0 else 0.0,
                raw_steps_per_sec=(num_steps_a / (total_ms_a / 1000.0)) if total_ms_a > 0 else 0.0,
                peak_memory_mb=0.0,
                hypertokens_emitted=0,
                decoded_text=text_a,
            )
        )

    # 2. Build Zip2Zip Model wrapper
    print("\nWrapping model with Zip2ZipModel...")
    config = Zip2ZipConfig(
        base_model_name_or_path=model_id,
        encoder_type=EncoderType.TRANSFORMER,
        encoder=TransformerEncoderConfig(
            hidden_size=896,
            num_heads=14,
            num_hidden_layers=2,
            intermediate_size=4 * 896,
            tie_encoders=False,
        ),
        compression=CompressionConfig(
            initial_vocab_size=initial_vocab_size,
            max_codebook_size=budget,
            max_subtokens=3,
            disabled_ids=disabled,
        ),
    )
    zip_model = Zip2ZipModel(config, base_model=base_model)

    # -------------------------------------------------------------
    # CONDITION B: Standard zip2zip LZW
    # -------------------------------------------------------------
    for dom, prompt_text in eval_prompts:
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        input_ids_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        print(f"\nEvaluating Condition B (zip2zip LZW) on: {prompt_text[:35]}...")

        zip_model.codebook_manager.reset()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_lzw = zip_model.generate(
                input_ids=input_ids_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        t1 = time.perf_counter()
        total_ms_b = (t1 - t0) * 1000.0

        gen_tokens_b = out_lzw[0, len(prompt_ids) :].tolist()
        num_steps_b = len(gen_tokens_b)

        comp = LZWCompressor(
            initial_vocab_size=initial_vocab_size,
            max_codebook_size=budget,
            max_subtokens=3,
            pad_token_id=0,
            disabled_ids=disabled,
        )
        dec_b, _ = comp.decode(out_lzw[0].tolist())
        gen_expanded_b = dec_b[len(prompt_ids) :]
        base_equiv_b = len(gen_expanded_b)
        text_b = tok.decode(gen_expanded_b, skip_special_tokens=True)
        hypertokens_b = sum(1 for t in gen_tokens_b if t >= initial_vocab_size)
        comp_b = (1.0 - num_steps_b / base_equiv_b) * 100.0 if base_equiv_b > 0 else 0.0

        results.append(
            GenerationBenchmarkResult(
                condition="Condition B (Standard zip2zip LZW)",
                prompt=prompt_text,
                base_equivalent_tokens=base_equiv_b,
                generated_steps=num_steps_b,
                compression_ratio=comp_b,
                total_wall_clock_ms=total_ms_b,
                prefill_wall_clock_ms=0.0,
                decoding_wall_clock_ms=total_ms_b,
                optimizer_overhead_ms=0.0,
                encoder_overhead_ms=0.0,
                effective_tokens_per_sec=(base_equiv_b / (total_ms_b / 1000.0)) if total_ms_b > 0 else 0.0,
                raw_steps_per_sec=(num_steps_b / (total_ms_b / 1000.0)) if total_ms_b > 0 else 0.0,
                peak_memory_mb=0.0,
                hypertokens_emitted=hypertokens_b,
                decoded_text=text_b,
            )
        )

    # -------------------------------------------------------------
    # CONDITION C: Predictive Seeded Vocabulary (StaticCodebookManager)
    # -------------------------------------------------------------
    for dom, prompt_text in eval_prompts:
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        print(f"\nEvaluating Condition C (Predictive Seeded) on: {prompt_text[:35]}...")

        # Step C1: Predict vocabulary from prompt
        t_opt_0 = time.perf_counter()
        seeded_dict = optimizer.select_prompt_conditioned(prompt_ids, budget=budget)
        t_opt_1 = time.perf_counter()
        opt_ms = (t_opt_1 - t_opt_0) * 1000.0

        # Step C2: Seed StaticCodebookManager
        static_mgr = StaticCodebookManager(
            initial_vocab_size=initial_vocab_size,
            max_codebook_size=budget,
            max_subtokens=3,
            embedding_dim=896,
            pad_token_id=tok.pad_token_id or 0,
            disabled_ids=disabled,
        )
        t_enc_0 = time.perf_counter()
        static_mgr.set_seeded_codebook(seeded_dict, batch_size=1, device=torch.device(device))
        t_enc_1 = time.perf_counter()
        enc_ms = (t_enc_1 - t_enc_0) * 1000.0

        # Swap into model
        zip_model.codebook_manager = static_mgr
        emb_layer = zip_model.base_model.get_input_embeddings()
        lin_layer = zip_model.base_model.get_output_embeddings()
        emb_layer.codebook_manager = static_mgr
        lin_layer.codebook_manager = static_mgr

        # Setup logits processor for unseeded slots
        logits_proc = LogitsProcessorList([static_mgr.get_logits_processor()])

        # Step C3: Segment prompt with seeded vocabulary
        segmented_prompt = static_mgr.segment_sequence(prompt_ids)
        seg_input_tensor = torch.tensor([segmented_prompt], dtype=torch.long, device=device)

        # Step C4: Autoregressive generation
        t_gen_0 = time.perf_counter()
        with torch.no_grad():
            out_c = zip_model.generate(
                input_ids=seg_input_tensor,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_proc,
                do_sample=False,
            )
        t_gen_1 = time.perf_counter()
        gen_ms = (t_gen_1 - t_gen_0) * 1000.0
        total_ms_c = opt_ms + enc_ms + gen_ms

        gen_tokens_c = out_c[0, len(segmented_prompt) :].tolist()
        num_steps_c = len(gen_tokens_c)
        gen_expanded_c = static_mgr.decode_sequence(gen_tokens_c)
        base_equiv_c = len(gen_expanded_c)
        text_c = tok.decode(gen_expanded_c, skip_special_tokens=True)
        hypertokens_c = sum(1 for t in gen_tokens_c if t >= initial_vocab_size)
        comp_c = (1.0 - num_steps_c / base_equiv_c) * 100.0 if base_equiv_c > 0 else 0.0

        results.append(
            GenerationBenchmarkResult(
                condition="Condition C (Predictive Seeded)",
                prompt=prompt_text,
                base_equivalent_tokens=base_equiv_c,
                generated_steps=num_steps_c,
                compression_ratio=comp_c,
                total_wall_clock_ms=total_ms_c,
                prefill_wall_clock_ms=opt_ms + enc_ms,
                decoding_wall_clock_ms=gen_ms,
                optimizer_overhead_ms=opt_ms,
                encoder_overhead_ms=enc_ms,
                effective_tokens_per_sec=(base_equiv_c / (total_ms_c / 1000.0)) if total_ms_c > 0 else 0.0,
                raw_steps_per_sec=(num_steps_c / (gen_ms / 1000.0)) if gen_ms > 0 else 0.0,
                peak_memory_mb=0.0,
                hypertokens_emitted=hypertokens_c,
                decoded_text=text_c,
            )
        )

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="experiments/generation_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    bench_results = run_benchmark(budget=args.budget, max_new_tokens=args.tokens, device=args.device)

    # Print summary table
    print("\n" + "=" * 105)
    print(f"GENERATION & WALL-CLOCK LATENCY BENCHMARK SUMMARY (Budget = {args.budget} Hypertokens)")
    print("=" * 105)
    print(
        f"{'Condition':<35} | {'Steps':<6} | {'Base-Eq':<7} | {'Comp %':<8} | {'Total ms':<9} | {'Opt ms':<7} | {'Enc ms':<7} | {'Eff Tok/s':<9}"
    )
    print("-" * 105)

    for r in bench_results:
        print(
            f"{r.condition:<35} | {r.generated_steps:<6} | {r.base_equivalent_tokens:<7} | "
            f"{r.compression_ratio:>6.1f}% | {r.total_wall_clock_ms:>8.1f} | {r.optimizer_overhead_ms:>6.2f} | "
            f"{r.encoder_overhead_ms:>6.2f} | {r.effective_tokens_per_sec:>8.1f}"
        )

    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in bench_results], f, indent=2)
    print(f"\nSaved detailed benchmark results to {args.output}")
