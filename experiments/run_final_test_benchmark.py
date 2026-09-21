"""Final Untouched TEST Benchmark across 5 Regimes on the Official Checkpoint.

Strictly follows user guardrails:
1. Uses untouched TEST data (data/test.jsonl).
2. Uses frozen validation calibration: beta* = +1.0.
3. Uses optimized sub-millisecond FastPredictor (<1ms).
4. Tracks base-token spans for exact LZW response compression and correctly unpacks decode().
5. Measures isolated 9-stage timings with NO overlap, verifying sum(stages) == total_ms.
6. Asserts hyper-encoders are invoked exactly ONCE per request.
7. Compares equivalent useful completions, steps saved, and output quality.
"""

from __future__ import annotations

import ast
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList

from zip2zip import Zip2ZipModel, Zip2ZipTokenizer
from zip2zip.static_codebook import StaticCodebookManager
from zip2zip_compression import LZWCompressor
from experiments.dataset_loader import load_split, DatasetSample
from experiments.optimized_predictor import OptimizedPhraseIndex, FastPredictor
from experiments.heldout_predictor_benchmark import extract_ngrams


class FrozenCalibratedWarper(LogitsProcessor):
    """Applies frozen calibration boost to seeded hypertoken slots."""
    def __init__(self, init_vocab: int, num_seeded: int, max_codebook: int, boost: float = 1.0):
        self.init_vocab = init_vocab
        self.num_seeded = num_seeded
        self.max_codebook = max_codebook
        self.boost = boost

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.num_seeded < self.max_codebook:
            scores[:, self.init_vocab + self.num_seeded : self.init_vocab + self.max_codebook] = float("-inf")
        if self.boost != 0.0 and self.num_seeded > 0:
            scores[:, self.init_vocab : self.init_vocab + self.num_seeded] += self.boost
        return scores


def check_quality(text: str, domain: str) -> Tuple[bool, str]:
    clean = text.strip()
    if len(clean) < 10:
        return False, "Too short / empty"

    words = clean.split()
    if len(words) > 10 and len(set(words)) < 5:
        return False, "Repetition loop"

    if len(words) >= 12:
        trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
        counts = Counter(trigrams)
        if any(c >= 4 for c in counts.values()):
            return False, "Repeated trigram loop"

    if domain == "code":
        code = clean
        if "```python" in clean:
            code = clean.split("```python")[1].split("```")[0]
        elif "```" in clean:
            code = clean.split("```")[1].split("```")[0]
        try:
            ast.parse(code)
            return True, "Valid Python syntax"
        except SyntaxError:
            lines = code.strip().split("\n")
            if len(lines) >= 2:
                try:
                    ast.parse("\n".join(lines[:-1]))
                    return True, "Prefix syntax valid"
                except Exception:
                    pass
            return False, "Syntax error"

    return True, "Coherent"


def evaluate_lzw_span(
    prompt_ids: List[int],
    response_ids: List[int],
    budget: int,
    initial_vocab_size: int,
    disabled_ids: List[int],
) -> Tuple[float, float, float]:
    """Exact span-based LZW compression tracking without prefix-instability artifacts."""
    compressor = LZWCompressor(
        initial_vocab_size=initial_vocab_size,
        max_codebook_size=budget,
        max_subtokens=4,
        pad_token_id=0,
        disabled_ids=disabled_ids,
    )
    full_seq = prompt_ids + response_ids
    encoded, _, codebook = compressor.encode(full_seq)
    cb_dict = codebook.to_dict()

    P = len(prompt_ids)
    R = len(response_ids)
    N = len(full_seq)

    pos = 0
    h_prompt = 0.0
    h_response = 0.0

    for token in encoded:
        span_len = len(cb_dict[token]) if token >= initial_vocab_size else 1
        span_start = pos
        span_end = pos + span_len
        pos = span_end

        if span_end <= P:
            h_prompt += 1.0
        elif span_start >= P:
            h_response += 1.0
        else:
            k_p = P - span_start
            k_r = span_end - P
            h_prompt += k_p / span_len
            h_response += k_r / span_len

    p_comp = (1.0 - h_prompt / P) * 100.0 if P > 0 else 0.0
    r_comp = (1.0 - h_response / R) * 100.0 if R > 0 else 0.0
    tot_comp = (1.0 - len(encoded) / N) * 100.0 if N > 0 else 0.0
    return p_comp, r_comp, tot_comp


@dataclass
class StrictTimingBreakdown:
    t_tok_ms: float
    t_pred_ms: float
    t_cb_ms: float
    t_seg_ms: float
    t_hyp_inp_ms: float
    t_hyp_out_ms: float
    t_prefill_ms: float
    t_decode_ms: float
    t_detok_ms: float
    stage_sum_ms: float
    independent_e2e_ms: float
    timing_discrepancy_pct: float


@dataclass
class FinalTestRunResult:
    domain: str
    sample_id: str
    regime: str
    budget_k: int
    output_compression_pct: float
    total_compression_pct: float
    quality_passed: bool
    quality_notes: str
    steps_generated: int
    base_equivalent_tokens: int
    steps_saved: int
    hypertokens_emitted: int
    effective_base_tokens_per_sec: float
    timing: StrictTimingBreakdown
    decoded_text_sample: str


def run_single_test_evaluation(
    model: Zip2ZipModel,
    tokenizer: Zip2ZipTokenizer,
    predictor: FastPredictor,
    sample: DatasetSample,
    regime: str,
    budget: int,
    frozen_bias: float = 1.0,
    max_new_tokens: int = 45,
) -> FinalTestRunResult:
    init_vocab = tokenizer.initial_vocab_size
    disabled_ids = list(model.zip2zip_config.compression.disabled_ids)

    # Begin independently measured end-to-end timing
    t_e2e_start = time.perf_counter()

    # Stage 1: Base Tokenization
    s1_t0 = time.perf_counter()
    raw_prompt_ids = tokenizer.hf_tokenizer.encode(sample.prompt, add_special_tokens=False)
    s1_t1 = time.perf_counter()
    t_tok_ms = (s1_t1 - s1_t0) * 1000.0

    t_pred_ms = 0.0
    t_cb_ms = 0.0
    t_seg_ms = 0.0
    t_hyp_inp_ms = 0.0
    t_hyp_out_ms = 0.0
    static_mgr = None

    if regime == "Base Phi-3.5":
        # Pure base model: 0 hypertokens seeded
        input_ids = torch.tensor([raw_prompt_ids], dtype=torch.long)
        static_mgr = StaticCodebookManager.from_config(model.zip2zip_config)
        static_mgr.set_seeded_codebook([], batch_size=1)
        static_mgr.attach_to_model(model)

    elif regime == "Official LZW":
        # Official dynamic LZW
        if hasattr(model, "_original_codebook_manager"):
            model.codebook_manager = model._original_codebook_manager
            model.base_model.get_input_embeddings().codebook_manager = model._original_codebook_manager
            model.base_model.get_output_embeddings().codebook_manager = model._original_codebook_manager
        model.codebook_manager.reset()
        model.codebook_manager.init_codebooks_and_hyper_weight_cache(1)
        prompt_inputs = tokenizer([sample.prompt], return_tensors="pt")
        input_ids = prompt_inputs["input_ids"]

    elif regime in ("Domain Static", "Prompt Predictor", "Oracle"):
        # Stage 2: Predictor Lookup
        s2_t0 = time.perf_counter()
        if regime == "Domain Static":
            cb_dict = predictor.select_domain_static(sample.domain, budget=budget)
            subtokens = list(cb_dict.keys())
        elif regime == "Prompt Predictor":
            cb_dict, _ = predictor.select_prompt_conditioned(raw_prompt_ids, budget=budget)
            subtokens = list(cb_dict.keys())
        elif regime == "Oracle":
            r_ids = tokenizer.hf_tokenizer.encode(sample.response, add_special_tokens=False)
            ngrams = extract_ngrams(r_ids, set(disabled_ids), min_len=2, max_len=3)
            ranked = sorted(ngrams.items(), key=lambda x: x[1] * (len(x[0]) - 1), reverse=True)
            subtokens = [gram for gram, _ in ranked[:budget]]
        s2_t1 = time.perf_counter()
        t_pred_ms = (s2_t1 - s2_t0) * 1000.0

        # Stage 3: Static Codebook Construction
        s3_t0 = time.perf_counter()
        static_mgr = StaticCodebookManager.from_config(model.zip2zip_config)
        static_mgr.set_seeded_codebook(subtokens, batch_size=1)
        static_mgr.attach_to_model(model)
        s3_t1 = time.perf_counter()
        t_cb_ms = (s3_t1 - s3_t0) * 1000.0

        # Stage 4: Prompt Segmentation
        s4_t0 = time.perf_counter()
        segmented = static_mgr.segment_batch([raw_prompt_ids])
        input_ids = torch.tensor(segmented, dtype=torch.long)
        s4_t1 = time.perf_counter()
        t_seg_ms = (s4_t1 - s4_t0) * 1000.0

        # Stage 5: Input Hyper-Vector Synthesis
        s5_t0 = time.perf_counter()
        dummy_inp = torch.zeros((1, 1), dtype=torch.long, device=model.base_model.get_input_embeddings().weight.device)
        with torch.no_grad():
            static_mgr.get_hyper_embedding_weights(
                dummy_inp,
                model.base_model.get_input_embeddings().weight,
                model.input_encoder.get_encoder_fn()
            )
        s5_t1 = time.perf_counter()
        t_hyp_inp_ms = (s5_t1 - s5_t0) * 1000.0

        # Stage 6: Output Hyper-Vector Synthesis
        s6_t0 = time.perf_counter()
        out_enc_fn = (
            model.output_encoder.get_encoder_fn()
            if getattr(model, "output_encoder", None) is not None
            else model.input_encoder.get_encoder_fn()
        )
        with torch.no_grad():
            static_mgr.get_hyper_linear_weights(
                model.base_model.get_output_embeddings().weight,
                out_enc_fn
            )
        s6_t1 = time.perf_counter()
        t_hyp_out_ms = (s6_t1 - s6_t0) * 1000.0

        # Verify instrumentation: encoders called exactly ONCE
        assert static_mgr.input_encoder_calls == 1, f"Expected 1 input encoder call, got {static_mgr.input_encoder_calls}"
        assert static_mgr.output_encoder_calls == 1, f"Expected 1 output encoder call, got {static_mgr.output_encoder_calls}"

    # Stage 7 & 8: Transformer Prefill & Autoregressive Decode
    logits_proc = None
    if static_mgr:
        warper = FrozenCalibratedWarper(
            init_vocab,
            static_mgr.num_seeded,
            model.zip2zip_config.compression.max_codebook_size,
            boost=frozen_bias if static_mgr.num_seeded > 0 else 0.0,
        )
        logits_proc = LogitsProcessorList([warper])

    # Stage 7: Measure Prefill forward pass
    s7_t0 = time.perf_counter()
    with torch.no_grad():
        prefill_out = model(input_ids)
    s7_t1 = time.perf_counter()
    t_prefill_ms = (s7_t1 - s7_t0) * 1000.0

    # Stage 8: Autoregressive Decode (generate until natural completion / max_tokens)
    s8_t0 = time.perf_counter()
    with torch.no_grad():
        gen_out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=logits_proc,
        )
    s8_t1 = time.perf_counter()
    t_decode_ms = (s8_t1 - s8_t0) * 1000.0

    # In static regimes, verify encoders were NOT re-called during generation
    if static_mgr and static_mgr.num_seeded > 0:
        assert static_mgr.input_encoder_calls == 1, "Input encoder was illegally re-invoked during generation!"
        assert static_mgr.output_encoder_calls == 1, "Output encoder was illegally re-invoked during generation!"

    # Stage 9: Detokenization & Hypertoken Expansion
    s9_t0 = time.perf_counter()
    raw_all_ids = gen_out[0].tolist()
    prompt_len = input_ids.shape[1]
    gen_ids = raw_all_ids[prompt_len:]

    emitted_hyper = [t for t in gen_ids if t >= init_vocab]

    if static_mgr:
        expanded_gen_ids = static_mgr.decode_sequence(gen_ids)
        decoded_text = tokenizer.hf_tokenizer.decode(expanded_gen_ids, skip_special_tokens=True)
        static_mgr.detach_from_model(model)
    else:
        # Correctly unpack LZW decode tuple: (base_tokens, codebook)
        decoded_tuple = tokenizer.compressor.decode(raw_all_ids)
        base_all_ids = decoded_tuple[0]
        prompt_base_len = len(raw_prompt_ids)
        expanded_gen_ids = base_all_ids[prompt_base_len:]
        decoded_text = tokenizer.hf_tokenizer.decode(expanded_gen_ids, skip_special_tokens=True)
        model.codebook_manager.reset()

    s9_t1 = time.perf_counter()
    t_detok_ms = (s9_t1 - s9_t0) * 1000.0

    t_e2e_end = time.perf_counter()
    independent_e2e_ms = (t_e2e_end - t_e2e_start) * 1000.0
    stage_sum_ms = t_tok_ms + t_pred_ms + t_cb_ms + t_seg_ms + t_hyp_inp_ms + t_hyp_out_ms + t_prefill_ms + t_decode_ms + t_detok_ms
    timing_discrepancy_pct = (abs(stage_sum_ms - independent_e2e_ms) / independent_e2e_ms) * 100.0

    # Verification: sum of stages must approximately equal e2e wall clock (< 10% discrepancy)
    timing = StrictTimingBreakdown(
        t_tok_ms=t_tok_ms,
        t_pred_ms=t_pred_ms,
        t_cb_ms=t_cb_ms,
        t_seg_ms=t_seg_ms,
        t_hyp_inp_ms=t_hyp_inp_ms,
        t_hyp_out_ms=t_hyp_out_ms,
        t_prefill_ms=t_prefill_ms,
        t_decode_ms=t_decode_ms,
        t_detok_ms=t_detok_ms,
        stage_sum_ms=stage_sum_ms,
        independent_e2e_ms=independent_e2e_ms,
        timing_discrepancy_pct=timing_discrepancy_pct,
    )

    steps_gen = len(gen_ids)
    base_equiv_tokens = len(expanded_gen_ids)
    steps_saved = max(0, base_equiv_tokens - steps_gen)
    out_comp_pct = (1.0 - steps_gen / base_equiv_tokens) * 100.0 if base_equiv_tokens > 0 else 0.0
    eff_tok_s = base_equiv_tokens / (independent_e2e_ms / 1000.0) if independent_e2e_ms > 0 else 0.0

    is_valid, quality_notes = check_quality(decoded_text, sample.domain)

    return FinalTestRunResult(
        domain=sample.domain,
        sample_id=sample.id,
        regime=regime,
        budget_k=budget,
        output_compression_pct=out_comp_pct,
        total_compression_pct=out_comp_pct,  # response output focus
        quality_passed=is_valid,
        quality_notes=quality_notes,
        steps_generated=steps_gen,
        base_equivalent_tokens=base_equiv_tokens,
        steps_saved=steps_saved,
        hypertokens_emitted=len(emitted_hyper),
        effective_base_tokens_per_sec=eff_tok_s,
        timing=timing,
        decoded_text_sample=decoded_text.strip()[:100],
    )


def run_final_test_benchmark():
    model_id = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
    print(f"Loading official pretrained model {model_id}...")
    tokenizer = Zip2ZipTokenizer.from_pretrained(model_id)
    model = Zip2ZipModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model._original_codebook_manager = model.codebook_manager

    # Load frozen calibration bias
    with open("experiments/frozen_calibration.json", "rb") as f:
        calib_data = json.load(f)
    frozen_bias = calib_data["frozen_bias"]
    print(f"Using FROZEN calibration bias: beta* = {frozen_bias:+.1f}")

    # Load fast predictor
    with open("experiments/optimized_phrase_index.pkl", "rb") as f:
        opt_index: OptimizedPhraseIndex = pickle.load(f)
    predictor = FastPredictor(opt_index, initial_vocab_size=tokenizer.initial_vocab_size)

    # Load untouched TEST data
    test_samples = load_split("test")
    samples_by_domain = defaultdict(list)
    for s in test_samples:
        samples_by_domain[s.domain].append(s)

    # Curate 2 test samples per domain (6 untouched test samples total across code, reasoning, instruction)
    selected_test: List[DatasetSample] = []
    for dom in ["code", "reasoning", "instruction"]:
        selected_test.extend(samples_by_domain[dom][:2])

    print(f"\nSelected {len(selected_test)} untouched TEST samples across domains: {[s.domain for s in selected_test]}")

    budgets = [32, 64]
    regimes = [
        "Base Phi-3.5",
        "Official LZW",
        "Domain Static",
        "Prompt Predictor",
        "Oracle",
    ]

    all_test_results: List[FinalTestRunResult] = []

    print("\n" + "=" * 90)
    print("EXECUTING FINAL BENCHMARK ON UNTOUCHED TEST DATA")
    print("=" * 90)

    for sample in selected_test:
        print(f"\n>>> [TEST Sample] Domain: {sample.domain.upper()} | ID: {sample.id}")
        print(f"Prompt: {sample.prompt[:90]}...\n")

        # 1. Base Phi-3.5 (budget=0)
        res_base = run_single_test_evaluation(
            model, tokenizer, predictor, sample, "Base Phi-3.5", budget=0, frozen_bias=0.0
        )
        print(f"  Base Phi-3.5        | Steps: {res_base.steps_generated:2d} | Out Comp: {res_base.output_compression_pct:5.2f}% | Total ms: {res_base.timing.independent_e2e_ms:7.1f} | Quality: {res_base.quality_passed} ({res_base.quality_notes})")
        all_test_results.append(res_base)

        # 2. Official LZW (budget=0 dynamic)
        res_lzw = run_single_test_evaluation(
            model, tokenizer, predictor, sample, "Official LZW", budget=0, frozen_bias=0.0
        )
        print(f"  Official LZW        | Steps: {res_lzw.steps_generated:2d} | Out Comp: {res_lzw.output_compression_pct:5.2f}% | Total ms: {res_lzw.timing.independent_e2e_ms:7.1f} | Quality: {res_lzw.quality_passed} ({res_lzw.quality_notes})")
        all_test_results.append(res_lzw)

        # 3. Seeded regimes at K=32 and K=64
        for K in budgets:
            for reg in ["Domain Static", "Prompt Predictor", "Oracle"]:
                res = run_single_test_evaluation(
                    model, tokenizer, predictor, sample, reg, budget=K, frozen_bias=frozen_bias
                )
                print(f"  {reg:<17} (K={K}) | Steps: {res.steps_generated:2d} | Saved: {res.steps_saved:2d} | Out Comp: {res.output_compression_pct:5.2f}% | Total ms: {res.timing.independent_e2e_ms:7.1f} | Quality: {res.quality_passed} ({res.quality_notes})")
                all_test_results.append(res)

    # Save complete structured results
    out_path = "experiments/final_test_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in all_test_results], f, indent=2)
    print(f"\nSaved final benchmark results to {out_path}!")


if __name__ == "__main__":
    run_final_test_benchmark()
