"""Rigorous Checkpoint Generation Study across 6 Regimes.

Evaluates epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1 on held-out test samples.
Compares:
1. Base Phi-3.5 (all hypertoken logits masked to -inf)
2. Official Zip2Zip (Dynamic LZW codebook)
3. Global Static Top-K (pre-seeded static codebook from train.jsonl)
4. Domain Static Top-K (pre-seeded domain-specific static codebook from train.jsonl)
5. Prompt Predictor (pre-seeded prompt-conditioned static codebook from train.jsonl)
6. Oracle Seeded (pre-seeded with target response n-grams)

Measures:
- Total base tokens generated
- Total autoregressive generation steps
- Effective compression ratio: (1 - steps / base_tokens)
- Emitted hypertokens count & identities
- Pipeline stage latencies: tokenization, prediction, hyper-encoder, decode, expansion
- Logit calibration diagnostics (mean/max base vs hyper logits, probability mass)
- Decoded text quality and coherence
"""

import sys
import os
import time
import json
import pickle
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
import torch.nn.functional as F
from transformers import LogitsProcessorList

from zip2zip import Zip2ZipModel, Zip2ZipTokenizer
from zip2zip.static_codebook import StaticCodebookManager, StaticCodebookLogitsWarper
from experiments.dataset_loader import DatasetSample, load_split
from experiments.heldout_predictor_benchmark import StrictHeldOutPhraseBank, StrictPredictor, extract_ngrams


@dataclass
class StageLatencies:
    tokenization_ms: float
    prediction_ms: float
    hyper_encoding_ms: float
    autoregressive_decode_ms: float
    detokenization_ms: float
    total_wall_clock_ms: float
    steps_per_second: float
    effective_tokens_per_second: float


@dataclass
class CalibrationDiagnostic:
    base_logit_max: float
    base_logit_mean: float
    hyper_logit_max: float
    hyper_logit_mean: float
    logit_gap: float  # hyper_max - base_max
    base_prob_mass: float
    hyper_prob_mass: float


@dataclass
class SampleRegimeResult:
    domain: str
    sample_id: str
    prompt: str
    regime: str
    budget_k: int
    generation_steps: int
    base_tokens_generated: int
    compression_ratio_pct: float
    hypertokens_emitted_count: int
    hypertokens_emitted_ids: List[int]
    hypertokens_emitted_phrases: List[str]
    latencies: StageLatencies
    calibration: CalibrationDiagnostic
    decoded_text: str
    is_coherent: bool


def extract_oracle_subtokens(
    response_text: str,
    tokenizer: Zip2ZipTokenizer,
    disabled_ids: Set[int],
    budget: int,
    max_len: int = 3,
) -> List[Tuple[int, ...]]:
    """Extract top frequent n-grams directly from the reference solution."""
    r_ids = tokenizer.hf_tokenizer.encode(response_text, add_special_tokens=False)
    ngrams = extract_ngrams(r_ids, disabled_ids, min_len=2, max_len=max_len)
    ranked = sorted(ngrams.items(), key=lambda x: x[1] * (len(x[0]) - 1), reverse=True)
    return [gram for gram, _ in ranked[:budget]]


def evaluate_regime_on_sample(
    model: Zip2ZipModel,
    tokenizer: Zip2ZipTokenizer,
    predictor: StrictPredictor,
    sample: DatasetSample,
    regime: str,
    budget: int,
    max_new_tokens: int = 70,
) -> SampleRegimeResult:
    """Run end-to-end generation and diagnostics for a single sample under a regime."""
    disabled_ids = set(model.zip2zip_config.compression.disabled_ids)
    init_vocab = tokenizer.initial_vocab_size
    max_codebook = model.zip2zip_config.compression.max_codebook_size

    # 1. Tokenization and Codebook Setup Stage
    t_tok0 = time.perf_counter()
    prompt_ids_raw = tokenizer.hf_tokenizer.encode(sample.prompt, add_special_tokens=False)
    prediction_ms = 0.0
    hyper_encoding_ms = 0.0
    static_manager = None

    if regime == "Official LZW":
        # Standard dynamic LZW: uses LZW compressor for prompt
        if hasattr(model, "_original_codebook_manager"):
            model.codebook_manager = model._original_codebook_manager
            model.base_model.get_input_embeddings().codebook_manager = model._original_codebook_manager
            model.base_model.get_output_embeddings().codebook_manager = model._original_codebook_manager
        model.codebook_manager.reset()
        model.codebook_manager.init_codebooks_and_hyper_weight_cache(1)
        prompt_inputs = tokenizer([sample.prompt], return_tensors="pt")
        input_ids = prompt_inputs["input_ids"]
        t_tok1 = time.perf_counter()
        tokenization_ms = (t_tok1 - t_tok0) * 1000

    elif regime == "Base Phi-3.5":
        # Pure base model: standard HF tokenizer, 0 hypertokens seeded, unseeded slots masked
        prompt_inputs = tokenizer.hf_tokenizer([sample.prompt], return_tensors="pt")
        input_ids = prompt_inputs["input_ids"]
        t_tok1 = time.perf_counter()
        tokenization_ms = (t_tok1 - t_tok0) * 1000

        static_manager = StaticCodebookManager.from_config(model.zip2zip_config)
        static_manager.set_seeded_codebook([], batch_size=1)
        static_manager.attach_to_model(model)

    elif regime in ("Global Static", "Domain Static", "Prompt Predictor", "Oracle"):
        # Select phrases
        t_pred0 = time.perf_counter()
        if regime == "Global Static":
            codebook_dict = predictor.select_global_static(budget)
            subtokens_list = list(codebook_dict.keys())
        elif regime == "Domain Static":
            codebook_dict = predictor.select_domain_static(sample.domain, budget)
            subtokens_list = list(codebook_dict.keys())
        elif regime == "Prompt Predictor":
            codebook_dict, _ = predictor.select_prompt_conditioned(prompt_ids_raw, budget)
            subtokens_list = list(codebook_dict.keys())
        elif regime == "Oracle":
            subtokens_list = extract_oracle_subtokens(sample.response, tokenizer, disabled_ids, budget)
        t_pred1 = time.perf_counter()
        prediction_ms = (t_pred1 - t_pred0) * 1000

        # Install static manager
        static_manager = StaticCodebookManager.from_config(model.zip2zip_config)
        static_manager.set_seeded_codebook(subtokens_list, batch_size=1)
        static_manager.attach_to_model(model)

        # Segment prompt using the seeded codebook (prompt compression)
        prompt_inputs = tokenizer.hf_tokenizer([sample.prompt], return_tensors="pt")
        raw_prompt_ids = prompt_inputs["input_ids"].tolist()
        segmented_prompt_ids = static_manager.segment_batch(raw_prompt_ids)
        input_ids = torch.tensor(segmented_prompt_ids, dtype=torch.long)
        t_tok1 = time.perf_counter()
        tokenization_ms = (t_tok1 - t_tok0) * 1000

        # Precompute hyper-encoder representations
        t_enc0 = time.perf_counter()
        with torch.no_grad():
            static_manager.get_hyper_linear_weights(
                model.base_model.get_output_embeddings().weight,
                model.output_encoder.get_encoder_fn() if model.output_encoder else model.input_encoder.get_encoder_fn()
            )
            static_manager.get_hyper_embedding_weights(
                input_ids,
                model.base_model.get_input_embeddings().weight,
                model.input_encoder.get_encoder_fn()
            )
        t_enc1 = time.perf_counter()
        hyper_encoding_ms = (t_enc1 - t_enc0) * 1000

    # 2. Calibration Diagnostic (Step 0 logits)
    with torch.no_grad():
        fwd_out = model(input_ids)
        step0_logits = fwd_out.logits[0, -1, :].float()
        base_logits = step0_logits[:init_vocab]
        
        b_max = base_logits.max().item()
        b_mean = base_logits.mean().item()

        if static_manager and static_manager.num_seeded > 0:
            h_slice = step0_logits[init_vocab : init_vocab + static_manager.num_seeded]
            h_max = h_slice.max().item()
            h_mean = h_slice.mean().item()
            all_active = torch.cat([base_logits, h_slice])
            probs = F.softmax(all_active, dim=-1)
            b_prob = probs[:init_vocab].sum().item()
            h_prob = probs[init_vocab:].sum().item()
        else:
            h_max = float("-inf")
            h_mean = float("-inf")
            b_prob = 1.0
            h_prob = 0.0

        calib = CalibrationDiagnostic(
            base_logit_max=b_max,
            base_logit_mean=b_mean,
            hyper_logit_max=h_max,
            hyper_logit_mean=h_mean,
            logit_gap=h_max - b_max if h_max != float("-inf") else -999.0,
            base_prob_mass=b_prob,
            hyper_prob_mass=h_prob,
        )

    # 3. Autoregressive Generation Stage
    t_gen0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    t_gen1 = time.perf_counter()
    decode_ms = (t_gen1 - t_gen0) * 1000

    # 4. Detokenization & Expansion Stage
    t_det0 = time.perf_counter()
    raw_all_ids = outputs[0].tolist()
    prompt_len = input_ids.shape[1]
    gen_ids = raw_all_ids[prompt_len:]

    emitted_hyper_ids = [t for t in gen_ids if t >= init_vocab]
    emitted_phrases = []

    if static_manager:
        expanded_gen_ids = static_manager.decode_sequence(gen_ids)
        decoded_text = tokenizer.hf_tokenizer.decode(expanded_gen_ids, skip_special_tokens=True)
        for h_id in emitted_hyper_ids:
            sub = static_manager.decode_hypertoken(h_id)
            emitted_phrases.append(tokenizer.hf_tokenizer.decode(sub))
        static_manager.detach_from_model(model)
    else:
        # LZW decoding
        expanded_full_ids = tokenizer.compressor.decode(raw_all_ids)
        prompt_base_len = len(tokenizer.hf_tokenizer.encode(sample.prompt, add_special_tokens=False))
        expanded_gen_ids = expanded_full_ids[prompt_base_len:]
        decoded_text = tokenizer.hf_tokenizer.decode(expanded_gen_ids, skip_special_tokens=True)
        for h_id in emitted_hyper_ids:
            emitted_phrases.append(f"LZW_ID_{h_id}")
        model.codebook_manager.reset()

    t_det1 = time.perf_counter()
    detok_ms = (t_det1 - t_det0) * 1000

    gen_steps = len(gen_ids)
    base_tokens_gen = len(expanded_gen_ids)
    compression_pct = (1.0 - (gen_steps / base_tokens_gen)) * 100 if base_tokens_gen > 0 else 0.0
    total_ms = tokenization_ms + prediction_ms + hyper_encoding_ms + decode_ms + detok_ms
    steps_per_sec = gen_steps / (decode_ms / 1000.0) if decode_ms > 0 else 0.0
    eff_tokens_per_sec = base_tokens_gen / (total_ms / 1000.0) if total_ms > 0 else 0.0

    latencies = StageLatencies(
        tokenization_ms=tokenization_ms,
        prediction_ms=prediction_ms,
        hyper_encoding_ms=hyper_encoding_ms,
        autoregressive_decode_ms=decode_ms,
        detokenization_ms=detok_ms,
        total_wall_clock_ms=total_ms,
        steps_per_second=steps_per_sec,
        effective_tokens_per_second=eff_tokens_per_sec,
    )

    # Coherence check (basic heuristic: non-empty, contains alpha characters, no infinite repetition)
    is_coherent = len(decoded_text.strip()) > 5 and not (len(set(decoded_text.split())) < 4 and len(decoded_text.split()) > 10)

    return SampleRegimeResult(
        domain=sample.domain,
        sample_id=sample.id,
        prompt=sample.prompt,
        regime=regime,
        budget_k=budget,
        generation_steps=gen_steps,
        base_tokens_generated=base_tokens_gen,
        compression_ratio_pct=compression_pct,
        hypertokens_emitted_count=len(emitted_hyper_ids),
        hypertokens_emitted_ids=emitted_hyper_ids,
        hypertokens_emitted_phrases=emitted_phrases,
        latencies=latencies,
        calibration=calib,
        decoded_text=decoded_text.strip(),
        is_coherent=is_coherent,
    )


def run_full_study():
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

    # Load phrase bank & predictor
    cache_path = "experiments/strict_phrase_bank.pkl"
    print(f"Loading PhraseBank from {cache_path}...")
    with open(cache_path, "rb") as f:
        bank = pickle.load(f)

    predictor = StrictPredictor(
        bank=bank,
        initial_vocab_size=tokenizer.initial_vocab_size,
        max_subtokens=model.zip2zip_config.compression.max_subtokens,
    )

    # Load test samples
    test_samples = load_split("test")
    # Curate 1 sample per domain for full end-to-end multi-regime model evaluation
    selected_samples: List[DatasetSample] = []
    domains_seen = set()
    for s in test_samples:
        if s.domain not in domains_seen:
            selected_samples.append(s)
            domains_seen.add(s.domain)
        if len(domains_seen) == 3:
            break

    print(f"\nSelected {len(selected_samples)} representative held-out test samples across domains: {[s.domain for s in selected_samples]}")

    budgets = [16, 32, 64]
    regimes = [
        "Base Phi-3.5",
        "Official LZW",
        "Global Static",
        "Domain Static",
        "Prompt Predictor",
        "Oracle",
    ]

    all_results: List[Dict] = []

    print("\n" + "=" * 80)
    print("STARTING END-TO-END CHECKPOINT GENERATION BENCHMARK")
    print("=" * 80)

    for sample in selected_samples:
        print(f"\n>>> [Domain: {sample.domain.upper()}] Prompt ID: {sample.id}")
        print(f"Prompt text: {sample.prompt[:120]}...\n")

        # 1. Base Model Baseline (budget=0)
        print("  Evaluating Base Phi-3.5 (No Hypertokens)...")
        res_base = evaluate_regime_on_sample(
            model, tokenizer, predictor, sample, "Base Phi-3.5", budget=0, max_new_tokens=40
        )
        print(f"    Base Steps: {res_base.generation_steps}, Base Tokens: {res_base.base_tokens_generated}, Speed: {res_base.latencies.steps_per_second:.1f} st/s")
        print(f"    Text Preview: {res_base.decoded_text[:100]}...")
        all_results.append(asdict(res_base))

        # 2. Official LZW Baseline (budget=0, dynamic LZW)
        print("  Evaluating Official Zip2Zip (Dynamic LZW)...")
        res_lzw = evaluate_regime_on_sample(
            model, tokenizer, predictor, sample, "Official LZW", budget=0, max_new_tokens=40
        )
        print(f"    LZW Steps: {res_lzw.generation_steps}, Emitted Hypertokens: {res_lzw.hypertokens_emitted_count}, Compression: {res_lzw.compression_ratio_pct:.1f}%")
        print(f"    Text Preview: {res_lzw.decoded_text[:100]}...")
        all_results.append(asdict(res_lzw))

        # 3. Seeded Regimes across Budgets
        for k in budgets:
            for regime in ["Global Static", "Domain Static", "Prompt Predictor", "Oracle"]:
                print(f"  Evaluating [{regime}] (K={k})...")
                res = evaluate_regime_on_sample(
                    model, tokenizer, predictor, sample, regime, budget=k, max_new_tokens=40
                )
                print(f"    Steps: {res.generation_steps}, Emitted: {res.hypertokens_emitted_count} {res.hypertokens_emitted_phrases[:3]}, Comp: {res.compression_ratio_pct:.1f}%")
                print(f"    Calibration: hyper_max={res.calibration.hyper_logit_max:.1f}, gap={res.calibration.logit_gap:.1f}, hyper_prob={res.calibration.hyper_prob_mass*100:.2f}%")
                print(f"    Text Preview: {res.decoded_text[:100]}...")
                all_results.append(asdict(res))

    # Save complete study results
    out_file = "experiments/checkpoint_study_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll benchmark results saved to {out_file}!")


if __name__ == "__main__":
    run_full_study()
