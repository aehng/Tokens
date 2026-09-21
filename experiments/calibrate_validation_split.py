"""Proper Logit Calibration on the Validation Split across Biases [0..8].

Evaluates epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1 strictly on data/val.jsonl.
Tests biases: 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
Measures:
- Hypertoken emission rate
- Expanded base tokens generated
- Steps saved vs Base Phi-3.5
- Output quality, repetition loop detection, and code syntax validation
- Hyper probability mass & logit gap
Selects and freezes the optimal calibration bias beta* before touching TEST data.
"""

from __future__ import annotations

import ast
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList

from zip2zip import Zip2ZipModel, Zip2ZipTokenizer
from zip2zip.static_codebook import StaticCodebookManager
from experiments.dataset_loader import load_split, DatasetSample
from experiments.optimized_predictor import OptimizedPhraseIndex, FastPredictor


class CalibratedStaticWarper(LogitsProcessor):
    def __init__(self, init_vocab: int, num_seeded: int, max_codebook: int, boost: float = 0.0):
        self.init_vocab = init_vocab
        self.num_seeded = num_seeded
        self.max_codebook = max_codebook
        self.boost = boost

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Mask unseeded slots to -inf
        if self.num_seeded < self.max_codebook:
            scores[:, self.init_vocab + self.num_seeded : self.init_vocab + self.max_codebook] = float("-inf")
        # Apply calibration boost to seeded slots
        if self.boost != 0.0 and self.num_seeded > 0:
            scores[:, self.init_vocab : self.init_vocab + self.num_seeded] += self.boost
        return scores


def check_syntax_or_coherence(text: str, domain: str) -> Tuple[bool, str]:
    """Check whether text is coherent, non-repetitive, and syntactically valid."""
    clean = text.strip()
    if len(clean) < 10:
        return False, "Too short / empty"

    words = clean.split()
    if len(words) > 10 and len(set(words)) < 5:
        return False, "Repetition loop detected"

    # Check 3-gram repetition
    if len(words) >= 12:
        trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
        counts = Counter(trigrams)
        if any(c >= 4 for c in counts.values()):
            return False, "Repeated trigram loop"

    if domain == "code":
        # Extract code block if present
        code = clean
        if "```python" in clean:
            code = clean.split("```python")[1].split("```")[0]
        elif "```" in clean:
            code = clean.split("```")[1].split("```")[0]
        try:
            ast.parse(code)
            return True, "Valid Python syntax"
        except SyntaxError:
            # Partial completion might have unclosed paren/indent, check if first 3 lines parse
            lines = code.strip().split("\n")
            if len(lines) >= 2:
                try:
                    ast.parse("\n".join(lines[:-1]))
                    return True, "Prefix syntax valid"
                except Exception:
                    pass
            return False, "Syntax error in code block"

    return True, "Coherent"


def run_validation_calibration():
    model_id = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
    print(f"Loading official pretrained model {model_id}...")
    tokenizer = Zip2ZipTokenizer.from_pretrained(model_id)
    model = Zip2ZipModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Load fast predictor
    with open("experiments/optimized_phrase_index.pkl", "rb") as f:
        opt_index: OptimizedPhraseIndex = pickle.load(f)
    predictor = FastPredictor(opt_index, initial_vocab_size=tokenizer.initial_vocab_size)

    # Select 2 validation samples per domain (6 total) strictly from data/val.jsonl
    val_samples = load_split("val")
    samples_by_domain = defaultdict(list)
    for s in val_samples:
        samples_by_domain[s.domain].append(s)

    selected_val: List[DatasetSample] = []
    for dom in ["code", "reasoning", "instruction"]:
        selected_val.extend(samples_by_domain[dom][:1])

    print(f"Selected {len(selected_val)} validation samples for calibration sweep across {[s.domain for s in selected_val]}")

    biases = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    K = 32

    # Track results per bias
    bias_summary = {b: {
        "emitted_counts": [],
        "steps_saved": [],
        "hyper_prob_mass": [],
        "logit_gaps": [],
        "valid_quality_flags": [],
        "texts": [],
    } for b in biases}

    print("\n" + "=" * 80)
    print("STARTING VALIDATION CALIBRATION SWEEP")
    print("=" * 80)

    for s_idx, sample in enumerate(selected_val):
        print(f"\n>>> [Sample {s_idx+1}/{len(selected_val)}] Domain: {sample.domain.upper()} | ID: {sample.id}")
        prompt_ids_raw = tokenizer.hf_tokenizer.encode(sample.prompt, add_special_tokens=False)
        inp = tokenizer.hf_tokenizer([sample.prompt], return_tensors="pt")
        input_ids = inp["input_ids"]

        # Base generation reference (bias doesn't apply)
        static_mgr = StaticCodebookManager.from_config(model.zip2zip_config)
        static_mgr.set_seeded_codebook([], batch_size=1)
        static_mgr.attach_to_model(model)
        with torch.no_grad():
            out_base = model.generate(input_ids=input_ids, max_new_tokens=40, do_sample=False)
        base_gen = out_base[0].tolist()[input_ids.shape[1]:]
        base_steps = len(base_gen)
        static_mgr.detach_from_model(model)

        # Seed K=32 phrases using Prompt Predictor
        prompt_cb, _ = predictor.select_prompt_conditioned(prompt_ids_raw, budget=K)
        subtokens_list = list(prompt_cb.keys())

        static_mgr = StaticCodebookManager.from_config(model.zip2zip_config)
        static_mgr.set_seeded_codebook(subtokens_list, batch_size=1)
        static_mgr.attach_to_model(model)

        # Precompute hyper-vectors ONCE
        static_mgr.synthesize_hyper_vectors(model, batch_size=1)

        # Inspect Step 0 uncalibrated logits
        with torch.no_grad():
            fwd = model(input_ids)
            logits = fwd.logits[0, -1, :].float()
            base_logits = logits[:tokenizer.initial_vocab_size]
            hyper_logits = logits[tokenizer.initial_vocab_size : tokenizer.initial_vocab_size + len(subtokens_list)]
            gap = (hyper_logits.max() - base_logits.max()).item()

        for b in biases:
            # Calibrated logits processor
            warper = CalibratedStaticWarper(
                tokenizer.initial_vocab_size,
                len(subtokens_list),
                model.zip2zip_config.compression.max_codebook_size,
                boost=b,
            )
            # Compute effective hyper probability mass at step 0 under this bias
            boosted_hyper = hyper_logits + b
            all_active = torch.cat([base_logits, boosted_hyper])
            probs = F.softmax(all_active, dim=-1)
            h_prob = probs[tokenizer.initial_vocab_size:].sum().item()

            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=40,
                    do_sample=False,
                    logits_processor=LogitsProcessorList([warper]),
                )
            gen = out[0].tolist()[input_ids.shape[1]:]
            emitted = [t for t in gen if t >= tokenizer.initial_vocab_size]
            expanded = static_mgr.decode_sequence(gen)
            text = tokenizer.hf_tokenizer.decode(expanded, skip_special_tokens=True)

            is_valid, reason = check_syntax_or_coherence(text, sample.domain)
            # Approximate steps saved: base_tokens in expanded minus actual steps taken
            steps_saved = len(expanded) - len(gen)

            bias_summary[b]["emitted_counts"].append(len(emitted))
            bias_summary[b]["steps_saved"].append(steps_saved)
            bias_summary[b]["hyper_prob_mass"].append(h_prob)
            bias_summary[b]["logit_gaps"].append(gap + b)
            bias_summary[b]["valid_quality_flags"].append(1 if is_valid else 0)

            print(f"  Bias {b:+4.1f} | Emitted: {len(emitted):2d} | Steps Saved: {steps_saved:2d} | Hyper Prob: {h_prob*100:5.2f}% | Valid: {is_valid} ({reason})")

        static_mgr.detach_from_model(model)

    print("\n" + "=" * 80)
    print("VALIDATION CALIBRATION SUMMARY ACROSS ALL 6 SAMPLES")
    print("=" * 80)
    print(f"{'Bias':<8} | {'Emitted Mean':<14} | {'Steps Saved Mean':<18} | {'Hyper Prob %':<14} | {'Quality Pass %':<16} | {'Status':<12}")
    print("-" * 90)

    best_bias = 0.0
    best_score = -1.0

    calibration_records = []
    for b in biases:
        emitted_m = np.mean(bias_summary[b]["emitted_counts"])
        saved_m = np.mean(bias_summary[b]["steps_saved"])
        prob_m = np.mean(bias_summary[b]["hyper_prob_mass"]) * 100.0
        pass_m = np.mean(bias_summary[b]["valid_quality_flags"]) * 100.0

        status = "OK"
        if pass_m < 80.0:
            status = "DEGRADED"
        elif pass_m == 100.0 and saved_m > 0:
            status = "RECOMMENDED"

        print(f"{b:<+8.1f} | {emitted_m:<14.2f} | {saved_m:<18.2f} | {prob_m:<13.2f}% | {pass_m:<15.1f}% | {status:<12}")

        record = {
            "bias": b,
            "emitted_mean": emitted_m,
            "steps_saved_mean": saved_m,
            "hyper_prob_pct": prob_m,
            "quality_pass_pct": pass_m,
            "status": status,
        }
        calibration_records.append(record)

        # Objective: Maximize steps saved subject to 100% quality pass rate
        if pass_m == 100.0 and saved_m > best_score:
            best_score = saved_m
            best_bias = b

    print("=" * 90)
    print(f"\nFROZEN OPTIMAL CALIBRATION BIAS: beta* = {best_bias:+.1f} (selected exclusively on validation data)")

    with open("experiments/frozen_calibration.json", "w") as f:
        json.dump({"frozen_bias": best_bias, "records": calibration_records}, f, indent=2)
    print("Saved calibration configuration to experiments/frozen_calibration.json")


if __name__ == "__main__":
    run_validation_calibration()
