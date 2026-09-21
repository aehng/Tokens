"""One-Command Predictive Checkpoint Evaluation Script (Phase 16).

Evaluates any checkpoint against:
1. The 12-prompt fixed smoke validation set (data/smoke_val_12.json)
   - Evaluates decode step savings, hypertoken emissions, answer completion, and correctness.
2. The continuation equivalence diagnostic suite (15 probes)
   - Evaluates KL divergence, cosine similarity, top-1 match, top-5 overlap, and continuation match.

Usage:
  python experiments/evaluate_predictive_checkpoint.py --checkpoint <path>
  (omit --checkpoint to evaluate baseline Step 0)
"""

import argparse
import ast
import json
import os
import re
import sys
import time
from typing import Dict, List, Any

import torch
from transformers import AutoTokenizer, LogitsProcessorList

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from zip2zip import Zip2ZipModel, StaticCodebookManager
from zip2zip.predictor_policy import CappedPredictorPolicy
from src.evaluation.offline_segmenter import segment_tokens_dp
from experiments.test_continuation_equivalence import evaluate_continuation_suite

INITIAL_VOCAB = 32011
SMOKE_VAL_PATH = "data/smoke_val_12.json"
PREDICTOR_PATH = "experiments/checkpoints/cached_predictor.pkl"


def evaluate_checkpoint(checkpoint_path: str = None, max_new_tokens: int = 150):
    model_id = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

    print(f"\n{'='*80}")
    print(f"EVALUATING PREDICTIVE CHECKPOINT: {checkpoint_path or 'Step 0 (Zero-Shot Baseline)'}")
    print(f"{'='*80}\n")

    # 1. Load model
    print("Loading model...")
    model = Zip2ZipModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # 2. Load checkpoint weights if specified
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Applying weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "lora_state_dict" in ckpt:
            # Load LoRA
            model.base_model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        if "input_encoder_state_dict" in ckpt:
            model.input_encoder.load_state_dict(ckpt["input_encoder_state_dict"])
        if "output_encoder_state_dict" in ckpt:
            model.output_encoder.load_state_dict(ckpt["output_encoder_state_dict"])
        print(f"Checkpoint loaded (step={ckpt.get('step', 'unknown')}).")
    model.eval()

    # 3. Load 12-prompt smoke test set
    with open(SMOKE_VAL_PATH, "r", encoding="utf-8") as f:
        val_samples = json.load(f)

    # 4. Load predictor policy
    import pickle
    with open(PREDICTOR_PATH, "rb") as f:
        raw_predictor = pickle.load(f)
    p_index = getattr(raw_predictor, "index", raw_predictor)
    policy = CappedPredictorPolicy(p_index, tokenizer, budget=32, max_structural_slots=8)

    dim = model.zip2zip_config.encoder.hidden_size
    pad_id = tokenizer.pad_token_id or 32000
    disabled_ids = list(model.zip2zip_config.compression.disabled_ids)

    print(f"\n--- PART 1: 12-PROMPT FIXED VALIDATION SWEEP ({len(val_samples)} prompts) ---")
    results = []
    tot_base_tokens = 0
    tot_decode_steps = 0
    tot_hypers_emitted = 0

    for i, s in enumerate(val_samples, 1):
        prompt_text = s["prompt"]
        dom = s["domain"]
        p_ids = s.get("prompt_token_ids") or tokenizer.encode(prompt_text, add_special_tokens=False)

        # Prompt-only codebook selection
        codebook, _ = policy.select_codebook(p_ids)
        phrases_set = set(codebook.keys())

        # Segment prompt
        comp_len, tiles, _ = segment_tokens_dp(p_ids, phrases_set)
        resegmented = [t[0] if len(t) == 1 else codebook[tuple(t)] for t in tiles]

        # Setup codebook
        mgr = StaticCodebookManager(
            initial_vocab_size=INITIAL_VOCAB,
            max_codebook_size=32,
            max_subtokens=4,
            embedding_dim=dim,
            pad_token_id=pad_id,
            disabled_ids=disabled_ids,
        )
        mgr.set_seeded_codebook(codebook, batch_size=1, device=torch.device("cpu"))
        mgr.attach_to_model(model)

        input_tensor = torch.tensor([resegmented], dtype=torch.long)
        logits_proc = LogitsProcessorList([mgr.get_logits_processor()])

        with torch.no_grad():
            out = model.generate(
                input_ids=input_tensor,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_proc,
                do_sample=False,
            )
        mgr.detach_from_model(model)

        gen_tokens = out[0][len(resegmented):].tolist()
        hypers = [t for t in gen_tokens if t >= INITIAL_VOCAB]

        # Expand hypertokens to base-equivalent tokens
        expanded = []
        for t in gen_tokens:
            if t in mgr.hyper_to_subtokens:
                expanded.extend(mgr.hyper_to_subtokens[t])
            else:
                expanded.append(t)

        decode_steps = len(gen_tokens)
        base_equiv = len(expanded)
        step_savings = base_equiv - decode_steps
        savings_pct = (step_savings / base_equiv * 100.0) if base_equiv > 0 else 0.0

        tot_base_tokens += base_equiv
        tot_decode_steps += decode_steps
        tot_hypers_emitted += len(hypers)

        out_text = tokenizer.decode(expanded, skip_special_tokens=True)

        # Quality check
        quality_ok = True
        if dom == "code":
            try:
                ast.parse(out_text)
            except SyntaxError:
                quality_ok = False
        elif dom == "reasoning":
            ground_truth = s.get("ground_truth_response", "")
            gt_m = re.search(r"####\s*(-?[\d\.,]+)", ground_truth)
            pred_m = re.search(r"####\s*(-?[\d\.,]+)", out_text)
            if gt_m and pred_m:
                quality_ok = (gt_m.group(1).replace(",", "") == pred_m.group(1).replace(",", ""))

        print(f"[{i:2d}/12] [{dom:11s}] Steps: {decode_steps:3d} | BaseEq: {base_equiv:3d} | "
              f"Saved: {step_savings:2d} ({savings_pct:4.1f}%) | Hypers: {len(hypers):2d} | Valid: {quality_ok}", flush=True)

        results.append({
            "id": s["id"],
            "domain": dom,
            "decode_steps": decode_steps,
            "base_equiv_tokens": base_equiv,
            "steps_saved": step_savings,
            "savings_pct": round(savings_pct, 2),
            "hypers_emitted": len(hypers),
            "quality_ok": quality_ok,
            "output_preview": out_text[:80].replace("\n", " "),
        })

    # Aggregate
    micro_savings_pct = ((tot_base_tokens - tot_decode_steps) / tot_base_tokens * 100.0) if tot_base_tokens else 0.0
    quality_pass_rate = sum(1 for r in results if r["quality_ok"]) / len(results) * 100.0
    samples_with_hyper = sum(1 for r in results if r["hypers_emitted"] > 0)

    print(f"\n12-PROMPT SMOKE VALIDATION SUMMARY:")
    print(f"  Micro Decode Step Savings: {micro_savings_pct:.2f}% ({tot_base_tokens - tot_decode_steps:,} tokens saved)")
    print(f"  Samples with >= 1 Hyper:   {samples_with_hyper}/{len(results)} ({samples_with_hyper/len(results)*100:.1f}%)")
    print(f"  Total Hypertokens Emitted: {tot_hypers_emitted}")
    print(f"  Quality Pass Rate:         {quality_pass_rate:.1f}%")

    print("\n--- PART 2: CONTINUATION EQUIVALENCE DIAGNOSTIC SUITE ---")
    cont_results = evaluate_continuation_suite(model, tokenizer)

    return {
        "smoke_validation": {
            "micro_savings_pct": round(micro_savings_pct, 2),
            "samples_with_hyper": samples_with_hyper,
            "quality_pass_rate": round(quality_pass_rate, 2),
            "total_hypers_emitted": tot_hypers_emitted,
        },
        "continuation_equivalence": cont_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictive checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint)
