"""
60-Prompt Held-Out Validation Sweep — Adaptation Ladder Study
==============================================================

Part of the Predictive Hypertoken Calibration Study.
See RESEARCH_LOG.md and PREDICTIVE_HYPERTOKEN_STUDY.md for full context.

PURPOSE
-------
Evaluate 5 conditions on 60 frozen held-out prompts to measure how much
output_encoder calibration training improves predictive hypertoken emission
while preserving output quality.

CONDITIONS (in order)
---------------------
  1. base            — Base Phi-3.5 (hypertoken logits masked; wrapper still present)
  2. reactive_zip2zip— Official EPFL Reactive Zip2Zip (dynamic LZW codebook)
  3. zero_shot       — Pure Predictive K=32, Step 0 (original output_encoder weights)
  4. step50          — Pure Predictive K=32, Step 50 checkpoint
  5. step100         — Pure Predictive K=32, Step 100 checkpoint

VALIDATION SET (data/cached_pure_pred_val_60.json)
---------------------------------------------------
  - 60 prompts, stratified: 20 code (MBPP), 20 instruction (Alpaca), 20 reasoning (GSM8k)
  - NEVER used for training or hyperparameter tuning
  - Fields: id, domain, prompt, ground_truth_response, prompt_token_ids, base_prompt_len

CHECKPOINTS
-----------
  - experiments/checkpoints/pure_pred_k32_step50.pt
  - experiments/checkpoints/pure_pred_k32_step100.pt

QUALITY GATES
-------------
  - Reasoning (GSM8k):   final-answer exact match (extract from '#### <number>')
  - Code (MBPP):         Python ast.parse() syntax validity
  - Instruction (Alpaca): truncation (<5 output words), repetition (trigram ≥4×),
                          corruption (>30% non-ASCII)

OUTPUT
------
  - experiments/checkpoints/eval_60prompt_results.json  (incremental checkpoint)
  - Per-prompt records: all metrics listed in PREDICTIVE_HYPERTOKEN_STUDY.md
  - Aggregate stats: MICRO/MACRO compression, bootstrap 95% CI, domain breakdown

USAGE
-----
  python experiments/eval_60prompt_validation.py

ESTIMATED RUNTIME
-----------------
  ~4–8 hours on CPU (300 generations × 8–16 s avg)
  Results are checkpointed after each (prompt, condition) pair — safe to interrupt.
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList
from zip2zip_compression import LZWCompressor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from zip2zip import Zip2ZipModel, StaticCodebookManager
from src.evaluation.offline_segmenter import segment_tokens_dp

# ─── Paths ────────────────────────────────────────────────────────────────────
VAL_DATA_PATH     = "data/cached_pure_pred_val_60.json"
PREDICTOR_PATH    = "experiments/checkpoints/cached_predictor.pkl"
CKPT_STEP50_PATH  = "experiments/checkpoints/pure_pred_k32_step50_encoder.pt"
CKPT_STEP100_PATH = "experiments/checkpoints/pure_pred_k32_step100_encoder.pt"
RESULTS_PATH      = "experiments/checkpoints/eval_60prompt_results.json"

MODEL_NAME   = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
BUDGET_K     = 32
MAX_NEW_TOKENS = 200          # enough for code/math/instruction answers
DEVICE       = "cpu"
INITIAL_VOCAB = 32011

class MaskAboveVocab(LogitsProcessor):
    """Force a base-vocabulary decode by blocking every hypertoken id."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[-1] > self.vocab_size:
            scores = scores.clone()
            scores[..., self.vocab_size :] = float("-inf")
        return scores


def _hit_max(gen_tokens: Sequence[int], eos_id: Optional[int], max_new_tokens: int) -> bool:
    if len(gen_tokens) < max_new_tokens:
        return False
    if eos_id is None:
        return True
    return not gen_tokens or gen_tokens[-1] != eos_id


def _offline_dp(token_ids: Sequence[int], phrases: Sequence[Tuple[int, ...]]) -> Tuple[int, float]:
    if not token_ids:
        return 0, 0.0
    comp_len, _, _ = segment_tokens_dp(list(token_ids), set(phrases))
    savings = len(token_ids) - comp_len
    pct = 100.0 * savings / len(token_ids)
    return savings, pct


def _expand_lzw(full_seq: List[int], prompt_len: int, disabled_ids: List[int]) -> List[int]:
    compressor = LZWCompressor(
        initial_vocab_size=INITIAL_VOCAB,
        max_codebook_size=BUDGET_K,
        max_subtokens=3,
        pad_token_id=32000,
        disabled_ids=list(disabled_ids),
    )
    decoded_full, _ = compressor.batch_decode([full_seq])[0]
    return list(decoded_full[prompt_len:])


# ─── Architecture Audit ───────────────────────────────────────────────────────
def architecture_audit(model: Zip2ZipModel, ckpt_path: str) -> None:
    print("\n" + "=" * 70)
    print("ARCHITECTURE AUDIT")
    print("=" * 70)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable_params / total_params if total_params else 0.0
    print(f"  Total model parameters:    {total_params:,}")
    print(f"  Trainable parameters:      {trainable_params:,} ({pct:.4f}%)")

    print("\n  Trainable parameter names:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"    {name}  shape={list(p.shape)}  dtype={p.dtype}")

    base_frozen = all(not p.requires_grad for p in model.base_model.parameters())
    in_frozen   = all(not p.requires_grad for p in model.input_encoder.parameters())
    out_trained = any(p.requires_grad for p in model.output_encoder.parameters())
    has_lora    = hasattr(model.base_model, "peft_type") or any(
        "lora" in n.lower() for n, _ in model.named_modules()
    )
    print(f"\n  base_model frozen:      {base_frozen}")
    print(f"  input_encoder frozen:   {in_frozen}")
    print(f"  output_encoder trained: {out_trained}")
    print(f"  LoRA/adapters present:  {has_lora}")

    if os.path.exists(ckpt_path):
        # Header-only read: the slim checkpoint is still ~900 MB, so print size
        # here and load weights later, one condition at a time.
        ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
        print(f"\n  Encoder checkpoint: {ckpt_path}")
        print(f"  Encoder checkpoint file size: {ckpt_size_mb:.1f} MB")
    print("=" * 70 + "\n", flush=True)


# ─── Quality Evaluation ───────────────────────────────────────────────────────
def eval_quality_gsm8k(output_text: str, ground_truth: str) -> Dict:
    """Extract final numerical answer from #### marker or last number."""
    def extract_answer(text: str) -> Optional[str]:
        m = re.search(r"####\s*([\-\d,\.]+)", text)
        if m:
            return m.group(1).replace(",", "").strip()
        nums = re.findall(r"[\-\d,\.]+", text)
        return nums[-1].replace(",", "").strip() if nums else None

    pred_ans = extract_answer(output_text)
    true_ans = extract_answer(ground_truth)
    correct = (pred_ans == true_ans) if (pred_ans and true_ans) else False
    return {
        "quality_label": "correct" if correct else "incorrect",
        "quality_score": int(correct),
        "pred_answer": pred_ans,
        "true_answer": true_ans,
    }

def eval_quality_mbpp(output_text: str) -> Dict:
    """Check Python syntax."""
    # Extract code block if present
    code_match = re.search(r"```(?:python)?\n(.*?)```", output_text, re.DOTALL)
    code = code_match.group(1) if code_match else output_text
    try:
        ast.parse(code)
        return {"quality_label": "syntax_ok", "quality_score": 1}
    except SyntaxError as e:
        return {"quality_label": f"syntax_error: {e}", "quality_score": 0}

def eval_quality_instruction(output_text: str, max_expected_tokens: int = 300) -> Dict:
    """Check for truncation, repetition, and obvious corruption."""
    issues = []
    # Truncation: very short output
    if len(output_text.split()) < 5:
        issues.append("truncated")
    # Repetition: look for 3+ consecutive repeated trigrams
    words = output_text.split()
    for i in range(len(words) - 8):
        trigram = tuple(words[i:i+3])
        count = sum(1 for j in range(i, min(i+30, len(words)-2))
                    if tuple(words[j:j+3]) == trigram)
        if count >= 4:
            issues.append("repetition")
            break
    # Corruption: if mostly non-ASCII
    non_ascii = sum(1 for c in output_text if ord(c) > 127)
    if non_ascii > len(output_text) * 0.3 and len(output_text) > 10:
        issues.append("corruption")
    label = "ok" if not issues else ",".join(issues)
    return {"quality_label": label, "quality_score": int(not issues)}


def evaluate_quality(domain: str, output_text: str, ground_truth: str) -> Dict:
    if domain == "reasoning":
        return eval_quality_gsm8k(output_text, ground_truth)
    elif domain == "code":
        return eval_quality_mbpp(output_text)
    else:
        return eval_quality_instruction(output_text)


# ─── Bootstrap CI ─────────────────────────────────────────────────────────────
def bootstrap_ci(data: List[float], n_boot: int = 2000, ci: float = 0.95) -> Tuple[float, float]:
    if not data:
        return (float("nan"), float("nan"))
    arr = np.array(data)
    boot_means = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return (float(lo), float(hi))


# ─── One-prompt generation helper ─────────────────────────────────────────────
def run_base(
    model: Zip2ZipModel,
    prompt_ids: List[int],
    tokenizer,
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: str = DEVICE,
) -> Dict:
    """Condition 1: base vocabulary only.

    The Zip2Zip wrapper still owns the LM head, so hypertoken logits are masked.
    Latency therefore includes wrapper overhead and is not a naked Phi-3.5 timing.
    """
    t_setup0 = time.perf_counter()
    if hasattr(model, "codebook_manager") and hasattr(model.codebook_manager, "reset"):
        model.codebook_manager.reset()
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits_proc = LogitsProcessorList([MaskAboveVocab(INITIAL_VOCAB)])
    t_setup1 = time.perf_counter()

    t_gen0 = time.perf_counter()
    with torch.no_grad():
        out = model.base_model.generate(
            input_ids=input_tensor,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_proc,
            do_sample=False,
        )
    t_gen1 = time.perf_counter()

    gen_tokens = out[0][len(prompt_ids):].tolist()
    hypers = [t for t in gen_tokens if t >= INITIAL_VOCAB]
    output_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return {
        "setup_ms":   (t_setup1 - t_setup0) * 1000,
        "prefill_ms": 0.0,          # merged into gen for base
        "decode_ms":  (t_gen1 - t_gen0) * 1000,
        "total_ms":   (t_gen1 - t_setup0) * 1000,
        "base_decode_steps": len(gen_tokens),
        "base_equiv_output_tokens": len(gen_tokens),
        "output_text": output_text,
        "output_len_tokens": len(gen_tokens),
        "hypertokens_emitted": len(hypers),
        "hypertoken_ids": hypers,
        "compressed_prompt_positions": len(prompt_ids),
        "prompt_compression_pct": 0.0,
        "decode_step_reduction_pct": 0.0,
        "step_savings": 0,
        "realization_ratio": 0.0,
        "offline_output_savings": 0,
        "offline_output_compression_pct": 0.0,
        "hit_max_new_tokens": _hit_max(gen_tokens, tokenizer.eos_token_id, max_new_tokens),
    }


def run_reactive_zip2zip(
    model: Zip2ZipModel,
    prompt_ids: List[int],
    tokenizer,
    disabled_ids: List[int],
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: str = DEVICE,
) -> Dict:
    """Condition 2: official reactive Zip2Zip — model.generate() with dynamic LZW codebook."""
    t_setup0 = time.perf_counter()
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    t_setup1 = time.perf_counter()

    t_gen0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    t_gen1 = time.perf_counter()

    full_seq = out[0].tolist()
    gen_tokens = full_seq[len(prompt_ids):]
    hypers = [t for t in gen_tokens if t >= INITIAL_VOCAB]
    expanded = _expand_lzw(full_seq, len(prompt_ids), disabled_ids)
    actual_decode_steps = len(gen_tokens)
    base_equiv_out = len(expanded)
    step_savings = base_equiv_out - actual_decode_steps
    decode_reduction_pct = 100.0 * step_savings / base_equiv_out if base_equiv_out > 0 else 0.0
    output_text = tokenizer.decode(expanded, skip_special_tokens=True)
    return {
        "setup_ms":   (t_setup1 - t_setup0) * 1000,
        "prefill_ms": 0.0,
        "decode_ms":  (t_gen1 - t_gen0) * 1000,
        "total_ms":   (t_gen1 - t_setup0) * 1000,
        "base_decode_steps": actual_decode_steps,
        "base_equiv_output_tokens": base_equiv_out,
        "output_text": output_text,
        "output_len_tokens": actual_decode_steps,
        "hypertokens_emitted": len(hypers),
        "hypertoken_ids": hypers,
        "compressed_prompt_positions": len(prompt_ids),
        "prompt_compression_pct": 0.0,
        "decode_step_reduction_pct": decode_reduction_pct,
        "step_savings": step_savings,
        "realization_ratio": None,
        "offline_output_savings": None,
        "offline_output_compression_pct": None,
        "hit_max_new_tokens": _hit_max(gen_tokens, tokenizer.eos_token_id, max_new_tokens),
    }


def run_pure_predictive(
    model: Zip2ZipModel,
    predictor,
    prompt_ids: List[int],
    tokenizer,
    dim: int,
    pad_id: int,
    disabled_ids: List[int],
    ground_truth: str = "",
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: str = DEVICE,
) -> Dict:
    """Conditions 3/4: pure predictive K=32. Shared logic — call with current output_encoder."""
    t_setup0 = time.perf_counter()

    # Build prompt-conditioned codebook
    p_dict, _ = predictor.select_prompt_conditioned(prompt_ids, budget=BUDGET_K)
    pred_phrases = list(p_dict.keys())
    comp_len, tiles, _ = segment_tokens_dp(prompt_ids, set(pred_phrases))
    seeded_dict = {p: INITIAL_VOCAB + i for i, p in enumerate(pred_phrases)}
    resegmented = [tile[0] if len(tile) == 1 else seeded_dict[tile] for tile in tiles]

    # Build hypertoken map: seeded_dict id -> list of base token ids
    hyper_to_subtokens = {v: list(k) for k, v in seeded_dict.items()}
    n_phrases = len(pred_phrases)

    static_mgr = StaticCodebookManager(
        initial_vocab_size=INITIAL_VOCAB,
        max_codebook_size=BUDGET_K,
        max_subtokens=3,
        embedding_dim=dim,
        pad_token_id=pad_id,
        disabled_ids=disabled_ids,
    )
    static_mgr.set_seeded_codebook(seeded_dict, batch_size=1, device=torch.device(device))
    static_mgr.attach_to_model(model)

    pred_tensor = torch.tensor([resegmented], dtype=torch.long, device=device)
    logits_proc = LogitsProcessorList([static_mgr.get_logits_processor()])
    t_setup1 = time.perf_counter()

    t_gen0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids=pred_tensor,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_proc,
            do_sample=False,
        )
    t_gen1 = time.perf_counter()
    static_mgr.detach_from_model(model)

    gen_tokens = out[0][len(resegmented):].tolist()
    hypers_in_gen = [t for t in gen_tokens if t >= INITIAL_VOCAB]
    hyper_ids_emitted = hypers_in_gen  # list of hypertoken vocab IDs

    # Expand to base-equivalent tokens
    expanded = []
    for tid in gen_tokens:
        subs = hyper_to_subtokens.get(tid) or static_mgr.hyper_to_subtokens.get(tid)
        if subs:
            expanded.extend(subs)
        else:
            expanded.append(tid)

    actual_decode_steps = len(gen_tokens)
    base_equiv_out = len(expanded)
    step_savings = base_equiv_out - actual_decode_steps

    prompt_comp_pct = 100.0 * (len(prompt_ids) - comp_len) / len(prompt_ids) if prompt_ids else 0.0
    decode_reduction_pct = 100.0 * step_savings / base_equiv_out if base_equiv_out > 0 else 0.0

    offline_prompt_savings = sum(len(tile) - 1 for tile in tiles if len(tile) > 1)
    # Upper bound on THIS generated text: optimal DP tiling with the same prompt codebook.
    offline_output_savings, offline_output_pct = _offline_dp(expanded, pred_phrases)
    if offline_output_savings > 0:
        realization_ratio = step_savings / offline_output_savings
    else:
        realization_ratio = 1.0 if step_savings == 0 else None

    gt_ids = tokenizer.encode(ground_truth, add_special_tokens=False) if ground_truth else []
    offline_gt_savings, offline_gt_pct = _offline_dp(gt_ids, pred_phrases)

    output_text = tokenizer.decode(expanded, skip_special_tokens=True)

    return {
        "setup_ms":   (t_setup1 - t_setup0) * 1000,
        "prefill_ms": 0.0,  # prefill is inside gen for now
        "decode_ms":  (t_gen1 - t_gen0) * 1000,
        "total_ms":   (t_gen1 - t_setup0) * 1000,
        "base_decode_steps": actual_decode_steps,
        "base_equiv_output_tokens": base_equiv_out,
        "output_text": output_text,
        "output_len_tokens": actual_decode_steps,
        "hypertokens_emitted": len(hypers_in_gen),
        "hypertoken_ids": [int(t) for t in hypers_in_gen],
        "n_codebook_entries": n_phrases,
        "compressed_prompt_positions": comp_len,
        "prompt_compression_pct": prompt_comp_pct,
        "decode_step_reduction_pct": decode_reduction_pct,
        "step_savings": step_savings,
        "realization_ratio": realization_ratio,
        "offline_prompt_savings": offline_prompt_savings,
        "offline_output_savings": offline_output_savings,
        "offline_output_compression_pct": offline_output_pct,
        "offline_gt_savings": offline_gt_savings,
        "offline_gt_compression_pct": offline_gt_pct,
        "offline_gt_tokens": len(gt_ids),
        "hit_max_new_tokens": _hit_max(gen_tokens, tokenizer.eos_token_id, max_new_tokens),
    }


# ─── Aggregate statistics ─────────────────────────────────────────────────────
def aggregate_stats(records: List[Dict], condition: str, domain_filter: Optional[str] = None) -> Dict:
    subset = [r for r in records if r.get("condition") == condition]
    if domain_filter:
        subset = [r for r in subset if r.get("domain") == domain_filter]
    if not subset:
        return {}

    def safe_mean(vals):
        v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(v)) if v else float("nan")

    def safe_median(vals):
        v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
        return float(np.median(v)) if v else float("nan")

    fields = [
        "prompt_compression_pct", "decode_step_reduction_pct",
        "total_ms", "setup_ms", "decode_ms",
        "hypertokens_emitted", "quality_score",
        "base_equiv_output_tokens", "base_decode_steps",
    ]

    result = {"n": len(subset)}
    for f in fields:
        vals = [r.get(f) for r in subset]
        clean = [v for v in vals if v is not None and isinstance(v, (int, float)) and not np.isnan(v)]
        result[f + "_mean"]   = safe_mean(clean)
        result[f + "_median"] = safe_median(clean)
        lo, hi = bootstrap_ci(clean)
        result[f + "_ci95_lo"] = lo
        result[f + "_ci95_hi"] = hi

    # Hypertoken emission counts
    result["n_with_0_hypers"]  = sum(1 for r in subset if r.get("hypertokens_emitted", 0) == 0)
    result["n_with_1plus_hypers"] = sum(1 for r in subset if r.get("hypertokens_emitted", 0) >= 1)
    result["n_with_2plus_hypers"] = sum(1 for r in subset if r.get("hypertokens_emitted", 0) >= 2)
    result["n_with_3plus_hypers"] = sum(1 for r in subset if r.get("hypertokens_emitted", 0) >= 3)

    # Quality pass rate
    result["quality_pass_rate"] = safe_mean([r.get("quality_score") for r in subset])
    result["quality_labels"] = [r.get("quality_label", "") for r in subset]

    # MICRO compression (token-weighted)
    total_base_prompt = sum(r.get("base_prompt_len", 0) for r in subset)
    total_comp_prompt  = sum(r.get("compressed_prompt_positions", 0) for r in subset)
    result["micro_prompt_compression_pct"] = (
        100.0 * (total_base_prompt - total_comp_prompt) / total_base_prompt
        if total_base_prompt > 0 else 0.0
    )

    total_base_decode = sum(r.get("base_equiv_output_tokens") or 0 for r in subset)
    total_actual_steps = sum(r.get("base_decode_steps") or 0 for r in subset)
    result["micro_decode_reduction_pct"] = (
        100.0 * (total_base_decode - total_actual_steps) / total_base_decode
        if total_base_decode > 0 else 0.0
    )
    live_savings = total_base_decode - total_actual_steps
    offline_savings = sum(r.get("offline_output_savings") or 0 for r in subset)
    gt_savings = sum(r.get("offline_gt_savings") or 0 for r in subset)
    gt_tokens = sum(r.get("offline_gt_tokens") or 0 for r in subset)
    result["micro_realization_ratio"] = (
        live_savings / offline_savings if offline_savings > 0 else None
    )
    result["micro_offline_output_compression_pct"] = (
        100.0 * offline_savings / total_base_decode if total_base_decode > 0 else 0.0
    )
    result["micro_offline_gt_compression_pct"] = (
        100.0 * gt_savings / gt_tokens if gt_tokens > 0 else None
    )
    result["hit_max_new_tokens_rate"] = safe_mean(
        [1.0 if r.get("hit_max_new_tokens") else 0.0 for r in subset]
    )
    result["total_hypertokens_emitted"] = int(sum(r.get("hypertokens_emitted") or 0 for r in subset))
    result["total_step_savings"] = int(live_savings)

    return result


# ─── Main sweep ───────────────────────────────────────────────────────────────
def _round_robin_by_domain(records: List[Dict]) -> List[Dict]:
    """Interleave domains so the first few prompts already cover code, instruction, and reasoning."""
    buckets: Dict[str, List[Dict]] = {}
    for record in records:
        buckets.setdefault(record["domain"], []).append(record)
    domains = [d for d in ("code", "instruction", "reasoning") if d in buckets]
    domains.extend(d for d in buckets if d not in domains)
    ordered: List[Dict] = []
    index = 0
    while True:
        added = False
        for domain in domains:
            if index < len(buckets[domain]):
                ordered.append(buckets[domain][index])
                added = True
        if not added:
            return ordered
        index += 1


def _restore_pretrained_encoder(model: Zip2ZipModel) -> None:
    print("  Restoring original Zip2Zip output_encoder...", flush=True)
    model.load_pretrained_hyper_encoders(MODEL_NAME)
    model.output_encoder.to(torch.float32)
    gc.collect()


def _write_live_summary(
    records: List[Dict],
    conditions: List[str],
    results_path: str,
    started_at: float,
) -> None:
    """Refresh a small decision file after every generation."""
    domains = ["code", "instruction", "reasoning"]
    agg: Dict[str, Dict] = {}
    for cond in conditions:
        agg[cond] = {"all": aggregate_stats(records, cond)}
        for domain in domains:
            agg[cond][domain] = aggregate_stats(records, cond, domain)

    by_prompt: Dict[str, set] = {}
    for record in records:
        by_prompt.setdefault(record["prompt_id"], set()).add(record["condition"])
    n_complete = sum(1 for seen in by_prompt.values() if set(conditions) <= seen)
    elapsed_min = (time.time() - started_at) / 60.0
    summary = {
        "prompts_with_all_conditions": n_complete,
        "records": len(records),
        "elapsed_min": elapsed_min,
        "aggregate": agg,
    }
    live_json = os.path.splitext(results_path)[0] + "_live_summary.json"
    tmp = live_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, live_json)

    lines = [
        f"prompts_complete={n_complete} records={len(records)} elapsed_min={elapsed_min:.1f}",
        f"{'condition':<32} {'n':>4} {'quality':>8} {'decode_red':>10} {'hypers':>8} {'latency_s':>10}",
    ]
    for cond in conditions:
        stats = agg[cond].get("all") or {}
        n = stats.get("n", 0)
        if not n:
            continue
        quality = stats.get("quality_pass_rate", float("nan"))
        decode_red = stats.get("micro_decode_reduction_pct", float("nan"))
        hypers = stats.get("hypertokens_emitted_mean", float("nan"))
        latency_s = (stats.get("total_ms_mean") or 0.0) / 1000.0
        lines.append(
            f"{cond:<32} {n:4d} {quality:8.1%} {decode_red:9.2f}% {hypers:8.2f} {latency_s:10.1f}"
        )
    for domain in domains:
        lines.append(f"[{domain}]")
        for cond in conditions:
            stats = agg[cond].get(domain) or {}
            n = stats.get("n", 0)
            if not n:
                continue
            quality = stats.get("quality_pass_rate", float("nan"))
            decode_red = stats.get("micro_decode_reduction_pct", float("nan"))
            hypers = stats.get("hypertokens_emitted_mean", float("nan"))
            lines.append(
                f"  {cond:<30} n={n:<3d} q={quality:.0%} red={decode_red:.2f}% hyp={hypers:.2f}"
            )
    live_txt = os.path.splitext(results_path)[0] + "_live_summary.txt"
    with open(live_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("LIVE " + lines[0], flush=True)
    if len(lines) > 2:
        print("LIVE " + " | ".join(lines[2:2 + len(conditions)]), flush=True)


def _load_encoder_checkpoint(model: Zip2ZipModel, path: str) -> None:
    gc.collect()
    try:
        import psutil
        avail = psutil.virtual_memory().available / (1024 ** 3)
        print(f"  Available RAM before checkpoint load: {avail:.2f} GB", flush=True)
    except Exception:
        pass
    print(f"  Loading output_encoder from {path}...", flush=True)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.output_encoder.load_state_dict(ckpt["output_encoder_state_dict"])
    model.output_encoder.to(torch.float32)
    print(
        f"  Loaded step={ckpt.get('step')} loss={ckpt.get('loss')} "
        f"params={ckpt.get('n_params')}",
        flush=True,
    )
    del ckpt
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="If >0, evaluate only the first N prompts.")
    parser.add_argument("--one-per-domain", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--results", type=str, default=RESULTS_PATH)
    args = parser.parse_args()
    results_path = args.results
    max_new_tokens = args.max_new_tokens

    print("=" * 70)
    print("60-PROMPT HELD-OUT VALIDATION SWEEP")
    print("=" * 70, flush=True)

    torch.set_num_threads(8)

    # Load validation data
    with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
        val_records = json.load(f)
    if args.one_per_domain:
        picked = []
        seen = set()
        for record in val_records:
            if record["domain"] not in seen:
                picked.append(record)
                seen.add(record["domain"])
        val_records = picked
    if args.limit and args.limit > 0:
        val_records = val_records[: args.limit]
    print(f"Loaded {len(val_records)} held-out prompts. max_new_tokens={max_new_tokens}", flush=True)
    domain_counts = {}
    for r in val_records:
        domain_counts[r["domain"]] = domain_counts.get(r["domain"], 0) + 1
    print(f"Domain breakdown: {domain_counts}", flush=True)

    # Load predictor
    with open(PREDICTOR_PATH, "rb") as f:
        predictor = pickle.load(f)
    print("Predictor loaded.", flush=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

    # Load model (float16 backbone)
    print("Loading Zip2Zip model (float16 backbone)...", flush=True)
    t0 = time.time()
    model = Zip2ZipModel.from_pretrained(
        MODEL_NAME,
        max_codebook_size=BUDGET_K,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.output_encoder.to(torch.float32)
    print(f"Model loaded in {time.time() - t0:.1f}s.", flush=True)

    dim = model.zip2zip_config.encoder.hidden_size
    pad_id = 32000
    disabled_ids = list(model.zip2zip_config.compression.disabled_ids)

    # Architecture audit
    architecture_audit(model, CKPT_STEP100_PATH)

    # Load incremental checkpoint if exists
    existing_results = []
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
            existing_results = existing.get("per_prompt_records", [])
        print(f"Resuming from {len(existing_results)} existing records.", flush=True)

    # Track which (prompt_id, condition) already done
    done_set = {(r["prompt_id"], r["condition"]) for r in existing_results}

    CONDITIONS = [
        "base",
        "reactive_zip2zip",
        "zero_shot_pred_k32",
        "calibrated_pred_k32_step50",
        "calibrated_pred_k32_step100",
    ]
    encoder_ckpts = {
        "calibrated_pred_k32_step50": CKPT_STEP50_PATH,
        "calibrated_pred_k32_step100": CKPT_STEP100_PATH,
    }

    # Prompt-major, domain-round-robin order. Each new prompt updates every
    # condition, so a decision can be made long before all 300 generations finish.
    val_records = _round_robin_by_domain(val_records)
    all_records = list(existing_results)
    n_prompts = len(val_records)
    n_conditions = len(CONDITIONS)
    total_runs = n_prompts * n_conditions
    print(f"Total runs to do: {total_runs} ({n_prompts} prompts × {n_conditions} conditions)", flush=True)
    print(f"Already done:     {len(done_set)}", flush=True)
    print("Order: round-robin domains, all conditions per prompt.", flush=True)

    model.eval()
    loaded_encoder = "pretrained"
    sweep_started = time.time()

    for prompt_idx, record in enumerate(val_records):
        pid = record["id"]
        domain = record["domain"]
        prompt_ids = record["prompt_token_ids"]
        base_prompt_len = len(prompt_ids)
        ground_truth = record.get("ground_truth_response", "")

        for condition in CONDITIONS:
            if (pid, condition) in done_set:
                continue

            if condition in encoder_ckpts:
                if loaded_encoder != condition:
                    _load_encoder_checkpoint(model, encoder_ckpts[condition])
                    loaded_encoder = condition
            elif loaded_encoder != "pretrained":
                _restore_pretrained_encoder(model)
                loaded_encoder = "pretrained"

            t_prompt_start = time.time()
            print(
                f"  [{prompt_idx+1:2d}/{n_prompts}] {condition:<28} {pid[:24]:24s} ({domain:11s}) ...",
                end=" ", flush=True
            )

            try:
                if condition == "base":
                    metrics = run_base(
                        model, prompt_ids, tokenizer, max_new_tokens=max_new_tokens,
                    )

                elif condition == "reactive_zip2zip":
                    metrics = run_reactive_zip2zip(
                        model, prompt_ids, tokenizer, disabled_ids,
                        max_new_tokens=max_new_tokens,
                    )

                elif condition in (
                    "zero_shot_pred_k32",
                    "calibrated_pred_k32_step50",
                    "calibrated_pred_k32_step100",
                ):
                    metrics = run_pure_predictive(
                        model, predictor, prompt_ids, tokenizer,
                        dim, pad_id, disabled_ids,
                        ground_truth=ground_truth,
                        max_new_tokens=max_new_tokens,
                    )

                else:
                    raise ValueError(f"Unknown condition: {condition}")

                quality = evaluate_quality(domain, metrics["output_text"], ground_truth)
                metrics.update(quality)

                row = {
                    "prompt_id":   pid,
                    "domain":      domain,
                    "condition":   condition,
                    "base_prompt_len": base_prompt_len,
                    **{k: v for k, v in metrics.items() if k != "hypertoken_ids"},
                    "hypertoken_ids": metrics.get("hypertoken_ids", []),
                }
                all_records.append(row)
                done_set.add((pid, condition))

                elapsed = time.time() - t_prompt_start
                print(
                    f"OK  | t={elapsed:.1f}s | steps={metrics.get('base_decode_steps', 0)} | "
                    f"hypers={metrics.get('hypertokens_emitted', 0)} | "
                    f"red={metrics.get('decode_step_reduction_pct', 0.0):.2f}% | "
                    f"q={metrics.get('quality_label', '?')}",
                    flush=True
                )

            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                _save_incremental(all_records, len(all_records), total_runs, results_path)
                raise

            _save_incremental(all_records, len(all_records), total_runs, results_path)
            _write_live_summary(all_records, CONDITIONS, results_path, sweep_started)

    # Final aggregate statistics
    print("\n" + "=" * 70, flush=True)
    print("COMPUTING AGGREGATE STATISTICS", flush=True)
    print("=" * 70, flush=True)

    domain_list = ["code", "instruction", "reasoning"]
    agg_report = {}
    for cond in CONDITIONS:
        agg_report[cond] = {}
        agg_report[cond]["all"] = aggregate_stats(all_records, cond)
        for dom in domain_list:
            agg_report[cond][dom] = aggregate_stats(all_records, cond, dom)

    # Print summary tables
    _print_summary_table(all_records, CONDITIONS, agg_report)

    # Answer the 9 key questions
    _answer_key_questions(all_records, agg_report)

    # Save final results
    final = {
        "per_prompt_records": all_records,
        "aggregate": agg_report,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"\nFull results saved to {results_path}", flush=True)


def _save_incremental(records: List[Dict], done: int, total: int, results_path: str = RESULTS_PATH) -> None:
    tmp = results_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"completed": done, "total": total, "per_prompt_records": records}, f)
    os.replace(tmp, results_path)


def _print_summary_table(all_records, conditions, agg_report):
    print("\n" + "=" * 90)
    print("SUMMARY TABLE — ALL 60 PROMPTS")
    print(f"{'Condition':<35} {'QualityPass':>11} {'PromptComp%':>11} {'DecodeRed%':>10} {'LatencyMs':>10} {'HyperEm/P':>10}")
    print("-" * 90)
    for cond in conditions:
        a = agg_report[cond].get("all", {})
        q    = a.get("quality_pass_rate", float("nan"))
        pc   = a.get("micro_prompt_compression_pct", float("nan"))
        dr   = a.get("micro_decode_reduction_pct", float("nan"))
        lat  = a.get("total_ms_mean", float("nan"))
        hyp  = a.get("hypertokens_emitted_mean", float("nan"))
        print(f"  {cond:<33} {q:>10.1%} {pc:>10.2f}% {dr:>9.2f}% {lat:>9.1f}ms {hyp:>9.2f}")

    print("\n" + "=" * 90)
    print("BY DOMAIN")
    for dom in ["code", "instruction", "reasoning"]:
        print(f"\n  {dom.upper()}")
        print(f"  {'Condition':<33} {'QualityPass':>11} {'PromptComp%':>11} {'DecodeRed%':>10} {'HyperEm/P':>10}")
        print("  " + "-" * 70)
        for cond in conditions:
            a = agg_report[cond].get(dom, {})
            if not a:
                continue
            q   = a.get("quality_pass_rate", float("nan"))
            pc  = a.get("micro_prompt_compression_pct", float("nan"))
            dr  = a.get("micro_decode_reduction_pct", float("nan"))
            hyp = a.get("hypertokens_emitted_mean", float("nan"))
            print(f"  {cond:<33} {q:>10.1%} {pc:>10.2f}% {dr:>9.2f}% {hyp:>9.2f}")

    print("=" * 90, flush=True)


def _answer_key_questions(all_records, agg_report):
    print("\n" + "=" * 70)
    print("KEY QUESTIONS (Step-100 Calibrated vs Zero-Shot)")
    print("=" * 70)

    zs   = agg_report.get("zero_shot_pred_k32", {})
    cal  = agg_report.get("calibrated_pred_k32_step100", {})
    base = agg_report.get("base", {})

    def get(d, domain, key):
        return d.get(domain, {}).get(key, float("nan"))

    zs_q_all  = get(zs,  "all",        "quality_pass_rate")
    cal_q_all = get(cal, "all",        "quality_pass_rate")
    base_q_all= get(base,"all",        "quality_pass_rate")

    print(f"\n1. Did calibration generalize beyond the 3-prompt probe?")
    cal_hyp = get(cal, "all", "hypertokens_emitted_mean")
    zs_hyp  = get(zs,  "all", "hypertokens_emitted_mean")
    print(f"   Zero-shot hypers/prompt: {zs_hyp:.2f}  |  Step-100 hypers/prompt: {cal_hyp:.2f}")
    print(f"   -> {'YES — calibrated emits more hypertokens' if cal_hyp > zs_hyp else 'NO — no improvement in hypertoken emission'}")

    print(f"\n2. Did it improve code?")
    zs_code  = get(zs,  "code", "quality_pass_rate")
    cal_code = get(cal, "code", "quality_pass_rate")
    print(f"   Code quality: zero-shot={zs_code:.1%}  calibrated={cal_code:.1%}")

    print(f"\n3. Did it improve reasoning?")
    zs_rsn  = get(zs,  "reasoning", "quality_pass_rate")
    cal_rsn = get(cal, "reasoning", "quality_pass_rate")
    cal_dr_rsn = get(cal, "reasoning", "micro_decode_reduction_pct")
    print(f"   Reasoning quality: zero-shot={zs_rsn:.1%}  calibrated={cal_rsn:.1%}")
    print(f"   Decode reduction (calibrated/reasoning): {cal_dr_rsn:.2f}%")

    print(f"\n4. Did it improve instruction/general?")
    zs_ins  = get(zs,  "instruction", "quality_pass_rate")
    cal_ins = get(cal, "instruction", "quality_pass_rate")
    print(f"   Instruction quality: zero-shot={zs_ins:.1%}  calibrated={cal_ins:.1%}")

    print(f"\n5. What % of offline opportunity was realized?")
    # Compute from records
    cal_recs = [r for r in all_records if r.get("condition") == "calibrated_pred_k32_step100"]
    real_ratios = [r.get("realization_ratio") for r in cal_recs if r.get("realization_ratio") is not None]
    mean_real = float(np.mean(real_ratios)) if real_ratios else float("nan")
    print(f"   Mean realization ratio (hypers used / hypers in codebook): {mean_real:.3f} ({mean_real*100:.1f}%)")

    print(f"\n6. Was quality preserved?")
    print(f"   Base quality: {base_q_all:.1%}  |  Calibrated quality: {cal_q_all:.1%}")
    preserved = abs(cal_q_all - base_q_all) < 0.05 if not np.isnan(cal_q_all) else False
    print(f"   -> {'YES (within 5pp)' if preserved else 'WARNING: quality gap >5pp'}")

    print(f"\n7. Did wall-clock latency improve?")
    base_lat  = get(base, "all", "total_ms_mean")
    cal_lat   = get(cal,  "all", "total_ms_mean")
    print(f"   Base latency: {base_lat:.1f}ms  |  Calibrated latency: {cal_lat:.1f}ms")
    print(f"   Delta = {base_lat - cal_lat:+.1f}ms (positive = calibrated faster)")

    print(f"\n8. Is Step-100 undertrained, sufficient, or overfitting?")
    s50 = agg_report.get("calibrated_pred_k32_step50", {})
    s50_hyp = get(s50, "all", "hypertokens_emitted_mean")
    s50_dr = get(s50, "all", "micro_decode_reduction_pct")
    cal_dr_all = get(cal, "all", "micro_decode_reduction_pct")
    cal_real = get(cal, "all", "micro_realization_ratio")
    print(f"   Decode step reduction (MICRO): step50={s50_dr:.2f}% step100={cal_dr_all:.2f}%")
    print(f"   Hypers emitted: step-0={zs_hyp:.2f}  step-50={s50_hyp:.2f}  step-100={cal_hyp:.2f}")
    print(f"   Step-100 micro realization (live savings / DP opportunity on generated text): {cal_real}")
    if cal_hyp < 0.5:
        print("   -> UNDERTRAINED — model barely emits hypertokens at step 100.")
    elif cal_hyp >= 2.0:
        print("   -> POSSIBLY SUFFICIENT — emitting meaningful hypertokens; check quality.")
    else:
        print("   -> MARGINAL — some emission but likely benefits from more training.")

    print(f"\n9. Next step recommendation:")
    if cal_hyp < 0.3 and abs(cal_q_all - base_q_all) < 0.05:
        print("   -> A) TRAIN LONGER — emission too low, quality intact, safe to continue.")
    elif not preserved:
        print("   -> D) ALTER TRAINABLE MODULES — quality degraded, check architecture.")
    elif cal_hyp >= 2.0 and cal_dr_all > 5.0:
        print("   -> B) INCREASE K or C) TRAIN HYBRID — strong signal, push further.")
    else:
        print("   -> A) TRAIN LONGER at current config; consider increasing max_steps.")

    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
