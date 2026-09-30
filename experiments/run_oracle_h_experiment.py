"""Runner for Phase 1: Oracle Per-Example H Representation Test.

Evaluates whether there exists ANY single 3072-D vector H such that
frozen Vanilla Phi-3.5 reproduces the two-token state [A, B] without any
encoder generalization constraints.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from zip2zip.frozen_phi_h import (
    EVAL_OFFSETS,
    assert_model_strictly_frozen,
    compute_model_parameter_hash,
)
from zip2zip.oracle_h import (
    compute_teacher_reference,
    optimize_oracle_h_for_example,
)
from experiments.run_representation_experiment import (
    load_vanilla_phi,
    CANONICAL_MODEL_ID,
    CANONICAL_MODEL_REVISION,
    VERDICT_GREEN,
    VERDICT_YELLOW,
    VERDICT_RED,
    VERDICT_INCONCLUSIVE,
)


def select_stratified_12_dev_phrases(benchmark_path: Path) -> List[Dict[str, Any]]:
    """Select exactly 12 deterministic DEV benchmark phrases: 4 code, 4 reasoning, 4 instruction."""
    records = []
    with benchmark_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Pick the phrase at phrase_start_idx == 32 for each prompt
    selected = [r for r in records if r.get("phrase_start_idx") == 32]
    if len(selected) != 12:
        # Fallback to unique prompt IDs
        seen_prompts = set()
        selected = []
        for r in records:
            p_id = r["prompt_id"]
            if p_id not in seen_prompts:
                seen_prompts.add(p_id)
                selected.append(r)
            if len(selected) == 12:
                break

    # Count by domain
    domains = {}
    for r in selected:
        d = r["domain"]
        domains[d] = domains.get(d, 0) + 1

    print(f"Selected {len(selected)} DEV benchmark examples: {domains}", flush=True)
    return selected


def run_oracle_h_investigation(
    model: Any,
    tokenizer: Any,
    dev_benchmark_path: Path,
    device: torch.device,
    output_dir: Path,
    *,
    num_samples: int = 12,
    stage_a_steps: int = 300,
    stage_b_steps: int = 300,
) -> Dict[str, Any]:
    """Execute complete Phase 1 Oracle H representation investigation."""
    print("\n" + "=" * 65, flush=True)
    print("STARTING PHASE 1: ORACLE PER-EXAMPLE H REPRESENTATION TEST", flush=True)
    print(f"Stratified Samples: {num_samples}, Device: {device}", flush=True)
    print("=" * 65, flush=True)

    base_hash_before = compute_model_parameter_hash(model)
    assert_model_strictly_frozen(model)

    examples = select_stratified_12_dev_phrases(dev_benchmark_path)[:num_samples]

    per_example_results = []
    t_start = time.perf_counter()

    for idx, sample in enumerate(examples, 1):
        prompt_id = sample["prompt_id"]
        domain = sample["domain"]
        ctx = sample["context_token_ids"]
        t_a = sample["token_a"]
        t_b = sample["token_b"]
        fut = sample["future_token_ids"]

        tok_a_str = tokenizer.decode([t_a])
        tok_b_str = tokenizer.decode([t_b])

        print(
            f"\n--- [{idx:2d}/{len(examples)}] Prompt: {prompt_id} ({domain}) | Phrase: [{tok_a_str!r}, {tok_b_str!r}] ---",
            flush=True,
        )

        # 1. Precompute Teacher Reference Data
        t_ref_start = time.perf_counter()
        teacher_ref = compute_teacher_reference(
            model, ctx, t_a, t_b, fut, device, offsets=EVAL_OFFSETS, max_rollout=32
        )
        t_ref_time = time.perf_counter() - t_ref_start

        # 2. Multi-Init Two-Stage Optimization
        t_opt_start = time.perf_counter()
        opt_outcome = optimize_oracle_h_for_example(
            model,
            teacher_ref,
            device,
            stage_a_max_steps=stage_a_steps,
            stage_b_max_steps=stage_b_steps,
        )
        t_opt_time = time.perf_counter() - t_opt_start

        best_init = opt_outcome["best_init_name"]
        best_res = opt_outcome["best_result"]

        # Print per-example summary
        print(
            f"  Best Init: {best_init} | Stage A Best KL: {best_res.stage_a_best_kl:.4f} nats (Top-1: {best_res.stage_a_best_top1}) | "
            f"Stage B Best Loss: {best_res.stage_b_best_loss:.4f} (Offset 0 KL: {best_res.stage_b_best_offset0_kl:.4f})",
            flush=True,
        )
        print(
            f"  Rollout Agreement: 8={best_res.rollout_agreement_stage_b[8]*100:.1f}%, "
            f"16={best_res.rollout_agreement_stage_b[16]*100:.1f}%, "
            f"32={best_res.rollout_agreement_stage_b[32]*100:.1f}% | Divergence Index: {best_res.divergence_index_stage_b}",
            flush=True,
        )

        # Serialize per-example record
        per_example_results.append({
            "example_index": idx,
            "prompt_id": prompt_id,
            "domain": domain,
            "token_a": t_a,
            "token_b": t_b,
            "token_a_str": tok_a_str,
            "token_b_str": tok_b_str,
            "context_length": len(ctx),
            "best_init_name": best_init,
            "stage_a_initial_kl": best_res.stage_a_initial_kl,
            "stage_a_best_kl": best_res.stage_a_best_kl,
            "stage_a_best_top1": best_res.stage_a_best_top1,
            "stage_a_best_top5_overlap": best_res.stage_a_best_top5_overlap,
            "stage_a_max_logit_diff": best_res.stage_a_max_logit_diff,
            "stage_a_steps": best_res.stage_a_steps_run,
            "stage_b_initial_loss": best_res.stage_b_initial_loss,
            "stage_b_best_loss": best_res.stage_b_best_loss,
            "stage_b_best_offset0_kl": best_res.stage_b_best_offset0_kl,
            "stage_b_best_multi_offset_kl": best_res.stage_b_best_multi_offset_kl,
            "stage_b_top1_rates": {str(k): v for k, v in best_res.stage_b_best_top1_rates.items()},
            "stage_b_per_offset_kl": {str(k): v for k, v in best_res.stage_b_per_offset_kl.items()},
            "stage_b_steps": best_res.stage_b_steps_run,
            "h_norm_a": best_res.h_norm_a,
            "h_norm_b": best_res.h_norm_b,
            "cos_sim_a_with_tokens": best_res.cos_sim_a_with_tokens,
            "cos_sim_b_with_tokens": best_res.cos_sim_b_with_tokens,
            "rollout_agreement_stage_a": {str(k): v for k, v in best_res.rollout_agreement_stage_a.items()},
            "rollout_agreement_stage_b": {str(k): v for k, v in best_res.rollout_agreement_stage_b.items()},
            "divergence_index_stage_a": best_res.divergence_index_stage_a,
            "divergence_index_stage_b": best_res.divergence_index_stage_b,
            "vanilla_rollout_32": best_res.vanilla_rollout_32,
            "student_rollout_32_stage_a": best_res.student_rollout_32_stage_a,
            "student_rollout_32_stage_b": best_res.student_rollout_32_stage_b,
            "timing_s": round(t_ref_time + t_opt_time, 2),
        })

    elapsed_total = time.perf_counter() - t_start

    # Verify base parameter hash unchanged
    base_hash_after = compute_model_parameter_hash(model)
    if base_hash_after != base_hash_before:
        raise AssertionError("HARD FAIL: Base model parameters mutated during Oracle H optimization!")
    print(f"\nBase model parameter hash verified strictly invariant: {base_hash_before[:16]}... matches.", flush=True)

    # Compute aggregate metrics across best solutions
    stage_a_top1_rate = float(np.mean([r["stage_a_best_top1"] for r in per_example_results]))
    stage_a_mean_kl = float(np.mean([r["stage_a_best_kl"] for r in per_example_results]))

    stage_b_offset0_top1_rate = float(np.mean([r["stage_b_top1_rates"].get("0", False) for r in per_example_results]))
    stage_b_offset0_mean_kl = float(np.mean([r["stage_b_best_offset0_kl"] for r in per_example_results]))
    stage_b_multi_mean_kl = float(np.mean([r["stage_b_best_multi_offset_kl"] for r in per_example_results]))

    all_stage_b_top1s = []
    for r in per_example_results:
        all_stage_b_top1s.extend(list(r["stage_b_top1_rates"].values()))
    stage_b_overall_top1_rate = float(np.mean(all_stage_b_top1s)) if all_stage_b_top1s else 0.0

    rollout_16_rate_a = float(np.mean([r["rollout_agreement_stage_a"]["16"] for r in per_example_results]))
    rollout_16_rate_b = float(np.mean([r["rollout_agreement_stage_b"]["16"] for r in per_example_results]))
    rollout_32_rate_b = float(np.mean([r["rollout_agreement_stage_b"]["32"] for r in per_example_results]))

    # Per-offset aggregates for Stage B
    per_offset_top1_agg = {}
    per_offset_kl_agg = {}
    for off in EVAL_OFFSETS:
        per_offset_top1_agg[off] = float(np.mean([r["stage_b_top1_rates"].get(str(off), False) for r in per_example_results]))
        per_offset_kl_agg[off] = float(np.mean([r["stage_b_per_offset_kl"].get(str(off), 0.0) for r in per_example_results]))

    # Pareto tradeoff analysis:
    # Does optimizing continuation (Stage B) degrade immediate Offset 0?
    pareto_offset0_kl_delta = stage_b_offset0_mean_kl - stage_a_mean_kl
    pareto_offset0_top1_delta = stage_b_offset0_top1_rate - stage_a_top1_rate

    # Acceptance Band Classification
    # GREEN: immediate top-1 >= 95%, immediate mean KL <= 0.10, multi-offset mean KL <= 0.10, rollout-16 >= 80%
    if (
        (stage_a_top1_rate >= 0.95 or stage_b_offset0_top1_rate >= 0.95)
        and (stage_a_mean_kl <= 0.10 or stage_b_offset0_mean_kl <= 0.10)
        and stage_b_multi_mean_kl <= 0.10
        and rollout_16_rate_b >= 0.80
    ):
        verdict = VERDICT_GREEN
    # YELLOW: H can fit immediate well, but continuation significantly worse, OR Pareto tradeoff
    elif (
        (stage_a_top1_rate >= 0.80 or stage_a_mean_kl <= 0.25)
        and (stage_b_offset0_mean_kl > stage_a_mean_kl * 1.5 or rollout_16_rate_b < 0.60)
    ):
        verdict = VERDICT_YELLOW
    # RED: immediate top-1 fails frequently, immediate KL remains large, or rollout collapses
    else:
        verdict = VERDICT_RED

    print("\n" + "=" * 65, flush=True)
    print(f"PHASE 1 ORACLE INVESTIGATION COMPLETE: VERDICT = {verdict}", flush=True)
    print("=" * 65, flush=True)
    print(f"Stage A Immediate (Offset 0): Top-1 = {stage_a_top1_rate*100:.2f}%, Mean KL = {stage_a_mean_kl:.4f} nats")
    print(f"Stage B Multi-Offset: Overall Top-1 = {stage_b_overall_top1_rate*100:.2f}%, Mean KL = {stage_b_multi_mean_kl:.4f} nats")
    print(f"Stage B Offset 0: Top-1 = {stage_b_offset0_top1_rate*100:.2f}%, KL = {stage_b_offset0_mean_kl:.4f} nats")
    print(f"Rollout-16 Agreement: Stage A = {rollout_16_rate_a*100:.2f}%, Stage B = {rollout_16_rate_b*100:.2f}%")
    print(f"Pareto Tradeoff (Stage B vs Stage A on Offset 0): KL Delta = {pareto_offset0_kl_delta:+.4f} nats, Top-1 Delta = {pareto_offset0_top1_delta*100:+.2f}%")
    print(f"Total Execution Time: {elapsed_total:.2f}s ({elapsed_total/60:.2f} min)")

    summary = {
        "schema": "tokens_oracle_h_phase1_summary_v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device": str(device),
        "elapsed_seconds": round(elapsed_total, 2),
        "num_examples_tested": len(per_example_results),
        "final_verdict": verdict,
        "base_parameter_hash_invariant": True,
        "base_parameter_hash": base_hash_before,
        "stage_a_metrics": {
            "immediate_top1_rate": stage_a_top1_rate,
            "immediate_mean_kl_nats": stage_a_mean_kl,
            "rollout_16_rate": rollout_16_rate_a,
        },
        "stage_b_metrics": {
            "overall_top1_rate": stage_b_overall_top1_rate,
            "multi_offset_mean_kl_nats": stage_b_multi_mean_kl,
            "offset0_top1_rate": stage_b_offset0_top1_rate,
            "offset0_mean_kl_nats": stage_b_offset0_mean_kl,
            "rollout_16_rate": rollout_16_rate_b,
            "rollout_32_rate": rollout_32_rate_b,
            "per_offset_top1": {str(k): v for k, v in per_offset_top1_agg.items()},
            "per_offset_mean_kl": {str(k): v for k, v in per_offset_kl_agg.items()},
        },
        "pareto_tradeoff": {
            "offset0_kl_delta_nats": round(pareto_offset0_kl_delta, 4),
            "offset0_top1_delta_pct": round(pareto_offset0_top1_delta * 100, 2),
            "shows_pareto_conflict": bool(pareto_offset0_kl_delta > 0.10 or pareto_offset0_top1_delta < -0.05),
        },
        "per_example_records": per_example_results,
    }

    out_file = output_dir / "oracle_h_phase1_summary.json"
    out_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved complete Phase 1 summary to {out_file}", flush=True)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Oracle H representation test")
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--output_dir", type=str, default="data/oracle_h_results")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stage_a_steps", type=int, default=300)
    parser.add_argument("--stage_b_steps", type=int, default=300)
    args = parser.parse_args()

    dev_benchmark_path = ROOT / "data" / "phrase_training_dataset" / "dev_phrases_benchmark_48.jsonl"
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, tokenizer = load_vanilla_phi(device)
    run_oracle_h_investigation(
        model,
        tokenizer,
        dev_benchmark_path,
        device,
        out_path,
        num_samples=args.num_samples,
        stage_a_steps=args.stage_a_steps,
        stage_b_steps=args.stage_b_steps,
    )


if __name__ == "__main__":
    main()
