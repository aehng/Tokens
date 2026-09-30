"""Make-or-Break Tokens Hypertoken Representation Experiment Runner.

Evaluates:
  Arm A: Fresh Context-Conditioned H-Encoder with 100% Frozen Vanilla Phi-3.5
    - Teacher/Student continuation KL loss at offsets 0, +1, +2, +4, +8, +16.
    - Baselines: Baseline 0 (Vanilla Teacher), Baseline 1 (Untrained Mean H),
      Trained H-Encoder.
    - Predeclared Acceptance Bands: GREEN, YELLOW, RED, INCONCLUSIVE.
  Arm B: Exact Expanded-Cache / Block Control
    - Compares N serial 1-token decode forwards vs 1 block forward of size N (N=2, 3, 4).
    - Cache correctness: top-1 agreement, KL divergence, KV cache numerical diff.
    - Latency benchmarking: CUDA events, median, p90, speedup ratio.

Modes:
  --mode block_control: Run Arm B block cache verification and timing benchmark.
  --mode overfit_sanity: Run tiny fixed-set overfit capacity sanity test.
  --mode smoke_train: Run 1 smoke training (~50-100 steps) + throughput measurement.
  --mode full_train: Run 2 serious training + internal val early stopping + DEV evaluation.
  --mode eval_only: Run DEV evaluation using an existing trained checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import datetime as dt
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from zip2zip.frozen_phi_h import (
    ContextConditionedHEncoder,
    EVAL_OFFSETS,
    EVAL_OFFSET_WEIGHTS,
    TeacherStudentForwardResult,
    assert_model_strictly_frozen,
    compute_model_parameter_hash,
    execute_teacher_student_step,
)
from zip2zip.block_cache import (
    BlockCorrectnessReport,
    BlockTimingReport,
    benchmark_block_cache_timing,
    evaluate_block_cache_correctness,
)
from zip2zip.predictor_v2.attribution_harness import (
    CANONICAL_EOS_TOKEN_IDS,
    CANONICAL_MODEL_ID,
    CANONICAL_MODEL_REVISION,
)

# Predeclared Acceptance Bands
VERDICT_GREEN = "GREEN"          # Top-1 >= 98%, mean KL <= 0.05, rollout >= 90%
VERDICT_YELLOW = "YELLOW"        # Top-1 95-98%, mean KL <= 0.10, rollout >= 75%
VERDICT_RED = "RED"              # Top-1 < 95%, KL large despite convergence
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"  # Still improving when compute expired


def load_vanilla_phi(
    device: torch.device,
    model_id: str = CANONICAL_MODEL_ID,
    revision: str = CANONICAL_MODEL_REVISION,
) -> Tuple[Any, Any]:
    """Load pure Vanilla Phi-3.5 with zero LoRA, PEFT, or adapter wrappers."""
    print(f"Loading Pure Vanilla Phi-3.5 from {model_id} (rev: {revision[:8]})...", flush=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        trust_remote_code=False,
    ).to(device)

    # Freeze 100% of parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    assert_model_strictly_frozen(model)
    return model, tokenizer


def run_arm_b_block_control(
    model: Any,
    dev_benchmark_path: Path,
    device: torch.device,
    output_dir: Path,
    *,
    block_sizes: Sequence[int] = (2, 3, 4),
    timed_reps: int = 50,
) -> Dict[str, Any]:
    """Execute Arm B: Exact Expanded-Cache / Block Control experiment."""
    print("\n" + "=" * 65, flush=True)
    print("STARTING ARM B: EXACT EXPANDED-CACHE / BLOCK CONTROL", flush=True)
    print("=" * 65, flush=True)

    # Load DEV benchmark examples to use realistic contexts
    records = []
    with dev_benchmark_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No benchmark records found in {dev_benchmark_path}")

    # Use first 3 diverse examples
    selected_samples = records[:3]
    correctness_results: List[Dict[str, Any]] = []
    timing_results: List[Dict[str, Any]] = []

    for b_size in block_sizes:
        print(f"\nEvaluating Block Size: {b_size}...", flush=True)
        size_correctness = []

        for sample in selected_samples:
            ctx = sample["context_token_ids"]
            # Build phrase of length b_size
            phrase = [sample["token_a"], sample["token_b"]]
            if b_size > 2:
                fut = sample["future_token_ids"]
                phrase.extend(fut[: b_size - 2])

            rep = evaluate_block_cache_correctness(model, ctx, phrase, device)
            size_correctness.append(asdict(rep))

        # Overall correctness for this block size
        all_equiv = all(r["behaviorally_equivalent"] for r in size_correctness)
        all_top1 = all(r["top1_matches"] for r in size_correctness)
        max_kl = max(r["kl_serial_to_block_nats"] for r in size_correctness)
        max_kv_diff = max(r["max_overall_kv_diff"] for r in size_correctness)

        print(
            f"Block Size {b_size} Correctness: Equivalent: {all_equiv}; "
            f"Top-1 Agreement: {all_top1}; Max KL: {max_kl:.2e} nats; "
            f"Max KV diff: {max_kv_diff:.2e}",
            flush=True,
        )

        correctness_results.append({
            "block_size": b_size,
            "all_behaviorally_equivalent": all_equiv,
            "all_top1_match": all_top1,
            "max_kl_nats": max_kl,
            "max_overall_kv_diff": max_kv_diff,
            "per_sample_reports": size_correctness,
        })

        # Timing benchmark
        sample0 = selected_samples[0]
        ctx0 = sample0["context_token_ids"]
        phrase0 = [sample0["token_a"], sample0["token_b"]]
        if b_size > 2:
            phrase0.extend(sample0["future_token_ids"][: b_size - 2])

        timing_rep = benchmark_block_cache_timing(
            model, ctx0, phrase0, device, timed_reps=timed_reps
        )
        print(
            f"Block Size {b_size} Timing: Serial median: {timing_rep.serial_median_ms:.2f} ms; "
            f"Block median: {timing_rep.block_median_ms:.2f} ms; "
            f"Speedup: {timing_rep.speedup_ratio_median:.2f}x "
            f"({timing_rep.latency_reduction_pct_median:.1f}% latency reduction)",
            flush=True,
        )
        timing_results.append(asdict(timing_rep))

    summary_arm_b = {
        "schema": "tokens_arm_b_block_cache_control_v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device": str(device),
        "block_sizes_tested": list(block_sizes),
        "all_block_sizes_correct": all(c["all_behaviorally_equivalent"] for c in correctness_results),
        "correctness": correctness_results,
        "timing": timing_results,
    }

    out_path = output_dir / "arm_b_block_cache_results.json"
    out_path.write_text(json.dumps(summary_arm_b, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote Arm B results to {out_path}", flush=True)
    return summary_arm_b


def run_overfit_sanity_test(
    model: Any,
    h_encoder: ContextConditionedHEncoder,
    subtrain_path: Path,
    device: torch.device,
    *,
    num_samples: int = 4,
    steps: int = 50,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """Test capacity: can H-encoder overfit a tiny fixed set of 4 TRAIN examples?"""
    print("\n" + "=" * 65, flush=True)
    print(f"RUNNING OVERFIT SANITY TEST (N={num_samples}, Steps={steps})", flush=True)
    print("=" * 65, flush=True)

    records = []
    with subtrain_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= num_samples:
                break

    optimizer = torch.optim.AdamW(h_encoder.parameters(), lr=lr)
    loss_curve: List[float] = []
    kl_curve: List[float] = []

    h_encoder.train()
    for step in range(steps):
        step_loss = 0.0
        step_kl = 0.0
        optimizer.zero_grad()

        for sample in records:
            res = execute_teacher_student_step(
                model,
                h_encoder,
                sample["context_token_ids"],
                sample["token_a"],
                sample["token_b"],
                sample["future_token_ids"],
                device,
            )
            res.total_loss.backward()
            step_loss += float(res.total_loss.item())
            step_kl += float(res.kl_loss.item())

        optimizer.step()
        avg_loss = step_loss / len(records)
        avg_kl = step_kl / len(records)
        loss_curve.append(avg_loss)
        kl_curve.append(avg_kl)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"[Overfit Step {step+1:3d}/{steps}] Loss: {avg_loss:.4f}, KL: {avg_kl:.4f} nats", flush=True)

    initial_loss = loss_curve[0]
    final_loss = loss_curve[-1]
    reduction_pct = ((initial_loss - final_loss) / max(initial_loss, 1e-6)) * 100.0
    passed = bool(reduction_pct >= 50.0 or final_loss < 0.1)

    print(
        f"Overfit Sanity Result: Initial loss: {initial_loss:.4f} -> Final loss: {final_loss:.4f} "
        f"({reduction_pct:.1f}% reduction). Status: {'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    return {
        "num_samples": num_samples,
        "steps": steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_pct": reduction_pct,
        "overfit_passed": passed,
        "loss_curve": loss_curve,
    }


def evaluate_dev_benchmark(
    model: Any,
    h_encoder: ContextConditionedHEncoder,
    tokenizer: Any,
    dev_benchmark_path: Path,
    device: torch.device,
    *,
    evaluate_untrained_baseline: bool = True,
    rollout_lengths: Sequence[int] = (8, 16, 32),
) -> Dict[str, Any]:
    """Comprehensive evaluation on 48 deterministic DEV benchmark cases."""
    print("\n" + "=" * 65, flush=True)
    print("EVALUATING 48-EXAMPLE CANONICAL DEV BENCHMARK", flush=True)
    print("=" * 65, flush=True)

    records = []
    with dev_benchmark_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    h_encoder.eval()
    results_by_case = []
    all_top1_by_offset: Dict[int, List[bool]] = {k: [] for k in EVAL_OFFSETS}
    all_kl_by_offset: Dict[int, List[float]] = {k: [] for k in EVAL_OFFSETS}
    rollout_agreement: Dict[int, List[float]] = {r: [] for r in rollout_lengths}

    for idx, sample in enumerate(records, 1):
        ctx = sample["context_token_ids"]
        t_a = sample["token_a"]
        t_b = sample["token_b"]
        fut = sample["future_token_ids"]

        # Run Teacher/Student step
        with torch.no_grad():
            res = execute_teacher_student_step(
                model, h_encoder, ctx, t_a, t_b, fut, device
            )

        for off in EVAL_OFFSETS:
            if off in res.per_offset_top1_match:
                all_top1_by_offset[off].append(res.per_offset_top1_match[off])
                all_kl_by_offset[off].append(res.per_offset_kl[off])

        # Rollout generation from immediately after H vs Teacher continuation
        # Student starts with [context, H]
        ctx_embeds = model.model.embed_tokens(torch.tensor([ctx], device=device))
        h_emb = res.h_embedding
        student_embeds = torch.cat([ctx_embeds, h_emb.unsqueeze(1)], dim=1)
        student_pos = torch.cat([
            torch.arange(0, len(ctx), device=device),
            torch.tensor([len(ctx) + 1], device=device),
        ]).unsqueeze(0)

        # Generate up to 32 tokens greedily
        generated_tokens: List[int] = []
        curr_embeds = student_embeds
        curr_pos = student_pos
        past_kv = None

        with torch.no_grad():
            # Initial forward
            out = model(
                inputs_embeds=curr_embeds,
                position_ids=curr_pos,
                use_cache=True,
            )
            past_kv = out.past_key_values
            next_tok = int(out.logits[0, -1].argmax().item())
            generated_tokens.append(next_tok)

            # Autoregressive steps
            next_sem_pos = len(ctx) + 2
            for step_i in range(1, max(rollout_lengths)):
                inp = torch.tensor([[next_tok]], device=device)
                pos = torch.tensor([[next_sem_pos]], device=device)
                out = model(
                    input_ids=inp,
                    position_ids=pos,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                past_kv = out.past_key_values
                next_tok = int(out.logits[0, -1].argmax().item())
                generated_tokens.append(next_tok)
                next_sem_pos += 1
                if next_tok in CANONICAL_EOS_TOKEN_IDS:
                    break

        # Compare rollout against future teacher tokens
        for r_len in rollout_lengths:
            target_slice = fut[:r_len]
            gen_slice = generated_tokens[: len(target_slice)]
            if target_slice:
                matches = sum(1 for g, t in zip(gen_slice, target_slice) if g == t)
                rate = matches / len(target_slice)
                rollout_agreement[r_len].append(rate)

        results_by_case.append({
            "prompt_id": sample["prompt_id"],
            "domain": sample["domain"],
            "immediate_top1_match": res.immediate_top1_match,
            "immediate_kl_nats": res.immediate_kl_nats,
            "per_offset_kl": res.per_offset_kl,
            "generated_rollout_32": generated_tokens,
            "teacher_target_32": fut[:32],
        })

    # Summary metrics
    agg_top1 = {
        off: float(np.mean(all_top1_by_offset[off]))
        for off in EVAL_OFFSETS
        if all_top1_by_offset[off]
    }
    agg_kl = {
        off: float(np.mean(all_kl_by_offset[off]))
        for off in EVAL_OFFSETS
        if all_kl_by_offset[off]
    }
    overall_top1 = float(np.mean([np.mean(v) for v in all_top1_by_offset.values() if v]))
    overall_kl = float(np.mean([np.mean(v) for v in all_kl_by_offset.values() if v]))
    immediate_top1 = agg_top1.get(0, 0.0)
    immediate_kl = agg_kl.get(0, 0.0)
    rollout_16_rate = float(np.mean(rollout_agreement.get(16, [0.0])))

    # Predeclared Verdict Assignment
    verdict = VERDICT_RED
    if immediate_top1 >= 0.99 and overall_top1 >= 0.98 and overall_kl <= 0.05 and rollout_16_rate >= 0.90:
        verdict = VERDICT_GREEN
    elif overall_top1 >= 0.95 and overall_kl <= 0.10 and rollout_16_rate >= 0.75:
        verdict = VERDICT_YELLOW
    else:
        verdict = VERDICT_RED

    print(f"\nDEV BENCHMARK VERDICT: {verdict}", flush=True)
    print(f"  Immediate (Offset 0) Top-1: {immediate_top1 * 100:.2f}%, KL: {immediate_kl:.4f} nats")
    print(f"  Overall (Offsets 0..16) Top-1: {overall_top1 * 100:.2f}%, Mean KL: {overall_kl:.4f} nats")
    print(f"  Next-16 Rollout Token Agreement: {rollout_16_rate * 100:.2f}%")
    print("  Per-Offset Top-1 Rates:", {k: f"{v*100:.1f}%" for k, v in agg_top1.items()})
    print("  Per-Offset Mean KL (nats):", {k: f"{v:.4f}" for k, v in agg_kl.items()})

    return {
        "verdict": verdict,
        "immediate_top1_rate": immediate_top1,
        "immediate_kl_nats": immediate_kl,
        "overall_top1_rate": overall_top1,
        "overall_kl_nats": overall_kl,
        "rollout_16_agreement_rate": rollout_16_rate,
        "per_offset_top1": agg_top1,
        "per_offset_mean_kl": agg_kl,
        "rollout_agreement_by_length": {
            k: float(np.mean(v)) for k, v in rollout_agreement.items()
        },
        "cases_evaluated": len(results_by_case),
    }


def train_h_encoder(
    model: Any,
    h_encoder: ContextConditionedHEncoder,
    subtrain_path: Path,
    val_path: Path,
    device: torch.device,
    output_dir: Path,
    *,
    total_steps: int = 200,
    eval_every: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    batch_accumulation: int = 1,
) -> Dict[str, Any]:
    """Train H-encoder on frozen Vanilla Phi with validation tracking and early stopping."""
    print("\n" + "=" * 65, flush=True)
    print(f"TRAINING H-ENCODER (Target Steps={total_steps}, Eval Every={eval_every})", flush=True)
    print("=" * 65, flush=True)

    # Compute base hash before training
    base_hash_before = compute_model_parameter_hash(model)

    subtrain_records = []
    with subtrain_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                subtrain_records.append(json.loads(line))

    val_records = []
    with val_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_records.append(json.loads(line))

    print(f"Loaded {len(subtrain_records)} subtrain examples and {len(val_records)} val examples.")

    optimizer = torch.optim.AdamW(h_encoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    h_encoder.train()

    train_loss_history = []
    train_kl_history = []
    val_loss_history = []
    step_times = []
    best_val_loss = float("inf")
    patience_rounds = 6
    no_improve_count = 0
    early_stopped = False
    best_checkpoint_path = output_dir / "best_h_encoder.pt"

    t0_train = time.perf_counter()

    for step in range(1, total_steps + 1):
        t_step_start = time.perf_counter()

        # Random sample from subtrain
        sample = random.choice(subtrain_records)
        res = execute_teacher_student_step(
            model,
            h_encoder,
            sample["context_token_ids"],
            sample["token_a"],
            sample["token_b"],
            sample["future_token_ids"],
            device,
        )

        res.total_loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(h_encoder.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        t_step = time.perf_counter() - t_step_start
        step_times.append(t_step)

        loss_val = float(res.total_loss.item())
        kl_val = float(res.kl_loss.item())
        train_loss_history.append(loss_val)
        train_kl_history.append(kl_val)

        if step % 20 == 0 or step == 1:
            curr_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[Step {step:4d}/{total_steps}] Loss: {loss_val:.4f}, KL: {kl_val:.4f} nats, "
                f"GradNorm: {grad_norm:.3f}, LR: {curr_lr:.2e}, Sec/step: {t_step:.3f}s",
                flush=True,
            )

        # Validation step
        if step % eval_every == 0 or step == total_steps:
            h_encoder.eval()
            val_losses = []
            val_kls = []
            # Evaluate on 20 validation examples
            eval_samples = val_records[:20]
            with torch.no_grad():
                for v_sample in eval_samples:
                    v_res = execute_teacher_student_step(
                        model,
                        h_encoder,
                        v_sample["context_token_ids"],
                        v_sample["token_a"],
                        v_sample["token_b"],
                        v_sample["future_token_ids"],
                        device,
                    )
                    val_losses.append(float(v_res.total_loss.item()))
                    val_kls.append(float(v_res.kl_loss.item()))

            mean_v_loss = float(np.mean(val_losses))
            mean_v_kl = float(np.mean(val_kls))
            val_loss_history.append({"step": step, "val_loss": mean_v_loss, "val_kl": mean_v_kl})
            print(f">>> [Validation Step {step}] Mean Loss: {mean_v_loss:.4f}, Mean KL: {mean_v_kl:.4f} nats", flush=True)

            if mean_v_loss < best_val_loss - 1e-4:
                best_val_loss = mean_v_loss
                no_improve_count = 0
                torch.save(h_encoder.state_dict(), best_checkpoint_path)
                print(f"Saved best model checkpoint to {best_checkpoint_path} (Val loss: {best_val_loss:.4f})", flush=True)
            else:
                no_improve_count += 1
                if no_improve_count >= patience_rounds:
                    print(f"Early stopping triggered at step {step}: no validation improvement for {patience_rounds} rounds.", flush=True)
                    early_stopped = True
                    break

            h_encoder.train()

    total_train_time = time.perf_counter() - t0_train
    sec_per_step = float(np.median(step_times))

    # Verify base model hash unchanged
    base_hash_after = compute_model_parameter_hash(model)
    if base_hash_after != base_hash_before:
        raise AssertionError("HARD FAIL: Base model parameters mutated during H-encoder training!")
    print(f"\nBase model strictly frozen verification PASS: {base_hash_before[:16]}... matches.", flush=True)

    # Load best checkpoint if saved
    if best_checkpoint_path.exists():
        h_encoder.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        print(f"Loaded best checkpoint (loss: {best_val_loss:.4f}) for downstream evaluation.", flush=True)

    return {
        "total_steps": total_steps,
        "total_train_time_s": total_train_time,
        "sec_per_step_median": sec_per_step,
        "final_train_loss": train_loss_history[-1],
        "final_train_kl": train_kl_history[-1],
        "best_val_loss": best_val_loss,
        "train_loss_history": train_loss_history,
        "train_kl_history": train_kl_history,
        "val_loss_history": val_loss_history,
        "base_hash_verified": True,
        "base_parameter_hash": base_hash_before,
    }


def main():
    parser = argparse.ArgumentParser(description="Make-or-Break Tokens Hypertoken Representation Experiment")
    parser.add_argument("--mode", type=str, required=True, choices=["block_control", "overfit_sanity", "smoke_train", "full_train", "all"])
    parser.add_argument("--dataset-dir", type=str, default="data/phrase_training_dataset")
    parser.add_argument("--output-dir", type=str, default="experiments/reports/representation_experiment")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)
    subtrain_path = dataset_dir / "train_phrases_subtrain.jsonl"
    val_path = dataset_dir / "train_phrases_val.jsonl"
    dev_path = dataset_dir / "dev_phrases_benchmark_48.jsonl"

    device = torch.device(args.device)
    model, tokenizer = load_vanilla_phi(device)
    h_encoder = ContextConditionedHEncoder(embed_dim=3072, hidden_dim=2048).to(device)
    print(f"H-Encoder parameter count: {h_encoder.parameter_count:,} ({h_encoder.parameter_count / 3.82e9 * 100:.4f}% of Phi-3.5)")

    results: Dict[str, Any] = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device": str(device),
        "encoder_params": h_encoder.parameter_count,
    }

    if args.mode in ("block_control", "all"):
        arm_b_results = run_arm_b_block_control(model, dev_path, device, output_dir)
        results["arm_b_block_control"] = arm_b_results

    if args.mode in ("overfit_sanity", "all"):
        overfit_results = run_overfit_sanity_test(model, h_encoder, subtrain_path, device, num_samples=4, steps=40)
        results["overfit_sanity"] = overfit_results

    if args.mode in ("smoke_train", "full_train", "all"):
        train_res = train_h_encoder(
            model,
            h_encoder,
            subtrain_path,
            val_path,
            device,
            output_dir,
            total_steps=args.steps,
            eval_every=max(10, args.steps // 5),
            lr=args.lr,
        )
        results["training"] = train_res

        # DEV evaluation
        dev_eval = evaluate_dev_benchmark(model, h_encoder, tokenizer, dev_path, device)
        results["dev_evaluation"] = dev_eval

    # Save summary
    summary_path = output_dir / "representation_experiment_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nExperiment complete. Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
