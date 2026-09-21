"""
Phase 1: Pure Predictive Hypertoken Calibration Training (K=32).
Calibrates the model's dynamic LM head (output_encoder) on CPU to emit
pre-seeded hypertokens during decode without quality loss.

Key Guarantees:
1. Strict Prompt-Only Causality: Uses data pre-segmented strictly using prompt tokens.
2. Zero Backbone Drift: base_model is 100% frozen, preserving math, code, and syntax.
3. Zero Graph Leakage: Uses static_mgr.reset(clear_caches=True) between steps.
4. Watchdog Logging: Logs step time, loss, gradient norms, and interim decode hypertoken emission.
"""

import argparse
import json
import os
import pickle
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, LogitsProcessorList

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from zip2zip import Zip2ZipModel, StaticCodebookManager
from src.evaluation.offline_segmenter import segment_tokens_dp


def evaluate_interim(
    model: Zip2ZipModel,
    predictor,
    tokenizer,
    val_records: List[Dict],
    initial_vocab_size: int = 32011,
    budget_k: int = 32,
    max_new_tokens: int = 25,
    device: str = "cpu",
    dim: int = 3072,
    pad_token_id: int = 32000,
    disabled_ids: List[int] = None,
) -> Dict:
    """Evaluate live greedy generation on a subset of held-out validation prompts."""
    model.eval()
    print("\n--- RUNNING INTERIM VALIDATION EVALUATION ---")

    total_base_steps = 0
    total_pred_steps = 0
    total_hypers_emitted = 0
    total_base_expanded_tokens = 0

    results = []

    with torch.no_grad():
        for i, record in enumerate(val_records):
            p_id = record["id"]
            domain = record["domain"]
            prompt_ids = record["prompt_token_ids"]
            base_len = len(prompt_ids)

            # Predictor codebook (prompt-only)
            p_dict, _ = predictor.select_prompt_conditioned(prompt_ids, budget=budget_k)
            pred_phrases = list(p_dict.keys())
            comp_len, tiles, _ = segment_tokens_dp(prompt_ids, set(pred_phrases))
            seeded_dict = {p: initial_vocab_size + i for i, p in enumerate(pred_phrases)}
            resegmented = [tile[0] if len(tile) == 1 else seeded_dict[tile] for tile in tiles]

            static_mgr = StaticCodebookManager(
                initial_vocab_size=initial_vocab_size,
                max_codebook_size=budget_k,
                max_subtokens=3,
                embedding_dim=dim,
                pad_token_id=pad_token_id,
                disabled_ids=disabled_ids,
            )
            static_mgr.set_seeded_codebook(seeded_dict, batch_size=1, device=torch.device(device))
            static_mgr.attach_to_model(model)

            pred_tensor = torch.tensor([resegmented], dtype=torch.long, device=device)
            logits_proc = LogitsProcessorList([static_mgr.get_logits_processor()])

            out_pred = model.generate(
                input_ids=pred_tensor,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_proc,
                do_sample=False,
            )
            static_mgr.detach_from_model(model)

            gen_tokens = out_pred[0][len(resegmented) :].tolist()
            pred_steps = len(gen_tokens)
            hypers_in_gen = [t for t in gen_tokens if t >= initial_vocab_size]

            # Expand generated tokens to count equivalent base tokens
            expanded_tokens = []
            for tid in gen_tokens:
                if tid in static_mgr.hyper_to_subtokens:
                    expanded_tokens.extend(static_mgr.hyper_to_subtokens[tid])
                else:
                    expanded_tokens.append(tid)

            step_savings = len(expanded_tokens) - pred_steps
            total_pred_steps += pred_steps
            total_hypers_emitted += len(hypers_in_gen)
            total_base_expanded_tokens += len(expanded_tokens)

            print(
                f"  [Probe {i+1}/{len(val_records)}] {p_id:10s} ({domain:11s}) | "
                f"Steps: {pred_steps:2d} | Emitted Hypers: {len(hypers_in_gen):2d} | "
                f"Expanded: {len(expanded_tokens):2d} | Saved: {step_savings:2d}",
                flush=True,
            )

            results.append({
                "id": p_id,
                "domain": domain,
                "pred_steps": pred_steps,
                "expanded_tokens": len(expanded_tokens),
                "hypers_emitted": len(hypers_in_gen),
                "step_savings": step_savings,
            })

    model.train()
    for p in model.base_model.parameters():
        p.requires_grad = False
    for p in model.input_encoder.parameters():
        p.requires_grad = False
    for p in model.output_encoder.parameters():
        p.requires_grad = True
    model.base_model.lm_head.detach_input = True

    overall_savings_pct = (
        (total_base_expanded_tokens - total_pred_steps) / total_base_expanded_tokens * 100.0
        if total_base_expanded_tokens > 0
        else 0.0
    )

    print(f"Validation Summary ({len(val_records)} prompts):")
    print(f"  Expanded Base Tokens: {total_base_expanded_tokens}")
    print(f"  Actual Decode Steps:  {total_pred_steps}")
    print(f"  Decode Step Savings:  {overall_savings_pct:.2f}%")
    print(f"  Total Hypertokens Emitted: {total_hypers_emitted} ({total_hypers_emitted/len(val_records):.1f} per sample)")
    print("-" * 50)

    return {
        "total_prompts": len(val_records),
        "expanded_tokens": total_base_expanded_tokens,
        "decode_steps": total_pred_steps,
        "step_savings_pct": overall_savings_pct,
        "hypers_emitted": total_hypers_emitted,
        "per_sample": results,
    }


def train_predictive_calibration(
    train_data_path: str = "data/cached_pure_pred_train_2k.pkl",
    val_data_path: str = "data/cached_pure_pred_val_60.json",
    predictor_path: str = "experiments/checkpoints/cached_predictor.pkl",
    checkpoint_dir: str = "experiments/checkpoints",
    budget_k: int = 32,
    max_steps: int = 100,
    grad_accum_steps: int = 4,
    lr: float = 5e-5,
    eval_every: int = 50,
    save_every: int = 50,
    device: str = "cpu",
):
    print("=" * 80)
    print("PHASE 1: PURE PREDICTIVE CALIBRATION TRAINING (K=32)")
    print(f"Max Steps: {max_steps}, Grad Accum: {grad_accum_steps}, LR: {lr}, Device: {device}")
    print("=" * 80, flush=True)

    torch.set_num_threads(8)
    os.makedirs(checkpoint_dir, exist_ok=True)
    initial_vocab_size = 32011

    # 1. Load Tokenizer & Predictor
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    with open(predictor_path, "rb") as f:
        predictor = pickle.load(f)

    # 2. Load Cached Training & Validation Data
    print(f"Loading training data from {train_data_path}...", flush=True)
    with open(train_data_path, "rb") as f:
        train_samples = pickle.load(f)
    print(f"Loaded {len(train_samples)} training samples.", flush=True)

    with open(val_data_path, "r", encoding="utf-8") as f:
        val_records = json.load(f)
    # Stratified 3-prompt probe: 1 Code (MBPP), 1 Instruction (Alpaca), 1 Math (GSM8k)
    interim_val_records = [val_records[0], val_records[20], val_records[40]]

    # 3. Load Zip2Zip Model (Mixed Precision: float16 base for speed & low RAM, float32 output_encoder for CPU autograd)
    print("Loading Zip2Zip model on CPU (float16 backbone)...", flush=True)
    t0 = time.time()
    model = Zip2ZipModel.from_pretrained(
        "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1",
        max_codebook_size=budget_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    print(f"Model loaded in {time.time() - t0:.2f}s.", flush=True)

    dim = model.zip2zip_config.encoder.hidden_size
    pad_id = 32000
    disabled_ids = list(model.zip2zip_config.compression.disabled_ids)

    # 4. Freeze Backbone & Input Encoder; Train Output Encoder Only (cast to float32 for CPU autograd)
    model.train()
    model.output_encoder.to(torch.float32)
    for p in model.base_model.parameters():
        p.requires_grad = False
    for p in model.input_encoder.parameters():
        p.requires_grad = False
    for p in model.output_encoder.parameters():
        p.requires_grad = True

    model.base_model.lm_head.detach_input = True

    trainable_params = [p for p in model.output_encoder.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters (output_encoder only): {num_trainable:,} ({num_trainable/1e6:.2f}M)", flush=True)

    # 5. Optimizer & Scheduler
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-6)

    # 6. Pre-training Baseline Evaluation (Step 0)
    print("\n--- STEP 0 ZERO-SHOT BASELINE CHECK ---", flush=True)
    step0_val = evaluate_interim(
        model,
        predictor,
        tokenizer,
        interim_val_records,
        initial_vocab_size=initial_vocab_size,
        budget_k=budget_k,
        device=device,
        dim=dim,
        pad_token_id=pad_id,
        disabled_ids=disabled_ids,
    )

    # 7. Training Loop
    training_log = []
    running_loss = 0.0
    accumulated_loss = 0.0
    t_start_train = time.time()
    step = 0
    sample_idx = 0

    optimizer.zero_grad()

    while step < max_steps:
        sample = train_samples[sample_idx % len(train_samples)]
        sample_idx += 1

        input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
        labels = torch.tensor([sample["labels"]], dtype=torch.long, device=device)
        seeded_dict = sample["seeded_dict"]

        # Setup StaticCodebookManager
        static_mgr = StaticCodebookManager(
            initial_vocab_size=initial_vocab_size,
            max_codebook_size=budget_k,
            max_subtokens=3,
            embedding_dim=dim,
            pad_token_id=pad_id,
            disabled_ids=disabled_ids,
        )
        static_mgr.set_seeded_codebook(seeded_dict, batch_size=1, device=torch.device(device))
        static_mgr.attach_to_model(model)

        # Clear autograd caches to prevent graph reuse error
        static_mgr.reset(clear_caches=True)

        t_step0 = time.perf_counter()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / grad_accum_steps
        loss.backward()
        t_step1 = time.perf_counter()

        step_elapsed_ms = (t_step1 - t_step0) * 1000.0
        accumulated_loss += outputs.loss.item()
        running_loss += outputs.loss.item()

        static_mgr.detach_from_model(model)

        if sample_idx % grad_accum_steps == 0:
            step += 1
            grad_norm = nn.utils.clip_grad_norm_(trainable_params, 1.0).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            avg_loss = accumulated_loss / grad_accum_steps
            accumulated_loss = 0.0

            log_entry = {
                "step": step,
                "loss": avg_loss,
                "grad_norm": grad_norm,
                "lr": scheduler.get_last_lr()[0],
                "step_ms": step_elapsed_ms,
            }
            training_log.append(log_entry)

            print(
                f"Step {step:4d}/{max_steps} | Loss: {avg_loss:.4f} | "
                f"Grad Norm: {grad_norm:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"Step Time: {step_elapsed_ms:.1f}ms",
                flush=True,
            )

            # Interim Evaluation
            if step % eval_every == 0 or step == max_steps:
                val_res = evaluate_interim(
                    model,
                    predictor,
                    tokenizer,
                    interim_val_records,
                    initial_vocab_size=initial_vocab_size,
                    budget_k=budget_k,
                    device=device,
                    dim=dim,
                    pad_token_id=pad_id,
                    disabled_ids=disabled_ids,
                )
                log_entry["interim_val"] = val_res

            # Checkpoint
            if step % save_every == 0 or step == max_steps:
                ckpt_path = os.path.join(checkpoint_dir, f"pure_pred_k32_step{step}.pt")
                torch.save(
                    {
                        "step": step,
                        "output_encoder_state_dict": model.output_encoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_loss,
                    },
                    ckpt_path,
                )
                print(f"--> Saved checkpoint to {ckpt_path}", flush=True)

    total_train_sec = time.time() - t_start_train
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE: {max_steps} steps in {total_train_sec:.2f}s ({total_train_sec/max_steps:.2f}s/step)")
    print("=" * 80, flush=True)

    # Save training log
    log_path = os.path.join(checkpoint_dir, "pure_pred_k32_stageA_training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "step0_baseline": step0_val,
                "training_log": training_log,
                "total_time_sec": total_train_sec,
            },
            f,
            indent=2,
        )
    print(f"Saved full training log to {log_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    train_predictive_calibration(
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
