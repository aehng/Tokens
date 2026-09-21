"""One-Command Predictive Joint Training Script (Phase 16).

Launches joint training of:
1. Zip2Zip LoRA adapters (r=32)
2. input_encoder (hyper-embedding synthesis)
3. output_encoder (hyper-linear projection)
With the customer's base Phi-3.5 model weights 100% frozen.

Usage:
  python experiments/train_predictive_zip2zip.py --config configs/predictive_joint_pilot.yaml
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
from typing import Dict, List, Any

import torch
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from zip2zip import Zip2ZipModel, Zip2ZipConfig
from zip2zip.predictor_policy import CappedPredictorPolicy
from zip2zip.predictive_pipeline import PredictivePipeline
from zip2zip.training_objectives import (
    configure_joint_training_parameters,
    DifferentiableTrainingManager,
)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(config_path: str, max_steps_override: int = None, device_str: str = None):
    cfg = load_config(config_path)
    if max_steps_override:
        cfg["training"]["max_steps"] = max_steps_override

    # Device selection
    if device_str is None:
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device_str = "xpu"
        else:
            device_str = "cpu"
    device = torch.device(device_str)

    print(f"\n{'='*80}")
    print(f"PREDICTIVE ZIP2ZIP JOINT TRAINING PILOT")
    print(f"Device: {device} | Config: {config_path}")
    print(f"{'='*80}\n")

    # 1. Load tokenizer & predictor
    model_name = cfg["model"]["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

    predictor_path = "experiments/checkpoints/cached_predictor.pkl"
    with open(predictor_path, "rb") as f:
        raw_predictor = pickle.load(f)
    p_index = getattr(raw_predictor, "index", raw_predictor)

    policy = CappedPredictorPolicy(
        p_index,
        tokenizer,
        budget=cfg["compression"]["budget_k"],
        max_structural_slots=cfg["policy"]["max_structural_slots"],
    )
    pipeline = PredictivePipeline(
        policy,
        tokenizer,
        max_codebook_size=cfg["compression"]["budget_k"],
        max_subtokens=cfg["compression"]["max_subtokens"],
    )

    # 2. Load model
    dtype = torch.bfloat16 if cfg["model"]["torch_dtype"] == "bfloat16" and device.type != "cpu" else torch.float32
    print(f"Loading Zip2Zip model from {model_name} (dtype={dtype})...")
    model = Zip2ZipModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if device.type != "cpu":
        model.to(device)

    # 3. Configure joint trainable parameters
    report = configure_joint_training_parameters(model)
    print("\nPARAMETER CONFIGURATION:")
    print(f"  Total Parameters:       {report['total_parameters']:,}")
    print(f"  Trainable Parameters:   {report['trainable_parameters']:,} ({report['trainable_percentage']}%)")
    print(f"  Base Model (FROZEN):    {report['base_frozen_params']:,}")
    print(f"  LoRA Adapters (TRAIN):  {report['lora_trainable_params']:,}")
    print(f"  Input Encoder (TRAIN):  {report['input_encoder_params']:,}")
    print(f"  Output Encoder (TRAIN): {report['output_encoder_params']:,}\n")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    lr = float(cfg["training"]["learning_rate"])
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    training_mgr = DifferentiableTrainingManager(
        model,
        initial_vocab_size=cfg["compression"]["initial_vocab_size"],
        max_codebook_size=cfg["compression"]["budget_k"],
        max_subtokens=cfg["compression"]["max_subtokens"],
    )

    # Load training dataset
    train_file = "data/train.jsonl"
    print(f"Streaming training data from {train_file}...")
    with open(train_file, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line.strip()) for line in f]
    print(f"Loaded {len(raw_samples):,} training samples.")

    max_steps = cfg["training"]["max_steps"]
    grad_accum_steps = cfg["training"]["gradient_accumulation_steps"]
    recon_weight = float(cfg["loss"]["reconstruction_weight"])
    output_dir = cfg["checkpointing"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting training run ({max_steps} steps, grad_accum={grad_accum_steps})...\n")

    model.train()
    optimizer.zero_grad()
    step = 0
    accum_loss = 0.0
    accum_lm = 0.0
    accum_recon = 0.0
    sample_idx = 0
    t_start = time.time()

    while step < max_steps:
        s = raw_samples[sample_idx % len(raw_samples)]
        sample_idx += 1

        # Curriculum density & structural slots
        curr_density = 1.0
        curr_structural_slots = cfg["policy"].get("initial_structural_slots", 0)
        if cfg["curriculum"]["enabled"]:
            for stage in cfg["curriculum"]["stages"]:
                if step >= stage["step"]:
                    curr_density = stage["density"]
                    if "max_structural_slots" in stage:
                        curr_structural_slots = stage["max_structural_slots"]
        policy.max_structural_slots = curr_structural_slots

        try:
            processed = pipeline.process_sample(
                s["prompt"],
                s["response"],
                domain=s.get("domain", "general"),
                curriculum_density=curr_density,
            )
        except Exception:
            continue

        input_ids = torch.tensor([processed["input_ids"]], dtype=torch.long)
        labels = torch.tensor([processed["labels"]], dtype=torch.long)
        codebook_dict = {
            eval(k) if isinstance(k, str) else k: v
            for k, v in processed["codebook_dict"].items()
        }
        codebook_tensor = processed["codebook_tensor"]

        # Forward step (differentiable)
        loss, metrics = training_mgr.forward_step(
            input_ids=input_ids,
            labels=labels,
            codebook_dict=codebook_dict,
            codebook_tensor=codebook_tensor,
            recon_weight=recon_weight,
            device=device,
        )

        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        accum_loss += metrics["total_loss"] / grad_accum_steps
        accum_lm += metrics["lm_loss"] / grad_accum_steps
        accum_recon += metrics["recon_loss"] / grad_accum_steps

        if sample_idx % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                trainable_params, float(cfg["training"]["max_grad_norm"])
            )
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            # Log
            elapsed = time.time() - t_start
            rate = step / elapsed if elapsed > 0 else 0.0
            print(
                f"Step {step:4d}/{max_steps} | Loss: {accum_loss:6.4f} (LM: {accum_lm:6.4f}, Recon: {accum_recon:6.4f}) "
                f"| Dens: {curr_density:.2f} | Rate: {rate:4.2f} steps/s",
                flush=True,
            )

            # Checkpoint
            if step % cfg["checkpointing"]["save_steps"] == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
                # Save trainable weights (LoRA + input_encoder + output_encoder)
                lora_state = {
                    k: v for k, v in model.base_model.named_parameters() if "lora" in k.lower()
                }
                torch.save(
                    {
                        "step": step,
                        "lora_state_dict": lora_state,
                        "input_encoder_state_dict": model.input_encoder.state_dict(),
                        "output_encoder_state_dict": model.output_encoder.state_dict(),
                        "loss": accum_loss,
                        "config": cfg,
                    },
                    ckpt_path,
                )
                print(f"  [Checkpoint saved to {ckpt_path}]", flush=True)

            accum_loss = 0.0
            accum_lm = 0.0
            accum_recon = 0.0

    print(f"\nTraining completed in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train predictive Zip2Zip jointly.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/predictive_joint_pilot.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, xpu)")
    args = parser.parse_args()

    train(args.config, max_steps_override=args.max_steps, device_str=args.device)
