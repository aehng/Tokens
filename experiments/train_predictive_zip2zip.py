"""Predictive Joint Training & CPU Feasibility Probe Script.

Implements Phase 1–5:
1. Mixed-precision CPU execution:
   - Base model weights: FROZEN in float16
   - Trainable LoRA adapters: float32
   - Trainable input_encoder: float32
   - Trainable output_encoder: float32
2. RAM Preflight Audit:
   - System total, available, process RSS
   - Analytical parameter, gradient, optimizer state memory
3. Safety guards:
   - Base weight SHA256 byte-hash verification before/after steps
   - Base weight gradient assertions (must be None)
   - Finite loss and gradient assertions
4. Stepwise escalation probe:
   Stage A: 1 step -> Stage B: 3 steps -> Stage C: 5 steps -> Stage D: up to 10 steps max.
5. High-resolution timing instrumentation (prep, codebook, forward, backward, clip, opt).
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

import psutil
import torch
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from zip2zip import Zip2ZipModel, Zip2ZipConfig
from zip2zip.predictor_policy import CappedPredictorPolicy, classify_phrase
from zip2zip.predictive_pipeline import PredictivePipeline
from zip2zip.training_objectives import (
    configure_joint_training_parameters,
    DifferentiableTrainingManager,
)


def get_ram_info() -> Dict[str, float]:
    """Get system RAM and current process RSS in GB."""
    vm = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    return {
        "total_gb": round(vm.total / (1024 ** 3), 2),
        "available_gb": round(vm.available / (1024 ** 3), 2),
        "used_gb": round(vm.used / (1024 ** 3), 2),
        "percent": vm.percent,
        "process_rss_gb": round(proc.memory_info().rss / (1024 ** 3), 2),
    }


def compute_tensor_hash(tensor: torch.Tensor) -> str:
    """Compute SHA256 hash of tensor bytes."""
    data = tensor.detach().cpu().contiguous()
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()[:16]


def move_training_model_to_device(model: torch.nn.Module, device: str | torch.device) -> torch.nn.Module:
    """Move the fully configured training model to one explicit device."""
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {target} was requested but CUDA is unavailable")
    model.to(target)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    return model


def estimate_minimum_training_memory_bytes(model: torch.nn.Module) -> Dict[str, int]:
    """Lower-bound model, gradient, and AdamW storage; activations are excluded."""
    frozen_parameter_bytes = 0
    trainable_parameter_bytes = 0
    gradient_bytes = 0
    adamw_state_bytes = 0
    for parameter in model.parameters():
        parameter_bytes = parameter.numel() * parameter.element_size()
        if parameter.requires_grad:
            trainable_parameter_bytes += parameter_bytes
            gradient_bytes += parameter_bytes
            adamw_state_bytes += 2 * parameter_bytes
        else:
            frozen_parameter_bytes += parameter_bytes
    return {
        "frozen_parameter_bytes": frozen_parameter_bytes,
        "trainable_parameter_bytes": trainable_parameter_bytes,
        "gradient_bytes": gradient_bytes,
        "adamw_state_bytes": adamw_state_bytes,
        "static_minimum_bytes": (
            frozen_parameter_bytes
            + trainable_parameter_bytes
            + gradient_bytes
            + adamw_state_bytes
        ),
    }


def get_base_weight_hashes(model: Zip2ZipModel) -> Dict[str, str]:
    """Compute verification hashes for representative base model tensors."""
    hashes = {}
    base = model.base_model

    # PEFT and distributed wrappers may put one or more shells around the
    # causal LM. Prefer their explicit unwrapping APIs, then inspect common
    # wrapper attributes while looking for the transformer block structure.
    pending = [base]
    seen = set()
    backbone = None
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if hasattr(candidate, "layers"):
            backbone = candidate
            break

        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                pending.append(get_base_model())
            except (AttributeError, RuntimeError, TypeError):
                pass
        for attr in ("module", "base_model", "model"):
            child = getattr(candidate, attr, None)
            if child is not None:
                pending.append(child)

    excluded_ids = {
        id(param)
        for encoder_name in ("input_encoder", "output_encoder")
        for encoder in (getattr(model, encoder_name, None),)
        if encoder is not None
        for param in encoder.parameters()
    }
    lora_ids = {
        id(param)
        for name, param in base.named_parameters(remove_duplicate=False)
        if "lora" in name.lower()
    }

    def record(name: str, weight: Optional[torch.Tensor]) -> None:
        if weight is not None and id(weight) not in excluded_ids | lora_ids:
            hashes[name] = compute_tensor_hash(weight)

    # HyperEmbedding/HyperLinear retain the original backbone Parameter. Hash
    # that tensor, but exclude any parameter aliased to a trainable encoder.
    embedding = getattr(backbone, "embed_tokens", None) if backbone is not None else None
    record("embed_tokens", getattr(embedding, "weight", None))

    # Layer 0, Layer 16, Layer 31 projection weights
    layers = getattr(backbone, "layers", []) if backbone is not None else []
    for l_idx in [0, 16, 31]:
        if l_idx < len(layers):
            layer = layers[l_idx]
            self_attn = getattr(layer, "self_attn", None)
            qkv = getattr(self_attn, "qkv_proj", None)
            if qkv is not None:
                base_w = getattr(qkv, "base_layer", qkv)
                record(f"layer_{l_idx}_qkv_base", getattr(base_w, "weight", None))

    if not any(name.startswith("layer_") for name in hashes):
        raise RuntimeError(
            "Could not select any frozen transformer-layer tensors to hash; refusing to "
            "continue with vacuous frozen-weight checks."
        )
    return hashes


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preflight_memory_check(
    model: Zip2ZipModel,
    trainable_mode: str = "joint",
) -> Dict[str, Any]:
    """Calculate exact memory requirements and check against available RAM."""
    ram = get_ram_info()

    # Parameter counts
    base_frozen_params = sum(
        p.numel() for n, p in model.base_model.named_parameters() if "lora" not in n.lower()
    )
    lora_params = sum(
        p.numel() for n, p in model.base_model.named_parameters() if "lora" in n.lower()
    )
    in_enc_params = sum(p.numel() for p in model.input_encoder.parameters())
    out_enc_params = (
        sum(p.numel() for p in model.output_encoder.parameters())
        if getattr(model, "output_encoder", None) is not None
        else 0
    )

    if trainable_mode == "joint":
        trainable_params = lora_params + in_enc_params + out_enc_params
    elif trainable_mode == "encoders_only":
        trainable_params = in_enc_params + out_enc_params
    else:
        raise ValueError(f"Unknown trainable_mode: {trainable_mode}")

    # Memory in bytes
    # Base is fp16 (2 bytes), trainable is fp32 (4 bytes)
    base_mem_gb = (base_frozen_params * 2) / (1024 ** 3)
    trainable_param_mem_gb = (trainable_params * 4) / (1024 ** 3)
    total_param_mem_gb = base_mem_gb + trainable_param_mem_gb

    # Gradients: float32 (4 bytes per trainable param)
    grad_mem_gb = (trainable_params * 4) / (1024 ** 3)

    # AdamW state: m (4 bytes) + v (4 bytes) = 8 bytes per trainable param
    adamw_state_gb = (trainable_params * 8) / (1024 ** 3)

    # Estimated total training footprint (params + grads + optimizer)
    total_estimated_gb = total_param_mem_gb + grad_mem_gb + adamw_state_gb

    report = {
        "system_ram_total_gb": ram["total_gb"],
        "system_ram_available_gb": ram["available_gb"],
        "process_rss_gb": ram["process_rss_gb"],
        "base_frozen_params": base_frozen_params,
        "base_mem_fp16_gb": round(base_mem_gb, 2),
        "lora_params": lora_params,
        "input_encoder_params": in_enc_params,
        "output_encoder_params": out_enc_params,
        "trainable_params": trainable_params,
        "trainable_param_mem_fp32_gb": round(trainable_param_mem_gb, 2),
        "total_param_mem_gb": round(total_param_mem_gb, 2),
        "estimated_grad_mem_gb": round(grad_mem_gb, 2),
        "estimated_adamw_mem_gb": round(adamw_state_gb, 2),
        "total_estimated_gb": round(total_estimated_gb, 2),
        "is_safe_for_adamw": total_estimated_gb < ram["total_gb"] * 0.95,
    }
    return report


def run_cpu_probe(
    config_path: str = "configs/predictive_joint_pilot.yaml",
    trainable_mode: str = "joint",
    target_steps: int = 10,
    device_str: str = "cpu",
    output_dir: str = "experiments/checkpoints/predictive_joint_pilot",
) -> Dict[str, Any]:
    """Execute purpose-built CPU probe with stepwise escalation."""
    cfg = load_config(config_path)
    device = torch.device(device_str)

    print(f"\n{'='*80}")
    print(f"PREDICTIVE ZIP2ZIP CPU TIMING & DIRECTIONALITY PROBE")
    print(f"Trainable Mode: {trainable_mode.upper()} | Target Steps: {target_steps} | Device: {device}")
    print(f"{'='*80}\n")

    # 1. Print resolved config
    print("RESOLVED CONFIGURATION:")
    for section, values in cfg.items():
        print(f"  [{section}]")
        if isinstance(values, dict):
            for k, v in values.items():
                print(f"    {k}: {v}")
        else:
            print(f"    {values}")
    print()

    # 2. Tokenizer & Predictor Policy
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
        allow_numeric=cfg["policy"].get("allow_numeric", True),
        filter_bare_punctuation=cfg["policy"].get("filter_bare_punctuation", True),
    )
    pipeline = PredictivePipeline(
        policy,
        tokenizer,
        max_codebook_size=cfg["compression"]["budget_k"],
        max_subtokens=cfg["compression"]["max_subtokens"],
    )

    # 3. Model Loading with Mixed Precision (Backbone float16)
    print(f"Loading base Zip2Zip model (backbone float16)...")
    t_load_start = time.time()
    model = Zip2ZipModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    t_load = time.time() - t_load_start
    print(f"Model loaded in {t_load:.2f}s.")

    # 4. Cast trainable modules to float32 for CPU backward compatibility
    print("Casting trainable modules to float32 on CPU...")
    model.input_encoder.to(torch.float32)
    if getattr(model, "output_encoder", None) is not None:
        model.output_encoder.to(torch.float32)

    if trainable_mode == "joint":
        for name, param in model.base_model.named_parameters():
            if "lora" in name.lower():
                param.data = param.data.to(torch.float32)
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif trainable_mode == "encoders_only":
        for name, param in model.base_model.named_parameters():
            param.requires_grad = False
    else:
        raise ValueError(f"Unknown trainable_mode: {trainable_mode}")

    # Set requires_grad for encoders
    for p in model.input_encoder.parameters():
        p.requires_grad = True
    if getattr(model, "output_encoder", None) is not None:
        for p in model.output_encoder.parameters():
            p.requires_grad = True

    # Un-detach lm_head
    if hasattr(model.base_model, "get_output_embeddings"):
        out_emb = model.base_model.get_output_embeddings()
        if hasattr(out_emb, "detach_input"):
            out_emb.detach_input = False

    # 5. Preflight Memory Audit
    preflight = preflight_memory_check(model, trainable_mode=trainable_mode)
    print("\n" + "="*50)
    print("PREFLIGHT RAM & PARAMETER AUDIT:")
    print(f"  Physical RAM Total:        {preflight['system_ram_total_gb']} GB")
    print(f"  Physical RAM Available:    {preflight['system_ram_available_gb']} GB")
    print(f"  Current Process RSS:       {preflight['process_rss_gb']} GB")
    print(f"  Base Model (FROZEN fp16):  {preflight['base_frozen_params']:,} params ({preflight['base_mem_fp16_gb']} GB)")
    print(f"  LoRA Adapters:             {preflight['lora_params']:,} params")
    print(f"  Input Encoder:             {preflight['input_encoder_params']:,} params")
    print(f"  Output Encoder:            {preflight['output_encoder_params']:,} params")
    print(f"  Total Trainable (fp32):    {preflight['trainable_params']:,} params ({preflight['trainable_param_mem_fp32_gb']} GB)")
    print(f"  Estimated Gradients:       {preflight['estimated_grad_mem_gb']} GB")
    print(f"  Estimated AdamW State:     {preflight['estimated_adamw_mem_gb']} GB")
    print(f"  Total Estimated Footprint: {preflight['total_estimated_gb']} GB")
    print("="*50 + "\n")

    # Record initial base tensor hashes
    base_hashes_initial = get_base_weight_hashes(model)
    print(f"Recorded initial base weight hashes: {list(base_hashes_initial.keys())}")

    # Load training data
    train_file = "data/train.jsonl"
    with open(train_file, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line.strip()) for line in f]

    training_mgr = DifferentiableTrainingManager(
        model,
        initial_vocab_size=cfg["compression"]["initial_vocab_size"],
        max_codebook_size=cfg["compression"]["budget_k"],
        max_subtokens=cfg["compression"]["max_subtokens"],
    )
    recon_weight = float(cfg["loss"]["reconstruction_weight"])
    max_grad_norm = float(cfg["training"]["max_grad_norm"])

    # -------------------------------------------------------------
    # DIAGNOSTIC PASS 1: FORWARD-ONLY TIMING & MEMORY
    # -------------------------------------------------------------
    print("\n--- DIAGNOSTIC PASS 1: FORWARD-ONLY ---")
    diag_sample = raw_samples[0]
    p_proc = pipeline.process_sample(
        diag_sample["prompt"],
        diag_sample["response"],
        domain=diag_sample.get("domain", "general"),
    )
    d_input_ids = torch.tensor([p_proc["input_ids"]], dtype=torch.long)
    d_labels = torch.tensor([p_proc["labels"]], dtype=torch.long)
    d_cb_dict = {
        eval(k) if isinstance(k, str) else k: v
        for k, v in p_proc["codebook_dict"].items()
    }
    d_cb_tensor = p_proc["codebook_tensor"]

    t_fwd_diag_start = time.perf_counter()
    with torch.no_grad():
        training_mgr.setup_differentiable_codebook(d_cb_dict, d_cb_tensor, batch_size=1, device=device)
        d_out = model(input_ids=d_input_ids.to(device), labels=d_labels.to(device))
        d_lm_loss = d_out.loss
    t_fwd_diag = time.perf_counter() - t_fwd_diag_start
    ram_after_fwd = get_ram_info()
    print(f"  Forward-only latency: {t_fwd_diag:.2f}s | LM Loss: {d_lm_loss.item():.4f}")
    print(f"  Process RSS after forward: {ram_after_fwd['process_rss_gb']} GB | Available RAM: {ram_after_fwd['available_gb']} GB\n")

    # -------------------------------------------------------------
    # DIAGNOSTIC PASS 2: FORWARD + BACKWARD (NO OPTIMIZER STATE)
    # -------------------------------------------------------------
    print("--- DIAGNOSTIC PASS 2: FORWARD + BACKWARD (NO OPTIMIZER ALLOCATION) ---")
    t_fwd_bwd_start = time.perf_counter()
    training_mgr.setup_differentiable_codebook(d_cb_dict, d_cb_tensor, batch_size=1, device=device)
    d_out2 = model(input_ids=d_input_ids.to(device), labels=d_labels.to(device))
    d_loss2 = d_out2.loss
    d_loss2.backward()
    t_fwd_bwd_diag = time.perf_counter() - t_fwd_bwd_start
    ram_after_bwd = get_ram_info()
    print(f"  Forward+Backward latency: {t_fwd_bwd_diag:.2f}s (Backward portion: {t_fwd_bwd_diag - t_fwd_diag:.2f}s)")
    print(f"  Process RSS after backward: {ram_after_bwd['process_rss_gb']} GB | Available RAM: {ram_after_bwd['available_gb']} GB\n")

    # Clear gradients from diagnostic pass
    for p in model.parameters():
        p.grad = None

    diagnostic_summary = {
        "forward_only_latency_s": round(t_fwd_diag, 2),
        "forward_backward_latency_s": round(t_fwd_bwd_diag, 2),
        "rss_after_forward_gb": ram_after_fwd["process_rss_gb"],
        "rss_after_backward_gb": ram_after_bwd["process_rss_gb"],
    }

    # -------------------------------------------------------------
    # OPTIMIZER SETUP & STEPWISE ESCALATION
    # -------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    lr = float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"]["weight_decay"])
    print("Allocating AdamW optimizer...")
    t_opt_alloc = time.perf_counter()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    t_opt_alloc = time.perf_counter() - t_opt_alloc
    ram_after_opt = get_ram_info()
    print(f"AdamW optimizer allocated in {t_opt_alloc:.2f}s for {len(trainable_params)} parameter tensors.")
    print(f"Process RSS after optimizer allocation: {ram_after_opt['process_rss_gb']} GB | Available RAM: {ram_after_opt['available_gb']} GB\n")

    # Stepwise escalation parameters
    escalation_stages = [1, 3, 5, min(10, target_steps)]
    step_records: List[Dict[str, Any]] = []

    model.train()
    optimizer.zero_grad()
    step = 0
    sample_idx = 0

    print(f"Starting stepwise escalation probe (Stages: {escalation_stages})...\n")

    for current_target in escalation_stages:
        if step >= target_steps:
            break

        stage_name = chr(ord('A') + escalation_stages.index(current_target))
        print(f"--- STAGE {stage_name}: Running to Step {current_target} ---")

        while step < current_target:
            s = raw_samples[sample_idx % len(raw_samples)]
            sample_idx += 1

            # Timing instrumentation
            t0 = time.perf_counter()

            # 1. Data & Predictor Prep
            t_prep_start = time.perf_counter()
            try:
                processed = pipeline.process_sample(
                    s["prompt"],
                    s["response"],
                    domain=s.get("domain", "general"),
                    curriculum_density=1.0,
                )
            except Exception as e:
                continue

            input_ids = torch.tensor([processed["input_ids"]], dtype=torch.long)
            labels = torch.tensor([processed["labels"]], dtype=torch.long)
            codebook_dict = {
                eval(k) if isinstance(k, str) else k: v
                for k, v in processed["codebook_dict"].items()
            }
            codebook_tensor = processed["codebook_tensor"]
            t_prep = time.perf_counter() - t_prep_start

            # 2. Codebook Setup
            t_cb_start = time.perf_counter()
            manager = training_mgr.setup_differentiable_codebook(
                codebook_dict, codebook_tensor, batch_size=input_ids.shape[0], device=device
            )
            t_cb = time.perf_counter() - t_cb_start

            # 3. Forward Pass (LM + Recon)
            t_fwd_start = time.perf_counter()
            out = model(input_ids=input_ids.to(device), labels=labels.to(device))
            lm_loss = out.loss

            base_m = getattr(model, "base_model", model)
            inp_emb = base_m.get_input_embeddings()
            from zip2zip.training_objectives import compute_reconstruction_loss
            recon_loss = compute_reconstruction_loss(
                model.input_encoder,
                codebook_tensor,
                inp_emb.weight,
                pad_token_id=cfg["compression"]["pad_token_id"],
            )

            total_loss = lm_loss + recon_weight * recon_loss
            t_fwd = time.perf_counter() - t_fwd_start

            # Check finite loss
            if not torch.isfinite(total_loss):
                raise ValueError(f"Non-finite loss encountered at step {step + 1}: {total_loss.item()}")

            # 4. Backward Pass
            t_bwd_start = time.perf_counter()
            total_loss.backward()
            t_bwd = time.perf_counter() - t_bwd_start

            # 5. Base Gradient Check (Assertion)
            for name, param in model.base_model.named_parameters():
                if "lora" not in name.lower():
                    assert param.grad is None, f"SAFETY VIOLATION: Base tensor {name} received grad!"

            # 6. Gradient Clipping
            t_clip_start = time.perf_counter()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            t_clip = time.perf_counter() - t_clip_start

            if not torch.isfinite(grad_norm):
                raise ValueError(f"Non-finite grad norm at step {step + 1}: {grad_norm.item()}")

            # 7. Optimizer Step
            t_opt_start = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad()
            t_opt = time.perf_counter() - t_opt_start

            step += 1
            t_total = time.perf_counter() - t0

            # 8. Base Hash Verification
            base_hashes_current = get_base_weight_hashes(model)
            for k, h_init in base_hashes_initial.items():
                h_curr = base_hashes_current.get(k)
                assert h_curr == h_init, f"SAFETY VIOLATION: Base tensor {k} modified! Initial={h_init}, Curr={h_curr}"

            ram_now = get_ram_info()

            rec = {
                "step": step,
                "total_loss": round(total_loss.item(), 4),
                "lm_loss": round(lm_loss.item(), 4),
                "recon_loss": round(recon_loss.item(), 4),
                "grad_norm": round(grad_norm.item(), 4),
                "time_prep_s": round(t_prep, 3),
                "time_codebook_s": round(t_cb, 3),
                "time_forward_s": round(t_fwd, 3),
                "time_backward_s": round(t_bwd, 3),
                "time_clip_s": round(t_clip, 3),
                "time_opt_s": round(t_opt, 3),
                "time_step_total_s": round(t_total, 3),
                "process_rss_gb": ram_now["process_rss_gb"],
                "available_ram_gb": ram_now["available_gb"],
            }
            step_records.append(rec)

            print(
                f"Step {step:2d} | Loss: {rec['total_loss']:6.4f} (LM: {rec['lm_loss']:6.4f}, Recon: {rec['recon_loss']:6.4f}) "
                f"| GradNorm: {rec['grad_norm']:5.2f} | Time: {rec['time_step_total_s']:5.2f}s "
                f"(Fwd: {rec['time_forward_s']:4.2f}s, Bwd: {rec['time_backward_s']:4.2f}s, Opt: {rec['time_opt_s']:4.2f}s) "
                f"| RSS: {rec['process_rss_gb']} GB",
                flush=True,
            )

        print(f"Stage {stage_name} complete. Validated: loss finite, base hash intact, base grads None.\n")

    # Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"probe_final_step_{step}.pt")
    lora_state = {
        k: v for k, v in model.base_model.named_parameters() if "lora" in k.lower()
    }
    torch.save(
        {
            "step": step,
            "trainable_mode": trainable_mode,
            "lora_state_dict": lora_state,
            "input_encoder_state_dict": model.input_encoder.state_dict(),
            "output_encoder_state_dict": model.output_encoder.state_dict(),
            "step_records": step_records,
            "config": cfg,
        },
        ckpt_path,
    )
    print(f"Probe checkpoint saved to {ckpt_path}\n")

    # Summary statistics
    step_times = [r["time_step_total_s"] for r in step_records]
    s1_time = step_times[0] if step_times else 0.0
    subsequent_times = step_times[1:] if len(step_times) > 1 else step_times
    median_time = float(torch.tensor(subsequent_times).median().item()) if subsequent_times else s1_time
    mean_time = sum(subsequent_times) / max(len(subsequent_times), 1)

    timing_summary = {
        "step_1_time_s": s1_time,
        "subsequent_median_time_s": round(median_time, 2),
        "subsequent_mean_time_s": round(mean_time, 2),
        "min_time_s": min(step_times) if step_times else 0.0,
        "max_time_s": max(step_times) if step_times else 0.0,
    }

    print("TIMING BREAKDOWN SUMMARY:")
    print(f"  Step 1 (cold cache):      {timing_summary['step_1_time_s']:.2f}s")
    print(f"  Subsequent Steps (median): {timing_summary['subsequent_median_time_s']:.2f}s")
    print(f"  Subsequent Steps (mean):   {timing_summary['subsequent_mean_time_s']:.2f}s")
    print(f"  Min / Max Step Time:       {timing_summary['min_time_s']:.2f}s / {timing_summary['max_time_s']:.2f}s\n")

    return {
        "status": "success",
        "preflight": preflight,
        "steps_completed": step,
        "step_records": step_records,
        "timing_summary": timing_summary,
        "checkpoint_path": ckpt_path,
    }


def run_training(
    config_path: str = "configs/predictive_joint_pilot.yaml",
    resume_from: Optional[str] = None,
    target_steps: int = 50,
    checkpoint_interval: int = 50,
    trainable_mode: str = "joint",
    device_str: str = "cpu",
    output_dir: str = "experiments/checkpoints/predictive_joint_pilot",
) -> Dict[str, Any]:
    """Execute cumulative predictive Zip2Zip training with exact resume."""
    cfg = load_config(config_path)
    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"PREDICTIVE ZIP2ZIP JOINT CUMULATIVE TRAINING")
    print(f"Target Steps: {target_steps} | Interval: {checkpoint_interval} | Mode: {trainable_mode.upper()} | Device: {device}")
    if resume_from:
        print(f"Resume Checkpoint: {resume_from}")
    print(f"{'='*80}\n")

    # 1. Tokenizer & Predictor Policy
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
        allow_numeric=cfg["policy"].get("allow_numeric", True),
        filter_bare_punctuation=cfg["policy"].get("filter_bare_punctuation", True),
    )
    pipeline = PredictivePipeline(
        policy,
        tokenizer,
        max_codebook_size=cfg["compression"]["budget_k"],
        max_subtokens=cfg["compression"]["max_subtokens"],
    )

    # 2. Model Loading (Backbone float16)
    print("Loading base Zip2Zip model (backbone float16)...")
    t_load_start = time.time()
    model = Zip2ZipModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    t_load = time.time() - t_load_start
    print(f"Model loaded in {t_load:.2f}s.")

    # 3. Cast trainable modules to float32 on CPU
    print("Casting trainable modules to float32 on CPU...")
    model.input_encoder.to(torch.float32)
    if getattr(model, "output_encoder", None) is not None:
        model.output_encoder.to(torch.float32)

    if trainable_mode == "joint":
        for name, param in model.base_model.named_parameters():
            if "lora" in name.lower():
                param.data = param.data.to(torch.float32)
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif trainable_mode == "encoders_only":
        for name, param in model.base_model.named_parameters():
            param.requires_grad = False
    else:
        raise ValueError(f"Unknown trainable_mode: {trainable_mode}")

    for p in model.input_encoder.parameters():
        p.requires_grad = True
    if getattr(model, "output_encoder", None) is not None:
        for p in model.output_encoder.parameters():
            p.requires_grad = True

    if hasattr(model.base_model, "get_output_embeddings"):
        out_emb = model.base_model.get_output_embeddings()
        if hasattr(out_emb, "detach_input"):
            out_emb.detach_input = False

    if device.type == "cuda":
        memory_estimate = estimate_minimum_training_memory_bytes(model)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        minimum_bytes = memory_estimate["static_minimum_bytes"]
        print(
            "GPU memory lower bound (excludes activations/workspaces): "
            f"{minimum_bytes / (1024 ** 3):.2f} GiB static of "
            f"{free_bytes / (1024 ** 3):.2f} GiB currently free "
            f"({total_bytes / (1024 ** 3):.2f} GiB total).",
            flush=True,
        )
        if minimum_bytes >= free_bytes:
            raise RuntimeError(
                "Requested GPU cannot hold even the model, trainable gradients, and AdamW states; "
                "no training steps were started. This lower bound excludes activations."
            )

    print(f"Moving configured model to training device {device}...", flush=True)
    model = move_training_model_to_device(model, device)
    if device.type == "cuda":
        print(
            f"Using {torch.cuda.get_device_name(device)}; "
            f"allocated after model placement: "
            f"{torch.cuda.memory_allocated(device) / (1024 ** 3):.2f} GiB.",
            flush=True,
        )

    # Record initial base tensor hashes
    base_hashes_initial = get_base_weight_hashes(model)

    # 4. Optimizer Allocation
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    lr = float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"]["weight_decay"])
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # 5. Handle Checkpoint Resume
    start_step = 0
    step_records: List[Dict[str, Any]] = []
    cumulative_time_s = 0.0

    if resume_from and os.path.exists(resume_from):
        print(f"\n--- RESUMING FROM CHECKPOINT: {resume_from} ---")
        ckpt = torch.load(resume_from, map_location="cpu")
        start_step = ckpt.get("step", 0)
        step_records = list(ckpt.get("step_records", []))
        cumulative_time_s = float(ckpt.get("cumulative_time_s", 0.0))

        # Restore LoRA
        if "lora_state_dict" in ckpt:
            for k, v in ckpt["lora_state_dict"].items():
                model.base_model.load_state_dict({k: v}, strict=False)
            print(f"  Restored {len(ckpt['lora_state_dict'])} LoRA weight tensors.")

        # Restore Input Encoder
        if "input_encoder_state_dict" in ckpt:
            model.input_encoder.load_state_dict(ckpt["input_encoder_state_dict"], strict=False)
            print("  Restored input_encoder state dict.")

        # Restore Output Encoder
        if "output_encoder_state_dict" in ckpt and getattr(model, "output_encoder", None) is not None:
            model.output_encoder.load_state_dict(ckpt["output_encoder_state_dict"], strict=False)
            print("  Restored output_encoder state dict.")

        # Restore Optimizer State
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("  Restored optimizer state from checkpoint.")
        else:
            print(f"  NOTICE: Optimizer state was not recoverable from historical checkpoint ({resume_from}).")
            print(f"  Initialized fresh AdamW optimizer with lr={lr}, weight_decay={weight_decay}.")

        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"])
            print("  Restored PyTorch RNG state.")
        if device.type == "cuda" and ckpt.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"], device=device)
            print(f"  Restored CUDA RNG state for {device}.")

        print(f"Checkpoint loaded. Starting at Step {start_step} -> Target Step {target_steps}.\n")

    if start_step >= target_steps:
        print(f"Start step {start_step} is already >= target_steps {target_steps}. Exiting.")
        return {
            "status": "already_completed",
            "start_step": start_step,
            "target_steps": target_steps,
        }

    # Verify base weights are unmodified
    base_hashes_current = get_base_weight_hashes(model)
    for k, h_init in base_hashes_initial.items():
        assert base_hashes_current.get(k) == h_init, f"Base tensor {k} differs after loading!"

    # 6. Load Training Data
    train_file = "data/train.jsonl"
    with open(train_file, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line.strip()) for line in f]

    training_mgr = DifferentiableTrainingManager(
        model,
        initial_vocab_size=cfg["compression"]["initial_vocab_size"],
        max_codebook_size=cfg["compression"]["budget_k"],
        max_subtokens=cfg["compression"]["max_subtokens"],
    )
    recon_weight = float(cfg["loss"]["reconstruction_weight"])
    max_grad_norm = float(cfg["training"]["max_grad_norm"])

    history_file = os.path.join(output_dir, "training_history.jsonl")

    model.train()
    optimizer.zero_grad()
    step = start_step
    sample_idx = start_step
    t_session_start = time.perf_counter()

    print(f"Beginning training loop: Step {step + 1} to {target_steps}...\n", flush=True)

    while step < target_steps:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        s = raw_samples[sample_idx % len(raw_samples)]
        sample_idx += 1

        t0 = time.perf_counter()

        # 1. Prep
        t_prep_start = time.perf_counter()
        try:
            processed = pipeline.process_sample(
                s["prompt"],
                s["response"],
                domain=s.get("domain", "general"),
                curriculum_density=1.0,
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
        t_prep = time.perf_counter() - t_prep_start

        # 2. Codebook Setup
        t_cb_start = time.perf_counter()
        manager = training_mgr.setup_differentiable_codebook(
            codebook_dict, codebook_tensor, batch_size=input_ids.shape[0], device=device
        )
        t_cb = time.perf_counter() - t_cb_start

        # 3. Forward Pass (LM + Recon)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_fwd_start = time.perf_counter()
        out = model(input_ids=input_ids.to(device), labels=labels.to(device))
        lm_loss = out.loss

        base_m = getattr(model, "base_model", model)
        inp_emb = base_m.get_input_embeddings()
        from zip2zip.training_objectives import compute_reconstruction_loss
        recon_loss = compute_reconstruction_loss(
            model.input_encoder,
            codebook_tensor,
            inp_emb.weight,
            pad_token_id=cfg["compression"]["pad_token_id"],
        )

        total_loss = lm_loss + recon_weight * recon_loss
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_fwd = time.perf_counter() - t_fwd_start

        if not torch.isfinite(total_loss):
            raise ValueError(f"Non-finite loss encountered at step {step + 1}: {total_loss.item()}")

        # 4. Backward Pass
        t_bwd_start = time.perf_counter()
        total_loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_bwd = time.perf_counter() - t_bwd_start

        # 5. Base Gradient Check
        for name, param in model.base_model.named_parameters():
            if "lora" not in name.lower():
                assert param.grad is None, f"SAFETY VIOLATION: Base tensor {name} received grad!"

        # 6. Gradient Clipping
        t_clip_start = time.perf_counter()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        t_clip = time.perf_counter() - t_clip_start

        if not torch.isfinite(grad_norm):
            raise ValueError(f"Non-finite grad norm at step {step + 1}: {grad_norm.item()}")

        # 7. Optimizer Step
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_opt_start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_opt = time.perf_counter() - t_opt_start

        step += 1
        t_total = time.perf_counter() - t0

        # 8. Base Hash Verification
        base_hashes_now = get_base_weight_hashes(model)
        for k, h_init in base_hashes_initial.items():
            h_curr = base_hashes_now.get(k)
            assert h_curr == h_init, f"SAFETY VIOLATION: Base tensor {k} modified!"

        ram_now = get_ram_info()

        rec = {
            "step": step,
            "total_loss": round(total_loss.item(), 4),
            "lm_loss": round(lm_loss.item(), 4),
            "recon_loss": round(recon_loss.item(), 4),
            "grad_norm": round(grad_norm.item(), 4),
            "time_prep_s": round(t_prep, 3),
            "time_codebook_s": round(t_cb, 3),
            "time_forward_s": round(t_fwd, 3),
            "time_backward_s": round(t_bwd, 3),
            "time_clip_s": round(t_clip, 3),
            "time_opt_s": round(t_opt, 3),
            "time_step_total_s": round(t_total, 3),
            "process_rss_gb": ram_now["process_rss_gb"],
            "available_ram_gb": ram_now["available_gb"],
            "gpu_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
        }
        step_records.append(rec)

        # Append to jsonl
        with open(history_file, "a", encoding="utf-8") as f_hist:
            f_hist.write(json.dumps(rec) + "\n")

        print(
            f"Step {step:3d}/{target_steps} | Loss: {rec['total_loss']:6.4f} "
            f"(LM: {rec['lm_loss']:6.4f}, Recon: {rec['recon_loss']:6.4f}) "
            f"| GradNorm: {rec['grad_norm']:5.2f} | Time: {rec['time_step_total_s']:5.2f}s "
            f"| RSS: {rec['process_rss_gb']} GB",
            flush=True,
        )

        # Checkpoint Saving
        if step % checkpoint_interval == 0 or step == target_steps:
            session_elapsed = time.perf_counter() - t_session_start
            total_cum_time = cumulative_time_s + session_elapsed

            lora_state = {
                k: v for k, v in model.base_model.named_parameters() if "lora" in k.lower()
            }
            ckpt_data = {
                "step": step,
                "trainable_mode": trainable_mode,
                "lora_state_dict": lora_state,
                "input_encoder_state_dict": model.input_encoder.state_dict(),
                "output_encoder_state_dict": model.output_encoder.state_dict() if getattr(model, "output_encoder", None) is not None else {},
                "optimizer_state_dict": optimizer.state_dict(),
                "step_records": step_records,
                "config": cfg,
                "base_hashes": base_hashes_now,
                "cumulative_time_s": round(total_cum_time, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                ),
            }
            ckpt_save_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
            torch.save(ckpt_data, ckpt_save_path)
            print(f"\n>>> FULL RESUMABLE CHECKPOINT SAVED TO {ckpt_save_path} <<<\n", flush=True)

    session_elapsed = time.perf_counter() - t_session_start
    total_cum_time = cumulative_time_s + session_elapsed

    summary = {
        "status": "success",
        "start_step": start_step,
        "target_steps": target_steps,
        "steps_trained_this_session": step - start_step,
        "session_elapsed_s": round(session_elapsed, 2),
        "cumulative_time_s": round(total_cum_time, 2),
        "latest_checkpoint": os.path.join(output_dir, f"checkpoint_step_{step}.pt"),
        "final_losses": {
            "total_loss": step_records[-1]["total_loss"] if step_records else None,
            "lm_loss": step_records[-1]["lm_loss"] if step_records else None,
            "recon_loss": step_records[-1]["recon_loss"] if step_records else None,
        },
    }

    summary_file = os.path.join(output_dir, "learning_curve_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train predictive Zip2Zip jointly or run CPU probe.")
    parser.add_argument("--config", type=str, default="configs/predictive_joint_pilot.yaml")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint pt file to resume from")
    parser.add_argument("--target-steps", type=int, default=50, help="Total target optimizer steps")
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Checkpoint save interval")
    parser.add_argument("--cpu-probe", action="store_true", help="Run lightweight CPU probe mode")
    parser.add_argument("--trainable-mode", type=str, default="joint", choices=["joint", "encoders_only"])
    parser.add_argument("--max-steps", type=int, default=10, help="Max steps for probe (hard cap 10)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="experiments/checkpoints/predictive_joint_pilot")
    args = parser.parse_args()

    if args.cpu_probe:
        res = run_cpu_probe(
            config_path=args.config,
            trainable_mode=args.trainable_mode,
            target_steps=min(10, args.max_steps),
            device_str=args.device,
            output_dir=args.output_dir,
        )
        with open("experiments/checkpoints/cpu_probe_metrics.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    else:
        res = run_training(
            config_path=args.config,
            resume_from=args.resume_from,
            target_steps=args.target_steps,
            checkpoint_interval=args.checkpoint_interval,
            trainable_mode=args.trainable_mode,
            device_str=args.device,
            output_dir=args.output_dir,
        )
        print("\nTraining run completed successfully.")
        print(json.dumps(res, indent=2))
