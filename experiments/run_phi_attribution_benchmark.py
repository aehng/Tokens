"""Executable Benchmark Runner for Phi Quality/Speed Attribution Experiments.

Supports running Conditions:
  A: Vanilla Phi-3.5-mini-instruct
  B0: Tokens wrapper with pinned Vanilla weights; no adapter loaded and H masked
  B1: Step-100 predictive checkpoint, H disabled
  C: ORACLE CODEBOOK LIVE (hindsight phrases; model chooses H or base tokens)
  D: Real Current Predictor (Phi-only candidate retrieval + PooledMLP ranker, K=32)

Enforces:
  - Hard constraint against accessing FINAL split.
  - Stratified prompt selection (Code, Reasoning, Instruction).
  - Exact provenance and machine-readable output schemas.
  - Robust resumption without duplicate generation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import gc
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
import pickle
import platform
import subprocess
import sys
import time
import weakref
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import zip2zip
import zip2zip.model
import zip2zip.static_codebook
import zip2zip.predictor_v2.attribution_harness
import zip2zip.predictor_v2.ablation_gates

sys.modules["src.zip2zip"] = zip2zip
sys.modules["src.zip2zip.model"] = zip2zip.model
sys.modules["src.zip2zip.static_codebook"] = zip2zip.static_codebook
sys.modules["src.zip2zip.predictor_v2.attribution_harness"] = zip2zip.predictor_v2.attribution_harness
sys.modules["src.zip2zip.predictor_v2.ablation_gates"] = zip2zip.predictor_v2.ablation_gates

from zip2zip.model import Zip2ZipModel
from zip2zip.static_codebook import StaticCodebookManager
from zip2zip.predictor_v2.attribution_harness import (
    ALL_CONDITIONS,
    ATTRIBUTION_RECORD_SCHEMA,
    CANONICAL_EOS_TOKEN_IDS,
    CANONICAL_MODEL_ID,
    CANONICAL_MODEL_REVISION,
    CANONICAL_ZIP2ZIP_ID,
    CANONICAL_ZIP2ZIP_REVISION,
    COND_A_VANILLA,
    COND_B0_TOKENS_VANILLA_WEIGHTS,
    COND_B1_UPSTREAM_EPFL_ADAPTER_H_DISABLED,
    COND_B2_STEP100_H_DISABLED,
    COND_B_H_DISABLED,
    COND_C_ORACLE,
    COND_CF_FORCED_ORACLE,
    COND_D_REAL_PREDICTOR,
    INITIAL_VOCAB_SIZE,
    MAX_NEW_TOKENS,
    PAD_TOKEN_ID,
    AttributionError,
    AttributionRecord,
    build_canonical_prompt_text,
    build_codebook_dict,
    check_split_safety,
    compute_attribution_summary,
    derive_oracle_codebook_phrases,
    evaluate_output_quality,
    find_first_divergence,
    select_stratified_dev_prompts,
    STRATIFIED_DEV12_PROMPT_IDS,
)
from zip2zip.predictor_v2.ablation_gates import (
    b0_b1_adapter_isolation_gate,
    b1_b2_checkpoint_isolation_gate,
    checkpoint_isolation_gate,
    forced_h_representation_gates,
    logit_parity_metrics,
    normalize_wrapper_logits,
    token_equivalence_gate,
)
from src.zip2zip.predictor_v2.canonical_dataset import (
    load_canonical_dataset,
    require_split_isolated_manifest,
    sha256_file,
)
from src.zip2zip.predictor_v2.attribution_reanalysis import oracle_ceiling
from src.zip2zip.predictor_v2.forced_oracle import force_oracle_substitutions, h_vs_base_prefix_pair
from src.zip2zip.predictor_v2.candidate_retrieval import (
    ConfigurableCandidateGenerator,
    RetrievalStrategy,
    TrainOnlyAssociationIndex,
)
from experiments.load_joint_checkpoint import load_joint_checkpoint
from experiments.generation_timing import TimingLogitsProcessor, synchronize_device


CONDITION_DISPLAY = {
    COND_A_VANILLA: "A — PURE VANILLA",
    COND_B0_TOKENS_VANILLA_WEIGHTS: "B0 — TOKENS ARCHITECTURE, VANILLA WEIGHTS",
    COND_B1_UPSTREAM_EPFL_ADAPTER_H_DISABLED: "B1 — UPSTREAM EPFL PEFT ADAPTER, H DISABLED",
    COND_B2_STEP100_H_DISABLED: "B2 — STEP-100 CHECKPOINT, H DISABLED",
    COND_B_H_DISABLED: "HISTORICAL B — TOKENS + STEP-100 CHECKPOINT, H DISABLED",
    COND_C_ORACLE: "C — ORACLE CODEBOOK LIVE",
    COND_CF_FORCED_ORACLE: "CF — FORCED ORACLE",
    COND_D_REAL_PREDICTOR: "D — REAL PREDICTOR",
}


def current_git_commit() -> str:
    try:
        import subprocess
        commit = os.environ.get("GIT_COMMIT")
        if commit:
            return commit
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def runtime_metadata(device: str) -> Dict[str, Any]:
    device_obj = torch.device(device)
    dtype = torch.float16 if device_obj.type == "cuda" else torch.float32
    info: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "torch_dtype": str(dtype),
        "attention_implementation": "UNRESOLVED_UNTIL_MODEL_LOAD",
        "source_archive_sha256": os.environ.get("SOURCE_ARCHIVE_SHA256", ""),
        "git_commit": current_git_commit(),
        "git_branch": os.environ.get("SOURCE_GIT_BRANCH", "") or current_git_branch(),
        "git_worktree_dirty": parse_bool_env("SOURCE_GIT_WORKTREE_DIRTY"),
        "git_status_sha256": os.environ.get("SOURCE_GIT_STATUS_SHA256", ""),
        "model_id": CANONICAL_MODEL_ID,
        "model_revision": CANONICAL_MODEL_REVISION,
        "zip2zip_id": CANONICAL_ZIP2ZIP_ID,
        "zip2zip_revision": CANONICAL_ZIP2ZIP_REVISION,
    }
    package_distributions = {
        "torch": "torch",
        "transformers": "transformers",
        "peft": "peft",
        "accelerate": "accelerate",
        "huggingface_hub": "huggingface-hub",
        "safetensors": "safetensors",
        "torchao": "torchao",
    }
    package_versions: Dict[str, Optional[str]] = {}
    for import_name, distribution_name in package_distributions.items():
        try:
            package_versions[import_name] = importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            package_versions[import_name] = None
    info["package_versions"] = package_versions
    required_runtime_packages = set(package_distributions) - {"torchao"}
    info["package_versions_complete"] = all(
        package_versions[name] for name in required_runtime_packages
    )
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return info


def parse_bool_env(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "dirty"}:
        return True
    if normalized in {"0", "false", "no", "clean"}:
        return False
    raise AttributionError(f"Environment variable {name} must be a boolean value, got {value!r}")


def current_git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def effective_attention_implementation(model: Any) -> str:
    """Describe the active attention module class and the resolved HF setting."""
    candidate = model
    if hasattr(model, "zip2zip_config"):
        candidate = getattr(model, "base_model", model)
    get_base_model = getattr(candidate, "get_base_model", None)
    if callable(get_base_model):
        candidate = get_base_model()
    config = getattr(candidate, "config", None)
    config_setting = (
        getattr(config, "_attn_implementation", None)
        or getattr(config, "_attn_implementation_internal", None)
        or "unset"
    )
    attention_class = "unknown"
    core = getattr(candidate, "model", None)
    layers = getattr(core, "layers", None)
    if layers is not None and len(layers):
        self_attention = getattr(layers[0], "self_attn", None)
        if self_attention is not None:
            attention_class = type(self_attention).__name__
    if attention_class == "unknown":
        raise AttributionError("Could not identify the model's effective attention module")
    return f"{attention_class};config={config_setting}"





def load_vanilla_model_and_tokenizer(device: str) -> Tuple[Any, Any]:
    print(f"Loading Vanilla Phi-3.5 from {CANONICAL_MODEL_ID} (rev: {CANONICAL_MODEL_REVISION[:8]})...", flush=True)
    device_obj = torch.device(device)
    dtype = torch.float16 if device_obj.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        CANONICAL_MODEL_ID,
        revision=CANONICAL_MODEL_REVISION,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        CANONICAL_MODEL_ID,
        revision=CANONICAL_MODEL_REVISION,
        torch_dtype=dtype,
        trust_remote_code=False,
    ).to(device_obj)
    model.eval()
    return model, tokenizer


def load_predictive_bundle(
    checkpoint_path: Path,
    device: str,
    *,
    load_peft_adapter: bool,
    apply_joint_checkpoint: bool = True,
    k: int = 32,
) -> Tuple[Zip2ZipModel, Any, Dict[str, Any]]:
    print(f"Loading Zip2Zip Predictive Bundle with checkpoint {checkpoint_path}...", flush=True)
    device_obj = torch.device(device)
    dtype = torch.float16 if device_obj.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        CANONICAL_MODEL_ID,
        revision=CANONICAL_MODEL_REVISION,
        trust_remote_code=False,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        CANONICAL_MODEL_ID,
        revision=CANONICAL_MODEL_REVISION,
        torch_dtype=dtype,
        trust_remote_code=False,
    )
    model = Zip2ZipModel.from_pretrained(
        CANONICAL_ZIP2ZIP_ID,
        base_model=base_model,
        revision=CANONICAL_ZIP2ZIP_REVISION,
        torch_dtype=dtype,
        codebook_backend="static",
        max_codebook_size=k,
        load_peft_adapter=load_peft_adapter,
    ).to(device_obj)
    loaded_position_mode = str(getattr(model.zip2zip_config, "position_mode", "compressed"))
    model.enable_base_token_positions()

    if apply_joint_checkpoint:
        report = load_joint_checkpoint(
            model,
            str(checkpoint_path),
            expected_step=100,
            expected_model_id=CANONICAL_ZIP2ZIP_ID,
        )
        print(f"Loaded checkpoint step {report.get('step')} into Zip2ZipModel.", flush=True)
    else:
        report = {"checkpoint_applied": False, "checkpoint_step": None}
        report["adapter_present"] = bool(getattr(model.base_model, "peft_config", None))
        report["step100_h_encoder_checkpoint_loaded"] = False
    peft_configs = getattr(model.base_model, "peft_config", None) or {}
    report["adapter_modules_to_save"] = {
        str(adapter_name): sorted(str(name) for name in (getattr(config, "modules_to_save", None) or []))
        for adapter_name, config in peft_configs.items()
    }
    active_adapters = getattr(model.base_model, "active_adapter", None)
    if isinstance(active_adapters, str):
        active_adapters = [active_adapters]
    report["active_adapter_names"] = list(active_adapters or [])
    report["adapter_disabled_state_after_load"] = getattr(model.base_model, "_adapters_disabled", None)
    report["adapter_load_requested"] = load_peft_adapter

    try:
        import peft
        peft_version = getattr(peft, "__version__", "unknown")
    except ImportError:
        peft_version = "not_installed"

    adapter_param_names = [n for n, p in model.named_parameters() if "lora" in n.lower()]
    adapter_param_count = sum(p.numel() for n, p in model.named_parameters() if "lora" in n.lower())
    adapter_param_shapes = {n: list(p.shape) for n, p in model.named_parameters() if "lora" in n.lower()}

    report["peft_version"] = peft_version
    report["adapter_param_count"] = adapter_param_count
    report["adapter_param_names"] = adapter_param_names
    report["adapter_param_shapes"] = adapter_param_shapes
    report["upstream_adapter_id"] = CANONICAL_ZIP2ZIP_ID if load_peft_adapter else None
    report["upstream_adapter_revision"] = CANONICAL_ZIP2ZIP_REVISION if load_peft_adapter else None
    report["active_lora_during_generation"] = bool(load_peft_adapter and adapter_param_count > 0)
    report["loaded_position_mode"] = loaded_position_mode
    report["effective_position_mode"] = model.zip2zip_config.position_mode
    report["codebook_backend"] = getattr(model.zip2zip_config, "codebook_backend", None)
    model.eval()
    return model, tokenizer, report


def generation_policy(max_new_tokens: int) -> Dict[str, Any]:
    """Return the shared greedy decoding configuration used by every arm."""
    return {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "pad_token_id": PAD_TOKEN_ID,
        "eos_token_id": list(CANONICAL_EOS_TOKEN_IDS),
    }


def generation_policy_sha256(max_new_tokens: int) -> str:
    payload = {
        **generation_policy(max_new_tokens),
        "model_id": CANONICAL_MODEL_ID,
        "model_revision": CANONICAL_MODEL_REVISION,
        "prompt_builder": "build_canonical_prompt_text",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tokenizer_template_sha256(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        template = json.dumps(template, sort_keys=True, ensure_ascii=False, default=str)
    return text_sha256(template)


def prompt_runtime_metadata(tokenizer: Any, prompt_text: str, input_ids: Sequence[int], max_new_tokens: int) -> Dict[str, Any]:
    return {
        "input_token_ids": [int(token_id) for token_id in input_ids],
        "input_token_ids_sha256": token_ids_sha256(input_ids),
        "rendered_prompt_text": prompt_text,
        "rendered_prompt_sha256": text_sha256(prompt_text),
        "tokenizer_chat_template_sha256": tokenizer_template_sha256(tokenizer),
        "tokenizer_id": CANONICAL_MODEL_ID,
        "tokenizer_revision": CANONICAL_MODEL_REVISION,
        "generation_policy_sha256": generation_policy_sha256(max_new_tokens),
        "generation_config": generation_policy(max_new_tokens),
        "eos_token_ids": list(CANONICAL_EOS_TOKEN_IDS),
        "max_new_tokens": int(max_new_tokens),
    }


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps([int(token_id) for token_id in token_ids], separators=(",", ":")).encode("ascii")).hexdigest()


def base_phi_weight_sha256(model: Any) -> str:
    """Hash every underlying Phi parameter, excluding LoRA and Zip2Zip encoders."""
    # Zip2ZipModel forwards attributes through __getattr__, so `hasattr` alone
    # can mistake the wrapper for a native CausalLM and retain its base_model.
    if hasattr(model, "zip2zip_config"):
        base = getattr(model, "base_model", model)
    elif hasattr(model, "lm_head") and hasattr(model, "get_input_embeddings"):
        base = model
    else:
        base = getattr(model, "base_model", model)
    get_base_model = getattr(base, "get_base_model", None)
    if callable(get_base_model):
        base = get_base_model()
    parameters = []
    for name, parameter in base.named_parameters():
        lowered = name.lower()
        if (
            "lora" in lowered
            or "input_encoder" in lowered
            or "output_encoder" in lowered
            or ".modules_to_save." in lowered
        ):
            continue
        canonical_name = name.replace(".base_layer.", ".").replace(".original_module.", ".")
        parameters.append((canonical_name, parameter))
    if not parameters:
        raise RuntimeError("No underlying Phi parameters found for the base-weight fingerprint")
    digest = hashlib.sha256()
    previous_names: Set[str] = set()
    for name, parameter in sorted(parameters, key=lambda item: item[0]):
        if name in previous_names:
            raise RuntimeError(f"Base-weight key normalization produced duplicate name {name!r}")
        previous_names.add(name)
        tensor = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast("B"))
    digest.update(str(len(parameters)).encode("ascii"))
    return digest.hexdigest()


def adapter_runtime_metadata(model: Any) -> Dict[str, Any]:
    """Expose installed/active PEFT state so B0 can require no adapter at all."""
    base = getattr(model, "base_model", model)
    configs = getattr(base, "peft_config", None) or {}
    active_adapters = getattr(base, "active_adapter", None)
    if isinstance(active_adapters, str):
        active_adapters = [active_adapters]
    modules_to_save = {
        str(adapter_name): sorted(str(name) for name in (getattr(config, "modules_to_save", None) or []))
        for adapter_name, config in configs.items()
    }
    return {
        "adapter_present_in_wrapper": bool(configs),
        "active_adapter_names": list(active_adapters or []),
        "adapter_modules_to_save": modules_to_save,
        "adapters_disabled_state": getattr(base, "_adapters_disabled", None),
    }


def b0_adapter_absence_verified(load_report: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    """Require the B0 loader and live model to contain no PEFT adapter state."""
    return (
        load_report.get("adapter_load_requested") is False
        and load_report.get("adapter_present") is False
        and not load_report.get("active_adapter_names")
        and not load_report.get("adapter_modules_to_save")
        and state.get("adapter_present_in_wrapper") is False
        and not state.get("active_adapter_names")
        and not state.get("adapter_modules_to_save")
        and state.get("adapters_disabled_state") is None
    )


def verify_empty_static_h_mask(static_mgr: StaticCodebookManager, device: torch.device) -> bool:
    """Check the configured empty H interval is actually masked to negative infinity."""
    if int(static_mgr.num_seeded) != 0:
        return False
    start = int(static_mgr.initial_vocab_size)
    end = start + int(static_mgr.max_codebook_size)
    probe = torch.zeros((1, end + 1), dtype=torch.float32, device=device)
    masked = static_mgr.mask_unused_logits(probe)
    h_rows = masked[..., start:end]
    base_rows = torch.cat((masked[..., :start], masked[..., end:]), dim=-1)
    return bool(torch.isneginf(h_rows).all().item() and torch.isfinite(base_rows).all().item())


MODEL_TEARDOWN_MAX_ALLOCATED_BYTES = 512 * 1024 * 1024


def cuda_memory_snapshot(device: torch.device) -> Dict[str, Any]:
    """Return allocator measurements for one CUDA device, or explicit N/A values."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"cuda_available": False, "allocated_bytes": None, "reserved_bytes": None}
    torch.cuda.synchronize(device)
    return {
        "cuda_available": True,
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def assert_model_phase_teardown(
    model_ref: Any,
    device: torch.device,
    *,
    phase: str,
    before: Optional[Mapping[str, Any]] = None,
    max_allocated_bytes: int = MODEL_TEARDOWN_MAX_ALLOCATED_BYTES,
    manager_ref: Any = None,
) -> Dict[str, Any]:
    """Collect and measure a released model before another full model may load."""
    gc.collect()
    cuda_before_release = cuda_memory_snapshot(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    after = cuda_memory_snapshot(device)
    model_alive = model_ref() is not None
    manager_alive = manager_ref is not None and manager_ref() is not None
    report = {
        "phase": phase,
        "model_object_alive_after_gc": model_alive,
        "manager_object_alive_after_gc": manager_alive,
        "before_teardown": dict(before or {}),
        "after_gc_before_empty_cache": cuda_before_release,
        "after_teardown": after,
        "max_allocated_bytes": int(max_allocated_bytes),
    }
    allocated = after.get("allocated_bytes")
    if model_alive or manager_alive or (
        allocated is not None and int(allocated) > int(max_allocated_bytes)
    ):
        report["status"] = "FAIL"
        raise AttributionError(
            "Model teardown failed before the next model load: "
            + json.dumps(report, sort_keys=True)
        )
    report["status"] = "PASS"
    return report


def make_empty_b0_manager(
    *, k: int, embedding_dim: int, device: torch.device
) -> StaticCodebookManager:
    """Create the one empty static H manager required by B0."""
    manager = StaticCodebookManager(
        initial_vocab_size=INITIAL_VOCAB_SIZE,
        max_codebook_size=k,
        max_subtokens=4,
        embedding_dim=embedding_dim,
        pad_token_id=PAD_TOKEN_ID,
    )
    manager.set_seeded_codebook({}, batch_size=1, device=device)
    if not verify_empty_static_h_mask(manager, device):
        raise AttributionError("B0 static codebook did not mask every empty H logit")
    return manager


def prepare_empty_b0_manager(
    model: Any, *, k: int, device: torch.device
) -> StaticCodebookManager:
    """Prepare the manager already installed in the Tokens wrapper for B0."""
    manager = getattr(model, "codebook_manager", None)
    if manager is None or not (
        isinstance(manager, StaticCodebookManager)
        or type(manager).__name__ == "StaticCodebookManager"
    ):
        raise AttributionError("B0 requires the wrapper's installed StaticCodebookManager")
    if manager.initial_vocab_size != INITIAL_VOCAB_SIZE or manager.max_codebook_size != k:
        embedding_dim = getattr(manager, "embedding_dim", 3072)
        pad_token_id = getattr(manager, "pad_token_id", PAD_TOKEN_ID)
        manager = StaticCodebookManager(
            initial_vocab_size=INITIAL_VOCAB_SIZE,
            max_codebook_size=k,
            max_subtokens=4,
            embedding_dim=embedding_dim,
            pad_token_id=pad_token_id,
        )
        model.codebook_manager = manager
        if hasattr(model, "base_model"):
            in_emb = model.base_model.get_input_embeddings()
            if hasattr(in_emb, "codebook_manager"):
                in_emb.codebook_manager = manager
            out_emb = model.base_model.get_output_embeddings()
            if hasattr(out_emb, "codebook_manager"):
                out_emb.codebook_manager = manager
    if manager.initial_vocab_size != INITIAL_VOCAB_SIZE or manager.max_codebook_size != k:
        raise AttributionError(
            f"B0 wrapper manager does not match the pinned H vocabulary layout: "
            f"initial_vocab_size={getattr(manager, 'initial_vocab_size', None)} (expected {INITIAL_VOCAB_SIZE}), "
            f"max_codebook_size={getattr(manager, 'max_codebook_size', None)} (expected {k})"
        )
    manager.set_seeded_codebook({}, batch_size=1, device=device)
    if not verify_empty_static_h_mask(manager, device):
        raise AttributionError("B0 static codebook did not mask every empty H logit")
    return manager


def reset_empty_b0_manager(manager: StaticCodebookManager, device: torch.device) -> None:
    """Reset per-request positions and caches while retaining the empty codebook."""
    manager.reset(clear_caches=True)
    if (
        manager.num_seeded != 0
        or manager.hyper_to_subtokens
        or manager.subtokens_to_hyper
        or not verify_empty_static_h_mask(manager, device)
    ):
        raise AttributionError("B0 empty H manager acquired seeded or selectable H state")


def remove_model_generation_hooks(model: Any) -> bool:
    """Break wrapper/base-model generation-hook references before model release."""
    remove_hook = getattr(model, "remove_base_position_generation_hook", None)
    return bool(remove_hook()) if callable(remove_hook) else False


def offline_a_b0_token_gate(
    records: Sequence[Mapping[str, Any]], expected_prompt_ids: Sequence[str]
) -> Dict[str, Any]:
    """Validate saved A/B0 records without loading a tokenizer or model."""
    expected_count = len(expected_prompt_ids)
    counts = {
        condition: sum(1 for row in records if row.get("condition") == condition)
        for condition in (COND_A_VANILLA, COND_B0_TOKENS_VANILLA_WEIGHTS)
    }
    unexpected_conditions = [
        str(row.get("condition"))
        for row in records
        if row.get("condition") not in (COND_A_VANILLA, COND_B0_TOKENS_VANILLA_WEIGHTS)
    ]
    gate = token_equivalence_gate(records, expected_prompt_ids=expected_prompt_ids)
    inventory_ok = (
        len(records) == 2 * expected_count
        and counts[COND_A_VANILLA] == expected_count
        and counts[COND_B0_TOKENS_VANILLA_WEIGHTS] == expected_count
        and not unexpected_conditions
    )
    gate["record_inventory"] = {
        "status": "PASS" if inventory_ok else "FAIL",
        "record_count": len(records),
        "expected_record_count": 2 * expected_count,
        "condition_counts": counts,
        "unexpected_conditions": unexpected_conditions,
    }
    if not inventory_ok:
        gate["status"] = "FAIL"
    return gate


class ForcedScheduleLogitsProcessor:
    """Record free greedy preferences, then force a precomputed CF token schedule."""

    def __init__(
        self,
        prompt_length: int,
        forced_ids: Sequence[int],
        *,
        static_mgr: Optional[StaticCodebookManager] = None,
        h_position_expectations: Optional[Mapping[int, Mapping[str, int]]] = None,
    ):
        self.prompt_length = int(prompt_length)
        self.forced_ids = [int(token_id) for token_id in forced_ids]
        self.unforced_top1: List[int] = []
        self.static_mgr = static_mgr
        self.h_position_expectations = dict(h_position_expectations or {})
        self.semantic_position_checks: List[Dict[str, int]] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # HuggingFace cached generate passes only the newest token, so the
        # schedule index cannot be recovered from input_ids.shape[1].
        step = len(self.unforced_top1)
        if step < 0 or step >= len(self.forced_ids):
            raise RuntimeError(f"CF forced schedule step {step} is out of range")
        if step > 0:
            schedule_index = step - 1
            expectation = self.h_position_expectations.get(schedule_index)
            if expectation is not None and self.static_mgr is not None:
                positions = self.static_mgr.position_ids
                offsets = self.static_mgr.base_position_offset
                if positions is None or offsets is None:
                    raise RuntimeError("CF manager did not expose semantic position/cache accounting")
                self.semantic_position_checks.append({
                    "schedule_index": schedule_index,
                    "token_id": int(input_ids[0, -1].item()),
                    "expected_position": int(expectation["expected_position"]),
                    "observed_position": int(positions[0, -1].item()),
                    "expected_next_offset": int(expectation["expected_next_offset"]),
                    "observed_next_offset": int(offsets[0, 0].item()),
                })
        self.unforced_top1.append(int(torch.argmax(scores[0]).item()))
        scores = scores.clone()
        scores.fill_(float("-inf"))
        scores[:, self.forced_ids[step]] = 0.0
        return scores


def _first_eos_prefix(token_ids: Sequence[int], eos_ids: Sequence[int]) -> List[int]:
    values = [int(token_id) for token_id in token_ids]
    for index, token_id in enumerate(values):
        if token_id in eos_ids:
            return values[:index]
    return values


def capture_prefix_logits(
    model: Any,
    prompt_ids: Sequence[int],
    continuation_ids: Sequence[int],
    *,
    device: torch.device,
    inserted_h_count: int,
    static_mgr: Optional[StaticCodebookManager] = None,
) -> Dict[int, torch.Tensor]:
    """Capture next-token logits at deterministic prefixes of a known output."""
    prompt = [int(token_id) for token_id in prompt_ids]
    continuation = _first_eos_prefix(continuation_ids, CANONICAL_EOS_TOKEN_IDS)
    if not prompt:
        raise AttributionError("Cannot compare next-token logits for an empty rendered prompt")
    prefix_lengths = [length for length in (0, 1, 4, 16) if length <= len(continuation)]
    if not prefix_lengths:
        prefix_lengths = [0]
    full_ids = prompt + continuation[: max(prefix_lengths)]
    input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    if static_mgr is not None:
        static_mgr.reset()
    with torch.no_grad():
        output = model(input_ids=input_tensor, attention_mask=attention_mask, use_cache=False)
    logits = output.logits if hasattr(output, "logits") else output[0]
    if static_mgr is not None:
        output_layer = model.base_model.get_output_embeddings()
        base_vocab_size = int(output_layer.weight.shape[0])
        logits = normalize_wrapper_logits(
            logits,
            base_vocab_size=base_vocab_size,
            initial_vocab_size=INITIAL_VOCAB_SIZE,
            inserted_h_count=inserted_h_count,
        )
    return {
        prefix_length: logits[0, len(prompt) + prefix_length - 1].detach().to(device="cpu", dtype=torch.float32)
        for prefix_length in prefix_lengths
    }


def capture_last_token_logits(
    model: Any,
    token_ids: Sequence[int],
    *,
    device: torch.device,
    inserted_h_count: int = 0,
    static_mgr: Optional[StaticCodebookManager] = None,
) -> torch.Tensor:
    """Return native-vocabulary logits for the token after ``token_ids``."""
    if not token_ids:
        raise AttributionError("Cannot capture logits for an empty prefix")
    input_tensor = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    if static_mgr is not None:
        static_mgr.reset()
    with torch.no_grad():
        output = model(input_ids=input_tensor, attention_mask=attention_mask, use_cache=False)
    logits = output.logits if hasattr(output, "logits") else output[0]
    last = logits[0, -1]
    if static_mgr is not None:
        output_layer = model.base_model.get_output_embeddings()
        last = normalize_wrapper_logits(
            last,
            base_vocab_size=int(output_layer.weight.shape[0]),
            initial_vocab_size=INITIAL_VOCAB_SIZE,
            inserted_h_count=inserted_h_count,
        )
    return last.detach().to(device="cpu", dtype=torch.float32)


def parity_tolerance(dtype: str) -> float:
    normalized = str(dtype).lower()
    if "bfloat16" in normalized:
        return 2e-2
    if "float16" in normalized or "half" in normalized:
        return 2e-3
    return 1e-5


def checkpoint_isolation_preflight(
    load_report: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    base_hash: str,
    expected_prompt_ids: Sequence[str],
) -> Dict[str, Any]:
    """Check the checkpoint loader changed only its declared trained groups."""
    b0_records = [r for r in records if r.get("condition") == COND_B0_TOKENS_VANILLA_WEIGHTS]
    b0_prompt_ids = {str(r.get("prompt_id")) for r in b0_records}
    b0_coverage = b0_prompt_ids == {str(prompt_id) for prompt_id in expected_prompt_ids}
    b0_hashes = {r.get("runtime", {}).get("base_phi_weight_sha256") for r in b0_records}
    component_reports = load_report.get("components", {})
    changes: Dict[str, Dict[str, Any]] = {}
    expected_groups = ("lora", "input_encoder", "output_encoder")
    reports_complete = bool(component_reports) and all(group in component_reports for group in expected_groups)
    reports_valid = reports_complete
    for group in expected_groups:
        component = component_reports.get(group, {})
        names = list(component.get("changed_parameter_names", []))
        shapes = dict(component.get("changed_parameter_shapes", {}))
        changes[group] = {
            "names": names,
            "shapes": shapes,
            "before_sha256": component.get("before_sha256"),
            "after_sha256": component.get("after_sha256"),
            "changed_tensor_count": int(component.get("changed_tensor_count", 0)),
            "missing_keys": list(component.get("missing_keys", [])),
            "unexpected_keys": list(component.get("unexpected_keys", [])),
        }
        reports_valid = reports_valid and not changes[group]["missing_keys"] and not changes[group]["unexpected_keys"]
        reports_valid = reports_valid and len(names) == changes[group]["changed_tensor_count"]
        reports_valid = reports_valid and set(names) == set(shapes)
    lora_active = bool(load_report.get("lora_tensors", 0)) and bool(changes.get("lora", {}).get("names"))
    b0_clean = b0_coverage and bool(b0_records) and all(
        r.get("runtime", {}).get("checkpoint_applied") is False
        and r.get("runtime", {}).get("step100_h_encoder_checkpoint_loaded") is False
        for r in b0_records
    )
    base_hash_match = bool(base_hash) and b0_hashes == {base_hash}
    frozen_base_ok = bool(load_report.get("frozen_base_unchanged_during_load"))
    passed = reports_valid and lora_active and b0_clean and base_hash_match and frozen_base_ok
    return {
        "status": "PASS" if passed else ("FAIL" if load_report else "NOT_TESTED"),
        "base_weight_hash_matches_b0": base_hash_match,
        "b0_prompt_coverage_complete": b0_coverage,
        "b0_checkpoint_and_step100_h_state_absent": b0_clean,
        "b1_lora_delta_active": lora_active,
        "changed_parameter_report_complete": reports_complete and reports_valid,
        "loader_frozen_base_unchanged": frozen_base_ok,
        "frozen_base_parameter_sha256": load_report.get("frozen_base_parameter_sha256"),
        "changed_parameters": changes,
    }


def run_a_b0_logit_parity(
    target_records: Sequence[Any],
    existing_records: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    device: str,
    *,
    max_new_tokens: int,
    k: int,
) -> Dict[str, Any]:
    """Measure same-prefix A/B0 logits for every selected DEV prompt."""
    indexed: Dict[Tuple[str, str], Mapping[str, Any]] = {
        (str(row.get("prompt_id")), str(row.get("condition"))): row for row in existing_records
    }
    selected, missing_prompt_ids = select_a_b0_logit_samples(target_records, existing_records)
    if missing_prompt_ids:
        return {
            "status": "FAIL",
            "reason": "Matched-prefix logit comparison requires an A and B0 record for every selected prompt",
            "checked_prompt_count": 0,
            "missing_prompt_ids": missing_prompt_ids,
        }
    if not selected:
        return {"status": "FAIL", "reason": "No selected A/B0 prompts were available for logit capture", "checked_prompt_count": 0}

    device_obj = torch.device(device)
    memory_lifecycle: Dict[str, Any] = {"A_before_load": cuda_memory_snapshot(device_obj)}
    vanilla_model, vanilla_tokenizer = load_vanilla_model_and_tokenizer(device)
    special_token_ids = sorted({int(token_id) for token_id in (getattr(vanilla_tokenizer, "all_special_ids", []) or [])} | set(CANONICAL_EOS_TOKEN_IDS))
    vanilla_capture: Dict[str, Dict[int, torch.Tensor]] = {}
    prompt_inputs: Dict[str, Tuple[List[int], List[int]]] = {}
    vanilla_model_ref = weakref.ref(vanilla_model)
    try:
        for sample in selected:
            pid = str(sample.prompt_id)
            a_record = indexed[(pid, COND_A_VANILLA)]
            prompt_text = build_canonical_prompt_text(asdict(sample))
            prompt_ids = vanilla_tokenizer.encode(prompt_text, add_special_tokens=False)
            a_runtime = a_record.get("runtime", {})
            if (
                prompt_text != a_runtime.get("rendered_prompt_text")
                or prompt_ids != a_runtime.get("input_token_ids")
                or token_ids_sha256(prompt_ids) != a_runtime.get("input_token_ids_sha256")
            ):
                return {"status": "FAIL", "reason": f"A prompt tokenization changed for {pid}", "checked_prompt_count": 0}
            generated_ids = [int(token_id) for token_id in a_record.get("generated_token_ids", [])]
            vanilla_capture[pid] = capture_prefix_logits(
                vanilla_model,
                prompt_ids,
                generated_ids,
                device=device_obj,
                inserted_h_count=0,
            )
            prompt_inputs[pid] = (prompt_ids, generated_ids)
    finally:
        a_before_teardown = cuda_memory_snapshot(device_obj)
        del vanilla_model
        memory_lifecycle["A_after_teardown"] = assert_model_phase_teardown(
            vanilla_model_ref,
            device_obj,
            phase="A logit capture",
            before=a_before_teardown,
        )

    # The A teardown guard above must pass before the second full model loads.
    memory_lifecycle["B0_before_load"] = cuda_memory_snapshot(device_obj)
    b0_model, b0_tokenizer, b0_report = load_predictive_bundle(
        checkpoint_path, device, load_peft_adapter=False, apply_joint_checkpoint=False, k=k
    )
    b0_model_ref = weakref.ref(b0_model)
    if (
        b0_report.get("checkpoint_applied") is not False
        or b0_report.get("step100_h_encoder_checkpoint_loaded") is not False
        or not b0_adapter_absence_verified(b0_report, adapter_runtime_metadata(b0_model))
    ):
        return {
            "status": "FAIL",
            "reason": "B0 matched-prefix load has checkpoint or PEFT adapter state",
            "checked_prompt_count": 0,
            "b0_load_report": b0_report,
        }
    b0_model_hash = base_phi_weight_sha256(b0_model)
    b0_dtype = str(b0_model.base_model.get_input_embeddings().weight.dtype)
    rows: List[Dict[str, Any]] = []
    vanilla_logits: List[torch.Tensor] = []
    wrapped_logits: List[torch.Tensor] = []
    b0_manager = prepare_empty_b0_manager(b0_model, k=k, device=device_obj)
    try:
        for sample in selected:
            pid = str(sample.prompt_id)
            b0_record = indexed.get((pid, COND_B0_TOKENS_VANILLA_WEIGHTS))
            if b0_record is None:
                continue
            prompt_ids, generated_ids = prompt_inputs[pid]
            prompt_text = build_canonical_prompt_text(asdict(sample))
            b0_prompt_ids = b0_tokenizer.encode(prompt_text, add_special_tokens=False)
            b0_runtime = b0_record.get("runtime", {})
            if (
                b0_prompt_ids != prompt_ids
                or prompt_text != b0_runtime.get("rendered_prompt_text")
                or b0_prompt_ids != b0_runtime.get("input_token_ids")
                or token_ids_sha256(b0_prompt_ids) != b0_runtime.get("input_token_ids_sha256")
            ):
                return {"status": "FAIL", "reason": f"A/B0 prompt tokenization differs for {pid}", "checked_prompt_count": len(rows)}
            reset_empty_b0_manager(b0_manager, device_obj)
            adapter_state = adapter_runtime_metadata(b0_model)
            if not b0_adapter_absence_verified(b0_report, adapter_state):
                return {
                    "status": "FAIL",
                    "reason": f"B0 had PEFT adapter state at matched prefix for {pid}",
                    "checked_prompt_count": len(rows),
                    "adapter_state": adapter_state,
                }
            wrapped_capture = capture_prefix_logits(
                b0_model,
                b0_prompt_ids,
                generated_ids,
                device=device_obj,
                inserted_h_count=k,
                static_mgr=b0_manager,
            )
            a_capture = vanilla_capture[pid]
            if set(a_capture) != set(wrapped_capture):
                return {"status": "FAIL", "reason": f"A/B0 prefix-state sets differ for {pid}", "checked_prompt_count": len(rows)}
            dtype = str(b0_model.base_model.get_input_embeddings().weight.dtype)
            atol = parity_tolerance(dtype)
            prefix_results = []
            for prefix_length in sorted(a_capture):
                vanilla_logits.append(a_capture[prefix_length])
                wrapped_logits.append(wrapped_capture[prefix_length])
                metric = logit_parity_metrics(
                    a_capture[prefix_length],
                    wrapped_capture[prefix_length],
                    eos_token_ids=CANONICAL_EOS_TOKEN_IDS,
                    special_token_ids=special_token_ids,
                    top_k=5,
                    atol=atol,
                )
                metric["continuation_prefix_length"] = prefix_length
                prefix_results.append(metric)
            rows.append({
                "prompt_id": pid,
                "domain": sample.domain,
                "base_weight_hash_matches": b0_record.get("runtime", {}).get("base_phi_weight_sha256") == b0_model_hash,
                "prefix_results": prefix_results,
            })
    finally:
        b0_before_teardown = cuda_memory_snapshot(device_obj)
        expects_position_hook = bool(getattr(b0_model, "uses_base_token_positions", False))
        position_hook_removed = remove_model_generation_hooks(b0_model)
        manager_ref = None
        if b0_manager is not None:
            b0_manager.reset(clear_dictionary=True, clear_caches=True)
            b0_manager.clear_weight_caches()
            manager_ref = weakref.ref(b0_manager)
            del b0_manager
        del b0_model
        memory_lifecycle["B0_after_teardown"] = assert_model_phase_teardown(
            b0_model_ref,
            device_obj,
            phase="B0 logit capture",
            before=b0_before_teardown,
            manager_ref=manager_ref,
        )
        memory_lifecycle["B0_after_teardown"]["position_hook_removed"] = position_hook_removed
        if expects_position_hook and not position_hook_removed:
            raise AttributionError("B0 teardown could not remove the span-aware generation hook")

    statuses = [metric["status"] for row in rows for metric in row["prefix_results"]]
    hashes_match = bool(rows) and all(row["base_weight_hash_matches"] for row in rows)
    dtype = b0_dtype
    atol = parity_tolerance(dtype)
    aggregate = (
        logit_parity_metrics(
            torch.stack(vanilla_logits),
            torch.stack(wrapped_logits),
            eos_token_ids=CANONICAL_EOS_TOKEN_IDS,
            special_token_ids=special_token_ids,
            top_k=5,
            atol=atol,
        )
        if vanilla_logits
        else {"status": "NOT_TESTED", "reason": "No matched prefix logits were captured"}
    )
    passed = (
        len(rows) == len(target_records)
        and bool(statuses)
        and all(status == "PASS" for status in statuses)
        and aggregate.get("status") == "PASS"
        and hashes_match
        and not b0_report.get("checkpoint_applied")
    )
    return {
        "status": "PASS" if passed else ("FAIL" if statuses else "NOT_TESTED"),
        "checked_prompt_count": len(rows),
        "checked_state_count": len(statuses),
        "dtype": dtype,
        "absolute_tolerance": atol,
        "base_weight_hashes_match": hashes_match,
        "b0_checkpoint_applied": bool(b0_report.get("checkpoint_applied")),
        "memory_lifecycle": memory_lifecycle,
        "aggregate": aggregate,
        "per_prompt": rows,
    }


def select_a_b0_logit_samples(
    target_records: Sequence[Any],
    existing_records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Any], List[str]]:
    """Select every requested prompt with both A and B0 records, preserving order."""
    present = {
        (str(record.get("prompt_id")), str(record.get("condition")))
        for record in existing_records
    }
    selected: List[Any] = []
    missing: List[str] = []
    for sample in target_records:
        prompt_id = str(getattr(sample, "prompt_id", ""))
        if (prompt_id, COND_A_VANILLA) in present and (prompt_id, COND_B0_TOKENS_VANILLA_WEIGHTS) in present:
            selected.append(sample)
        else:
            missing.append(prompt_id)
    return selected, missing


def run_a_b1_logit_fidelity(
    target_records: Sequence[Any],
    existing_records: Sequence[Mapping[str, Any]],
    b1_model: Any,
    b1_tokenizer: Any,
    device: str,
    *,
    k: int,
) -> Dict[str, Any]:
    """Measure Vanilla-to-B1 base-logit drift at identical Vanilla prefixes.

    This is a diagnostic, not an acceptance gate: A/B0 must already have
    passed before B1 is loaded, and these measurements quantify the trained
    checkpoint's base-token logit drift without letting generation drift
    compound the comparison.
    """
    indexed: Dict[Tuple[str, str], Mapping[str, Any]] = {
        (str(row.get("prompt_id")), str(row.get("condition"))): row for row in existing_records
    }
    selected: List[Any] = []
    domain_count = {"code": 0, "reasoning": 0, "instruction": 0}
    for sample in target_records:
        domain = str(getattr(sample, "domain", ""))
        pid = str(getattr(sample, "prompt_id", ""))
        if (
            domain in domain_count
            and domain_count[domain] < 2
            and (pid, COND_A_VANILLA) in indexed
            and (pid, COND_B0_TOKENS_VANILLA_WEIGHTS) in indexed
        ):
            selected.append(sample)
            domain_count[domain] += 1
    if not selected:
        return {"status": "NOT_TESTED", "reason": "No paired A/B0 DEV records are available", "checked_prompt_count": 0}

    device_obj = torch.device(device)
    vanilla_model, vanilla_tokenizer = load_vanilla_model_and_tokenizer(device)
    special_token_ids = sorted({
        int(token_id) for token_id in (getattr(vanilla_tokenizer, "all_special_ids", []) or [])
    } | set(CANONICAL_EOS_TOKEN_IDS))
    rows: List[Dict[str, Any]] = []
    vanilla_logits: List[torch.Tensor] = []
    b1_logits: List[torch.Tensor] = []
    try:
        dim = getattr(b1_model.config, "hidden_size", 3072)
        for sample in selected:
            pid = str(sample.prompt_id)
            a_record = indexed[(pid, COND_A_VANILLA)]
            prompt_text = build_canonical_prompt_text(asdict(sample))
            a_prompt_ids = vanilla_tokenizer.encode(prompt_text, add_special_tokens=False)
            if token_ids_sha256(a_prompt_ids) != a_record.get("runtime", {}).get("input_token_ids_sha256"):
                return {"status": "FAIL", "reason": f"A prompt tokenization changed for {pid}", "checked_prompt_count": len(rows)}
            b1_prompt_ids = b1_tokenizer.encode(prompt_text, add_special_tokens=False)
            if b1_prompt_ids != a_prompt_ids:
                return {"status": "FAIL", "reason": f"A/B1 prompt tokenization differs for {pid}", "checked_prompt_count": len(rows)}

            generated_ids = [int(token_id) for token_id in a_record.get("generated_token_ids", [])]
            a_capture = capture_prefix_logits(
                vanilla_model, a_prompt_ids, generated_ids,
                device=device_obj, inserted_h_count=0,
            )
            manager = StaticCodebookManager(
                initial_vocab_size=INITIAL_VOCAB_SIZE,
                max_codebook_size=k,
                max_subtokens=4,
                embedding_dim=dim,
                pad_token_id=PAD_TOKEN_ID,
            )
            manager.set_seeded_codebook({}, batch_size=1, device=device_obj)
            manager.attach_to_model(b1_model)
            b1_capture = capture_prefix_logits(
                b1_model, b1_prompt_ids, generated_ids,
                device=device_obj, inserted_h_count=k, static_mgr=manager,
            )
            if set(a_capture) != set(b1_capture):
                return {"status": "FAIL", "reason": f"A/B1 prefix-state sets differ for {pid}", "checked_prompt_count": len(rows)}

            prefix_results = []
            for prefix_length in sorted(a_capture):
                a_state = a_capture[prefix_length]
                b1_state = b1_capture[prefix_length]
                vanilla_logits.append(a_state)
                b1_logits.append(b1_state)
                prefix_results.append({
                    "continuation_prefix_length": prefix_length,
                    **logit_parity_metrics(
                        a_state, b1_state,
                        eos_token_ids=CANONICAL_EOS_TOKEN_IDS,
                        special_token_ids=special_token_ids,
                        top_k=5,
                        atol=0.0,
                    ),
                })
            rows.append({"prompt_id": pid, "domain": sample.domain, "prefix_results": prefix_results})
    finally:
        del vanilla_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not vanilla_logits:
        return {"status": "NOT_TESTED", "reason": "No paired prefix states were captured", "checked_prompt_count": len(rows)}
    dtype = str(b1_model.base_model.get_input_embeddings().weight.dtype)
    aggregate = logit_parity_metrics(
        torch.stack(vanilla_logits), torch.stack(b1_logits),
        eos_token_ids=CANONICAL_EOS_TOKEN_IDS,
        special_token_ids=special_token_ids,
        top_k=5,
        atol=parity_tolerance(dtype),
    )
    return {
        "status": "MEASURED",
        "strict_parity_status": aggregate["status"],
        "interpretation": "Diagnostic only; A/B0 is the architecture gate. This compares A and B1 at identical Vanilla-prefix states.",
        "checked_prompt_count": len(rows),
        "checked_state_count": len(vanilla_logits),
        "dtype": dtype,
        "aggregate": aggregate,
        "per_prompt": rows,
    }


DEV_B1_DIAGNOSTIC_SUBSET_PROMPT_IDS: Tuple[str, ...] = (
    "mbpp_113",
    "mbpp_168",
    "gsm_2032",
    "gsm_2491",
    "alpaca_1",
    "alpaca_1024",
)


def run_a_b1_matched_prefix_diagnostic(
    target_records: Sequence[Any],
    existing_records: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    device: str,
    *,
    k: int = 32,
    diagnostic_prefix_lengths: Sequence[int] = (0, 1, 4, 16),
) -> Dict[str, Any]:
    """Measure Vanilla vs B1 (upstream adapter only) matched-prefix logit drift sequentially.

    Runs Phase A (Vanilla) completely, frees memory, then runs Phase B1 (upstream adapter only)
    to prevent co-residence on GPU.
    """
    indexed: Dict[Tuple[str, str], Mapping[str, Any]] = {
        (str(row.get("prompt_id")), str(row.get("condition"))): row for row in existing_records
    }
    selected: List[Any] = []
    missing_ids: List[str] = []
    for sample in target_records:
        pid = str(getattr(sample, "prompt_id", ""))
        if (pid, COND_A_VANILLA) in indexed:
            selected.append(sample)
        else:
            missing_ids.append(pid)

    if not selected:
        return {
            "status": "NOT_TESTED",
            "reason": "No A_vanilla records are available for matched-prefix diagnostic",
            "checked_prompt_count": 0,
        }

    device_obj = torch.device(device)
    memory_lifecycle: Dict[str, Any] = {"A_before_load": cuda_memory_snapshot(device_obj)}

    # Phase A: Vanilla forward passes
    vanilla_model, vanilla_tok = load_vanilla_model_and_tokenizer(device)
    special_token_ids = sorted({
        int(token_id) for token_id in (getattr(vanilla_tok, "all_special_ids", []) or [])
    } | set(CANONICAL_EOS_TOKEN_IDS))
    vanilla_capture: Dict[str, Dict[int, torch.Tensor]] = {}
    prompt_inputs: Dict[str, Tuple[List[int], str, List[int]]] = {}
    vanilla_model_ref = weakref.ref(vanilla_model)
    try:
        for sample in selected:
            pid = str(sample.prompt_id)
            a_record = indexed[(pid, COND_A_VANILLA)]
            prompt_text = build_canonical_prompt_text(asdict(sample))
            prompt_ids = vanilla_tok.encode(prompt_text, add_special_tokens=False)
            generated_ids = [int(t) for t in a_record.get("generated_token_ids", [])]
            raw_capture = capture_prefix_logits(
                vanilla_model, prompt_ids, generated_ids,
                device=device_obj, inserted_h_count=0,
            )
            # Store on CPU to release GPU tensor memory
            vanilla_capture[pid] = {
                length: tensor.detach().to(device="cpu", dtype=torch.float32)
                for length, tensor in raw_capture.items()
                if length in diagnostic_prefix_lengths
            }
            prompt_inputs[pid] = (prompt_ids, prompt_text, generated_ids)
    finally:
        a_before_teardown = cuda_memory_snapshot(device_obj)
        del vanilla_model
        memory_lifecycle["A_after_teardown"] = assert_model_phase_teardown(
            vanilla_model_ref,
            device_obj,
            phase="A diagnostic prefix capture",
            before=a_before_teardown,
        )

    # Phase B1: Upstream PEFT adapter only (NO Step-100 checkpoint)
    memory_lifecycle["B1_before_load"] = cuda_memory_snapshot(device_obj)
    b1_model, b1_tok, b1_report = load_predictive_bundle(
        checkpoint_path, device, load_peft_adapter=True, apply_joint_checkpoint=False, k=k
    )
    b1_model_ref = weakref.ref(b1_model)
    b1_manager = prepare_empty_b0_manager(b1_model, k=k, device=device_obj)
    b1_capture: Dict[str, Dict[int, torch.Tensor]] = {}
    manager_ref = weakref.ref(b1_manager)
    try:
        for sample in selected:
            pid = str(sample.prompt_id)
            prompt_ids, prompt_text, generated_ids = prompt_inputs[pid]
            b1_prompt_ids = b1_tok.encode(prompt_text, add_special_tokens=False)
            reset_empty_b0_manager(b1_manager, device_obj)
            raw_capture = capture_prefix_logits(
                b1_model, b1_prompt_ids, generated_ids,
                device=device_obj, inserted_h_count=k, static_mgr=b1_manager,
            )
            b1_capture[pid] = {
                length: tensor.detach().to(device="cpu", dtype=torch.float32)
                for length, tensor in raw_capture.items()
                if length in diagnostic_prefix_lengths
            }
    finally:
        b1_before_teardown = cuda_memory_snapshot(device_obj)
        remove_model_generation_hooks(b1_model)
        b1_manager.reset(clear_dictionary=True, clear_caches=True)
        del b1_manager
        del b1_model
        memory_lifecycle["B1_after_teardown"] = assert_model_phase_teardown(
            b1_model_ref,
            device_obj,
            phase="B1 diagnostic prefix capture",
            before=b1_before_teardown,
            manager_ref=manager_ref,
        )

    # Comparison Phase
    rows: List[Dict[str, Any]] = []
    all_vanilla_logits: List[torch.Tensor] = []
    all_b1_logits: List[torch.Tensor] = []
    for sample in selected:
        pid = str(sample.prompt_id)
        a_states = vanilla_capture[pid]
        b1_states = b1_capture[pid]
        prefix_results = []
        for prefix_len in sorted(set(a_states) & set(b1_states)):
            a_vec = a_states[prefix_len]
            b1_vec = b1_states[prefix_len]
            all_vanilla_logits.append(a_vec)
            all_b1_logits.append(b1_vec)
            metrics = logit_parity_metrics(
                a_vec, b1_vec,
                eos_token_ids=CANONICAL_EOS_TOKEN_IDS,
                special_token_ids=special_token_ids,
                top_k=5,
                atol=0.0,
            )
            top1_a = int(a_vec.argmax(dim=-1).item())
            top1_b1 = int(b1_vec.argmax(dim=-1).item())
            eos_32007 = metrics.get("eos_special_logit_differences", {}).get("32007", {})
            prefix_results.append({
                "continuation_prefix_length": prefix_len,
                "top1_vanilla_id": top1_a,
                "top1_b1_id": top1_b1,
                "top1_agreement": bool(top1_a == top1_b1),
                "max_abs_logit_difference": metrics["max_abs_logit_difference"],
                "mean_abs_logit_difference": metrics["mean_abs_logit_difference"],
                "kl_vanilla_to_b1_nats": metrics["mean_kl_reference_to_candidate_nats"],
                "top_k_overlap": metrics["top_k_overlap_mean"],
                "mean_abs_top1_margin_difference": metrics["mean_abs_top1_margin_difference"],
                "eos_32007": eos_32007,
                "full_metrics": metrics,
            })
        rows.append({
            "prompt_id": pid,
            "domain": sample.domain,
            "prefix_results": prefix_results,
        })

    if not all_vanilla_logits:
        return {
            "status": "NOT_TESTED",
            "reason": "No matched prefix states were compared",
            "checked_prompt_count": 0,
        }

    stacked_a = torch.stack(all_vanilla_logits)
    stacked_b1 = torch.stack(all_b1_logits)
    aggregate = logit_parity_metrics(
        stacked_a, stacked_b1,
        eos_token_ids=CANONICAL_EOS_TOKEN_IDS,
        special_token_ids=special_token_ids,
        top_k=5,
        atol=0.0,
    )
    eos_32007_agg = aggregate.get("eos_special_logit_differences", {}).get("32007", {})
    divergence_detected = bool(
        aggregate.get("top1_agreement_rate", 1.0) < 0.999
        or aggregate.get("mean_kl_reference_to_candidate_nats", 0.0) > 0.05
        or abs(eos_32007_agg.get("mean_abs_difference", 0.0)) > 0.5
    )
    return {
        "status": "MEASURED",
        "divergence_detected": divergence_detected,
        "checked_prompt_count": len(rows),
        "checked_state_count": len(all_vanilla_logits),
        "aggregate": aggregate,
        "eos_32007_aggregate": eos_32007_agg,
        "per_prompt": rows,
        "memory_lifecycle": memory_lifecycle,
        "b1_provenance": {
            "upstream_adapter_id": CANONICAL_ZIP2ZIP_ID,
            "upstream_adapter_revision": CANONICAL_ZIP2ZIP_REVISION,
            "peft_version": b1_report.get("peft_version"),
            "adapter_param_count": b1_report.get("adapter_param_count"),
            "adapter_param_names_count": len(b1_report.get("adapter_param_names", [])),
            "checkpoint_applied": False,
            "step100_h_encoder_checkpoint_loaded": False,
        },
    }


def generate_single_prompt(
    model: Any,
    tokenizer: Any,
    input_ids: List[int],
    device: torch.device,
    static_mgr: Optional[StaticCodebookManager] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    extra_logits_processors: Sequence[Any] = (),
) -> Tuple[List[int], float, float, int]:
    """Execute forward generation loop and measure accurate TTFT, decode time, and steps."""
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    synchronize_device(device)
    t_start = time.perf_counter()

    timing_proc = TimingLogitsProcessor(t_start, static_mgr=static_mgr)
    proc_list = LogitsProcessorList([timing_proc, *extra_logits_processors])

    with torch.no_grad():
        out = model.generate(
            input_ids=input_tensor,
            logits_processor=proc_list,
            **generation_policy(max_new_tokens),
        )

    synchronize_device(device)
    t_end = time.perf_counter()
    total_gen_time = t_end - t_start
    ttft = timing_proc.ttft if timing_proc.ttft is not None else total_gen_time
    decode_time = max(0.0, total_gen_time - ttft)

    gen_ids = out[0, len(input_ids) :].tolist()
    steps = len(gen_ids)
    return gen_ids, ttft, decode_time, steps


def run_attribution_benchmark(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_attribution_records.jsonl"
    split_marker = output_dir / "attribution_split.txt"
    summary_path = output_dir / "attribution_summary.json"
    report_path = output_dir / "attribution_report.md"

    git_commit = current_git_commit()
    runtime = runtime_metadata(args.device)
    device_obj = torch.device(args.device)
    requested_live = {
        COND_B1_UPSTREAM_EPFL_ADAPTER_H_DISABLED,
        COND_B2_STEP100_H_DISABLED,
        COND_B_H_DISABLED,
        COND_C_ORACLE,
        COND_CF_FORCED_ORACLE,
        COND_D_REAL_PREDICTOR,
    }
    if getattr(args, "logit_only", False):
        if (
            args.split.upper() != "DEV"
            or set(args.conditions) != {COND_A_VANILLA, COND_B0_TOKENS_VANILLA_WEIGHTS}
            or args.prompt_limit != len(STRATIFIED_DEV12_PROMPT_IDS)
        ):
            raise AttributionError(
                "--logit-only is pinned to DEV12 and conditions A_vanilla plus B0_tokens_vanilla_weights"
            )
    if {COND_C_ORACLE, COND_D_REAL_PREDICTOR}.intersection(args.conditions) and COND_CF_FORCED_ORACLE not in args.conditions:
        raise AttributionError("C/D live H evaluation requires CF FORCED ORACLE to pass first")
    if {COND_CF_FORCED_ORACLE, COND_C_ORACLE, COND_D_REAL_PREDICTOR}.intersection(args.conditions) and not {
        COND_B1_UPSTREAM_EPFL_ADAPTER_H_DISABLED, COND_B2_STEP100_H_DISABLED, COND_B_H_DISABLED
    }.intersection(args.conditions):
        raise AttributionError("CF/C/D require B1/B2/B H-disabled records as the matched continuation target")
    if args.k != 32:
        raise AttributionError("The attribution gates are pinned to K=32; use --k 32")
    if raw_path.exists():
        if not split_marker.is_file():
            raise AttributionError("Refusing to resume an unmarked attribution file before checking its split")
        recorded_split = split_marker.read_text(encoding="utf-8").strip().upper()
        if recorded_split != args.split.upper():
            raise AttributionError(
                f"Attribution output is pinned to {recorded_split}, not requested {args.split.upper()}"
            )
    else:
        if split_marker.exists():
            recorded_split = split_marker.read_text(encoding="utf-8").strip().upper()
            if recorded_split != args.split.upper():
                raise AttributionError(
                    f"Attribution output is pinned to {recorded_split}, not requested {args.split.upper()}"
                )
        else:
            with split_marker.open("x", encoding="utf-8") as marker_file:
                marker_file.write(args.split.upper() + "\n")

    # 1. Load canonical dataset and split
    check_split_safety(args.split)
    manifest_path = Path(args.canonical_manifest)
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        input_manifest = json.load(manifest_file)
    require_split_isolated_manifest(input_manifest, args.split)
    runtime["dataset_manifest_sha256"] = sha256_file(manifest_path)
    runtime["canonical_dataset_sha256"] = str(input_manifest.get("dataset_sha256", ""))
    if {COND_A_VANILLA, COND_B0_TOKENS_VANILLA_WEIGHTS}.issubset(set(args.conditions)):
        required_provenance = (
            "git_commit",
            "git_branch",
            "git_worktree_dirty",
            "git_status_sha256",
            "source_archive_sha256",
            "dataset_manifest_sha256",
            "canonical_dataset_sha256",
        )
        missing_provenance = [
            key for key in required_provenance
            if runtime.get(key) is None or runtime.get(key) == ""
        ]
        if missing_provenance:
            raise AttributionError(
                "A/B0 requires complete source and dataset provenance before model loading; "
                "missing: " + ", ".join(missing_provenance)
            )
        if runtime.get("git_commit") == "UNKNOWN" or runtime.get("git_branch") == "UNKNOWN":
            raise AttributionError("A/B0 source provenance must identify a concrete Git commit and branch")
        for key in ("source_archive_sha256", "git_status_sha256", "dataset_manifest_sha256", "canonical_dataset_sha256"):
            value = str(runtime.get(key, ""))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
                raise AttributionError(f"A/B0 provenance field {key} is not a SHA-256 digest")
        if runtime.get("git_commit") != git_commit:
            raise AttributionError("Source provenance commit does not match the runner's git commit")
        if not runtime.get("package_versions_complete"):
            raise AttributionError("A/B0 requires complete versions for the model runtime packages")
    views, _ = load_canonical_dataset(args.canonical_dataset, args.canonical_manifest)
    if args.split.upper() == "DEV":
        split_records = getattr(views, "dev", getattr(views, "dev_records", None))
    elif args.split.upper() == "TRAIN":
        split_records = getattr(views, "train", getattr(views, "train_records", None))
    else:
        raise AttributionError(f"Unsupported split: {args.split}")

    # Stratified selection
    target_records = select_stratified_dev_prompts(split_records, limit=args.prompt_limit)
    print(f"Selected {len(target_records)} prompts from {args.split} split.")

    # Load existing completions for resume
    completed: Set[Tuple[str, str]] = set()
    existing_records: List[Dict[str, Any]] = []
    if raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        pair = (rec.get("prompt_id"), rec.get("condition"))
                        if pair[0] and pair[1]:
                            completed.add(pair)
                            existing_records.append(rec)
                    except json.JSONDecodeError:
                        pass
        print(f"Resumed: loaded {len(completed)} existing prompt-condition records from {raw_path}")

    if getattr(args, "logit_only", False):
        selected_ids = [str(record.prompt_id) for record in target_records]
        if selected_ids != list(STRATIFIED_DEV12_PROMPT_IDS):
            raise AttributionError("--logit-only target selection does not match the pinned DEV12 prompt order")
        offline_gate = offline_a_b0_token_gate(
            existing_records,
            expected_prompt_ids=STRATIFIED_DEV12_PROMPT_IDS,
        )
        expected_record_count = 2 * len(STRATIFIED_DEV12_PROMPT_IDS)
        if len(existing_records) != expected_record_count or offline_gate.get("status") != "PASS":
            raise AttributionError(
                "--logit-only requires exactly 24 complete, unique A/B0 records that pass the offline gate; "
                + json.dumps(offline_gate, sort_keys=True)
            )
        print(
            "Offline A/B0 token gate passed for 12 pinned prompts; entering matched-prefix logit-only mode.",
            flush=True,
        )

    # Build reference lookup for Vanilla continuations (from canonical dataset)
    vanilla_continuation_map: Dict[str, List[int]] = {
        r.prompt_id: list(r.continuation_token_ids) for r in split_records
    }

    # Load ground-truth references if available
    gt_map: Dict[str, str] = {}
    gt_path = getattr(args, "ground_truth", None)
    if not gt_path and args.canonical_dataset:
        candidate_gt = Path(args.canonical_dataset).parent / "dev_ground_truth.json"
        if candidate_gt.is_file():
            gt_path = str(candidate_gt)
    if gt_path and Path(gt_path).is_file():
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_map = json.load(f)
        print(f"Loaded ground truth references for {len(gt_map)} prompts.")

    # Initialize predictor and candidate generator if Condition D is active
    cand_index = None
    ranker_model = None
    if COND_D_REAL_PREDICTOR in args.conditions:
        print("Initializing Phi-only Candidate Retrieval Index and PooledMLP ranker...", flush=True)
        idx_path = Path(args.association_index)
        cand_index = TrainOnlyAssociationIndex.load(str(idx_path))
        with open(args.predictor_model, "rb") as pf:
            ranker_model = pickle.load(pf)

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_sha = sha256_file(checkpoint_path) if checkpoint_path.is_file() else ""

    # Open raw file in append mode
    raw_file = raw_path.open("a", encoding="utf-8", buffering=1)

    # 2. RUN CONDITION A: VANILLA (if selected and needed)
    if COND_A_VANILLA in args.conditions:
        pending_a = [r for r in target_records if (r.prompt_id, COND_A_VANILLA) not in completed]
        if pending_a:
            v_model, v_tok = load_vanilla_model_and_tokenizer(args.device)
            a_base_hash = base_phi_weight_sha256(v_model)
            for idx, sample in enumerate(pending_a, 1):
                pid = sample.prompt_id
                dom = sample.domain
                print(f"[A: Vanilla {idx}/{len(pending_a)}] Prompt {pid} ({dom})...", flush=True)

                prompt_text = build_canonical_prompt_text(asdict(sample))
                input_ids = v_tok.encode(prompt_text, add_special_tokens=False)

                t_start = time.perf_counter()
                gen_ids, ttft, dec_time, steps = generate_single_prompt(
                    v_model, v_tok, input_ids, device_obj, static_mgr=None, max_new_tokens=args.max_new_tokens
                )
                total_lat = time.perf_counter() - t_start

                eos_hit = bool(gen_ids and gen_ids[-1] in CANONICAL_EOS_TOKEN_IDS)
                term_reason = "eos" if eos_hit else "length"
                term_token = gen_ids[-1] if gen_ids else None

                output_text = v_tok.decode(gen_ids, skip_special_tokens=True)
                sample_dict = asdict(sample)
                if pid in gt_map:
                    sample_dict["reference_response"] = gt_map[pid]
                sample_dict["eos_reached"] = eos_hit
                pass_status, scores = evaluate_output_quality(dom, output_text, sample_dict)

                rec = AttributionRecord(
                    prompt_id=pid,
                    domain=dom,
                    split=sample.split,
                    condition=COND_A_VANILLA,
                    created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    git_commit=git_commit,
                    runtime={
                        **runtime,
                        "attention_implementation": effective_attention_implementation(v_model),
                        "torch_dtype": str(v_model.get_input_embeddings().weight.dtype),
                        "ablation_role": "A_PURE_VANILLA",
                        "base_phi_weight_sha256": a_base_hash,
                        **prompt_runtime_metadata(v_tok, prompt_text, input_ids, args.max_new_tokens),
                        "lora_delta_verified": False,
                    },
                    raw_output=output_text,
                    expanded_output=output_text,
                    generated_token_ids=gen_ids,
                    expanded_token_ids=gen_ids,
                    generated_token_count=len(gen_ids),
                    expanded_token_count=len(gen_ids),
                    termination_reason=term_reason,
                    termination_token_id=term_token,
                    eos_reached=eos_hit,
                    truncated=(not eos_hit and len(gen_ids) >= args.max_new_tokens),
                    task_quality_scores=scores,
                    quality_gate_pass=pass_status,
                    ttft_s=round(ttft, 4),
                    decode_time_s=round(dec_time, 4),
                    tpot_s=round(dec_time / max(steps, 1), 6) if steps > 0 else None,
                    total_latency_s=round(total_lat, 4),
                    time_to_eos_s=round(total_lat, 4),
                    transformer_decode_calls=steps,
                    raw_model_steps=steps,
                    words_per_sec=round(len(output_text.split()) / max(total_lat, 1e-4), 2),
                )
                rec_dict = rec.to_dict()
                raw_file.write(json.dumps(rec_dict, ensure_ascii=False) + "\n")
                completed.add((pid, COND_A_VANILLA))
                existing_records.append(rec_dict)

            a_memory_before_teardown = cuda_memory_snapshot(device_obj)
            v_model_ref = weakref.ref(v_model)
            del v_model
            a_teardown = assert_model_phase_teardown(
                v_model_ref,
                device_obj,
                phase="A generation",
                before=a_memory_before_teardown,
            )
            print(f"A model teardown: {json.dumps(a_teardown, sort_keys=True)}", flush=True)

    # 2b. RUN CONDITION B0: TOKENS WRAPPER WITH VANILLA WEIGHTS, NO CHECKPOINT
    b0_pending = [r for r in target_records if (r.prompt_id, COND_B0_TOKENS_VANILLA_WEIGHTS) not in completed]
    if COND_B0_TOKENS_VANILLA_WEIGHTS in args.conditions and b0_pending:
        b0_model, b0_tok, b0_load_report = load_predictive_bundle(
            checkpoint_path, args.device, load_peft_adapter=False, apply_joint_checkpoint=False, k=args.k
        )
        b0_adapter_state = adapter_runtime_metadata(b0_model)
        if (
            b0_load_report.get("checkpoint_applied") is not False
            or b0_load_report.get("step100_h_encoder_checkpoint_loaded") is not False
            or not b0_adapter_absence_verified(b0_load_report, b0_adapter_state)
        ):
            raise AttributionError(
                "B0 requires no Step-100 checkpoint and no PEFT adapter state"
            )
        b0_base_hash = base_phi_weight_sha256(b0_model)
        b0_runtime = {
            **runtime,
            "attention_implementation": effective_attention_implementation(b0_model),
            "torch_dtype": str(b0_model.base_model.get_input_embeddings().weight.dtype),
            "ablation_role": "B0_TOKENS_ARCHITECTURE_VANILLA_WEIGHTS",
            "base_phi_weight_sha256": b0_base_hash,
            "checkpoint_applied": bool(b0_load_report.get("checkpoint_applied")),
            "step100_h_encoder_checkpoint_loaded": b0_load_report.get("step100_h_encoder_checkpoint_loaded"),
            "adapter_load_requested": b0_load_report.get("adapter_load_requested"),
            "adapter_present_in_wrapper": bool(b0_load_report.get("adapter_present")),
            "adapter_modules_to_save": b0_load_report.get("adapter_modules_to_save", {}),
            "active_adapter_names": b0_load_report.get("active_adapter_names", []),
            "adapter_disabled_context_used": False,
            "lora_delta_verified": False,
            "h_seeded_slots": 0,
            "h_logits_masked_by_static_manager": True,
            "loaded_position_mode": b0_load_report.get("loaded_position_mode"),
            "effective_position_mode": b0_load_report.get("effective_position_mode"),
            "codebook_backend": b0_load_report.get("codebook_backend"),
        }
        static_mgr = None
        try:
            static_mgr = prepare_empty_b0_manager(b0_model, k=args.k, device=device_obj)
            for idx, sample in enumerate(b0_pending, 1):
                pid = sample.prompt_id
                dom = sample.domain
                print(f"[B0: Tokens wrapper / Vanilla weights {idx}/{len(b0_pending)}] Prompt {pid} ({dom})...", flush=True)
                reset_empty_b0_manager(static_mgr, device_obj)
                prompt_text = build_canonical_prompt_text(asdict(sample))
                input_ids = b0_tok.encode(prompt_text, add_special_tokens=False)
                h_mask_verified = verify_empty_static_h_mask(static_mgr, device_obj)
                t_start = time.perf_counter()
                adapter_state_during_generation = adapter_runtime_metadata(b0_model)
                if not b0_adapter_absence_verified(b0_load_report, adapter_state_during_generation):
                    raise AttributionError("B0 contains PEFT adapter state during generation")
                gen_ids, ttft, dec_time, steps = generate_single_prompt(
                    b0_model, b0_tok, input_ids, device_obj, static_mgr=static_mgr, max_new_tokens=args.max_new_tokens
                )
                total_lat = time.perf_counter() - t_start
                eos_hit = bool(gen_ids and gen_ids[-1] in CANONICAL_EOS_TOKEN_IDS)
                output_text = b0_tok.decode(gen_ids, skip_special_tokens=True)
                sample_dict = asdict(sample)
                if pid in gt_map:
                    sample_dict["reference_response"] = gt_map[pid]
                sample_dict["eos_reached"] = eos_hit
                pass_status, scores = evaluate_output_quality(dom, output_text, sample_dict)
                peak_vram = torch.cuda.max_memory_allocated(device_obj) if device_obj.type == "cuda" else None
                rec = AttributionRecord(
                    prompt_id=pid,
                    domain=dom,
                    split=sample.split,
                    condition=COND_B0_TOKENS_VANILLA_WEIGHTS,
                    created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    checkpoint_name="NONE",
                    checkpoint_sha256="",
                    git_commit=git_commit,
                    runtime={
                        **b0_runtime,
                        "h_seeded_slots": int(static_mgr.num_seeded),
                        "h_logits_masked_by_static_manager": h_mask_verified,
                        **prompt_runtime_metadata(b0_tok, prompt_text, input_ids, args.max_new_tokens),
                        "adapter_load_requested": b0_load_report.get("adapter_load_requested"),
                        "adapter_disabled_context_used": False,
                        "active_lora_during_generation": False,
                        "adapters_disabled_state_during_generation": adapter_state_during_generation.get(
                            "adapters_disabled_state"
                        ),
                    },
                    raw_output=output_text,
                    expanded_output=output_text,
                    generated_token_ids=gen_ids,
                    expanded_token_ids=gen_ids,
                    generated_token_count=len(gen_ids),
                    expanded_token_count=len(gen_ids),
                    termination_reason="eos" if eos_hit else "length",
                    termination_token_id=gen_ids[-1] if gen_ids else None,
                    eos_reached=eos_hit,
                    truncated=(not eos_hit and len(gen_ids) >= args.max_new_tokens),
                    task_quality_scores=scores,
                    quality_gate_pass=pass_status,
                    ttft_s=round(ttft, 4),
                    decode_time_s=round(dec_time, 4),
                    tpot_s=round(dec_time / max(steps, 1), 6) if steps > 0 else None,
                    total_latency_s=round(total_lat, 4),
                    time_to_eos_s=round(total_lat, 4),
                    transformer_decode_calls=steps,
                    raw_model_steps=steps,
                    words_per_sec=round(len(output_text.split()) / max(total_lat, 1e-4), 2),
                    setup_time_s=0.0,
                    peak_vram_bytes=peak_vram,
                )
                rec_dict = rec.to_dict()
                raw_file.write(json.dumps(rec_dict, ensure_ascii=False) + "\n")
                completed.add((pid, COND_B0_TOKENS_VANILLA_WEIGHTS))
                existing_records.append(rec_dict)
        finally:
            b0_memory_before_teardown = cuda_memory_snapshot(device_obj)
            expects_position_hook = bool(getattr(b0_model, "uses_base_token_positions", False))
            position_hook_removed = remove_model_generation_hooks(b0_model)
            manager_ref = None
            if static_mgr is not None:
                static_mgr.reset(clear_dictionary=True, clear_caches=True)
                static_mgr.clear_weight_caches()
                manager_ref = weakref.ref(static_mgr)
                del static_mgr
            b0_model_ref = weakref.ref(b0_model)
            del b0_model
            b0_teardown = assert_model_phase_teardown(
                b0_model_ref,
                device_obj,
                phase="B0 generation",
                before=b0_memory_before_teardown,
                manager_ref=manager_ref,
            )
            b0_teardown["position_hook_removed"] = position_hook_removed
            if expects_position_hook and not position_hook_removed:
                raise AttributionError("B0 teardown could not remove the span-aware generation hook")
            print(f"B0 model teardown: {json.dumps(b0_teardown, sort_keys=True)}", flush=True)

    # 2c. HARD A/B0 TOKEN GATE AND DEFERRED LOGIT GATE.
    expected_prompt_ids = [str(record.prompt_id) for record in target_records]
    a_b0_token_gate = token_equivalence_gate(existing_records, expected_prompt_ids=expected_prompt_ids)
    if a_b0_token_gate.get("status") != "PASS":
        a_b0_logit_gate = {
            "status": "NOT_RUN",
            "reason": "Offline A/B0 token gate failed; downstream evaluation blocked",
        }
    elif getattr(args, "run_b0_logit_parity", False) or getattr(args, "logit_only", False):
        a_b0_logit_gate = run_a_b0_logit_parity(
            target_records,
            existing_records,
            checkpoint_path,
            args.device,
            max_new_tokens=args.max_new_tokens,
            k=args.k,
        )
    else:
        a_b0_logit_gate = {
            "status": "DEFERRED",
            "reason": "Generation-level equivalence established; full logit-distribution equivalence deferred.",
        }
    ablation_gate_report: Dict[str, Any] = {
        "a_b0_token_equality": a_b0_token_gate,
        "a_b0_logit_parity": a_b0_logit_gate,
        "a_b1_matched_prefix_diagnostic": {"status": "NOT_TESTED"},
        "a_b1_logit_fidelity": {"status": "NOT_TESTED"},
        "b0_b1_adapter_isolation": {"status": "NOT_TESTED"},
        "b1_b2_checkpoint_isolation": {"status": "NOT_TESTED"},
        "b0_b1_parameter_isolation": {"status": "NOT_TESTED"},
        "forced_h_representation": forced_h_representation_gates(
            existing_records, expected_prompt_ids=expected_prompt_ids
        ),
    }

    # B1 and subsequent causal evaluation proceed once generation-level A/B0 equivalence passes.
    a_b0_pass = a_b0_token_gate.get("status") == "PASS"

    # Matched-prefix diagnostic for B1 (sequential, no dual model residency)
    if getattr(args, "b1_diagnostic", False):
        if not a_b0_pass:
            print("A/B0 token gate did not pass; skipping B1 matched-prefix diagnostic.", flush=True)
        else:
            print("\n--- Running Sequential Matched-Prefix Diagnostic: Vanilla A vs B1 ---", flush=True)
            diagnostic_res = run_a_b1_matched_prefix_diagnostic(
                target_records,
                existing_records,
                checkpoint_path,
                args.device,
                k=args.k,
            )
            ablation_gate_report["a_b1_matched_prefix_diagnostic"] = diagnostic_res
            ablation_gate_report["a_b1_logit_fidelity"] = diagnostic_res
            diag_path = output_dir / "a_b1_matched_prefix_diagnostic.json"
            with diag_path.open("w", encoding="utf-8") as df:
                json.dump(diagnostic_res, df, indent=2, ensure_ascii=False)
            print(f"Wrote B1 matched-prefix diagnostic to {diag_path}", flush=True)

    # 2d. RUN TRUE B1 (Upstream EPFL adapter only, H disabled, NO Step-100 checkpoint)
    b1_pending = [
        r for r in target_records
        if (r.prompt_id, COND_B1_UPSTREAM_EPFL_ADAPTER_H_DISABLED) not in completed
    ]
    if getattr(args, "b1_generate_subset", False):
        subset_set = set(DEV_B1_DIAGNOSTIC_SUBSET_PROMPT_IDS)
        b1_pending = [r for r in b1_pending if str(r.prompt_id) in subset_set]

    if COND_B1_UPSTREAM_EPFL_ADAPTER_H_DISABLED in args.conditions and b1_pending and a_b0_pass:
        print(f"\n--- Running True B1 Generation (Upstream EPFL Adapter Only) on {len(b1_pending)} prompts ---", flush=True)
        b1_model, b1_tok, b1_load_report = load_predictive_bundle(
            checkpoint_path, args.device, load_peft_adapter=True, apply_joint_checkpoint=False, k=args.k
        )
        b1_base_hash = base_phi_weight_sha256(b1_model)
        dim = getattr(b1_model.config, "hidden_size", 3072)
        static_mgr = prepare_empty_b0_manager(b1_model, k=args.k, device=device_obj)
        try:
            for idx, sample in enumerate(b1_pending, 1):
                pid = sample.prompt_id
                dom = sample.domain
                print(f"[B1 {idx}/{len(b1_pending)}] Prompt {pid} ({dom})...", flush=True)
                prompt_text = build_canonical_prompt_text(asdict(sample))
                input_ids = b1_tok.encode(prompt_text, add_special_tokens=False)
                reset_empty_b0_manager(static_mgr, device_obj)
                t_start = time.perf_counter()
                gen_ids, ttft, dec_time, steps = generate_single_prompt(
                    b1_model, b1_tok, input_ids, device_obj, static_mgr=static_mgr, max_new_tokens=args.max_new_tokens
                )
                total_lat = time.perf_counter() - t_start
                eos_hit = bool(gen_ids and gen_ids[-1] in CANONICAL_EOS_TOKEN_IDS)
                output_text = b1_tok.decode(gen_ids, skip_special_tokens=True)
                sample_dict = asdict(sample)
                if pid in gt_map:
                    sample_dict["reference_response"] = gt_map[pid]
                sample_dict["eos_reached"] = eos_hit
                pass_status, scores = evaluate_output_quality(dom, output_text, sample_dict)
                peak_vram = torch.cuda.max_memory_allocated(device_obj) if device_obj.type == "cuda" else None

                vanilla_ids = vanilla_continuation_map.get(pid, [])
                div_pos = find_first_divergence(gen_ids, vanilla_ids)

                rec = AttributionRecord(
                    prompt_id=pid,
                    domain=dom,
                    split=sample.split,
                    condition=COND_B1_UPSTREAM_EPFL_ADAPTER_H_DISABLED,
                    created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    checkpoint_name="UPSTREAM_EPFL_PEFT_ONLY",
                    checkpoint_sha256="",
                    git_commit=git_commit,
                    runtime={
                        **runtime,
                        "base_phi_weight_sha256": b1_base_hash,
                        "generation_policy_sha256": generation_policy_sha256(args.max_new_tokens),
                        "tokenizer_id": CANONICAL_MODEL_ID,
                        "tokenizer_revision": CANONICAL_MODEL_REVISION,
                        "checkpoint_applied": False,
                        "step100_h_encoder_checkpoint_loaded": False,
                        "checkpoint_step": None,
                        "adapter_load_requested": True,
                        "adapter_present_in_wrapper": bool(b1_load_report.get("adapter_present")),
                        "active_lora_during_generation": True,
                        "active_adapter_names": b1_load_report.get("active_adapter_names", []),
                        "adapter_param_count": b1_load_report.get("adapter_param_count"),
                        "adapter_param_names": b1_load_report.get("adapter_param_names", []),
                        "adapter_param_shapes": b1_load_report.get("adapter_param_shapes", {}),
                        "upstream_adapter_id": CANONICAL_ZIP2ZIP_ID,
                        "upstream_adapter_revision": CANONICAL_ZIP2ZIP_REVISION,
                        "peft_version": b1_load_report.get("peft_version"),
                        "h_seeded_slots": 0,
                        "h_logits_masked_by_static_manager": True,
                        "codebook_backend": "static",
                        "effective_position_mode": "base_token_end",
                        **prompt_runtime_metadata(b1_tok, prompt_text, input_ids, args.max_new_tokens),
                    },
                    raw_output=output_text,
                    expanded_output=output_text,
                    generated_token_ids=gen_ids,
                    expanded_token_ids=gen_ids,
                    generated_token_count=len(gen_ids),
                    expanded_token_count=len(gen_ids),
                    termination_reason="eos" if eos_hit else "length",
                    termination_token_id=gen_ids[-1] if gen_ids else None,
                    eos_reached=eos_hit,
                    truncated=(not eos_hit and len(gen_ids) >= args.max_new_tokens),
                    task_quality_scores=scores,
                    quality_gate_pass=pass_status,
                    first_divergence_from_vanilla_pos=div_pos,
                    ttft_s=round(ttft, 4),
                    decode_time_s=round(dec_time, 4),
                    tpot_s=round(dec_time / max(steps, 1), 6) if steps > 0 else None,
                    total_latency_s=round(total_lat, 4),
                    time_to_eos_s=round(total_lat, 4),
                    transformer_decode_calls=steps,
                    raw_model_steps=steps,
                    words_per_sec=round(len(output_text.split()) / max(total_lat, 1e-4), 2),
                    setup_time_s=0.0,
                    peak_vram_bytes=peak_vram,
                )
                rec_dict = rec.to_dict()
                raw_file.write(json.dumps(rec_dict, ensure_ascii=False) + "\n")
                completed.add((pid, COND_B1_UPSTREAM_EPFL_ADAPTER_H_DISABLED))
                existing_records.append(rec_dict)
        finally:
            b1_memory_before_teardown = cuda_memory_snapshot(device_obj)
            position_hook_removed = remove_model_generation_hooks(b1_model)
            manager_ref = None
            if static_mgr is not None:
                static_mgr.reset(clear_dictionary=True, clear_caches=True)
                manager_ref = weakref.ref(static_mgr)
                del static_mgr
            b1_model_ref = weakref.ref(b1_model)
            del b1_model
            b1_teardown = assert_model_phase_teardown(
                b1_model_ref,
                device_obj,
                phase="B1 generation",
                before=b1_memory_before_teardown,
                manager_ref=manager_ref,
            )
            print(f"B1 model teardown: {json.dumps(b1_teardown, sort_keys=True)}", flush=True)

    # 3. RUN B2 / Historical B / CF / C / D (Trained Step-100 Bundle)
    pred_conditions = [
        c for c in (COND_B2_STEP100_H_DISABLED, COND_B_H_DISABLED, COND_CF_FORCED_ORACLE, COND_C_ORACLE, COND_D_REAL_PREDICTOR)
        if c in args.conditions
    ]
    if pred_conditions and not a_b0_pass:
        print("A/B0 token gate did not pass; skipping B2/C/CF/D causal evaluation.", flush=True)
        pred_conditions = []
    has_pending_pred = any((r.prompt_id, c) not in completed for r in target_records for c in pred_conditions)

    if pred_conditions and has_pending_pred:
        p_model, p_tok, load_rep = load_predictive_bundle(
            checkpoint_path, args.device, load_peft_adapter=True, k=args.k
        )
        p_base_hash = base_phi_weight_sha256(p_model)
        lora_component = load_rep.get("components", {}).get("lora", {})
        lora_delta_verified = bool(
            load_rep.get("lora_tensors", 0)
            and lora_component.get("changed_tensor_count", 0)
            and lora_component.get("changed_parameter_names")
            and lora_component.get("changed_parameter_shapes")
            and not lora_component.get("missing_keys")
            and not lora_component.get("unexpected_keys")
        )
        parameter_gate_preflight = checkpoint_isolation_preflight(
            load_rep, existing_records, p_base_hash, expected_prompt_ids
        )
        ablation_gate_report["b0_b1_parameter_isolation_preflight"] = parameter_gate_preflight
        if parameter_gate_preflight.get("status") != "PASS":
            print("B0/B1 parameter isolation gate failed; skipping B1/C/CF/D causal evaluation.", flush=True)
            pred_conditions = []
        else:
            ablation_gate_report["a_b1_logit_fidelity"] = run_a_b1_logit_fidelity(
                target_records,
                existing_records,
                p_model,
                p_tok,
                args.device,
                k=args.k,
            )
        p_runtime = {
            **runtime,
            "base_phi_weight_sha256": p_base_hash,
            "generation_policy_sha256": generation_policy_sha256(args.max_new_tokens),
            "tokenizer_id": CANONICAL_MODEL_ID,
            "tokenizer_revision": CANONICAL_MODEL_REVISION,
            "checkpoint_applied": True,
            "step100_h_encoder_checkpoint_loaded": True,
            "active_lora_during_generation": True,
            "checkpoint_step": load_rep.get("step"),
            "lora_delta_verified": lora_delta_verified,
            "lora_tensor_count": load_rep.get("lora_tensors", 0),
            "lora_changed_tensor_count": lora_component.get("changed_tensor_count", 0),
            "checkpoint_changed_parameters": parameter_gate_preflight.get("changed_parameters", {}),
            "frozen_base_parameter_sha256": load_rep.get("frozen_base_parameter_sha256"),
            "frozen_base_unchanged_during_load": load_rep.get("frozen_base_unchanged_during_load"),
            "loaded_position_mode": load_rep.get("loaded_position_mode"),
            "effective_position_mode": load_rep.get("effective_position_mode"),
            "codebook_backend": load_rep.get("codebook_backend"),
        }
        p_runtime.update({
            "generation_config": generation_policy(args.max_new_tokens),
            "eos_token_ids": list(CANONICAL_EOS_TOKEN_IDS),
            "max_new_tokens": int(args.max_new_tokens),
            "tokenizer_chat_template_sha256": tokenizer_template_sha256(p_tok),
        })
        dim = getattr(p_model.config, "hidden_size", 3072)

        for cond in pred_conditions:
            if cond in (COND_C_ORACLE, COND_D_REAL_PREDICTOR):
                forced_gate = forced_h_representation_gates(
                    existing_records, expected_prompt_ids=expected_prompt_ids
                )
                ablation_gate_report["forced_h_representation"] = forced_gate
                if forced_gate.get("status") != "PASS":
                    print(f"{cond} blocked because FORCED ORACLE representation gates did not pass.", flush=True)
                    continue
            pending = [r for r in target_records if (r.prompt_id, cond) not in completed]
            if not pending:
                continue

            if cond == COND_CF_FORCED_ORACLE:
                b1_by_prompt = {
                    record.get("prompt_id"): record
                    for record in existing_records
                    if record.get("condition") == COND_B_H_DISABLED
                }
                for idx, sample in enumerate(pending, 1):
                    pid = sample.prompt_id
                    target_record = b1_by_prompt.get(pid)
                    if target_record is None or not target_record.get("expanded_token_ids"):
                        raise AttributionError(f"CF requires an existing B1 H-disabled DEV record for {pid}")
                    target_ids = [int(token_id) for token_id in target_record["expanded_token_ids"]]
                    print(f"[CF: FORCED ORACLE {idx}/{len(pending)}] Prompt {pid} ({sample.domain})...", flush=True)
                    prompt_text = build_canonical_prompt_text(asdict(sample))
                    input_ids = p_tok.encode(prompt_text, add_special_tokens=False)
                    selected_phrases = [
                        tuple(int(token) for token in phrase)
                        for phrase in oracle_ceiling(target_ids, min_length=2, max_length=4)["selected_phrases"]
                    ]
                    codebook_dict = build_codebook_dict(selected_phrases, INITIAL_VOCAB_SIZE)
                    forced = force_oracle_substitutions(
                        target_ids,
                        {phrase: h_id for phrase, h_id in codebook_dict.items()},
                    )
                    static_mgr = StaticCodebookManager(
                        initial_vocab_size=INITIAL_VOCAB_SIZE,
                        max_codebook_size=args.k,
                        max_subtokens=4,
                        embedding_dim=dim,
                        pad_token_id=PAD_TOKEN_ID,
                    )
                    static_mgr.set_seeded_codebook(codebook_dict, batch_size=1, device=device_obj)
                    static_mgr.attach_to_model(p_model)
                    h_position_expectations = {
                        int(emission["schedule_index"]): {
                            "expected_position": len(input_ids) + int(emission["base_start"]) + int(emission["span"]) - 1,
                            "expected_next_offset": len(input_ids) + int(emission["base_start"]) + int(emission["span"]),
                        }
                        for emission in forced["h_emissions"]
                    }
                    forced_processor = ForcedScheduleLogitsProcessor(
                        len(input_ids),
                        forced["forced_generation_ids"],
                        static_mgr=static_mgr,
                        h_position_expectations=h_position_expectations,
                    )
                    started = time.perf_counter()
                    gen_ids, ttft, dec_time, steps = generate_single_prompt(
                        p_model,
                        p_tok,
                        input_ids,
                        device_obj,
                        static_mgr=static_mgr,
                        max_new_tokens=len(forced["forced_generation_ids"]),
                        extra_logits_processors=(forced_processor,),
                    )
                    total_lat = time.perf_counter() - started
                    schedule_matches = gen_ids == forced["forced_generation_ids"]

                    hyper_to_subtokens = {h_id: list(phrase) for phrase, h_id in codebook_dict.items()}
                    expanded_ids: List[int] = []
                    h_emissions: List[Dict[str, Any]] = []
                    for position, token_id in enumerate(gen_ids):
                        phrase = hyper_to_subtokens.get(token_id)
                        if phrase is None:
                            expanded_ids.append(token_id)
                        else:
                            expanded_ids.extend(phrase)
                            h_emissions.append({
                                "pos": position,
                                "id": token_id,
                                "phrase": p_tok.decode(phrase),
                                "subtokens": phrase,
                            })
                    roundtrip_ok = expanded_ids == target_ids
                    immediate_checks = []
                    for emission in forced["h_emissions"]:
                        schedule_index = int(emission["schedule_index"])
                        next_index = schedule_index + 1
                        if next_index < len(gen_ids) and next_index < len(forced_processor.unforced_top1):
                            immediate_checks.append({
                                "h_schedule_index": schedule_index,
                                "expected_next_token_id": int(forced["forced_generation_ids"][next_index]),
                                "unforced_top1_token_id": int(forced_processor.unforced_top1[next_index]),
                                "top1_matches": int(forced_processor.unforced_top1[next_index]) == int(forced["forced_generation_ids"][next_index]),
                            })
                    continuation_stable = bool(immediate_checks) and all(row["top1_matches"] for row in immediate_checks)
                    state_checks = []
                    for emission in forced["h_emissions"]:
                        prefixes = h_vs_base_prefix_pair(input_ids, forced, emission)
                        static_mgr.reset()
                        base_logits = capture_last_token_logits(
                            p_model,
                            prefixes["base_prefix_ids"],
                            device=device_obj,
                            inserted_h_count=args.k,
                            static_mgr=static_mgr,
                        )
                        static_mgr.reset()
                        h_logits = capture_last_token_logits(
                            p_model,
                            prefixes["h_prefix_ids"],
                            device=device_obj,
                            inserted_h_count=args.k,
                            static_mgr=static_mgr,
                        )
                        metric = logit_parity_metrics(
                            base_logits,
                            h_logits,
                            eos_token_ids=CANONICAL_EOS_TOKEN_IDS,
                            atol=parity_tolerance(str(p_model.base_model.get_input_embeddings().weight.dtype)),
                        )
                        metric["h_id"] = int(emission["id"])
                        metric["base_prefix_length"] = len(prefixes["base_prefix_ids"])
                        metric["h_prefix_length"] = len(prefixes["h_prefix_ids"])
                        state_checks.append(metric)
                    cf_state_equivalent = bool(state_checks) and all(row["status"] == "PASS" for row in state_checks)
                    semantic_positions_ok = (
                        bool(h_position_expectations)
                        and len(forced_processor.semantic_position_checks) == len(h_position_expectations)
                        and all(
                            row["token_id"] == int(forced["forced_generation_ids"][row["schedule_index"]])
                            and row["observed_position"] == row["expected_position"]
                            and row["observed_next_offset"] == row["expected_next_offset"]
                            for row in forced_processor.semantic_position_checks
                        )
                    )
                    output_text = p_tok.decode(expanded_ids, skip_special_tokens=True)
                    sample_dict = asdict(sample)
                    if pid in gt_map:
                        sample_dict["reference_response"] = gt_map[pid]
                    eos_hit = bool(gen_ids and gen_ids[-1] in CANONICAL_EOS_TOKEN_IDS)
                    sample_dict["eos_reached"] = eos_hit
                    pass_status, quality_scores = evaluate_output_quality(sample.domain, output_text, sample_dict)
                    peak_vram = torch.cuda.max_memory_allocated(device_obj) if device_obj.type == "cuda" else None
                    rec = AttributionRecord(
                        prompt_id=pid,
                        domain=sample.domain,
                        split=sample.split,
                        condition=COND_CF_FORCED_ORACLE,
                        created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                        checkpoint_name=checkpoint_path.name,
                        checkpoint_sha256=checkpoint_sha,
                        git_commit=git_commit,
                        runtime={
                            **p_runtime,
                            **prompt_runtime_metadata(p_tok, prompt_text, input_ids, args.max_new_tokens),
                            "forced_schedule_matches": schedule_matches,
                            "forced_schedule_roundtrip_ok": roundtrip_ok,
                            "forced_schedule_length": len(forced["forced_generation_ids"]),
                            "cf_semantic_position_checks": forced_processor.semantic_position_checks,
                            "cf_continuation_checks": immediate_checks,
                            "cf_h_vs_base_logit_checks": state_checks,
                            "cf_h_vs_base_state_equivalent": cf_state_equivalent,
                            "cf_post_h_top1_matches_target": (sum(row["top1_matches"] for row in immediate_checks) / len(immediate_checks)) if immediate_checks else None,
                            "cf_post_h_boundary_count": len(immediate_checks),
                        },
                        raw_output=output_text,
                        expanded_output=output_text,
                        generated_token_ids=gen_ids,
                        expanded_token_ids=expanded_ids,
                        generated_token_count=len(gen_ids),
                        expanded_token_count=len(expanded_ids),
                        termination_reason="eos" if eos_hit else "length",
                        termination_token_id=gen_ids[-1] if gen_ids else None,
                        eos_reached=eos_hit,
                        truncated=(not eos_hit and len(gen_ids) >= args.max_new_tokens),
                        task_quality_scores=quality_scores,
                        quality_gate_pass=pass_status,
                        forced_oracle_roundtrip_ok=roundtrip_ok and schedule_matches,
                        cf_semantic_positions_ok=semantic_positions_ok,
                        cf_continuation_stable=continuation_stable and cf_state_equivalent,
                        candidates_supplied=len(selected_phrases),
                        selected_h_slots=[list(phrase) for phrase in selected_phrases],
                        phrase_length_per_slot=[len(phrase) for phrase in selected_phrases],
                        h_emissions=h_emissions,
                        first_h_emission_pos=h_emissions[0]["pos"] if h_emissions else None,
                        continuation_tokens_after_last_h=(len(gen_ids) - 1 - h_emissions[-1]["pos"]) if h_emissions else None,
                        ttft_s=round(ttft, 4),
                        decode_time_s=round(dec_time, 4),
                        tpot_s=round(dec_time / max(steps, 1), 6) if steps > 0 else None,
                        total_latency_s=round(total_lat, 4),
                        time_to_eos_s=round(total_lat, 4),
                        transformer_decode_calls=steps,
                        raw_model_steps=steps,
                        words_per_sec=round(len(output_text.split()) / max(total_lat, 1e-4), 2),
                        setup_time_s=0.0,
                        peak_vram_bytes=peak_vram,
                    )
                    rec_dict = rec.to_dict()
                    raw_file.write(json.dumps(rec_dict, ensure_ascii=False) + "\n")
                    completed.add((pid, COND_CF_FORCED_ORACLE))
                    existing_records.append(rec_dict)
                continue

            print(f"\n--- Running Condition {cond} on {len(pending)} prompts ---", flush=True)

            for idx, sample in enumerate(pending, 1):
                pid = sample.prompt_id
                dom = sample.domain
                print(f"[{cond} {idx}/{len(pending)}] Prompt {pid} ({dom})...", flush=True)

                prompt_text = build_canonical_prompt_text(asdict(sample))
                input_ids = p_tok.encode(prompt_text, add_special_tokens=False)

                t_setup_start = time.perf_counter()
                selected_phrases: List[Tuple[int, ...]] = []
                cands_supplied = 0

                if cond in (COND_B2_STEP100_H_DISABLED, COND_B_H_DISABLED):
                    codebook_dict = {}
                elif cond == COND_C_ORACLE:
                    # Derived from canonical Vanilla continuation
                    cont_tokens = vanilla_continuation_map.get(pid, [])
                    selected_phrases = derive_oracle_codebook_phrases(
                        cont_tokens, k=args.k, min_len=2, max_len=4
                    )
                    codebook_dict = build_codebook_dict(selected_phrases, INITIAL_VOCAB_SIZE)
                    cands_supplied = len(selected_phrases)
                elif cond == COND_D_REAL_PREDICTOR:
                    # Phi-only candidate retrieval + PooledMLP ranker
                    cand_gen = ConfigurableCandidateGenerator(cand_index, tokenizer=p_tok)
                    legacy_rec = sample.to_legacy_record(p_tok)
                    cand_records = cand_gen.build_candidate_records(
                        legacy_rec,
                        strategy=RetrievalStrategy.EXPANDED_ASSOCIATIONS,
                        target_pool_size=args.pool_size,
                    )
                    cands_supplied = len(cand_records)
                    ranked = ranker_model.rank_codebook(input_ids, cand_records, domain=dom, k=args.k)
                    selected_phrases = [tuple(c.tokens) for c, _ in ranked]
                    codebook_dict = build_codebook_dict(selected_phrases, INITIAL_VOCAB_SIZE)

                static_mgr = StaticCodebookManager(
                    initial_vocab_size=INITIAL_VOCAB_SIZE,
                    max_codebook_size=32,
                    max_subtokens=4,
                    embedding_dim=dim,
                    pad_token_id=PAD_TOKEN_ID,
                )
                static_mgr.set_seeded_codebook(codebook_dict, batch_size=1, device=device_obj)
                static_mgr.attach_to_model(p_model)
                setup_time_s = time.perf_counter() - t_setup_start

                t_gen_start = time.perf_counter()
                gen_ids, ttft, dec_time, steps = generate_single_prompt(
                    p_model, p_tok, input_ids, device_obj, static_mgr=static_mgr, max_new_tokens=args.max_new_tokens
                )
                gen_time_s = time.perf_counter() - t_gen_start
                total_lat = setup_time_s + gen_time_s

                # Hypertoken expansion
                hyper_to_subtokens = {v: list(k) for k, v in codebook_dict.items()}
                expanded_tokens: List[int] = []
                h_emissions: List[Dict[str, Any]] = []
                emitted_hyper_ids: Set[int] = set()
                first_h_pos = None

                for pos, tid in enumerate(gen_ids):
                    if tid in hyper_to_subtokens:
                        if first_h_pos is None:
                            first_h_pos = pos
                        phrase_toks = hyper_to_subtokens[tid]
                        phrase_text = p_tok.decode(phrase_toks)
                        h_emissions.append({"pos": pos, "id": tid, "phrase": phrase_text, "subtokens": phrase_toks})
                        emitted_hyper_ids.add(tid)
                        expanded_tokens.extend(phrase_toks)
                    else:
                        expanded_tokens.append(tid)

                raw_output = p_tok.decode(gen_ids, skip_special_tokens=True)
                expanded_output = p_tok.decode(expanded_tokens, skip_special_tokens=True)

                eos_hit = bool(gen_ids and gen_ids[-1] in CANONICAL_EOS_TOKEN_IDS)
                term_reason = "eos" if eos_hit else "length"
                term_token = gen_ids[-1] if gen_ids else None

                sample_dict = asdict(sample)
                if pid in gt_map:
                    sample_dict["reference_response"] = gt_map[pid]
                sample_dict["eos_reached"] = eos_hit
                pass_status, scores = evaluate_output_quality(dom, expanded_output, sample_dict)

                # Diagnostics
                vanilla_ids = vanilla_continuation_map.get(pid, [])
                vanilla_str = p_tok.decode(vanilla_ids)
                phrase_occurs = [
                    bool(p_tok.decode(list(p)) in vanilla_str) for p in selected_phrases
                ]
                dead_slots = [
                    codebook_dict[p] for p in selected_phrases if codebook_dict[p] not in emitted_hyper_ids
                ]
                utilization = (
                    (len(emitted_hyper_ids) / len(selected_phrases) * 100.0)
                    if selected_phrases
                    else 0.0
                )
                div_pos = find_first_divergence(expanded_tokens, vanilla_ids)
                tokens_after_h = (len(gen_ids) - 1 - h_emissions[-1]["pos"]) if h_emissions else None

                peak_vram = torch.cuda.max_memory_allocated(device_obj) if device_obj.type == "cuda" else None

                rec = AttributionRecord(
                    prompt_id=pid,
                    domain=dom,
                    split=sample.split,
                    condition=cond,
                    created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    checkpoint_name=checkpoint_path.name,
                    checkpoint_sha256=checkpoint_sha,
                    git_commit=git_commit,
                    runtime={
                        **p_runtime,
                        **prompt_runtime_metadata(p_tok, prompt_text, input_ids, args.max_new_tokens),
                    },
                    raw_output=raw_output,
                    expanded_output=expanded_output,
                    generated_token_ids=gen_ids,
                    expanded_token_ids=expanded_tokens,
                    generated_token_count=len(gen_ids),
                    expanded_token_count=len(expanded_tokens),
                    termination_reason=term_reason,
                    termination_token_id=term_token,
                    eos_reached=eos_hit,
                    truncated=(not eos_hit and len(gen_ids) >= args.max_new_tokens),
                    task_quality_scores=scores,
                    quality_gate_pass=pass_status,
                    candidates_supplied=cands_supplied,
                    selected_h_slots=[list(p) for p in selected_phrases],
                    phrase_length_per_slot=[len(p) for p in selected_phrases],
                    phrase_occurs_in_vanilla=phrase_occurs,
                    h_emissions=h_emissions,
                    dead_h_slots=dead_slots,
                    h_utilization_pct=round(utilization, 2),
                    first_h_emission_pos=first_h_pos,
                    first_divergence_from_vanilla_pos=div_pos,
                    continuation_tokens_after_last_h=tokens_after_h,
                    ttft_s=round(ttft, 4),
                    decode_time_s=round(dec_time, 4),
                    tpot_s=round(dec_time / max(steps, 1), 6) if steps > 0 else None,
                    total_latency_s=round(total_lat, 4),
                    time_to_eos_s=round(total_lat, 4),
                    transformer_decode_calls=steps,
                    raw_model_steps=steps,
                    words_per_sec=round(len(expanded_output.split()) / max(total_lat, 1e-4), 2),
                    setup_time_s=round(setup_time_s, 4),
                    peak_vram_bytes=peak_vram,
                )
                rec_dict = rec.to_dict()
                raw_file.write(json.dumps(rec_dict, ensure_ascii=False) + "\n")
                completed.add((pid, cond))
                existing_records.append(rec_dict)

        p_memory_before_teardown = cuda_memory_snapshot(device_obj)
        remove_model_generation_hooks(p_model)
        p_model_ref = weakref.ref(p_model)
        del p_model
        p_teardown = assert_model_phase_teardown(
            p_model_ref,
            device_obj,
            phase="Predictive bundle generation",
            before=p_memory_before_teardown,
        )
        print(f"Predictive model teardown: {json.dumps(p_teardown, sort_keys=True)}", flush=True)

    raw_file.close()

    # 4. Generate Machine-Readable Summary and Markdown Report
    ablation_gate_report["b0_b1_adapter_isolation"] = b0_b1_adapter_isolation_gate(
        existing_records, expected_prompt_ids=expected_prompt_ids
    )
    ablation_gate_report["b1_b2_checkpoint_isolation"] = b1_b2_checkpoint_isolation_gate(
        existing_records, expected_prompt_ids=expected_prompt_ids
    )
    ablation_gate_report["b0_b1_parameter_isolation"] = checkpoint_isolation_gate(
        existing_records, expected_prompt_ids=expected_prompt_ids
    )
    ablation_gate_report["forced_h_representation"] = forced_h_representation_gates(
        existing_records, expected_prompt_ids=expected_prompt_ids
    )
    summary = compute_attribution_summary(existing_records, external_gates=ablation_gate_report)
    summary["ablation_gates"] = ablation_gate_report
    summary["evaluated_prompts_count"] = len({r["prompt_id"] for r in existing_records})
    summary["split"] = args.split
    summary["git_commit"] = git_commit
    summary["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()

    with summary_path.open("w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2, ensure_ascii=False)
    print(f"\nWrote machine-readable summary to {summary_path}")

    # Generate Markdown Report
    a_b1_fidelity = ablation_gate_report.get("a_b1_logit_fidelity", {})
    a_b1_aggregate = a_b1_fidelity.get("aggregate", {})
    a_b1_row_status = str(a_b1_fidelity.get("status", "NOT_TESTED"))
    if a_b1_fidelity.get("status") == "MEASURED":
        a_b1_row_status += (
            f" (top1={a_b1_aggregate.get('top1_agreement_rate')}; "
            f"mean KL(Vanilla || B1)={a_b1_aggregate.get('mean_kl_reference_to_candidate_nats')} nats)"
        )
    md_lines = [
        f"# Phi Quality/Speed Attribution Benchmark Report",
        f"",
        f"- **Generated at**: `{summary['completed_at_utc']}`",
        f"- **Git Commit**: `{git_commit}`",
        f"- **Split**: `{args.split}` (Total Prompts: `{summary['evaluated_prompts_count']}`)",
        f"- **Records Generated**: `{len(existing_records)}`",
        f"",
        f"---",
        f"",
        f"## 1. Executive Summary & Causal Diagnosis",
        f"",
        f"- **Primary Bottleneck**: `{summary.get('causal_diagnosis', {}).get('primary_bottleneck', 'unclear')}`",
        f"- **Diagnosis**: {summary.get('causal_diagnosis', {}).get('detailed_diagnosis', 'N/A')}",
        f"",
        f"## Required Causal Gates",
        f"",
        f"| Gate | Status |",
        f"|---|---|",
        f"| A/B0 token equality | {ablation_gate_report.get('a_b0_token_equality', {}).get('status', 'NOT_TESTED')} |",
        f"| A/B0 logit parity | {ablation_gate_report.get('a_b0_logit_parity', {}).get('status', 'NOT_TESTED')} |",
        f"| A/B1 matched-prefix diagnostic | {ablation_gate_report.get('a_b1_matched_prefix_diagnostic', {}).get('status', 'NOT_TESTED')} |",
        f"| A/B1 base-logit fidelity | {a_b1_row_status} |",
        f"| B0/B1 upstream adapter isolation | {ablation_gate_report.get('b0_b1_adapter_isolation', {}).get('status', 'NOT_TESTED')} |",
        f"| B1/B2 Step-100 checkpoint isolation | {ablation_gate_report.get('b1_b2_checkpoint_isolation', {}).get('status', 'NOT_TESTED')} |",
        f"| B0/B2 parameter isolation (legacy) | {ablation_gate_report.get('b0_b1_parameter_isolation', {}).get('status', 'NOT_TESTED')} |",
        f"| Forced-H exact expansion | {ablation_gate_report.get('forced_h_representation', {}).get('forced_h_exact_expansion', {}).get('status', 'NOT_TESTED')} |",
        f"| Forced-H semantic position/cache state | {ablation_gate_report.get('forced_h_representation', {}).get('forced_h_semantic_position_and_state', {}).get('status', 'NOT_TESTED')} |",
        f"| Forced-H immediate continuation stability | {ablation_gate_report.get('forced_h_representation', {}).get('forced_h_continuation_stability', {}).get('status', 'NOT_TESTED')} |",
        f"",
        f"---",
        f"",
        f"## 2. Condition Overview",
        f"",
        f"| Condition | Prompts | Agg Quality | MBPP Pass | GSM8K Pass | Alpaca Pass | Mean Latency (s) | Total Steps | Live Realized Compression | H Emissions |",
        f"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for c in ALL_CONDITIONS:
        s = summary["conditions"].get(c)
        if not s:
            continue
        dq = s.get("domain_quality", {})
        code_p = f"{dq.get('code', {}).get('rate', 0.0) * 100:.1f}%" if 'code' in dq else "N/A"
        gsm_p = f"{dq.get('reasoning', {}).get('rate', 0.0) * 100:.1f}%" if 'reasoning' in dq else "N/A"
        alp_p = f"{dq.get('instruction', {}).get('rate', 0.0) * 100:.1f}%" if 'instruction' in dq else "N/A"
        md_lines.append(
            f"| **{CONDITION_DISPLAY.get(c, c)}** | {s['record_count']} | **{s['aggregate_quality_rate']*100:.1f}%** | "
            f"{code_p} | {gsm_p} | {alp_p} | {s['mean_total_latency_s']:.3f} | "
            f"{s['total_decode_calls']} | {s['realized_compression_pct']:.1f}% | {s['total_h_emissions']} |"
        )

    md_lines.extend([
        f"",
        f"---",
        f"",
        f"## 3. Comparison vs Vanilla (<= 3% Quality Gate)",
        f"",
        f"| Condition | Abs Quality Diff | Rel Quality Drop | Meets <=3% Gate? | Speedup vs Vanilla | Faster? | Steps Saved |",
        f"|---|---:|---:|:---:|---:|:---:|---:|",
    ])

    for c in (
        COND_B0_TOKENS_VANILLA_WEIGHTS,
        COND_B1_UPSTREAM_EPFL_ADAPTER_H_DISABLED,
        COND_B2_STEP100_H_DISABLED,
        COND_B_H_DISABLED,
        COND_C_ORACLE,
        COND_D_REAL_PREDICTOR,
    ):
        comp = summary["comparison_vs_vanilla"].get(c)
        if not comp:
            continue
        gate_icon = "PASS" if comp["meets_3pct_quality_gate"] else "**FAIL**"
        faster_icon = "YES" if comp["is_faster_than_vanilla"] else "NO"
        md_lines.append(
            f"| **{CONDITION_DISPLAY.get(c, c)}** | {comp['absolute_quality_diff']*100:+.1f}% | {comp['relative_quality_drop_pct']:.2f}% | "
            f"{gate_icon} | {comp['speedup_pct']:+.1f}% | {faster_icon} | {comp['decode_steps_saved']} |"
        )

    md_lines.append("")
    with report_path.open("w", encoding="utf-8") as rf:
        rf.write("\n".join(md_lines) + "\n")
    print(f"Wrote human-readable report to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled Phi quality/speed attribution experiment.")
    parser.add_argument(
        "--canonical-dataset",
        default="scratch/attribution_dataset_staged/dev_canonical_continuations.jsonl",
        help="Must contain only the requested split; the default is a DEV-only artifact.",
    )
    parser.add_argument(
        "--canonical-manifest",
        default="scratch/attribution_dataset_staged/dev_canonical_manifest.json",
        help="Manifest for a split-isolated artifact; mixed split files are rejected before rows are read.",
    )
    parser.add_argument("--ground-truth", default=None, help="Optional path to DEV ground truth references JSON")
    parser.add_argument("--association-index", default="experiments/checkpoints/train_only_association_index.pkl")
    parser.add_argument("--checkpoint", default="experiments/checkpoints/predictive_joint_pilot/checkpoint_step_100.pt")
    parser.add_argument("--predictor-model", default="experiments/checkpoints/predictor_v2/pooledmlp.pkl")
    parser.add_argument("--conditions", nargs="+", default=list(ALL_CONDITIONS))
    parser.add_argument("--split", default="DEV", choices=["TRAIN", "DEV"], help="Split to run. FINAL is forbidden.")
    parser.add_argument("--prompt-limit", type=int, default=None, help="Number of prompts to evaluate.")
    parser.add_argument("--output-dir", required=True, help="Directory to store raw results and summaries.")
    parser.add_argument("--device", default="cpu", help="Device to use, e.g. cpu or cuda:0")
    parser.add_argument("--k", type=int, default=32, help="Hypertoken budget K.")
    parser.add_argument("--pool-size", type=int, default=1024, help="Candidate pool size.")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--logit-only",
        action="store_true",
        help="Require complete pinned DEV12 A/B0 records, gate them offline, then run matched-prefix logits without generation.",
    )
    parser.add_argument(
        "--b1-diagnostic",
        action="store_true",
        help="Run matched-prefix diagnostic comparing Vanilla (A) vs UPSTREAM EPFL adapter (B1) across prefix states [0, 1, 4, 16] without generating tokens.",
    )
    parser.add_argument(
        "--b1-generate-subset",
        action="store_true",
        help="If set with B1 in conditions, evaluate B1 greedy generation on DEV_B1_DIAGNOSTIC_SUBSET_PROMPT_IDS (6 prompts) instead of all prompts.",
    )
    parser.add_argument(
        "--run-b0-logit-parity",
        action="store_true",
        help="Explicitly run B0 matched-prefix logit parity check during the run.",
    )
    args = parser.parse_args()

    run_attribution_benchmark(args)


if __name__ == "__main__":
    main()
