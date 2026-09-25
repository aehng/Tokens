"""One-session vLLM 0.30.0 predictive hypertoken compatibility proof.

This is not a benchmark. Each phase writes an atomic JSON file and stops
the process on the first failure.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

# Admission logs record the id passed to add_request. vLLM 0.30.0 otherwise
# appends a random suffix, which the proof cannot match to a phase.
os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PHI_ID = "microsoft/Phi-3.5-mini-instruct"
PHI_REV = "2fe192450127e6a83f7441aef6e3ca586c338b77"
VLLM_SHA = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
MAX_NEW = 16
PROOF_BLOCK_SIZE = 16
# T4 Phi-3.5 fp16 weights are about 7.14 GiB. 0.42 left a negative KV budget.
# BASE=0.90 is used only for engines that do not receive hyperencoders (stock and Phase 1 H-disabled).
# PREDICTIVE=0.75 leaves ~3.6 GiB outside vLLM allocation for the ~1.73 GiB hyperencoders plus safety headroom.
BASE_GPU_MEMORY_UTILIZATION = 0.90
PREDICTIVE_GPU_MEMORY_UTILIZATION = 0.75
DEFAULT_HEADROOM_SAFETY_MARGIN_BYTES = 512 * 1024 * 1024  # 512 MiB
GPU_MEMORY_UTILIZATION = PREDICTIVE_GPU_MEMORY_UTILIZATION


def _result_root() -> Path:
    kaggle = Path("/kaggle/working")
    if kaggle.is_dir():
        root = kaggle / "vllm_predictive_proof"
    else:
        root = REPO_ROOT / "experiments" / "checkpoints" / "vllm_predictive_proof"
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_sha() -> str:
    sha_file = REPO_ROOT / "SOURCE_SHA.txt"
    if sha_file.is_file():
        return sha_file.read_text(encoding="utf-8").strip()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _source_provenance() -> dict[str, Any]:
    path = REPO_ROOT / "source_provenance.json"
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {}
    return loaded


def environment() -> dict[str, Any]:
    import vllm

    cuda = torch.cuda.is_available()
    provenance = _source_provenance()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "vllm": vllm.__version__,
        "vllm_tag": "v0.30.0",
        "vllm_sha": VLLM_SHA,
        "source_sha": _source_sha(),
        "source_dataset_slug": provenance.get("source_dataset_slug"),
        "source_dataset_path": provenance.get("source_dataset_path"),
        "source_archive_sha256": provenance.get("source_archive_sha256"),
        "v2_env": os.environ.get("VLLM_USE_V2_MODEL_RUNNER"),
    }


def shutdown_vllm_engine(llm) -> str:
    """Shut down one vLLM 0.30 engine. The caller must ``del`` its own name."""
    if llm is None:
        return "none"
    engine = getattr(llm, "llm_engine", None)
    client = getattr(engine, "engine_core", None)
    shutdown = getattr(client, "shutdown", None)
    if not callable(shutdown):
        return "no-shutdown"
    shutdown()
    return "engine_core.shutdown"


def cuda_after_collect() -> dict[str, int | None]:
    gc.collect()
    if not torch.cuda.is_available():
        return {"allocated": None, "reserved": None}
    torch.cuda.empty_cache()
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
    }


def _record_teardown(root: Path, engine_name: str, method: str, cuda: dict[str, int | None]) -> None:
    path = root / "engine_teardown.json"
    events: list[dict[str, Any]] = []
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("events"), list):
            events = loaded["events"]
    events.append({"engine": engine_name, "shutdown": method, **cuda})
    write_json(path, {"events": events})


def _sampling(extra: dict | None = None, max_tokens: int = MAX_NEW):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        detokenize=False,
        logprobs=None,
        extra_args=extra,
    )


def _llm_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        tokenizer=PHI_ID,
        tokenizer_revision=PHI_REV,
        dtype="float16",
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        gpu_memory_utilization=PREDICTIVE_GPU_MEMORY_UTILIZATION,
        max_num_seqs=4,
        trust_remote_code=False,
        hf_overrides={
            "vocab_size": 32096,
            "architectures": ["PredictivePhi3ForCausalLM"],
        },
    )
    kwargs.update(overrides)
    return kwargs


def _make_llm(model: str, *, h_enabled: bool, **overrides: Any):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TOKENS_PREDICTIVE_H_ENABLED"] = "1" if h_enabled else "0"
    from vllm import LLM

    llm = LLM(model=model, **_llm_kwargs(**overrides))
    if not llm.llm_engine.vllm_config.use_v2_model_runner:
        raise RuntimeError("Model Runner V2 is not active")
    vocab = llm.llm_engine.model_config.get_vocab_size()
    if vocab != 32096:
        raise RuntimeError(f"logical vocab is {vocab}, expected 32096")
    return llm


def _ids(output) -> list[int]:
    return [int(token) for token in output.outputs[0].token_ids]


def _is_finished(output) -> bool:
    flag = getattr(output, "finished", False)
    return bool(flag() if callable(flag) else flag)


def _record_predictive_warmup(llm, root: Path) -> dict[str, Any]:
    def audit(model):
        events = list(model.predictive_state.admission_log)
        warmup = [
            event
            for event in events
            if event.get("event") == "add" and event.get("mode") == "vllm_warmup"
        ]
        synthesized = [event for event in events if event.get("event") == "add" and event.get("h_enabled")]
        return {
            "request_ids": [event["req_id"] for event in warmup],
            "all_inactive": all(event.get("h_enabled") is False for event in warmup),
            "codebooks_synthesized": len(synthesized),
        }

    payload = _apply(llm, audit)
    write_json(root / "predictive_warmup.json", payload)
    return payload


def _apply(llm, fn):
    results = llm.apply_model(fn)
    return results[0]


def phase_01_and_02(root: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(PHI_ID, revision=PHI_REV)
    prompt_ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
    stock_params = _sampling()
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    stock = LLM(
        model=PHI_ID,
        revision=PHI_REV,
        tokenizer=PHI_ID,
        tokenizer_revision=PHI_REV,
        dtype="float16",
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_model_len=128,
        max_num_seqs=2,
        gpu_memory_utilization=BASE_GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=1,
    )
    stock_out = stock.generate(
        [{"prompt_token_ids": prompt_ids}],
        SamplingParams(temperature=0.0, max_tokens=MAX_NEW, detokenize=False),
    )
    stock_ids = _ids(stock_out[0])
    stock_vocab = stock.llm_engine.model_config.get_vocab_size()
    stock_shutdown = shutdown_vllm_engine(stock)
    del stock
    cuda_after_stock = cuda_after_collect()

    ours = _make_llm(
        PHI_ID,
        h_enabled=False,
        revision=PHI_REV,
        max_model_len=128,
        max_num_seqs=2,
        gpu_memory_utilization=BASE_GPU_MEMORY_UTILIZATION,
    )
    ours_out = ours.generate([{"prompt_token_ids": prompt_ids}], stock_params)
    ours_logical_ids = _ids(ours_out[0])
    from tokens_vllm.contract import logical_output_to_base_ids

    conversion_error = None
    try:
        ours_base_ids = logical_output_to_base_ids(ours_logical_ids)
    except ValueError as exc:
        ours_base_ids = []
        conversion_error = str(exc)

    def inspect(model):
        seen: dict[str, list[int]] = {}
        original = model.model.embed_tokens.forward

        def wrapped(token_ids):
            seen["physical"] = token_ids.detach().to("cpu").tolist()
            return original(token_ids)

        logical = torch.tensor(
            [32010, 32011, 32042, 32043, 32095], device=model.model.embed_tokens.weight.device
        )
        model.model.embed_tokens.forward = wrapped
        try:
            embedded = model.embed_input_ids(logical)
        finally:
            model.model.embed_tokens.forward = original
        hidden = torch.zeros(
            1, model.config.hidden_size, dtype=embedded.dtype, device=embedded.device
        )
        logits = model.compute_logits(hidden)
        return {
            "embed_rows": int(model.model.embed_tokens.weight.shape[0]),
            "head_rows": int(model.lm_head.weight.shape[0]),
            "physical_ids": seen.get("physical"),
            "logits_width": None if logits is None else int(logits.shape[-1]),
            "h_slice_all_neg_inf": bool(
                torch.isneginf(logits[0, 32011:32043]).all().item()
            ),
            "state": type(model.predictive_state).__name__,
            "h_enabled": bool(model.h_enabled),
        }

    inspected = _apply(ours, inspect)
    ours_shutdown = shutdown_vllm_engine(ours)
    del ours
    cuda_after_ours = cuda_after_collect()
    phase1 = {
        "status": "PASS",
        "stock_vocab": stock_vocab,
        "logical_vocab": 32096,
        "stock_ids": stock_ids,
        "ours_logical_ids": ours_logical_ids,
        "ours_base_ids": ours_base_ids,
        "base_conversion_error": conversion_error,
        "greedy_match": conversion_error is None and ours_base_ids == stock_ids,
        "embed_rows": inspected["embed_rows"],
        "head_rows": inspected["head_rows"],
        "logits_width": inspected["logits_width"],
        "h_masked": inspected["h_slice_all_neg_inf"],
        "state": inspected["state"],
        "hyperencoders_invoked": False,
        "base_gpu_memory_utilization": BASE_GPU_MEMORY_UTILIZATION,
        "predictive_gpu_memory_utilization": PREDICTIVE_GPU_MEMORY_UTILIZATION,
        "gpu_memory_utilization": BASE_GPU_MEMORY_UTILIZATION,
        "stock_shutdown": stock_shutdown,
        "ours_shutdown": ours_shutdown,
        "cuda_after_stock": cuda_after_stock,
        "cuda_after_ours": cuda_after_ours,
    }
    if not phase1["greedy_match"] or phase1["embed_rows"] != 32064 or phase1["head_rows"] != 32064:
        phase1["status"] = "FAIL"
    if phase1["logits_width"] != 32096 or not phase1["h_masked"]:
        phase1["status"] = "FAIL"
    write_json(root / "phase_01_model_load.json", phase1)
    if phase1["status"] != "PASS":
        raise RuntimeError(f"phase 1 failed: {phase1}")

    expected_physical = [32010, 0, 0, 32011, 32063]
    phase2 = {
        "status": "PASS" if inspected["physical_ids"] == expected_physical else "FAIL",
        "logical_ids": [32010, 32011, 32042, 32043, 32095],
        "physical_ids": inspected["physical_ids"],
        "expected_physical": expected_physical,
        "max_physical": max(inspected["physical_ids"] or [0]),
    }
    write_json(root / "phase_02_vocab_mapping.json", phase2)
    if phase2["status"] != "PASS":
        raise RuntimeError(f"phase 2 failed: {phase2}")
    return {"phase_01": phase1, "phase_02": phase2}


def _load_samples() -> dict[str, dict]:
    path = REPO_ROOT / "data" / "cached_pure_pred_val_60.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["id"]: row for row in rows}


def _checkpoint_path() -> Path:
    from experiments.kaggle_step100_mount import (
        STEP100_CHECKPOINT_NAME,
        resolve_approved_step100_mount,
        validate_step100_artifact_manifest,
    )

    mount = resolve_approved_step100_mount()
    manifest_path = mount / "manifest.json"
    if manifest_path.is_file():
        validate_step100_artifact_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    checkpoint = mount / STEP100_CHECKPOINT_NAME
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _predictor_path() -> Path:
    from experiments.kaggle_step100_mount import STEP100_PREDICTOR_NAME, resolve_approved_step100_mount

    path = resolve_approved_step100_mount() / STEP100_PREDICTOR_NAME
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _codebook_payload(phrases: list[list[int]], *, max_subtokens: int, disabled_ids: list[int], pad_id: int) -> dict:
    from tokens_vllm.contract import codebook_sha256, validate_codebook

    payload = {
        "version": 1,
        "k": 32,
        "max_subtokens": max_subtokens,
        "pad_token_id": pad_id,
        "disabled_ids": disabled_ids,
        "phrases": phrases,
        "sha256": codebook_sha256(phrases),
    }
    validate_codebook(payload, disabled_ids=disabled_ids)
    return payload


def _build_hf_references(root: Path) -> dict[str, Any]:
    os.environ.setdefault("PREDICTOR_PATH", str(_predictor_path()))
    # The benchmark loader reads its module-level predictor constant.
    import experiments.run_quality_benchmark as benchmark
    from zip2zip import StaticCodebookManager

    benchmark.PREDICTOR_PATH = _predictor_path()
    checkpoint = _checkpoint_path()
    bundle = benchmark.load_predictive_model_bundle(
        str(checkpoint), device="cuda", expected_step=100
    )
    model = bundle["model"]
    max_subtokens = int(model.zip2zip_config.compression.max_subtokens)
    disabled_ids = list(model.zip2zip_config.compression.disabled_ids)
    pad_id = int(bundle["pad_id"])
    position_mode = model.zip2zip_config.position_mode
    samples = _load_samples()
    chosen = []
    for prompt_id in ("gsm_2956", "gsm_3022", "gsm_6613"):
        if prompt_id in samples:
            chosen.append(prompt_id)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        raise RuntimeError(f"need two prompts, found {chosen}")

    references = []
    for prompt_id in chosen:
        sample = samples[prompt_id]
        prompt_ids = [int(token) for token in sample["prompt_token_ids"]]
        codebook_dict, _meta = bundle["policy"].select_codebook(prompt_ids)
        ordered = sorted(codebook_dict.items(), key=lambda item: item[1])
        phrases = [list(phrase) for phrase, _hid in ordered]
        if len(phrases) != 32:
            raise RuntimeError(f"{prompt_id} codebook has {len(phrases)} phrases, expected 32")
        payload = _codebook_payload(
            phrases, max_subtokens=max_subtokens, disabled_ids=disabled_ids, pad_id=pad_id
        )
        manager = StaticCodebookManager(
            initial_vocab_size=32011,
            max_codebook_size=32,
            max_subtokens=max_subtokens,
            embedding_dim=bundle["embedding_dim"],
            pad_token_id=pad_id,
            disabled_ids=disabled_ids,
        )
        manager.set_seeded_codebook(codebook_dict, batch_size=1, device=torch.device("cuda"))
        manager.attach_to_model(model)
        started = time.perf_counter()
        manager.prepare_inference_tables(model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        setup_ms = (time.perf_counter() - started) * 1000.0
        h_in = manager.effective_embedding_weight_cache[32011:32043].detach().float().cpu()
        h_out = manager.effective_linear_weight_cache[32011:32043].detach().float().cpu()
        compressed = manager.prepare_input_sequence(prompt_ids, compress=True)
        position_tensor = manager.prepare_input_ids(
            torch.tensor([compressed], dtype=torch.long, device="cuda")
        )
        positions = position_tensor[0].detach().cpu().tolist()
        manager.base_position_offset = None
        manager.position_ids = None

        captured: list[torch.Tensor] = []

        from transformers import LogitsProcessor, LogitsProcessorList

        class _Capture(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if len(captured) < 1:
                    captured.append(scores[0].detach().float().cpu())
                return scores

        with torch.no_grad():
            generated = model.generate(
                input_ids=torch.tensor([compressed], dtype=torch.long, device="cuda"),
                max_new_tokens=MAX_NEW,
                do_sample=False,
                pad_token_id=bundle["tokenizer"].eos_token_id,
                logits_processor=LogitsProcessorList([_Capture()]),
            )
        new_ids = generated[0, len(compressed) :].detach().cpu().tolist()
        manager.detach_from_model(model)
        model.codebook_manager.reset()
        torch.save(
            {"h_input": h_in, "h_output": h_out, "positions": positions},
            root / f"hf_{prompt_id}.pt",
        )
        references.append(
            {
                "prompt_id": prompt_id,
                "payload": payload,
                "compressed_ids": [int(token) for token in compressed],
                "positions": [int(pos) for pos in positions],
                "new_ids": [int(token) for token in new_ids],
                "setup_ms": setup_ms,
                "first_logits_path": str(root / f"hf_{prompt_id}_logits.pt"),
            }
        )
        if captured:
            torch.save(captured[0], root / f"hf_{prompt_id}_logits.pt")
    # Keep encoders and merge the base for vLLM.
    encoder_path = root / "encoders.pt"
    torch.save(
        {
            "input_state": {k: v.detach().cpu() for k, v in model.input_encoder.state_dict().items()},
            "output_state": {k: v.detach().cpu() for k, v in model.output_encoder.state_dict().items()},
            "encoder_config": model.zip2zip_config.encoder,
            "compression_config": model.zip2zip_config.compression,
            "pad_id": pad_id,
            "disabled_ids": disabled_ids,
            "max_subtokens": max_subtokens,
            "position_mode": position_mode,
        },
        encoder_path,
    )
    from tokens_vllm.proof_harness import build_encoder_memory_plan
    plan = build_encoder_memory_plan(
        encoder_path,
        base_utilization=BASE_GPU_MEMORY_UTILIZATION,
        predictive_utilization=PREDICTIVE_GPU_MEMORY_UTILIZATION,
    )
    write_json(root / "encoder_memory_plan.json", plan)
    base = model.base_model
    merged = base.merge_and_unload() if hasattr(base, "merge_and_unload") else base
    merged_dir = root / "merged_phi"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    del model
    del base
    del merged
    cuda_after_collect()
    return {
        "references": references,
        "merged_dir": str(merged_dir),
        "encoder_path": str(encoder_path),
        "max_subtokens": max_subtokens,
        "position_mode": position_mode,
        "pad_id": pad_id,
        "disabled_ids": disabled_ids,
    }


def _install_encoders(llm, encoder_path: str, *, root: Path | None = None) -> dict[str, Any]:
    blob_path = encoder_path

    def install(model):
        from zip2zip.nn.encoders.base import BaseEncoder
        from tokens_vllm.proof_harness import (
            DEFAULT_HEADROOM_SAFETY_MARGIN_BYTES,
            check_encoder_headroom,
            inspect_parameter_footprint,
        )

        blob = torch.load(blob_path, map_location="cpu", weights_only=False)
        device = model.model.embed_tokens.weight.device

        in_fp = inspect_parameter_footprint(blob["input_state"])
        out_fp = inspect_parameter_footprint(blob["output_state"])
        total_encoder_bytes = in_fp["bytes"] + out_fp["bytes"]

        def _cuda_snapshot():
            if not torch.cuda.is_available() or device.type != "cuda":
                return {"free": None, "total": None, "allocated": None, "reserved": None}
            free, total = torch.cuda.mem_get_info(device)
            return {
                "free": int(free),
                "total": int(total),
                "allocated": int(torch.cuda.memory_allocated(device)),
                "reserved": int(torch.cuda.memory_reserved(device)),
            }

        cuda_before = _cuda_snapshot()
        if cuda_before["free"] is not None:
            check_encoder_headroom(
                cuda_before["free"],
                total_encoder_bytes,
                safety_margin_bytes=DEFAULT_HEADROOM_SAFETY_MARGIN_BYTES,
            )

        input_encoder = BaseEncoder.from_config(
            blob["encoder_config"], blob["compression_config"]
        )
        output_encoder = BaseEncoder.from_config(
            blob["encoder_config"], blob["compression_config"]
        )
        input_encoder.load_state_dict(blob["input_state"])
        output_encoder.load_state_dict(blob["output_state"])

        model.input_encoder = input_encoder.to(device).eval()
        cuda_after_input = _cuda_snapshot()

        model.output_encoder = output_encoder.to(device).eval()
        cuda_after_output = _cuda_snapshot()

        model.pad_token_id = int(blob["pad_id"])
        model.h_enabled = True
        model.model.h_enabled = True

        audit = {
            "max_subtokens": int(blob["max_subtokens"]),
            "position_mode": blob["position_mode"],
            "input_parameter_count": in_fp["count"],
            "output_parameter_count": out_fp["count"],
            "input_parameter_bytes": in_fp["bytes"],
            "output_parameter_bytes": out_fp["bytes"],
            "total_parameter_bytes": total_encoder_bytes,
            "dtypes": sorted(list(set(in_fp["dtypes"] + out_fp["dtypes"]))),
            "safety_margin_bytes": DEFAULT_HEADROOM_SAFETY_MARGIN_BYTES,
            "cuda_memory_before": cuda_before,
            "cuda_memory_after_input_encoder": cuda_after_input,
            "cuda_memory_after_output_encoder": cuda_after_output,
        }
        return audit

    info = _apply(llm, install)
    if root is not None:
        write_json(root / "encoder_gpu_memory.json", info)
    return info


def _compare_vectors(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    delta = (reference.float() - actual.float()).abs()
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean())}


def _install_debug_hooks(model):
    """Record RoPE positions and the first logits on the worker model."""
    debug: dict[str, Any] = {"rope_positions": [], "first_logits": None}
    model._tokens_debug = debug

    def pre_hook(_module, args, debug=debug):
        debug["rope_positions"].append(args[0].detach().to("cpu").tolist())

    handle = model.model.layers[0].self_attn.rotary_emb.register_forward_pre_hook(pre_hook)
    model._tokens_debug_handle = handle
    if not hasattr(model, "_orig_compute_logits"):
        model._orig_compute_logits = model.compute_logits
    original = model._orig_compute_logits

    def wrapped(hidden, debug=debug, original=original):
        logits = original(hidden)
        if logits is not None and debug["first_logits"] is None:
            debug["first_logits"] = logits[0].detach().float().cpu()
        return logits

    model.compute_logits = wrapped
    return True


def _consume_debug(model):
    """Return CPU copies of worker debug state and remove the hooks."""
    debug = getattr(model, "_tokens_debug", None) or {}
    handle = getattr(model, "_tokens_debug_handle", None)
    if handle is not None:
        handle.remove()
    model._tokens_debug = None
    model._tokens_debug_handle = None
    if hasattr(model, "_orig_compute_logits"):
        model.compute_logits = model._orig_compute_logits
        delattr(model, "_orig_compute_logits")
    logits = debug.get("first_logits")
    if logits is not None:
        logits = logits.detach().float().cpu()
    return {
        "rope_positions": list(debug.get("rope_positions") or []),
        "first_logits": logits,
        "log": list(model.predictive_state.admission_log),
        "setup_ms": dict(model.predictive_state.setup_ms),
    }


def _capture_active_request(model, request_id: str):
    """Copy H state only while this request still owns its slot."""
    state = model.predictive_state
    slot = state.req_id_to_index.get(request_id)
    if slot is None:
        return {"active": False, "req_id": request_id}
    return {
        "active": True,
        "req_id": request_id,
        "req_index": int(slot),
        "sha256": state.codebook_sha.get(slot),
        "h_input": state.h_input[slot].detach().float().cpu(),
        "h_output": state.h_output[slot].detach().float().cpu(),
        "spans": [int(span) for span in state.h_spans[slot].detach().cpu().tolist()],
    }


def _scheduler_diagnostics(llm) -> dict[str, Any]:
    try:
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    def brief(request) -> dict[str, Any]:
        return {
            "request_id": request.request_id,
            "status": str(request.status),
            "num_computed_tokens": int(request.num_computed_tokens),
            "num_tokens": int(request.num_tokens),
            "num_preemptions": int(request.num_preemptions),
            "max_tokens": int(request.max_tokens),
        }

    pool = scheduler.kv_cache_manager.block_pool
    return {
        "running": [brief(request) for request in list(scheduler.running)],
        "waiting": [brief(request) for request in list(scheduler.waiting)],
        "free_blocks": int(pool.get_num_free_blocks()),
        "num_gpu_blocks": int(pool.num_gpu_blocks),
        "block_size": int(scheduler.block_size),
    }


def _fail_step_budget(root: Path, phase_name: str, llm, steps: int, max_steps: int) -> None:
    diagnostics = _scheduler_diagnostics(llm)
    payload = {
        "status": "FAIL",
        "reason": "engine step budget exceeded",
        "phase": phase_name,
        "steps": steps,
        "max_engine_steps": max_steps,
        "diagnostics": diagnostics,
    }
    write_json(root / f"{phase_name}.json", payload)
    raise RuntimeError(f"{phase_name} exceeded {max_steps} engine steps: {diagnostics}")


def _bounded_steps(
    engine,
    llm,
    *,
    root: Path,
    phase_name: str,
    max_steps: int,
    stop_when=None,
):
    from tokens_vllm.proof_harness import engine_step_decision

    collected = []
    steps = 0
    while True:
        decision = engine_step_decision(
            steps, max_steps, engine.has_unfinished_requests()
        )
        if decision == "stop":
            return collected, steps
        if decision == "budget_exceeded":
            _fail_step_budget(root, phase_name, llm, steps, max_steps)
        collected.extend(engine.step())
        steps += 1
        if stop_when is not None and stop_when(collected):
            return collected, steps


def _finished_ids(collected, request_id: str) -> list[int]:
    finished = [
        item
        for item in collected
        if getattr(item, "request_id", None) == request_id and _is_finished(item)
    ]
    if not finished:
        return []
    return _ids(finished[-1])


def _run_request(
    llm,
    compressed: list[int],
    payload: dict,
    max_tokens: int = MAX_NEW,
    *,
    request_id: str,
    root: Path,
    phase_name: str,
    capture_h: bool = False,
):
    from tokens_vllm.proof_harness import MAX_ENGINE_STEPS

    params = _sampling({"predictive_codebook": payload}, max_tokens=max_tokens)
    _apply(llm, _install_debug_hooks)
    captured = None
    token_ids: list[int] = []
    try:
        assigned = llm.llm_engine.add_request(
            request_id,
            {"prompt_token_ids": compressed},
            params,
        )
        if assigned != request_id:
            raise RuntimeError(
                f"engine assigned request id {assigned}, expected {request_id}"
            )

        def _capture_while_active(collected):
            nonlocal captured
            del collected
            if captured is not None:
                return False
            snap = _apply(
                llm,
                lambda model, request_id=request_id: _capture_active_request(
                    model, request_id
                ),
            )
            if snap.get("active"):
                captured = snap
            return False

        collected, _steps = _bounded_steps(
            llm.llm_engine,
            llm,
            root=root,
            phase_name=phase_name,
            max_steps=MAX_ENGINE_STEPS,
            stop_when=_capture_while_active if capture_h else None,
        )
        if capture_h and captured is None:
            raise RuntimeError(
                f"{request_id} finished before its H vectors could be captured"
            )
        token_ids = _finished_ids(collected, request_id)
    finally:
        snap = _apply(llm, _consume_debug)
    forwards = snap["rope_positions"]
    return {
        "request_id": request_id,
        "token_ids": token_ids,
        "positions": forwards[0] if forwards else [],
        "forwards": forwards,
        "logits": snap["first_logits"],
        "log": snap["log"],
        "setup_ms": snap["setup_ms"],
        "h_capture": captured,
    }


def _logit_report(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    width = min(reference.shape[-1], actual.shape[-1])
    ref = reference[:width].float()
    act = actual[:width].float()
    delta = (ref - act).abs()
    h_delta = (ref[32011:32043] - act[32011:32043]).abs()
    return {
        "max_abs": float(delta.max()),
        "h_max_abs": float(h_delta.max()) if h_delta.numel() else None,
        "base_example_abs": float(delta[100].item()) if width > 100 else None,
        "ref_top1": int(ref.argmax()),
        "act_top1": int(act.argmax()),
        "ref_top5": ref.topk(5).indices.tolist(),
        "act_top5": act.topk(5).indices.tolist(),
        "finite": bool(torch.isfinite(act).all()),
    }


def main() -> None:
    root = _result_root()
    manifest: dict[str, Any] = {"status": "RUNNING", "phases": {}}
    try:
        import vllm

        if not vllm.__version__.startswith("0.30.0"):
            raise RuntimeError(f"vLLM {vllm.__version__} is not 0.30.0")
        env = environment()
        write_json(root / "environment.json", env)
        write_json(root / "run_manifest.json", manifest)

        early = phase_01_and_02(root)
        manifest["phases"].update(early)
        write_json(root / "run_manifest.json", manifest)

        prepared = _build_hf_references(root)
        write_json(
            root / "phase_03_hf_reference.json",
            {
                "status": "PASS",
                "position_mode": prepared["position_mode"],
                "max_subtokens": prepared["max_subtokens"],
                "prompts": [row["prompt_id"] for row in prepared["references"]],
                "hf_setup_ms": {
                    row["prompt_id"]: row["setup_ms"] for row in prepared["references"]
                },
            },
        )
        llm = _make_llm(
            prepared["merged_dir"],
            h_enabled=True,
            max_model_len=512,
            max_num_seqs=4,
            max_num_batched_tokens=512,
            gpu_memory_utilization=PREDICTIVE_GPU_MEMORY_UTILIZATION,
        )
        _record_predictive_warmup(llm, root)
        encoder_info = _install_encoders(llm, prepared["encoder_path"], root=root)
        primary = prepared["references"][0]
        secondary = prepared["references"][1]
        primary_run = _run_request(
            llm,
            primary["compressed_ids"],
            primary["payload"],
            request_id="phase3-primary",
            root=root,
            phase_name="phase_03_single_request_state",
            capture_h=True,
        )
        hf_h = torch.load(root / f"hf_{primary['prompt_id']}.pt", weights_only=False)
        from tokens_vllm.contract import expected_hypertoken_spans

        synthesized = primary_run["h_capture"] or {"active": False}
        expected_spans = expected_hypertoken_spans(primary["payload"]["phrases"])
        got_spans = synthesized.get("spans")
        spans_match = got_spans == expected_spans
        vector_delta = None
        if synthesized.get("active"):
            vector_delta = {
                "h_input": _compare_vectors(hf_h["h_input"], synthesized["h_input"]),
                "h_output": _compare_vectors(hf_h["h_output"], synthesized["h_output"]),
            }
        phase3 = {
            "status": "PASS",
            "prompt_id": primary["prompt_id"],
            "encoder": encoder_info,
            "setup_ms": primary_run["setup_ms"],
            "captured_while_active": bool(synthesized.get("active")),
            "req_index": synthesized.get("req_index"),
            "codebook_sha256": synthesized.get("sha256"),
            "vector_delta": vector_delta,
            "spans": got_spans,
            "expected_spans": expected_spans,
            "spans_match": spans_match,
            "hyperencoders_in_decode": False,
        }
        vectors_ok = (
            vector_delta is not None
            and vector_delta["h_input"]["max_abs"] <= 1e-2
            and vector_delta["h_output"]["max_abs"] <= 1e-2
        )
        if not phase3["captured_while_active"] or not spans_match or not vectors_ok:
            phase3["status"] = "FAIL"
        write_json(root / "phase_03_single_request_state.json", phase3)
        if phase3["status"] != "PASS":
            raise RuntimeError(f"phase 3 H vectors diverged: {phase3}")

        ref_positions = primary["positions"]
        got_positions = primary_run["positions"] or []
        phase4 = {
            "status": "PASS" if got_positions[: len(ref_positions)] == ref_positions else "FAIL",
            "reference_positions": ref_positions,
            "rope_positions": got_positions[: len(ref_positions)],
            "rope_length": len(got_positions),
        }
        write_json(root / "phase_04_semantic_positions.json", phase4)
        if phase4["status"] != "PASS":
            raise RuntimeError(f"phase 4 positions diverged: {phase4}")

        logits_path = root / f"hf_{primary['prompt_id']}_logits.pt"
        logit_info = {"status": "FAIL", "reason": "missing logits"}
        if primary_run["logits"] is not None and logits_path.is_file():
            reference_logits = torch.load(logits_path, weights_only=False)
            logit_info = _logit_report(reference_logits, primary_run["logits"])
            logit_info["status"] = "PASS" if logit_info["ref_top1"] == logit_info["act_top1"] else "FAIL"
        expanded_match = primary_run["token_ids"] == primary["new_ids"]
        phase5 = {
            "status": "PASS" if logit_info.get("status") == "PASS" and expanded_match else "FAIL",
            "logical_match": expanded_match,
            "reference_ids": primary["new_ids"],
            "vllm_ids": primary_run["token_ids"],
            "logits": {k: v for k, v in logit_info.items() if k != "status"},
            "logit_status": logit_info.get("status"),
        }
        write_json(root / "phase_05_logits_and_greedy.json", phase5)
        if phase5["status"] != "PASS":
            raise RuntimeError(f"phase 5 greedy or logits diverged: {phase5}")

        alone_a = primary_run["token_ids"]
        alone_b = _run_request(
            llm,
            secondary["compressed_ids"],
            secondary["payload"],
            request_id="phase6-alone-b",
            root=root,
            phase_name="phase_06_two_codebooks",
        )["token_ids"]

        def isolation(model):
            logical = torch.tensor(
                [32016, 32016], device=model.model.embed_tokens.weight.device
            )
            slots = []
            for event in reversed(model.predictive_state.admission_log):
                if event["event"] == "add" and event.get("h_enabled") and event["req_index"] not in slots:
                    slots.append(event["req_index"])
                if len(slots) == 2:
                    break
            model.predictive_state.token_req_indices[0] = slots[0]
            model.predictive_state.token_req_indices[1] = slots[1]
            embedded = model.embed_input_ids(logical)
            expected_0 = model.predictive_state.h_input[slots[0], 5]
            expected_1 = model.predictive_state.h_input[slots[1], 5]
            return {
                "slots": [int(slot) for slot in slots],
                "row0_match": bool(torch.allclose(embedded[0], expected_0, atol=1e-3, rtol=1e-3)),
                "row1_match": bool(torch.allclose(embedded[1], expected_1, atol=1e-3, rtol=1e-3)),
                "rows_differ": bool((embedded[0] - embedded[1]).abs().max().item() > 1e-4),
            }

        from tokens_vllm.proof_harness import MAX_ENGINE_STEPS

        engine = llm.llm_engine
        # Two tokens so the first scheduled step cannot finish and clear the slots.
        engine.add_request(
            "iso-A",
            {"prompt_token_ids": primary["compressed_ids"]},
            _sampling({"predictive_codebook": primary["payload"]}, max_tokens=2),
        )
        engine.add_request(
            "iso-B",
            {"prompt_token_ids": secondary["compressed_ids"]},
            _sampling({"predictive_codebook": secondary["payload"]}, max_tokens=2),
        )

        def _both_live(_collected):
            presence = _apply(
                llm,
                lambda model: {
                    "a": "iso-A" in model.predictive_state.req_id_to_index,
                    "b": "iso-B" in model.predictive_state.req_id_to_index,
                },
            )
            return bool(presence["a"] and presence["b"])

        _bounded_steps(
            engine,
            llm,
            root=root,
            phase_name="phase_06_two_codebooks",
            max_steps=MAX_ENGINE_STEPS,
            stop_when=_both_live,
        )
        isolated = _apply(llm, isolation)
        engine.abort_request(["iso-A", "iso-B"])
        _bounded_steps(
            engine,
            llm,
            root=root,
            phase_name="phase_06_two_codebooks",
            max_steps=MAX_ENGINE_STEPS,
        )
        both = llm.generate(
            [
                {"prompt_token_ids": primary["compressed_ids"]},
                {"prompt_token_ids": secondary["compressed_ids"]},
            ],
            [
                _sampling({"predictive_codebook": primary["payload"]}),
                _sampling({"predictive_codebook": secondary["payload"]}),
            ],
        )
        pair_ab = [_ids(both[0]), _ids(both[1])]
        swapped = llm.generate(
            [
                {"prompt_token_ids": secondary["compressed_ids"]},
                {"prompt_token_ids": primary["compressed_ids"]},
            ],
            [
                _sampling({"predictive_codebook": secondary["payload"]}),
                _sampling({"predictive_codebook": primary["payload"]}),
            ],
        )
        pair_ba = [_ids(swapped[0]), _ids(swapped[1])]
        phase6 = {
            "status": "PASS",
            "a_match": pair_ab[0] == alone_a and pair_ba[1] == alone_a,
            "b_match": pair_ab[1] == alone_b and pair_ba[0] == alone_b,
            "isolation": isolated,
        }
        if not (phase6["a_match"] and phase6["b_match"] and isolated["row0_match"] and isolated["row1_match"]):
            phase6["status"] = "FAIL"
        write_json(root / "phase_06_two_codebooks.json", phase6)
        if phase6["status"] != "PASS":
            raise RuntimeError(f"phase 6 cross-request mismatch: {phase6}")

        third_phrases = [list(phrase) for phrase in primary["payload"]["phrases"]]
        third_phrases[0], third_phrases[1] = third_phrases[1], third_phrases[0]
        third_payload = _codebook_payload(
            third_phrases,
            max_subtokens=prepared["max_subtokens"],
            disabled_ids=prepared["disabled_ids"],
            pad_id=prepared["pad_id"],
        )
        standalone_c = _run_request(
            llm,
            primary["compressed_ids"],
            third_payload,
            max_tokens=4,
            request_id="phase7-standalone-c",
            root=root,
            phase_name="phase_07_slot_reuse",
        )
        engine = llm.llm_engine
        # C is admitted only after A has been removed.
        a_id = engine.add_request(
            "reuse-A2",
            {"prompt_token_ids": primary["compressed_ids"]},
            _sampling({"predictive_codebook": primary["payload"]}, max_tokens=1),
        )
        engine.add_request(
            "reuse-B2",
            {"prompt_token_ids": secondary["compressed_ids"]},
            _sampling({"predictive_codebook": secondary["payload"]}, max_tokens=6),
        )

        def _a_finished(step_outs, a_id=a_id):
            return any(
                getattr(item, "request_id", None) == a_id and _is_finished(item)
                for item in step_outs
            )

        _bounded_steps(
            engine,
            llm,
            root=root,
            phase_name="phase_07_slot_reuse",
            max_steps=MAX_ENGINE_STEPS,
            stop_when=_a_finished,
        )
        c_id = engine.add_request(
            "reuse-C",
            {"prompt_token_ids": primary["compressed_ids"]},
            _sampling({"predictive_codebook": third_payload}, max_tokens=4),
        )
        collected, _steps = _bounded_steps(
            engine,
            llm,
            root=root,
            phase_name="phase_07_slot_reuse",
            max_steps=MAX_ENGINE_STEPS,
        )
        c_ids = _finished_ids(collected, c_id)

        def reuse_snapshot(model):
            return list(model.predictive_state.admission_log)

        reuse_log = _apply(llm, reuse_snapshot)
        c_adds = [event for event in reuse_log if event["event"] == "add" and event["req_id"] == c_id]
        a_removes = [event for event in reuse_log if event["event"] == "remove" and event["req_id"] == a_id]
        phase7 = {
            "status": "FAIL",
            "c_ids": c_ids,
            "standalone_c": standalone_c["token_ids"],
            "c_was_clear": bool(c_adds and c_adds[-1].get("was_clear")),
            "a_removed": bool(a_removes),
            "same_slot": bool(
                c_adds and a_removes and c_adds[-1]["req_index"] == a_removes[-1]["req_index"]
            ),
        }
        if (
            phase7["c_was_clear"]
            and phase7["a_removed"]
            and phase7["same_slot"]
            and c_ids == standalone_c["token_ids"]
        ):
            phase7["status"] = "PASS"
        write_json(root / "phase_07_slot_reuse.json", phase7)
        if phase7["status"] != "PASS":
            raise RuntimeError(f"phase 7 slot reuse failed: {phase7}")
        del engine
        llm_shutdown = shutdown_vllm_engine(llm)
        del llm
        _record_teardown(root, "llm", llm_shutdown, cuda_after_collect())

        chunk_llm = _make_llm(
            prepared["merged_dir"],
            h_enabled=True,
            max_model_len=512,
            max_num_seqs=2,
            max_num_batched_tokens=16,
            enable_chunked_prefill=True,
            gpu_memory_utilization=PREDICTIVE_GPU_MEMORY_UTILIZATION,
        )
        _install_encoders(chunk_llm, prepared["encoder_path"], root=root)
        chunk_run = _run_request(
            chunk_llm,
            primary["compressed_ids"],
            primary["payload"],
            request_id="phase8-chunk",
            root=root,
            phase_name="phase_08_chunked_prefill",
        )
        chunk_positions: list[int] = []
        for forward in chunk_run["forwards"]:
            if len(chunk_positions) >= len(primary["positions"]):
                break
            chunk_positions.extend(int(pos) for pos in forward)
        chunk_positions = chunk_positions[: len(primary["positions"])]
        phase8 = {
            "status": "PASS" if chunk_run["token_ids"] == primary["new_ids"] else "FAIL",
            "chunk_ids": chunk_run["token_ids"],
            "reference_ids": primary["new_ids"],
            "chunk_positions": chunk_positions,
            "reference_positions": primary["positions"],
        }
        if chunk_positions != primary["positions"]:
            phase8["status"] = "FAIL"
        write_json(root / "phase_08_chunked_prefill.json", phase8)
        chunk_shutdown = shutdown_vllm_engine(chunk_llm)
        del chunk_llm
        _record_teardown(root, "chunk_llm", chunk_shutdown, cuda_after_collect())
        if phase8["status"] != "PASS":
            raise RuntimeError(f"phase 8 chunked prefill diverged: {phase8}")

        from tokens_vllm.proof_harness import preemption_block_budget, preempted_request_ids

        budget = preemption_block_budget(
            len(primary["compressed_ids"]),
            len(secondary["compressed_ids"]),
            PROOF_BLOCK_SIZE,
        )
        preempt_llm = _make_llm(
            prepared["merged_dir"],
            h_enabled=True,
            max_model_len=budget["max_model_len"],
            max_num_seqs=2,
            max_num_batched_tokens=budget["max_model_len"],
            block_size=PROOF_BLOCK_SIZE,
            num_gpu_blocks_override=budget["num_gpu_blocks"],
            gpu_memory_utilization=PREDICTIVE_GPU_MEMORY_UTILIZATION,
        )
        _install_encoders(preempt_llm, prepared["encoder_path"], root=root)
        idle = _scheduler_diagnostics(preempt_llm)
        budget_report = {
            "block_size": int(preempt_llm.llm_engine.vllm_config.cache_config.block_size),
            "available_blocks": idle.get("free_blocks"),
            "num_gpu_blocks": idle.get("num_gpu_blocks"),
            "a_required_blocks": budget["a_full_blocks"],
            "b_required_blocks": budget["b_full_blocks"],
            "a_prompt_blocks": budget["a_prompt_blocks"],
            "b_prompt_blocks": budget["b_prompt_blocks"],
            "combined_requirement": budget["combined_full_blocks"],
            "max_new_tokens": budget["max_new_tokens"],
            "prompt_a_tokens": budget["prompt_a_tokens"],
            "prompt_b_tokens": budget["prompt_b_tokens"],
        }
        budget_ok = (
            budget_report["block_size"] == PROOF_BLOCK_SIZE
            and budget_report["num_gpu_blocks"] == budget["num_gpu_blocks"]
            and budget_report["available_blocks"] == budget["usable_blocks"]
            and budget["a_full_blocks"] <= budget["usable_blocks"]
            and budget["b_full_blocks"] <= budget["usable_blocks"]
            and budget["combined_full_blocks"] > budget["usable_blocks"]
        )
        if not budget_ok:
            phase9 = {
                "status": "FAIL",
                "reason": "preemption block budget was not applied",
                **budget_report,
            }
            write_json(root / "phase_09_preemption.json", phase9)
            preempt_shutdown = shutdown_vllm_engine(preempt_llm)
            del preempt_llm
            _record_teardown(root, "preempt_llm", preempt_shutdown, cuda_after_collect())
            raise RuntimeError(f"phase 9 preemption budget was not applied: {phase9}")
        max_new = budget["max_new_tokens"]
        uninterrupted_a = _run_request(
            preempt_llm,
            primary["compressed_ids"],
            primary["payload"],
            max_tokens=max_new,
            request_id="pre-A-solo",
            root=root,
            phase_name="phase_09_preemption",
        )
        uninterrupted_b = _run_request(
            preempt_llm,
            secondary["compressed_ids"],
            secondary["payload"],
            max_tokens=max_new,
            request_id="pre-B-solo",
            root=root,
            phase_name="phase_09_preemption",
        )
        engine = preempt_llm.llm_engine
        engine.add_request(
            "pre-A",
            {"prompt_token_ids": primary["compressed_ids"]},
            _sampling({"predictive_codebook": primary["payload"]}, max_tokens=max_new),
        )
        engine.add_request(
            "pre-B",
            {"prompt_token_ids": secondary["compressed_ids"]},
            _sampling({"predictive_codebook": secondary["payload"]}, max_tokens=max_new),
        )
        preempt_collected, preempt_steps = _bounded_steps(
            engine,
            preempt_llm,
            root=root,
            phase_name="phase_09_preemption",
            max_steps=MAX_ENGINE_STEPS,
        )
        final_ids = {
            "pre-A": _finished_ids(preempt_collected, "pre-A"),
            "pre-B": _finished_ids(preempt_collected, "pre-B"),
        }
        solo_ids = {
            "pre-A": uninterrupted_a["token_ids"],
            "pre-B": uninterrupted_b["token_ids"],
        }

        def preempt_log(model):
            return list(model.predictive_state.admission_log)

        events = _apply(preempt_llm, preempt_log)
        victims = [
            req_id
            for req_id in ("pre-A", "pre-B")
            if req_id in preempted_request_ids(events)
        ]
        trajectories_match = bool(victims) and all(
            final_ids[req_id] == solo_ids[req_id] for req_id in victims
        )
        if not victims:
            reason = "preemption not exercised"
        elif not trajectories_match:
            reason = "preempted trajectory diverged"
        else:
            reason = None
        phase9 = {
            "status": "PASS" if reason is None else "FAIL",
            "reason": reason,
            "preempted_request_ids": victims,
            "steps": preempt_steps,
            "final_ids": final_ids,
            "uninterrupted": solo_ids,
            **budget_report,
        }
        write_json(root / "phase_09_preemption.json", phase9)
        del engine
        preempt_shutdown = shutdown_vllm_engine(preempt_llm)
        del preempt_llm
        _record_teardown(root, "preempt_llm", preempt_shutdown, cuda_after_collect())
        if phase9["status"] != "PASS":
            raise RuntimeError(f"phase 9 preemption failed: {phase9}")

        rope_llm = _make_llm(
            prepared["merged_dir"],
            h_enabled=True,
            max_model_len=8,
            max_num_seqs=1,
            max_num_batched_tokens=8,
            gpu_memory_utilization=PREDICTIVE_GPU_MEMORY_UTILIZATION,
        )
        _install_encoders(rope_llm, prepared["encoder_path"], root=root)
        synthetic = []
        for slot in range(32):
            synthetic.append([100 + slot, 200 + slot, 300 + slot])
        synthetic_payload = _codebook_payload(
            synthetic,
            max_subtokens=prepared["max_subtokens"],
            disabled_ids=prepared["disabled_ids"],
            pad_id=prepared["pad_id"],
        )
        rope_ids = [32011] * 6
        rope_run = _run_request(
            rope_llm,
            rope_ids,
            synthetic_payload,
            max_tokens=1,
            request_id="phase10-rope",
            root=root,
            phase_name="phase_10_semantic_rope",
        )
        rope_positions = rope_run["positions"] or []
        phase10 = {
            "status": "FAIL",
            "max_model_len": 8,
            "physical_tokens": len(rope_ids),
            "rope_positions": rope_positions,
            "max_semantic": max(rope_positions) if rope_positions else None,
        }
        if rope_positions and max(rope_positions) > 8 and max(rope_positions) < 131072:
            phase10["status"] = "PASS"
        write_json(root / "phase_10_semantic_rope.json", phase10)
        rope_shutdown = shutdown_vllm_engine(rope_llm)
        del rope_llm
        _record_teardown(root, "rope_llm", rope_shutdown, cuda_after_collect())
        if phase10["status"] != "PASS":
            raise RuntimeError(f"phase 10 semantic RoPE failed: {phase10}")

        manifest["status"] = "PASS"
        manifest["phases"] = {
            "01": "PASS",
            "02": "PASS",
            "03": "PASS",
            "04": "PASS",
            "05": "PASS",
            "06": "PASS",
            "07": "PASS",
            "08": "PASS",
            "09": "PASS",
            "10": "PASS",
        }
        write_json(root / "run_manifest.json", manifest)
    except Exception:
        manifest["status"] = "FAIL"
        manifest["traceback"] = traceback.format_exc()
        write_json(root / "run_manifest.json", manifest)
        write_json(root / "failure.json", {"traceback": manifest["traceback"]})
        print(manifest["traceback"], flush=True)
        raise


if __name__ == "__main__":
    main()
