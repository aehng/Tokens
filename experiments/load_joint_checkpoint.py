"""Load a predictive joint checkpoint saved by experiments/train_predictive_zip2zip.py.

The trainer writes a nested mapping, not a flat Zip2Zip state_dict:

    lora_state_dict: parameters from model.base_model.named_parameters()
        whose names contain "lora" (PEFT keeps the adapter segment, e.g.
        ``lora_A.default.weight``)
    input_encoder_state_dict: model.input_encoder.state_dict()
    output_encoder_state_dict: model.output_encoder.state_dict()

Passing that outer mapping to ``model.load_state_dict(..., strict=False)``
ignores every trained tensor.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

CHECKPOINT_LOADER_ID = "joint_nested_v2"

HISTORICAL_BASELINE_STATUS = "historical_unverified"

_LORA_KEY = "lora_state_dict"
_INPUT_KEY = "input_encoder_state_dict"
_OUTPUT_KEY = "output_encoder_state_dict"
_REQUIRED = (_LORA_KEY, _INPUT_KEY, _OUTPUT_KEY)


def stamp_generation_record(record: dict) -> dict:
    """Mark a generation produced after the nested loader actually ran."""
    record["checkpoint_loader"] = CHECKPOINT_LOADER_ID
    return record


def mark_historical_baseline(record: dict) -> dict:
    """Label an old baseline for provisional reporting, never for cache reuse."""
    record["baseline_provenance"] = HISTORICAL_BASELINE_STATUS
    return record


def is_historical_provisional_baseline(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and record.get("condition") == "cond_a_baseline_k32"
        and record.get("baseline_provenance") == HISTORICAL_BASELINE_STATUS
    )


def accept_cached_generation(record: Any) -> bool:
    """True when a saved row is safe to reuse as a Step-100 generation.

    Unstamped rows from the four broken runners are rejected. Historical files
    are not modified; callers simply treat the row as a cache miss.
    """
    if not isinstance(record, dict):
        return False
    return record.get("checkpoint_loader") == CHECKPOINT_LOADER_ID


def load_joint_checkpoint(
    model: nn.Module,
    source: str | os.PathLike | Mapping,
    *,
    expected_step: int | None = None,
    expected_model_id: str | None = None,
) -> dict:
    """Install nested joint weights onto ``model`` and return a small report.

    ``source`` is a checkpoint path or an already-loaded mapping. Every
    checkpoint tensor must match the live tensor after the copy. Non-LoRA base
    parameters outside the aliased hyper-encoders must be unchanged.
    """
    checkpoint = _read_checkpoint(source)
    _require_nested_sections(checkpoint)
    step = _checkpoint_step(checkpoint)
    if expected_step is not None and step != expected_step:
        raise ValueError(
            f"wrong joint checkpoint step: expected {expected_step}, got {step}"
        )
    checkpoint_model_id = _checkpoint_model_id(checkpoint)
    if expected_model_id is not None and checkpoint_model_id != expected_model_id:
        raise ValueError(
            "wrong joint checkpoint model ID: "
            f"expected {expected_model_id!r}, got {checkpoint_model_id!r}"
        )

    lora_state = _tensor_mapping(checkpoint[_LORA_KEY], _LORA_KEY)
    input_state = _tensor_mapping(checkpoint[_INPUT_KEY], _INPUT_KEY)
    output_state = _tensor_mapping(checkpoint[_OUTPUT_KEY], _OUTPUT_KEY)

    base_model = model.base_model
    input_encoder = model.input_encoder
    output_encoder = getattr(model, "output_encoder", None)
    if output_encoder is None:
        raise RuntimeError("joint checkpoint loading requires model.output_encoder")

    lora_params = _lora_parameters(base_model)
    _validate_lora(lora_params, lora_state)
    _validate_module_state(input_encoder, input_state, _INPUT_KEY)
    _validate_module_state(output_encoder, output_state, _OUTPUT_KEY)

    before = {
        "lora": _tensor_group_summary(lora_params),
        "input_encoder": _tensor_group_summary(input_encoder.state_dict()),
        "output_encoder": _tensor_group_summary(output_encoder.state_dict()),
    }

    encoder_parameter_ids = {
        id(param)
        for encoder in (input_encoder, output_encoder)
        if encoder is not None
        for param in encoder.parameters()
    }
    base_fingerprint = _parameter_fingerprint(base_model, encoder_parameter_ids)

    _assign_named(lora_params, lora_state, "lora")
    _assign_module_state(input_encoder, input_state, _INPUT_KEY)
    if output_encoder is not None:
        _assign_module_state(output_encoder, output_state, _OUTPUT_KEY)

    _assert_installed(lora_params, lora_state, "lora")
    _assert_module_installed(input_encoder, input_state, _INPUT_KEY)
    _assert_module_installed(output_encoder, output_state, _OUTPUT_KEY)

    after = {
        "lora": _tensor_group_summary(lora_params),
        "input_encoder": _tensor_group_summary(input_encoder.state_dict()),
        "output_encoder": _tensor_group_summary(output_encoder.state_dict()),
    }
    components = {
        "lora": _component_report(before["lora"], after["lora"], lora_state),
        "input_encoder": _component_report(before["input_encoder"], after["input_encoder"], input_state),
        "output_encoder": _component_report(before["output_encoder"], after["output_encoder"], output_state),
    }
    changed_count = sum(component["changed_tensor_count"] for component in components.values())
    if changed_count == 0:
        raise RuntimeError(
            "joint checkpoint tensors exactly match all live trainable weights before loading; "
            "the checkpoint load did not change any expected parameter"
        )

    after = _parameter_fingerprint(base_model, encoder_parameter_ids)
    if after != base_fingerprint:
        raise AssertionError("Frozen base weights changed while loading the joint checkpoint")

    return {
        "step": step,
        "trainable_mode": checkpoint.get("trainable_mode"),
        "model_id": checkpoint_model_id,
        "base_hash_status": (
            "present_unverified"
            if isinstance(checkpoint.get("base_hashes"), Mapping) and checkpoint["base_hashes"]
            else "missing"
        ),
        "lora_tensors": len(lora_state),
        "input_encoder_tensors": len(input_state),
        "output_encoder_tensors": len(output_state),
        "missing_keys": {name: values["missing_keys"] for name, values in components.items()},
        "unexpected_keys": {name: values["unexpected_keys"] for name, values in components.items()},
        "components": components,
        "changed_tensor_count": changed_count,
        "checkpoint_loader": CHECKPOINT_LOADER_ID,
    }


def _checkpoint_step(checkpoint: Mapping) -> int:
    step = checkpoint.get("step", checkpoint.get("global_step"))
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(
            "joint checkpoint must contain a non-negative integer `step` or `global_step`; "
            f"got {step!r}"
        )
    return step


def _checkpoint_model_id(checkpoint: Mapping) -> str | None:
    config = checkpoint.get("config")
    model_config = config.get("model") if isinstance(config, Mapping) else None
    if not isinstance(model_config, Mapping):
        return None
    value = model_config.get("name_or_path")
    return value if isinstance(value, str) and value else None


def _read_checkpoint(source: str | os.PathLike | Mapping) -> Mapping:
    if isinstance(source, Mapping):
        return source
    try:
        return torch.load(
            os.fspath(source),
            # Keep the full archive, including optimizer state, on CPU. Only
            # tensors in the three model sections are copied to live devices.
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "This PyTorch build does not support safe torch.load(mmap=True, weights_only=True); "
            "refusing a non-memory-mapped fallback."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Could not load joint checkpoint with safe memory-mapped torch.load "
            "(mmap=True, weights_only=True). The checkpoint may use an unsupported format or "
            "contain unsupported Python objects; refusing non-memory-mapped or unrestricted-pickle fallback."
        ) from exc


def _require_nested_sections(checkpoint: Mapping) -> None:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"joint checkpoint must be a mapping, got {type(checkpoint).__name__}")
    missing = [key for key in _REQUIRED if key not in checkpoint]
    if missing:
        raise KeyError(
            "joint checkpoint is missing "
            f"{missing}. Trainers save lora_state_dict, input_encoder_state_dict, and "
            "output_encoder_state_dict; the outer mapping is not a model state_dict."
        )


def _tensor_mapping(state: Any, label: str) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise TypeError(f"{label} must be a mapping, got {type(state).__name__}")
    tensors: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise TypeError(f"{label} key {key!r} is not a string")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label}[{key!r}] is {type(value).__name__}, expected a tensor")
        tensors[key] = value
    return tensors


def _lora_parameters(base_model: nn.Module) -> dict[str, nn.Parameter]:
    # Same filter the trainer uses when it builds lora_state_dict.
    return {name: param for name, param in base_model.named_parameters() if "lora" in name.lower()}


def _validate_lora(live: Mapping[str, torch.Tensor], incoming: Mapping[str, torch.Tensor]) -> None:
    if not incoming:
        raise RuntimeError("lora_state_dict is empty")
    missing = sorted(set(live) - set(incoming))
    unexpected = sorted(set(incoming) - set(live))
    if missing or unexpected:
        raise RuntimeError(
            "LoRA key mismatch against model.base_model.named_parameters() "
            f"(trainer format, adapter segment kept). missing={missing} unexpected={unexpected}"
        )
    _validate_shapes(live, incoming, "lora_state_dict")


def _validate_module_state(module: nn.Module, incoming: Mapping[str, torch.Tensor], label: str) -> None:
    live = module.state_dict()
    missing = sorted(set(live) - set(incoming))
    unexpected = sorted(set(incoming) - set(live))
    if missing or unexpected:
        raise RuntimeError(f"{label} key mismatch. missing={missing} unexpected={unexpected}")
    _validate_shapes(live, incoming, label)


def _validate_shapes(live: Mapping[str, torch.Tensor], incoming: Mapping[str, torch.Tensor], label: str) -> None:
    for key, value in incoming.items():
        live_shape = tuple(live[key].shape)
        ckpt_shape = tuple(value.shape)
        if live_shape != ckpt_shape:
            raise RuntimeError(
                f"{label} shape mismatch for {key}: checkpoint {ckpt_shape} live {live_shape}"
            )


def _assign_named(live: Mapping[str, torch.Tensor], incoming: Mapping[str, torch.Tensor], label: str) -> None:
    for key, value in incoming.items():
        _assign_exact(live[key], value, f"{label}.{key}")


def _assign_module_state(module: nn.Module, incoming: Mapping[str, torch.Tensor], label: str) -> None:
    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    for key, value in incoming.items():
        if key in params:
            _assign_exact(params[key], value, f"{label}.{key}")
        elif key in buffers:
            _assign_exact(buffers[key], value, f"{label}.{key}")
        else:
            raise RuntimeError(f"{label} has no live parameter or buffer named {key}")


def _assign_exact(dest: torch.Tensor, src: torch.Tensor, name: str) -> None:
    cloned = src.detach().to(device=dest.device).clone()
    if tuple(dest.shape) != tuple(cloned.shape):
        raise RuntimeError(f"{name} shape mismatch during assign")
    # Keep the trained dtype. copy_() into a narrower live tensor would not
    # match the checkpoint bitwise.
    if dest.dtype != cloned.dtype:
        dest.data = cloned
    else:
        dest.data.copy_(cloned)


def _assert_installed(live: Mapping[str, torch.Tensor], incoming: Mapping[str, torch.Tensor], label: str) -> None:
    for key, value in incoming.items():
        if not _bitwise_equal(live[key], value):
            raise AssertionError(f"{label}.{key} does not match the checkpoint tensor after load")


def _assert_module_installed(module: nn.Module, incoming: Mapping[str, torch.Tensor], label: str) -> None:
    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    for key, value in incoming.items():
        dest = params[key] if key in params else buffers[key]
        if not _bitwise_equal(dest, value):
            raise AssertionError(f"{label}.{key} does not match the checkpoint tensor after load")


def _bitwise_equal(live: torch.Tensor, src: torch.Tensor) -> bool:
    a = live.detach().cpu()
    b = src.detach().cpu()
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    return bool(torch.equal(a, b))


def _tensor_digest(value: torch.Tensor) -> tuple[str, float]:
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise RuntimeError("checkpoint/live tensor contains NaN or infinity")

    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    byte_tensor = tensor.reshape(-1).view(torch.uint8)
    digest.update(memoryview(byte_tensor.numpy()).cast("B"))
    norm = float(torch.linalg.vector_norm(tensor.float()).item()) if tensor.numel() else 0.0
    return digest.hexdigest(), norm


def _tensor_group_summary(named_tensors: Mapping[str, torch.Tensor]) -> dict:
    digest = hashlib.sha256()
    tensor_digests: dict[str, str] = {}
    squared_norm = 0.0
    for name, tensor in sorted(named_tensors.items()):
        tensor_digest, norm = _tensor_digest(tensor)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_digest.encode("ascii"))
        tensor_digests[name] = tensor_digest
        squared_norm += norm * norm
    digest.update(str(len(tensor_digests)).encode("ascii"))
    return {
        "sha256": digest.hexdigest(),
        "l2_norm": math.sqrt(squared_norm),
        "tensor_digests": tensor_digests,
    }


def _component_report(before: Mapping, after: Mapping, incoming: Mapping[str, torch.Tensor]) -> dict:
    missing = sorted(set(before["tensor_digests"]) - set(incoming))
    unexpected = sorted(set(incoming) - set(before["tensor_digests"]))
    changed = [
        name
        for name, tensor_digest in before["tensor_digests"].items()
        if after["tensor_digests"].get(name) != tensor_digest
    ]
    return {
        "tensor_count": len(after["tensor_digests"]),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "before_sha256": before["sha256"],
        "after_sha256": after["sha256"],
        "before_l2_norm": before["l2_norm"],
        "after_l2_norm": after["l2_norm"],
        "changed_tensor_count": len(changed),
    }


def _parameter_fingerprint(base_model: nn.Module, excluded_ids: set[int]) -> str:
    """Hash non-LoRA base parameters, excluding encoder aliases by identity."""
    hasher = hashlib.sha256()
    count = 0
    for name, param in base_model.named_parameters():
        if "lora" in name.lower() or id(param) in excluded_ids:
            continue
        count += 1
        tensor = param.detach().to(device="cpu").contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        byte_tensor = tensor.reshape(-1).view(torch.uint8)
        hasher.update(memoryview(byte_tensor.numpy()).cast("B"))
    hasher.update(str(count).encode("utf-8"))
    return hasher.hexdigest()
