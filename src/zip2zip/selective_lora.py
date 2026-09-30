"""Surgical masking and parameter accounting for Selective LoRA configurations.

Provides mechanisms to enable/disable specific subsets of transformer layers and
target modules in pre-trained LoRA checkpoints without modifying on-disk weights.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
from torch import nn

PHI_35_TOTAL_PARAMETERS = 3_821_079_552
TOTAL_TRANSFORMER_LAYERS = 32

ALL_LORA_MODULES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
ATTN_LORA_MODULES = ("qkv_proj", "o_proj")
MLP_LORA_MODULES = ("gate_up_proj", "down_proj")


@dataclass(frozen=True)
class LoRAMaskConfig:
    """Specification for a surgical LoRA mask."""

    name: str
    enabled_layers: Sequence[int] = field(default_factory=tuple)
    enabled_modules: Sequence[str] = field(default_factory=tuple)
    description: str = ""

    def __post_init__(self):
        # Normalize to sorted tuples
        object.__setattr__(
            self, "enabled_layers", tuple(sorted(set(int(l) for l in self.enabled_layers)))
        )
        object.__setattr__(
            self, "enabled_modules", tuple(sorted(set(str(m) for m in self.enabled_modules)))
        )


# Standard Ladder Steps
LADDER_L0_ZERO = LoRAMaskConfig(
    name="L0_ZERO",
    enabled_layers=(),
    enabled_modules=(),
    description="Zero LoRA contribution anywhere; hyperencoders and codebook manager active.",
)

LADDER_L1_TINY_ATTN_LAST4 = LoRAMaskConfig(
    name="L1_TINY_ATTN_LAST4",
    enabled_layers=tuple(range(28, 32)),
    enabled_modules=ATTN_LORA_MODULES,
    description="Tiny selective LoRA: attention projections (qkv + o) only on last 4 blocks (28-31).",
)

LADDER_L2_SMALL_ATTN_LAST8 = LoRAMaskConfig(
    name="L2_SMALL_ATTN_LAST8",
    enabled_layers=tuple(range(24, 32)),
    enabled_modules=ATTN_LORA_MODULES,
    description="Small selective LoRA: attention projections (qkv + o) only on last 8 blocks (24-31).",
)

LADDER_L3_MOD_ALL_LAST8 = LoRAMaskConfig(
    name="L3_MOD_ALL_LAST8",
    enabled_layers=tuple(range(24, 32)),
    enabled_modules=ALL_LORA_MODULES,
    description="Moderate selective LoRA: all modules (attention + MLP) on last 8 blocks (24-31).",
)

LADDER_LFULL = LoRAMaskConfig(
    name="LFULL",
    enabled_layers=tuple(range(32)),
    enabled_modules=ALL_LORA_MODULES,
    description="Full existing Step-100 LoRA: all 32 transformer blocks, all 4 module families.",
)

STANDARD_LORA_LADDER = (
    LADDER_L0_ZERO,
    LADDER_L1_TINY_ATTN_LAST4,
    LADDER_L2_SMALL_ATTN_LAST8,
    LADDER_L3_MOD_ALL_LAST8,
    LADDER_LFULL,
)


def parse_lora_parameter_identity(param_name: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Parse (layer_idx, module_type, lora_component) from parameter name.

    Examples:
        'base_model.model.model.layers.0.self_attn.o_proj.lora_A.default.weight'
        -> (0, 'o_proj', 'lora_A')
    """
    parts = param_name.split(".")
    layer_idx: Optional[int] = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            layer_idx = int(parts[i + 1])
            break

    module_type: Optional[str] = None
    for m in ALL_LORA_MODULES:
        for p in parts:
            if m == p or m in p:
                module_type = m
                break
        if module_type is not None:
            break

    lora_component: Optional[str] = None
    for c in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
        for p in parts:
            if c == p or c in p:
                lora_component = c
                break
        if lora_component is not None:
            break

    return layer_idx, module_type, lora_component


def is_lora_module_enabled(
    layer_idx: Optional[int], module_type: Optional[str], config: LoRAMaskConfig
) -> bool:
    """Return True if the specific layer and module are active under config."""
    if layer_idx is None or module_type is None:
        return False
    return (layer_idx in config.enabled_layers) and (module_type in config.enabled_modules)


@dataclass
class LoRAMaskReport:
    """Full provenance and measurement report for an applied LoRA mask."""

    mask_name: str
    enabled_layers: List[int]
    enabled_modules: List[str]
    total_lora_parameters: int
    active_lora_parameters: int
    masked_lora_parameters: int
    active_parameter_pct_of_phi: float
    total_lora_tensors: int
    active_lora_tensors: int
    masked_lora_tensors: int
    active_weight_sha256: str
    effective_rank: int = 32

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mask_name": self.mask_name,
            "enabled_layers": list(self.enabled_layers),
            "enabled_modules": list(self.enabled_modules),
            "total_lora_parameters": self.total_lora_parameters,
            "active_lora_parameters": self.active_lora_parameters,
            "masked_lora_parameters": self.masked_lora_parameters,
            "active_parameter_pct_of_phi": round(self.active_parameter_pct_of_phi, 6),
            "total_lora_tensors": self.total_lora_tensors,
            "active_lora_tensors": self.active_lora_tensors,
            "masked_lora_tensors": self.masked_lora_tensors,
            "active_weight_sha256": self.active_weight_sha256,
            "effective_rank": self.effective_rank,
        }


def apply_lora_mask(
    model: nn.Module,
    config: LoRAMaskConfig,
    *,
    reference_lora_state_dict: Optional[Mapping[str, torch.Tensor]] = None,
) -> LoRAMaskReport:
    """Surgically enable or disable LoRA modules in ``model`` according to ``config``.

    Masking zeroes out ``lora_B`` weights so that (B @ A) @ x == 0.0 identically,
    leaving the base linear projection unaltered without modifying checkpoint files.
    """
    total_params = 0
    active_params = 0
    masked_params = 0
    total_tensors = 0
    active_tensors = 0
    masked_tensors = 0

    hasher = hashlib.sha256()

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora" not in name.lower():
                continue

            layer_idx, module_type, component = parse_lora_parameter_identity(name)
            if component is None:
                continue

            total_tensors += 1
            total_params += param.numel()

            enabled = is_lora_module_enabled(layer_idx, module_type, config)

            if enabled:
                if reference_lora_state_dict is not None and name in reference_lora_state_dict:
                    param.copy_(reference_lora_state_dict[name].to(device=param.device, dtype=param.dtype))
                active_params += param.numel()
                active_tensors += 1
                hasher.update(name.encode("utf-8"))
                hasher.update(param.detach().cpu().numpy().tobytes())
            else:
                # Mask out this module by zeroing lora_B
                if "lora_b" in component.lower():
                    param.zero_()
                elif reference_lora_state_dict is not None and name in reference_lora_state_dict:
                    # Restore lora_A to reference state so it doesn't drift if re-enabled later
                    param.copy_(reference_lora_state_dict[name].to(device=param.device, dtype=param.dtype))
                masked_params += param.numel()
                masked_tensors += 1

    pct_of_phi = (active_params / PHI_35_TOTAL_PARAMETERS) * 100.0 if PHI_35_TOTAL_PARAMETERS > 0 else 0.0

    return LoRAMaskReport(
        mask_name=config.name,
        enabled_layers=list(config.enabled_layers),
        enabled_modules=list(config.enabled_modules),
        total_lora_parameters=total_params,
        active_lora_parameters=active_params,
        masked_lora_parameters=masked_params,
        active_parameter_pct_of_phi=pct_of_phi,
        total_lora_tensors=total_tensors,
        active_lora_tensors=active_tensors,
        masked_lora_tensors=masked_tensors,
        active_weight_sha256=hasher.hexdigest(),
        effective_rank=32,
    )
