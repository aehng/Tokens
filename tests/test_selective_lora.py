"""Unit tests for surgical selective LoRA masking and parameter accounting."""

from types import SimpleNamespace
import pytest
import torch
from torch import nn

from zip2zip.selective_lora import (
    ALL_LORA_MODULES,
    ATTN_LORA_MODULES,
    LADDER_L0_ZERO,
    LADDER_L1_TINY_ATTN_LAST4,
    LADDER_L2_SMALL_ATTN_LAST8,
    LADDER_L3_MOD_ALL_LAST8,
    LADDER_LFULL,
    LoRAMaskConfig,
    apply_lora_mask,
    is_lora_module_enabled,
    parse_lora_parameter_identity,
)


def test_parse_lora_parameter_identity():
    name1 = "base_model.model.model.layers.0.self_attn.o_proj.lora_A.default.weight"
    layer, mod, comp = parse_lora_parameter_identity(name1)
    assert layer == 0
    assert mod == "o_proj"
    assert comp == "lora_A"

    name2 = "base_model.model.model.layers.31.mlp.gate_up_proj.lora_B.default.weight"
    layer, mod, comp = parse_lora_parameter_identity(name2)
    assert layer == 31
    assert mod == "gate_up_proj"
    assert comp == "lora_B"

    name3 = "base_model.model.model.layers.15.mlp.down_proj.lora_A.weight"
    layer, mod, comp = parse_lora_parameter_identity(name3)
    assert layer == 15
    assert mod == "down_proj"
    assert comp == "lora_A"

    name_non_lora = "base_model.model.model.layers.0.self_attn.qkv_proj.weight"
    layer, mod, comp = parse_lora_parameter_identity(name_non_lora)
    assert layer == 0
    assert mod == "qkv_proj"
    assert comp is None


def test_is_lora_module_enabled():
    config = LoRAMaskConfig(
        name="test",
        enabled_layers=(28, 29, 30, 31),
        enabled_modules=("qkv_proj", "o_proj"),
    )
    assert is_lora_module_enabled(28, "qkv_proj", config) is True
    assert is_lora_module_enabled(31, "o_proj", config) is True
    assert is_lora_module_enabled(27, "qkv_proj", config) is False  # layer 27 not in 28..31
    assert is_lora_module_enabled(28, "gate_up_proj", config) is False  # MLP not enabled


class MockPhiBlock(nn.Module):
    def __init__(self, d_model=16, r=4):
        super().__init__()
        # Attention
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.qkv_proj.weight.data.normal_(std=0.05)
        self.qkv_proj_lora_A = nn.Parameter(torch.randn(r, d_model) * 0.05)
        self.qkv_proj_lora_B = nn.Parameter(torch.randn(3 * d_model, r) * 0.05)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj.weight.data.normal_(std=0.05)
        self.o_proj_lora_A = nn.Parameter(torch.randn(r, d_model) * 0.05)
        self.o_proj_lora_B = nn.Parameter(torch.randn(d_model, r) * 0.05)
        # MLP
        self.gate_up_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.gate_up_proj.weight.data.normal_(std=0.05)
        self.gate_up_proj_lora_A = nn.Parameter(torch.randn(r, d_model) * 0.05)
        self.gate_up_proj_lora_B = nn.Parameter(torch.randn(2 * d_model, r) * 0.05)
        self.down_proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.down_proj.weight.data.normal_(std=0.05)
        self.down_proj_lora_A = nn.Parameter(torch.randn(r, 2 * d_model) * 0.05)
        self.down_proj_lora_B = nn.Parameter(torch.randn(d_model, r) * 0.05)

    def forward(self, x):
        # Attention with LoRA
        qkv = self.qkv_proj(x) + (x @ self.qkv_proj_lora_A.T) @ self.qkv_proj_lora_B.T
        attn_out = self.o_proj(qkv[:, :, :16]) + (qkv[:, :, :16] @ self.o_proj_lora_A.T) @ self.o_proj_lora_B.T
        x = x + attn_out
        # MLP with LoRA
        gu = self.gate_up_proj(x) + (x @ self.gate_up_proj_lora_A.T) @ self.gate_up_proj_lora_B.T
        mlp_out = self.down_proj(gu) + (gu @ self.down_proj_lora_A.T) @ self.down_proj_lora_B.T
        return x + mlp_out

    def forward_base_only(self, x):
        qkv = self.qkv_proj(x)
        attn_out = self.o_proj(qkv[:, :, :16])
        x = x + attn_out
        gu = self.gate_up_proj(x)
        mlp_out = self.down_proj(gu)
        return x + mlp_out


class Mock32LayerPhi(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.model = nn.Module()
        self.model.model.layers = nn.ModuleList([MockPhiBlock() for _ in range(32)])

    def forward(self, x):
        for layer in self.model.model.layers:
            x = layer(x)
        return x

    def forward_base_only(self, x):
        for layer in self.model.model.layers:
            x = layer.forward_base_only(x)
        return x


def test_lora_mask_zeroing_restores_base_identity():
    """Verify that applying L0_ZERO makes forward output 100% bitwise identical to forward_base_only."""
    model = Mock32LayerPhi()
    x = torch.randn(2, 4, 16)

    # Initially, LoRA is active and forward != forward_base_only
    with torch.no_grad():
        vanilla = model.forward_base_only(x)
        adapted = model(x)
        assert (adapted - vanilla).abs().max().item() > 1e-2

    # Save reference state dict
    ref_lora = {n: p.clone() for n, p in model.named_parameters() if "lora" in n}

    # Apply L0_ZERO
    report_l0 = apply_lora_mask(model, LADDER_L0_ZERO, reference_lora_state_dict=ref_lora)
    assert report_l0.active_lora_parameters == 0
    assert report_l0.active_lora_tensors == 0
    assert report_l0.masked_lora_tensors == len(ref_lora)

    with torch.no_grad():
        zeroed_out = model(x)
        delta = (zeroed_out - vanilla).abs().max().item()
        assert delta == 0.0, f"L0_ZERO must produce exact 0.0 delta vs vanilla, got {delta}"

    # Apply LFULL -> should restore exact original adapted output
    report_full = apply_lora_mask(model, LADDER_LFULL, reference_lora_state_dict=ref_lora)
    assert report_full.masked_lora_parameters == 0
    assert report_full.active_lora_tensors == len(ref_lora)

    with torch.no_grad():
        restored_out = model(x)
        restore_delta = (restored_out - adapted).abs().max().item()
        assert restore_delta == 0.0, f"LFULL must restore exact original adapted output, got {restore_delta}"


def test_lora_mask_parameter_accounting_on_step100_checkpoint():
    """Verify exact parameter counts for L0, L1, L2, L3, LFULL against real checkpoint."""
    from pathlib import Path

    ckpt_path = Path("experiments/checkpoints/predictive_joint_pilot/checkpoint_step_100.pt")
    if not ckpt_path.exists():
        pytest.skip("Step-100 checkpoint not present in local workspace")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    lora = ckpt["lora_state_dict"]
    assert len(lora) == 256
    total_params = sum(p.numel() for p in lora.values())
    assert total_params == 50_331_648

    # Create dummy module to test apply_lora_mask on real checkpoint keys
    dummy = nn.ParameterDict({k.replace(".", "_"): nn.Parameter(v.clone()) for k, v in lora.items()})

    # Wrap dummy named_parameters to return original keys
    orig_named_parameters = dummy.named_parameters

    def patched_named_parameters():
        for k, v in orig_named_parameters():
            # restore dots
            for prefix in ("base_model_model_model_layers_", "base_model_base_model_model_model_layers_"):
                pass
            # Find the actual original key
            for orig_k in lora:
                if orig_k.replace(".", "_") == k:
                    yield orig_k, v
                    break

    dummy.named_parameters = patched_named_parameters

    # Test L0
    rep_l0 = apply_lora_mask(dummy, LADDER_L0_ZERO, reference_lora_state_dict=lora)
    assert rep_l0.active_lora_parameters == 0
    assert rep_l0.active_parameter_pct_of_phi == 0.0

    # Test L1 (Last 4 layers, attention only)
    rep_l1 = apply_lora_mask(dummy, LADDER_L1_TINY_ATTN_LAST4, reference_lora_state_dict=lora)
    assert rep_l1.active_lora_parameters == 2_359_296  # 4 * 589,824
    assert rep_l1.active_lora_tensors == 16  # 4 layers * 2 modules * 2 tensors (A, B) = 16
    assert rep_l1.enabled_layers == [28, 29, 30, 31]
    assert rep_l1.enabled_modules == ["o_proj", "qkv_proj"]

    # Test L2 (Last 8 layers, attention only)
    rep_l2 = apply_lora_mask(dummy, LADDER_L2_SMALL_ATTN_LAST8, reference_lora_state_dict=lora)
    assert rep_l2.active_lora_parameters == 4_718_592  # 8 * 589,824

    # Test L3 (Last 8 layers, all modules)
    rep_l3 = apply_lora_mask(dummy, LADDER_L3_MOD_ALL_LAST8, reference_lora_state_dict=lora)
    assert rep_l3.active_lora_parameters == 12_582_912  # 8 * 1,572,864

    # Test LFULL
    rep_full = apply_lora_mask(dummy, LADDER_LFULL, reference_lora_state_dict=lora)
    assert rep_full.active_lora_parameters == 50_331_648
    assert rep_full.active_lora_tensors == 256
