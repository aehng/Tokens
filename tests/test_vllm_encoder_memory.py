"""Tests for vLLM encoder memory planning, headroom checking, and engine budget separation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tokens_vllm.proof_harness import (
    BASE_GPU_MEMORY_UTILIZATION,
    DEFAULT_HEADROOM_SAFETY_MARGIN_BYTES,
    PREDICTIVE_GPU_MEMORY_UTILIZATION,
    build_encoder_memory_plan,
    check_encoder_headroom,
    inspect_parameter_footprint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "experiments" / "kaggle" / "run_vllm_predictive_proof.py"
RUNNER_TEXT = RUNNER_PATH.read_text(encoding="utf-8")


def test_base_engines_use_point_nine_zero():
    """Base engines (stock and Phase-1 H-disabled) must use 0.90 utilization."""
    assert BASE_GPU_MEMORY_UTILIZATION == 0.90

    # stock engine uses BASE_GPU_MEMORY_UTILIZATION
    stock_block = re.search(r"stock = LLM\((.*?)\)", RUNNER_TEXT, re.DOTALL)
    assert stock_block is not None
    assert "gpu_memory_utilization=BASE_GPU_MEMORY_UTILIZATION" in stock_block.group(1)

    # ours engine in phase 1 (H-disabled) uses BASE_GPU_MEMORY_UTILIZATION
    ours_block = re.search(r"ours = _make_llm\((.*?)\)", RUNNER_TEXT, re.DOTALL)
    assert ours_block is not None
    assert "h_enabled=False" in ours_block.group(1)
    assert "gpu_memory_utilization=BASE_GPU_MEMORY_UTILIZATION" in ours_block.group(1)


def test_all_henabled_engines_use_point_seven_five():
    """All H-enabled/encoder-bearing engines must use 0.75 utilization."""
    assert PREDICTIVE_GPU_MEMORY_UTILIZATION == 0.75

    # Check default in _llm_kwargs
    llm_kwargs_block = re.search(r"def _llm_kwargs\(.*?\n(.*?)\n\n", RUNNER_TEXT, re.DOTALL)
    assert llm_kwargs_block is not None
    assert "gpu_memory_utilization=PREDICTIVE_GPU_MEMORY_UTILIZATION" in llm_kwargs_block.group(1)

    # Check primary llm in Phase 3
    primary_block = re.search(r"llm = _make_llm\(\s*prepared\[\"merged_dir\"\],.*?h_enabled=True,.*?\)", RUNNER_TEXT, re.DOTALL)
    assert primary_block is not None
    assert "gpu_memory_utilization=PREDICTIVE_GPU_MEMORY_UTILIZATION" in primary_block.group(0)

    # Check chunk_llm in Phase 8
    chunk_block = re.search(r"chunk_llm = _make_llm\(\s*prepared\[\"merged_dir\"\],.*?enable_chunked_prefill=True,.*?\)", RUNNER_TEXT, re.DOTALL)
    assert chunk_block is not None
    assert "gpu_memory_utilization=PREDICTIVE_GPU_MEMORY_UTILIZATION" in chunk_block.group(0)

    # Check preempt_llm in Phase 9
    preempt_block = re.search(r"preempt_llm = _make_llm\(\s*prepared\[\"merged_dir\"\],.*?block_size=PROOF_BLOCK_SIZE,.*?\)", RUNNER_TEXT, re.DOTALL)
    assert preempt_block is not None
    assert "gpu_memory_utilization=PREDICTIVE_GPU_MEMORY_UTILIZATION" in preempt_block.group(0)

    # Check rope_llm in Phase 10
    rope_block = re.search(r"rope_llm = _make_llm\(\s*prepared\[\"merged_dir\"\],.*?h_enabled=True,.*?\)", RUNNER_TEXT, re.DOTALL)
    assert rope_block is not None
    assert "gpu_memory_utilization=PREDICTIVE_GPU_MEMORY_UTILIZATION" in rope_block.group(0)


def test_encoder_byte_calculation_is_exact():
    """Footprint must use exact numel * element_size."""
    class ToyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20, bias=True)  # float32: 10*20 + 20 = 220 params = 880 bytes
            self.emb = nn.Embedding(50, 8)  # float32: 50*8 = 400 params = 1600 bytes

    toy = ToyModule()
    fp = inspect_parameter_footprint(toy)
    assert fp["count"] == 220 + 400  # 620
    assert fp["bytes"] == 880 + 1600  # 2480
    assert fp["dtypes"] == ["torch.float32"]

    # State dict test with mixed dtypes
    t_fp32 = torch.zeros(100, dtype=torch.float32)  # 400 bytes
    t_fp16 = torch.zeros(200, dtype=torch.float16)  # 400 bytes
    sd = {"w1": t_fp32, "w2": t_fp16}
    fp_sd = inspect_parameter_footprint(sd)
    assert fp_sd["count"] == 300
    assert fp_sd["bytes"] == 800
    assert sorted(fp_sd["dtypes"]) == ["torch.float16", "torch.float32"]


def test_build_encoder_memory_plan():
    """Verify build_encoder_memory_plan structure and field values."""
    in_state = {
        "weight": torch.zeros(1000, 3072, dtype=torch.float32),  # 3,072,000 params, 12,288,000 bytes
    }
    out_state = {
        "weight": torch.zeros(1000, 3072, dtype=torch.float32),  # 3,072,000 params, 12,288,000 bytes
    }
    blob = {
        "input_state": in_state,
        "output_state": out_state,
    }
    plan = build_encoder_memory_plan(blob)
    assert plan["input_parameter_count"] == 3_072_000
    assert plan["output_parameter_count"] == 3_072_000
    assert plan["input_parameter_bytes"] == 12_288_000
    assert plan["output_parameter_bytes"] == 12_288_000
    assert plan["total_parameter_bytes"] == 24_576_000
    assert plan["dtypes"] == ["torch.float32"]
    assert plan["base_gpu_memory_utilization"] == 0.90
    assert plan["predictive_gpu_memory_utilization"] == 0.75


def test_headroom_check_fails_cleanly_when_insufficient():
    """check_encoder_headroom must raise RuntimeError with clear diagnostic message."""
    total_encoder_bytes = 1_812_234_240  # ~1.73 GiB
    safety_margin = 512 * 1024 * 1024  # 512 MiB
    # Free memory less than encoders + margin
    free_bytes = total_encoder_bytes + safety_margin - 1024

    with pytest.raises(RuntimeError) as exc_info:
        check_encoder_headroom(free_bytes, total_encoder_bytes, safety_margin_bytes=safety_margin)

    msg = str(exc_info.value)
    assert "insufficient reserved GPU headroom for predictive encoders" in msg
    assert str(free_bytes) in msg
    assert str(total_encoder_bytes) in msg


def test_headroom_check_passes_when_sufficient():
    """check_encoder_headroom must pass cleanly when free memory >= encoders + margin."""
    total_encoder_bytes = 1_812_234_240
    safety_margin = 512 * 1024 * 1024
    free_bytes = total_encoder_bytes + safety_margin + 1024 * 1024

    # Should not raise
    check_encoder_headroom(free_bytes, total_encoder_bytes, safety_margin_bytes=safety_margin)


def test_no_encoder_dtype_conversion_in_installer():
    """Verify that _install_encoders preserves checkpoint dtypes without casting."""
    # Ensure no .half(), .bfloat16(), .float(), or dtype casting in _install_encoders
    install_func = RUNNER_TEXT.split("def _install_encoders", 1)[1].split("def _compare_vectors", 1)[0]
    assert ".half()" not in install_func
    assert ".bfloat16()" not in install_func
    assert ".float()" not in install_func
    assert "torch.float16" not in install_func
    assert "torch.bfloat16" not in install_func
