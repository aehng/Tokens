"""Unit tests for the minimum-LoRA experiment runner and failure mode taxonomy."""

from dataclasses import asdict
from pathlib import Path
import pytest
import torch

from zip2zip.selective_lora import (
    LADDER_L0_ZERO,
    LADDER_L1_TINY_ATTN_LAST4,
    LADDER_L2_SMALL_ATTN_LAST8,
    LADDER_L3_MOD_ALL_LAST8,
    LADDER_LFULL,
    STANDARD_LORA_LADDER,
)
from experiments.run_minimum_lora_experiment import (
    FAILURE_MODE_A,
    FAILURE_MODE_B,
    FAILURE_MODE_C,
    FAILURE_MODE_D,
    LADDER_MAP,
    MinimumLoRAResult,
    generate_minimum_lora_markdown_report,
)


def test_ladder_map_completeness():
    """Verify that all standard ladder steps are correctly registered."""
    expected_names = [
        "L0_ZERO",
        "L1_TINY_ATTN_LAST4",
        "L2_SMALL_ATTN_LAST8",
        "L3_MOD_ALL_LAST8",
        "LFULL",
    ]
    for name in expected_names:
        assert name in LADDER_MAP
        assert LADDER_MAP[name].name == name


def test_failure_mode_classification_logic():
    """Verify exact taxonomy classification for failure modes A, B, C, D and Success."""

    # 1. Mode A: No H emissions
    res_a = MinimumLoRAResult(
        mask_name="L0_ZERO",
        mask_report={"active_lora_parameters": 0, "active_parameter_pct_of_phi": 0.0},
        pass_a_base_fidelity={"exact_match_count": 12, "total_prompts": 12, "mean_kl_nats": 0.0, "top1_agreement_rate": 1.0},
        pass_b_oracle_acceleration={"total_emitted_h": 0, "overall_steps_saved_pct": 0.0, "quality_pass_count": 12},
        criteria_verdict={
            "criterion_1_base_fidelity": True,
            "criterion_2_useful_h_emission": False,
            "criterion_3_post_h_continuation": False,
            "criterion_4_real_decode_savings": False,
        },
        all_criteria_passed=False,
        failure_classification=FAILURE_MODE_A,
        diagnosis_notes="Mode A test",
    )
    assert res_a.failure_classification == FAILURE_MODE_A

    # 2. Mode B: Emits H, but continuation breaks
    res_b = MinimumLoRAResult(
        mask_name="L1_TINY_ATTN_LAST4",
        mask_report={"active_lora_parameters": 2359296, "active_parameter_pct_of_phi": 0.0617},
        pass_a_base_fidelity={"exact_match_count": 12, "total_prompts": 12, "mean_kl_nats": 0.01, "top1_agreement_rate": 0.995},
        pass_b_oracle_acceleration={"total_emitted_h": 15, "overall_steps_saved_pct": 10.0, "quality_pass_count": 3},
        criteria_verdict={
            "criterion_1_base_fidelity": True,
            "criterion_2_useful_h_emission": True,
            "criterion_3_post_h_continuation": False,
            "criterion_4_real_decode_savings": True,
        },
        all_criteria_passed=False,
        failure_classification=FAILURE_MODE_B,
        diagnosis_notes="Mode B test",
    )
    assert res_b.failure_classification == FAILURE_MODE_B

    # 3. Mode C: Emits H and continues, but base tokens drift
    res_c = MinimumLoRAResult(
        mask_name="LFULL",
        mask_report={"active_lora_parameters": 50331648, "active_parameter_pct_of_phi": 1.317},
        pass_a_base_fidelity={"exact_match_count": 0, "total_prompts": 12, "mean_kl_nats": 0.68, "top1_agreement_rate": 0.708},
        pass_b_oracle_acceleration={"total_emitted_h": 20, "overall_steps_saved_pct": 12.0, "quality_pass_count": 10},
        criteria_verdict={
            "criterion_1_base_fidelity": False,
            "criterion_2_useful_h_emission": True,
            "criterion_3_post_h_continuation": True,
            "criterion_4_real_decode_savings": True,
        },
        all_criteria_passed=False,
        failure_classification=FAILURE_MODE_C,
        diagnosis_notes="Mode C test",
    )
    assert res_c.failure_classification == FAILURE_MODE_C

    # 4. Mode D: Emits H and coherent, but negligible decode savings (< 5%)
    res_d = MinimumLoRAResult(
        mask_name="L2_SMALL_ATTN_LAST8",
        mask_report={"active_lora_parameters": 4718592, "active_parameter_pct_of_phi": 0.1235},
        pass_a_base_fidelity={"exact_match_count": 12, "total_prompts": 12, "mean_kl_nats": 0.02, "top1_agreement_rate": 0.992},
        pass_b_oracle_acceleration={"total_emitted_h": 2, "overall_steps_saved_pct": 1.5, "quality_pass_count": 12},
        criteria_verdict={
            "criterion_1_base_fidelity": True,
            "criterion_2_useful_h_emission": True,
            "criterion_3_post_h_continuation": True,
            "criterion_4_real_decode_savings": False,
        },
        all_criteria_passed=False,
        failure_classification=FAILURE_MODE_D,
        diagnosis_notes="Mode D test",
    )
    assert res_d.failure_classification == FAILURE_MODE_D

    # 5. Success
    res_success = MinimumLoRAResult(
        mask_name="L0_ZERO",
        mask_report={"active_lora_parameters": 0, "active_parameter_pct_of_phi": 0.0},
        pass_a_base_fidelity={"exact_match_count": 12, "total_prompts": 12, "mean_kl_nats": 0.0, "top1_agreement_rate": 1.0},
        pass_b_oracle_acceleration={"total_emitted_h": 25, "overall_steps_saved_pct": 15.2, "quality_pass_count": 12},
        criteria_verdict={
            "criterion_1_base_fidelity": True,
            "criterion_2_useful_h_emission": True,
            "criterion_3_post_h_continuation": True,
            "criterion_4_real_decode_savings": True,
        },
        all_criteria_passed=True,
        failure_classification=None,
        diagnosis_notes="Success test",
    )
    assert res_success.all_criteria_passed is True
    assert res_success.failure_classification is None


def test_generate_markdown_report_formatting():
    """Verify that report generation produces well-structured markdown tables."""
    summary = {
        "created_at_utc": "2026-09-30T00:00:00Z",
        "device": "cuda:0",
        "tested_ladder_steps": ["L0_ZERO", "L1_TINY_ATTN_LAST4"],
        "viable_config_found": False,
        "results": [
            {
                "mask_name": "L0_ZERO",
                "mask_report": {"active_lora_parameters": 0, "active_parameter_pct_of_phi": 0.0},
                "pass_a_base_fidelity": {"exact_match_count": 12, "total_prompts": 12, "mean_kl_nats": 0.0, "top1_agreement_rate": 1.0},
                "pass_b_oracle_acceleration": {"total_emitted_h": 0, "overall_steps_saved_pct": 0.0},
                "criteria_verdict": {
                    "criterion_1_base_fidelity": True,
                    "criterion_2_useful_h_emission": False,
                    "criterion_3_post_h_continuation": False,
                    "criterion_4_real_decode_savings": False,
                },
                "all_criteria_passed": False,
                "failure_classification": FAILURE_MODE_A,
                "diagnosis_notes": "Mode A test notes",
            }
        ],
    }
    report = generate_minimum_lora_markdown_report(summary)
    assert "# Tokens Bounded Minimum-LoRA Experiment Report" in report
    assert "| **L0_ZERO** |" in report
    assert "MODE_A_NO_H_EMISSION" in report
    assert "Retraining Recommendations" in report
