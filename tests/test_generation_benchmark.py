import pytest
import os
import json
from experiments.run_generation_benchmark import run_benchmark


def test_generation_benchmark_execution():
    """Verify that the generation benchmark runs cleanly and records valid metrics."""
    results = run_benchmark(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        budget=8,
        device="cpu",
        max_new_tokens=5,
    )

    # 3 conditions x 3 prompts = 9 results
    assert len(results) == 9

    conditions = {r.condition for r in results}
    assert "Condition A (Base Model)" in conditions
    assert "Condition B (Standard zip2zip LZW)" in conditions
    assert "Condition C (Predictive Seeded)" in conditions

    for r in results:
        assert r.generated_steps > 0
        assert r.total_wall_clock_ms > 0
        if "Condition C" in r.condition:
            assert r.optimizer_overhead_ms >= 0.0
            assert r.encoder_overhead_ms >= 0.0
            assert r.base_equivalent_tokens >= r.generated_steps
