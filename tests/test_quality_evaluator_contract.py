"""Evaluator-contract tests; these do not invoke a model or read the test split."""

from experiments.run_quality_benchmark import (
    EVALUATOR_VERSION,
    GENERATION_RECORD_SCHEMA,
    PROMPT_FORMATTER_VERSION,
    _write_generation_record,
    evaluate_alpaca_instruction,
    evaluate_mbpp_code,
    generate_full_benchmark_analytics,
    generation_health_fields,
)


def test_evaluator_contract_versions_are_explicit():
    assert EVALUATOR_VERSION == "phi_quality_evaluator_v2"
    assert PROMPT_FORMATTER_VERSION == "mbpp_task_signature_v2"
    assert GENERATION_RECORD_SCHEMA == "phi_generation_record_v2"


def test_mbpp_restricted_runner_executes_safe_reference_and_assertions():
    result = evaluate_mbpp_code(
        "def add(left, right):\n    return left + right\n",
        ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"],
    )

    assert result["syntax_valid"] is True
    assert result["problem_pass"] is True
    assert result["tests_passed"] == 2
    assert result["safety_rejected"] is False


def test_mbpp_extraction_retains_allowed_imports_before_function():
    result = evaluate_mbpp_code(
        "import math\n\ndef hypotenuse(a, b):\n    return math.sqrt(a*a + b*b)\n",
        ["assert hypotenuse(3, 4) == 5"],
    )

    assert result["problem_pass"] is True


def test_mbpp_restricted_runner_rejects_unsafe_code_and_tests():
    unsafe_code = evaluate_mbpp_code(
        "import os\ndef answer():\n    return 1\n",
        ["assert answer() == 1"],
    )
    unsafe_test = evaluate_mbpp_code(
        "def answer():\n    return 1\n",
        ["assert __import__('os').name"],
    )

    assert unsafe_code["safety_rejected"] is True
    assert unsafe_code["problem_pass"] is False
    assert unsafe_test["safety_rejected"] is True
    assert unsafe_test["problem_pass"] is False


def test_mbpp_restricted_runner_enforces_timeout_and_fails_closed_without_assertions():
    timeout = evaluate_mbpp_code(
        "def answer():\n    while True:\n        pass\n",
        ["assert answer() == 1"],
        timeout_s=0.25,
    )
    missing_tests = evaluate_mbpp_code("def answer():\n    return 1\n", [])

    assert timeout["timeout"] is True
    assert timeout["problem_pass"] is False
    assert missing_tests["problem_pass"] is False
    assert missing_tests["tests_available"] == 0


def test_instruction_checks_are_explicitly_mechanical_not_semantic():
    result = evaluate_alpaca_instruction("A short but relevant answer.")

    assert result["mechanical_instruction_pass"] is True
    assert result["mechanical_instruction_failure"] is False
    assert "instruction_failure" not in result


def test_generation_health_separates_eos_cap_truncation_and_length():
    result = generation_health_fields(
        "one two three",
        generated_ids=[10, 11, 99],
        eos_token_id=99,
        max_new_tokens=3,
        response_length_base_tokens=5,
    )

    assert result["eos_reached"] is True
    assert result["hit_max_length"] is True
    assert result["truncated"] is False
    assert result["response_length_base_tokens"] == 5
    assert result["response_length_chars"] == len("one two three")


def test_new_generation_records_are_stamped_and_legacy_records_are_rejected(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    record = {"prompt_id": "p1", "condition": "original_phi"}
    _write_generation_record(record, str(raw_path), None, None, None)
    written = raw_path.read_text(encoding="utf-8")

    assert f'"record_schema": "{GENERATION_RECORD_SCHEMA}"' in written
    assert f'"evaluator_version": "{EVALUATOR_VERSION}"' in written
    assert f'"prompt_formatter_version": "{PROMPT_FORMATTER_VERSION}"' in written

    raw_path.write_text(
        '{"prompt_id":"old","record_schema":"phi_generation_record_v1"}\n',
        encoding="utf-8",
    )
    try:
        generate_full_benchmark_analytics(str(tmp_path), str(raw_path))
    except ValueError as exc:
        assert "refusing to mix" in str(exc)
    else:
        raise AssertionError("legacy records must not enter v2 analytics")
