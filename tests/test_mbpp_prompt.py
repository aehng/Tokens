"""Focused tests for MBPP signature disclosure without reference execution."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mbpp_prompt import (  # noqa: E402
    MBPPPromptError,
    build_mbpp_prompt,
    extract_function_signature,
)


def test_signature_keeps_parameters_and_annotations_without_body_or_tests():
    sample = {
        "id": "synthetic_signature",
        "domain": "code",
        "prompt": "Write a function that adds two values.",
        "ground_truth_response": (
            "def add(left: int, right: int = 4) -> int:\n"
            "    return left + right\n\n"
            "# Tests\nassert add(1) == 5"
        ),
    }

    result = build_mbpp_prompt(sample)

    assert result == (
        "Write a function that adds two values.\n\n"
        "Implement this Python function using the required signature:\n"
        "```python\ndef add(left: int, right: int=4) -> int:\n```"
    )
    assert "return left + right" not in result
    assert "# Tests" not in result
    assert "assert add(1) == 5" not in result


def test_signature_parsing_never_executes_reference():
    reference = (
        "raise RuntimeError('reference must not run')\n"
        "def safe(value):\n"
        "    return value\n"
        "assert safe(1) == 1"
    )

    assert extract_function_signature(reference) == "def safe(value):"


def test_async_function_signature_is_supported():
    assert extract_function_signature(
        "async def fetch(url: str):\n    return url"
    ) == "async def fetch(url: str):"


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("# no function here", "found 0"),
        (
            "def first():\n    pass\ndef second():\n    pass",
            "found 2",
        ),
        ("def unfinished(:\n    pass", "not valid Python"),
    ],
)
def test_ambiguous_or_invalid_reference_fails_explicitly(reference, message):
    with pytest.raises(MBPPPromptError, match=message):
        extract_function_signature(reference, sample_id="bad_sample")


def test_non_code_sample_fails_explicitly():
    with pytest.raises(MBPPPromptError, match="domain must be 'code'"):
        build_mbpp_prompt(
            {
                "id": "not_code",
                "domain": "instruction",
                "prompt": "Task",
                "ground_truth_response": "answer",
            }
        )


def test_all_cached_code_samples_have_prompt_safe_signatures():
    data_path = Path(__file__).resolve().parents[1] / "data" / "cached_pure_pred_val_60.json"
    records = json.loads(data_path.read_text(encoding="utf-8"))
    samples = [record for record in records if record.get("domain") == "code"]

    assert len(samples) == 20
    for sample in samples:
        result = build_mbpp_prompt(sample)
        signature = extract_function_signature(
            sample["ground_truth_response"], sample_id=sample["id"]
        )
        assert f"```python\n{signature}\n```" in result, sample["id"]
        assert sample["ground_truth_response"].strip() not in result, sample["id"]
        assert "# Tests" not in result, sample["id"]
        assert "assert " not in result, sample["id"]
