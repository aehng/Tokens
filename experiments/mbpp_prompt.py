"""Build MBPP model prompts that include the required function signature."""

from __future__ import annotations

import ast
import copy
from collections.abc import Mapping
from typing import Any


class MBPPPromptError(ValueError):
    """Raised when a sample does not contain a usable Python function."""


def extract_function_signature(reference: str, *, sample_id: str = "<unknown>") -> str:
    """Return the sole top-level Python function signature in ``reference``.

    The reference is parsed as syntax only. It is never imported or executed, and
    the returned string contains no function body or trailing test code.
    """
    try:
        module = ast.parse(reference)
    except SyntaxError as exc:
        raise MBPPPromptError(
            f"Cannot determine Python function signature for {sample_id}: "
            f"reference is not valid Python ({exc.msg})."
        ) from exc

    functions = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(functions) != 1:
        raise MBPPPromptError(
            f"Cannot determine Python function signature for {sample_id}: "
            f"expected one top-level function, found {len(functions)}."
        )

    signature_node = copy.copy(functions[0])
    signature_node.decorator_list = []
    signature_node.body = [ast.Pass()]
    signature_node.type_comment = None
    ast.fix_missing_locations(signature_node)
    stub = ast.unparse(signature_node)
    signature = stub.removesuffix("\n    pass").rstrip()
    if not signature or not signature.endswith(":"):
        raise MBPPPromptError(
            f"Cannot determine Python function signature for {sample_id}: "
            "AST did not produce a function header."
        )
    return signature


def build_mbpp_prompt(sample: Mapping[str, Any]) -> str:
    """Add the required signature to a code-domain sample's task prompt."""
    sample_id = str(sample.get("id", "<unknown>"))
    if sample.get("domain") != "code":
        raise MBPPPromptError(
            f"Cannot build MBPP prompt for {sample_id}: sample domain must be 'code'."
        )

    prompt = sample.get("prompt")
    reference = sample.get("ground_truth_response")
    if not isinstance(prompt, str) or not prompt.strip():
        raise MBPPPromptError(
            f"Cannot build MBPP prompt for {sample_id}: prompt must be a non-empty string."
        )
    if not isinstance(reference, str) or not reference.strip():
        raise MBPPPromptError(
            f"Cannot build MBPP prompt for {sample_id}: "
            "ground_truth_response must be a non-empty string."
        )

    signature = extract_function_signature(reference, sample_id=sample_id)
    return (
        f"{prompt.rstrip()}\n\n"
        "Implement this Python function using the required signature:\n"
        f"```python\n{signature}\n```"
    )
