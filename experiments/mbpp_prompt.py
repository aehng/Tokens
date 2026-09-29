"""Build MBPP model prompts that include the required function signature."""

from __future__ import annotations

import ast
import copy
from collections.abc import Mapping
from typing import Any


class MBPPPromptError(ValueError):
    """Raised when a sample does not contain a usable Python function."""


def extract_function_signature(
    reference: str,
    *,
    sample_id: str = "<unknown>",
    test_assert_statements: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Return the signature for the function identified by the MBPP tests.

    Single-function references are unambiguous. If a reference includes helpers,
    top-level assertion calls identify the requested function. Ambiguous and
    zero-function references fail closed; the first function is never assumed
    to be the task target. Reference and test source are parsed but never run.
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
    if not functions:
        raise MBPPPromptError(
            f"Cannot determine Python function signature for {sample_id}: "
            "found 0 top-level functions."
        )

    target = functions[0]
    if len(functions) > 1:
        test_modules = []
        if test_assert_statements:
            for test in test_assert_statements:
                if isinstance(test, str):
                    try:
                        test_modules.append(ast.parse(test))
                    except SyntaxError as exc:
                        raise MBPPPromptError(
                            f"Cannot determine Python function signature for {sample_id}: "
                            f"test metadata is not valid Python ({exc.msg})."
                        ) from exc
        if not test_modules:
            # MBPP references in this frozen pool carry their tests after the
            # reference implementations as module-level assert statements.
            test_modules = [ast.Module(
                body=[node for node in module.body if isinstance(node, ast.Assert)],
                type_ignores=[],
            )]

        tested_names: set[str] = set()
        for test_module in test_modules:
            for statement in test_module.body:
                test_expression = statement.test if isinstance(statement, ast.Assert) else statement
                for node in ast.walk(test_expression):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        tested_names.add(node.func.id)

        selected = [function for function in functions if function.name in tested_names]
        if len(selected) != 1:
            names = ", ".join(function.name for function in functions)
            matched = ", ".join(function.name for function in selected) or "none"
            raise MBPPPromptError(
                f"Cannot determine Python function signature for {sample_id}: "
                f"top-level tests must identify exactly one target among [{names}], "
                f"but identified [{matched}]."
            )
        target = selected[0]

    signature_node = copy.copy(target)
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
    sample_id = str(sample.get("prompt_id", sample.get("id", "<unknown>")))
    if sample.get("domain") != "code":
        raise MBPPPromptError(
            f"Cannot build MBPP prompt for {sample_id}: sample domain must be 'code'."
        )

    prompt = sample.get("prompt_text", sample.get("prompt"))
    reference = sample.get("reference_response", sample.get("ground_truth_response"))
    if not isinstance(prompt, str) or not prompt.strip():
        raise MBPPPromptError(
            f"Cannot build MBPP prompt for {sample_id}: prompt must be a non-empty string."
        )
    if not isinstance(reference, str) or not reference.strip():
        raise MBPPPromptError(
            f"Cannot build MBPP prompt for {sample_id}: "
            "reference_response must be a non-empty string."
        )

    tests = sample.get("test_assert_statements")
    signature = extract_function_signature(
        reference,
        sample_id=sample_id,
        test_assert_statements=tests if isinstance(tests, (list, tuple)) else None,
    )
    return (
        f"{prompt.rstrip()}\n\n"
        "Implement this Python function using the required signature:\n"
        f"```python\n{signature}\n```"
    )


def canonical_task_text(sample: Mapping[str, Any]) -> str:
    """Build the exact user-task text used by the canonical Phi runner."""
    if sample.get("domain") != "code":
        prompt = sample.get("prompt_text", sample.get("prompt"))
        if not isinstance(prompt, str) or not prompt.strip():
            sample_id = str(sample.get("prompt_id", sample.get("id", "<unknown>")))
            raise MBPPPromptError(
                f"Cannot build prompt for {sample_id}: prompt must be a non-empty string."
            )
        return prompt
    return build_mbpp_prompt(sample)
