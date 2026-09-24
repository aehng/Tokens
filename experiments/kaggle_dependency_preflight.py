"""Kaggle-only dependency normalization before importing the benchmark stack.

This module deliberately uses only the Python standard library in the launcher
process. PEFT, Transformers, and the benchmark/model modules are imported in a
fresh child process only after the unused optional TorchAO distribution has
been removed.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Optional, TypeVar


EXPECTED_TRANSFORMERS_VERSION = "5.17.0"
EXPECTED_PEFT_VERSION = "0.19.1"
EXPECTED_ACCELERATE_VERSION = "1.13.0"
_PREFLIGHT_MARKER = "TOKENS_DEPENDENCY_PREFLIGHT="
_VERSION_UNSET = object()
_PREFLIGHT_SCRIPT = r'''
import importlib.metadata
import importlib.util
import json

if importlib.util.find_spec("torchao") is not None:
    raise RuntimeError("torchao is still importable after optional-package normalization")
print("TOKENS_TORCHAO_ABSENT=true", flush=True)

import torch
import transformers
import peft
import accelerate

cuda_available = bool(torch.cuda.is_available())
device_count = int(torch.cuda.device_count()) if cuda_available else 0
device_name = torch.cuda.get_device_name(0) if device_count else None
print("TOKENS_DEPENDENCY_PREFLIGHT=" + json.dumps({
    "torchao_absent": True,
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "peft_version": importlib.metadata.version("peft"),
    "accelerate_version": accelerate.__version__,
    "cuda_available": cuda_available,
    "cuda_device_count": device_count,
    "cuda_device_name": device_name,
}, sort_keys=True), flush=True)
'''


class DependencyPreflightError(RuntimeError):
    """Raised before model imports when Kaggle runtime normalization fails."""


def _distribution_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def installed_torchao_version() -> Optional[str]:
    """Inspect TorchAO package metadata without importing TorchAO or model code."""

    return _distribution_version("torchao")


def _write_report(path: Optional[str | os.PathLike[str]], report: dict[str, Any]) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _preflight_record(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_PREFLIGHT_MARKER):
            return json.loads(line[len(_PREFLIGHT_MARKER) :])
    raise DependencyPreflightError(
        "fresh-process dependency preflight did not emit its result record"
    )


def normalize_unused_torchao(
    *,
    expected_torch_version: str,
    expected_cuda_available: bool,
    expected_cuda_device_count: int,
    expected_cuda_device_name: str,
    python_executable: str = sys.executable,
    runner: Optional[Callable[..., Any]] = None,
    version_lookup: Optional[Callable[[str], Optional[str]]] = None,
    initial_torchao_version: Any = _VERSION_UNSET,
    report_path: Optional[str | os.PathLike[str]] = None,
) -> dict[str, Any]:
    """Remove unused TorchAO if installed, then verify imports in a new process."""

    run = runner or subprocess.run
    lookup = version_lookup or _distribution_version
    initial_version = (
        lookup("torchao")
        if initial_torchao_version is _VERSION_UNSET
        else initial_torchao_version
    )
    report: dict[str, Any] = {
        "schema": "tokens_kaggle_dependency_preflight_v1",
        "status": "RUNNING",
        "torchao_initial_version": initial_version,
        "torchao_action": (
            "uninstalling because unused optional dependency"
            if initial_version is not None
            else "no action; unused optional dependency not installed"
        ),
        "torchao_final_state": "unknown",
        "previous_failure_context": {
            "python_version": "3.12.13",
            "torch_version": "2.10.0+cu128",
            "transformers_version": EXPECTED_TRANSFORMERS_VERSION,
            "peft_version": EXPECTED_PEFT_VERSION,
            "accelerate_version": EXPECTED_ACCELERATE_VERSION,
            "cuda_runtime": "12.8",
            "gpu": "Tesla T4",
            "installed_torchao": "0.10.0",
            "failure": (
                "Found an incompatible version of torchao. Found version 0.10.0, "
                "but only versions above 0.16.0 are supported"
            ),
            "peft_comparison_semantics": (
                "PEFT rejects torchao_version < 0.16.0; exactly 0.16.0 satisfies "
                "the comparison"
            ),
            "compatible_release_context": (
                "upstream pairs torchao 0.16.0 with PyTorch 2.10.0; it is not "
                "installed because this benchmark does not use TorchAO"
            ),
            "resolution": "remove TorchAO because this benchmark does not use it",
        },
    }
    _write_report(report_path, report)

    try:
        if initial_version is not None:
            uninstall = run(
                [python_executable, "-m", "pip", "uninstall", "-y", "torchao"],
                capture_output=True,
                text=True,
            )
            report["torchao_uninstall_returncode"] = int(uninstall.returncode)
            report["torchao_uninstall_output"] = (uninstall.stdout or "") + (
                uninstall.stderr or ""
            )
            if uninstall.returncode != 0:
                raise DependencyPreflightError(
                    "uninstalling unused TorchAO failed; stopping before benchmark imports"
                )
        report["torchao_action"] = (
            "uninstalled because unused optional dependency"
            if initial_version is not None
            else "no action; unused optional dependency not installed"
        )
        _write_report(report_path, report)

        verification = run(
            [python_executable, "-c", _PREFLIGHT_SCRIPT],
            capture_output=True,
            text=True,
        )
        report["fresh_process_returncode"] = int(verification.returncode)
        report["fresh_process_output"] = (verification.stdout or "") + (
            verification.stderr or ""
        )
        if "TOKENS_TORCHAO_ABSENT=true" in (verification.stdout or ""):
            report["torchao_final_state"] = "absent"
        if verification.returncode != 0:
            raise DependencyPreflightError(
                "fresh-process dependency preflight failed after TorchAO normalization; "
                "stopping before benchmark imports"
            )

        environment = _preflight_record(verification.stdout or "")
        report["verified_environment"] = environment
        required = {
            "torchao_absent": True,
            "torch_version": expected_torch_version,
            "transformers_version": EXPECTED_TRANSFORMERS_VERSION,
            "peft_version": EXPECTED_PEFT_VERSION,
            "accelerate_version": EXPECTED_ACCELERATE_VERSION,
            "cuda_available": expected_cuda_available,
            "cuda_device_count": expected_cuda_device_count,
            "cuda_device_name": expected_cuda_device_name,
        }
        mismatches = {
            key: {"expected": value, "actual": environment.get(key)}
            for key, value in required.items()
            if environment.get(key) != value
        }
        if mismatches:
            report["mismatches"] = mismatches
            raise DependencyPreflightError(
                "fresh-process dependency/GPU preflight did not preserve the reviewed "
                f"runtime: {json.dumps(mismatches, sort_keys=True)}"
            )
        if not environment["cuda_available"] or environment["cuda_device_count"] != 1:
            raise DependencyPreflightError(
                "fresh-process preflight requires exactly one usable CUDA device"
            )
        if "T4" not in str(environment["cuda_device_name"]).upper():
            raise DependencyPreflightError(
                "fresh-process preflight requires the reviewed Tesla T4 device"
            )
        report.update(
            {
                "status": "PASSED",
                "torchao_final_state": "absent",
            }
        )
        _write_report(report_path, report)
        return report
    except BaseException as error:
        report["status"] = "FAILED"
        if report.get("torchao_final_state") != "absent":
            report["torchao_final_state"] = "unknown"
        report["failure"] = f"{type(error).__name__}: {error}"
        _write_report(report_path, report)
        raise


_T = TypeVar("_T")


def run_after_dependency_preflight(
    preflight: Callable[[], dict[str, Any]],
    import_reviewed_modules: Callable[[], _T],
) -> tuple[dict[str, Any], _T]:
    """Run model-stack imports only after the dependency preflight succeeds."""

    report = preflight()
    modules = import_reviewed_modules()
    return report, modules
