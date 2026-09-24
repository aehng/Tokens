import json
from types import SimpleNamespace

import pytest

from experiments import kaggle_dependency_preflight as preflight


def _child_stdout(**overrides):
    record = {
        "torchao_absent": True,
        "torch_version": "2.10.0+cu128",
        "transformers_version": "5.17.0",
        "peft_version": "0.19.1",
        "accelerate_version": "1.13.0",
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_device_name": "Tesla T4",
    }
    record.update(overrides)
    return "TOKENS_TORCHAO_ABSENT=true\n" + preflight._PREFLIGHT_MARKER + json.dumps(record)


def _run_normalization(monkeypatch, *, initial_version, runner, report_path=None):
    monkeypatch.setattr(
        preflight,
        "_distribution_version",
        lambda name: initial_version if name == "torchao" else None,
    )
    return preflight.normalize_unused_torchao(
        expected_torch_version="2.10.0+cu128",
        expected_cuda_available=True,
        expected_cuda_device_count=1,
        expected_cuda_device_name="Tesla T4",
        runner=runner,
        report_path=report_path,
    )


def test_absent_torchao_skips_uninstall_but_runs_fresh_dependency_preflight(
    monkeypatch,
):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=_child_stdout(), stderr="")

    report = _run_normalization(
        monkeypatch, initial_version=None, runner=runner
    )

    assert len(calls) == 1
    assert calls[0][1:3] == ["-c", preflight._PREFLIGHT_SCRIPT]
    assert report["torchao_action"].startswith("no action")
    assert report["torchao_final_state"] == "absent"
    assert report["status"] == "PASSED"
    prior = report["previous_failure_context"]
    assert prior["peft_comparison_semantics"].find("exactly 0.16.0 satisfies") >= 0
    assert "does not use TorchAO" in prior["compatible_release_context"]


@pytest.mark.parametrize("installed_version", ["0.10.0", "0.16.0", "0.99.0"])
def test_installed_torchao_is_removed_even_if_new_enough_for_peft(
    monkeypatch, installed_version
):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if "uninstall" in command:
            return SimpleNamespace(returncode=0, stdout="Successfully uninstalled", stderr="")
        return SimpleNamespace(returncode=0, stdout=_child_stdout(), stderr="")

    report = _run_normalization(
        monkeypatch, initial_version=installed_version, runner=runner
    )

    assert len(calls) == 2
    assert calls[0][1:] == ["-m", "pip", "uninstall", "-y", "torchao"]
    assert calls[1][1] == "-c"
    assert report["torchao_initial_version"] == installed_version
    assert report["torchao_action"] == "uninstalled because unused optional dependency"
    assert report["torchao_final_state"] == "absent"


def test_torchao_version_captured_before_later_environment_setup_is_preserved(
    monkeypatch,
):
    monkeypatch.setattr(preflight, "_distribution_version", lambda _name: None)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if "uninstall" in command:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout=_child_stdout(), stderr="")

    report = preflight.normalize_unused_torchao(
        expected_torch_version="2.10.0+cu128",
        expected_cuda_available=True,
        expected_cuda_device_count=1,
        expected_cuda_device_name="Tesla T4",
        initial_torchao_version="0.10.0",
        runner=runner,
    )
    assert report["torchao_initial_version"] == "0.10.0"
    assert "uninstall" in calls[0]


def test_failed_uninstall_stops_before_import_preflight_and_records_failure(
    monkeypatch, tmp_path
):
    calls = []
    report_path = tmp_path / "torchao-preflight.json"

    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout="permission denied", stderr="")

    with pytest.raises(preflight.DependencyPreflightError, match="uninstalling.*failed"):
        _run_normalization(
            monkeypatch,
            initial_version="0.10.0",
            runner=runner,
            report_path=report_path,
        )

    assert len(calls) == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "FAILED"
    assert report["torchao_initial_version"] == "0.10.0"
    assert "permission denied" in report["torchao_uninstall_output"]


def test_peft_import_failure_after_uninstall_stops_before_model_import(
    monkeypatch, tmp_path
):
    calls = []
    report_path = tmp_path / "torchao-preflight.json"

    def runner(command, **kwargs):
        calls.append(command)
        if "uninstall" in command:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(
            returncode=1,
            stdout="TOKENS_TORCHAO_ABSENT=true\n",
            stderr="ImportError: PEFT unavailable",
        )

    with pytest.raises(preflight.DependencyPreflightError, match="fresh-process.*failed"):
        _run_normalization(
            monkeypatch,
            initial_version="0.10.0",
            runner=runner,
            report_path=report_path,
        )

    assert len(calls) == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "FAILED"
    assert report["torchao_final_state"] == "absent"
    assert "PEFT unavailable" in report["fresh_process_output"]


def test_runtime_cuda_must_remain_usable_after_dependency_normalization(monkeypatch):
    def runner(command, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=_child_stdout(cuda_available=False, cuda_device_count=0, cuda_device_name=None),
            stderr="",
        )

    with pytest.raises(preflight.DependencyPreflightError, match="did not preserve"):
        _run_normalization(monkeypatch, initial_version=None, runner=runner)


def test_reviewed_module_import_waits_until_dependency_preflight_succeeds():
    events = []

    def preflight_ok():
        events.append("dependency-preflight")
        return {"status": "PASSED"}

    def import_model_stack():
        events.append("model-stack-import")
        return "loaded"

    report, modules = preflight.run_after_dependency_preflight(
        preflight_ok, import_model_stack
    )
    assert report["status"] == "PASSED"
    assert modules == "loaded"
    assert events == ["dependency-preflight", "model-stack-import"]

    events.clear()

    def preflight_fails():
        events.append("dependency-preflight")
        raise preflight.DependencyPreflightError("not safe to import")

    with pytest.raises(preflight.DependencyPreflightError):
        preflight.run_after_dependency_preflight(preflight_fails, import_model_stack)
    assert events == ["dependency-preflight"]
