"""CPU checks for the Kaggle bootstrap process and engine teardown."""

import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOOTSTRAP = (REPO / "experiments" / "kaggle" / "bootstrap_kaggle.py").read_text(encoding="utf-8")
RUNNER = (REPO / "experiments" / "kaggle" / "run_vllm_predictive_proof.py").read_text(encoding="utf-8")


def test_child_process_imports_tokens_vllm_and_loads_the_plugin():
    script = textwrap.dedent(
        """
        import importlib.metadata
        import tomllib
        from pathlib import Path

        root = Path(%r)
        declared = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        value = declared["project"]["entry-points"]["vllm.general_plugins"]["tokens_predictive"]
        installed = [
            item
            for item in importlib.metadata.entry_points(group="vllm.general_plugins")
            if item.name == "tokens_predictive"
        ]
        entry = installed[0] if installed else importlib.metadata.EntryPoint(
            name="tokens_predictive",
            value=value,
            group="vllm.general_plugins",
        )
        loaded = entry.load()
        import tokens_vllm

        if loaded.__module__ != "tokens_vllm.plugin" or loaded.__name__ != "register":
            raise SystemExit(f"unexpected plugin target {loaded}")
        print("import-ok", tokens_vllm.__file__)
        print("plugin-ok", entry.value)
        """
        % str(REPO)
    )
    env = dict(**{key: value for key, value in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(REPO / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "import-ok" in completed.stdout
    assert "plugin-ok tokens_vllm.plugin:register" in completed.stdout


def test_bootstrap_runs_the_proof_in_a_fresh_interpreter():
    assert "from experiments.kaggle.run_vllm_predictive_proof import" not in BOOTSTRAP
    assert '[sys.executable, "-m", "experiments.kaggle.run_vllm_predictive_proof"]' in BOOTSTRAP
    assert 'env["PYTHONPATH"]' in BOOTSTRAP


def test_engine_transitions_delete_the_owner_name():
    assert "_free(" not in RUNNER
    assert "BASE_GPU_MEMORY_UTILIZATION = 0.90" in RUNNER
    assert "PREDICTIVE_GPU_MEMORY_UTILIZATION = 0.75" in RUNNER
    assert "gpu_memory_utilization=0.42" not in RUNNER
    assert "gpu_memory_utilization=0.50" not in RUNNER
    for name in ("stock", "ours", "llm", "chunk_llm", "preempt_llm", "rope_llm"):
        assert f"shutdown_vllm_engine({name})" in RUNNER
        assert f"del {name}" in RUNNER
    shutdown = RUNNER.split("def shutdown_vllm_engine", 1)[1].split("def cuda_after_collect", 1)[0]
    assert "del llm" not in shutdown
