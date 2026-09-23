"""Kaggle entrypoint for controlled GPU runtime decomposition and matched 12-prompt benchmark."""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure both GPUs are visible to PyTorch
if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]

os.environ.setdefault("HF_HOME", "/kaggle/temp/tokens-hf-cache")
os.environ.setdefault("PIP_CACHE_DIR", "/kaggle/temp/tokens-pip-cache")

DATASET_ROOT = Path("/kaggle/input/tokens-step100-gpu-smoke")
DATASET_ID = "elikearl/tokens-step100-gpu-smoke"
OUTPUT_ROOT = Path("/kaggle/working/tokens-kaggle-output")
REPO_ROOT = Path("/kaggle/working/tokens-source")
PACKAGE_REPO = "https://github.com/aehng/Tokens.git"
LAUNCHER_VERSION = "v14-runtime-decomposition-fixed"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
                log_path: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "COMMAND: " + " ".join(command) + "\n\nSTDOUT\n" + result.stdout
            + "\nSTDERR\n" + result.stderr,
            encoding="utf-8",
        )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout tail:\n{result.stdout[-5000:]}\nstderr tail:\n{result.stderr[-5000:]}"
        )
    return result


class NvidiaSampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,driver_version,utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip() or f"nvidia-smi exit {result.returncode}")
                for line in result.stdout.strip().splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) >= 6:
                        self.samples.append(
                            {
                                "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
                                "gpu_index": int(parts[0]),
                                "gpu_name": parts[1],
                                "driver_version": parts[2],
                                "utilization_pct": int(parts[3].replace("%", "")),
                                "memory_used_mib": int(parts[4].replace("MiB", "")),
                                "memory_total_mib": int(parts[5].replace("MiB", "")),
                            }
                        )
            except Exception as error:
                self.errors.append(f"{type(error).__name__}: {error}")
            self.stop_event.wait(0.5)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=15)


def python_environment(repo_root: Path) -> dict[str, Any]:
    code = r'''
import importlib.metadata as m, json, platform, torch, transformers
from pathlib import Path
packages = {}
for name in ("accelerate", "peft", "omegaconf", "datasets", "zip2zip-compression", "sentencepiece"):
    try: packages[name] = m.version(name)
    except m.PackageNotFoundError: packages[name] = None
device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
gpus = []
for index in range(device_count):
    props = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    gpus.append({"index": index, "name": torch.cuda.get_device_name(index), "memory_bytes": int(props.total_memory),
                 "idle_allocated_bytes": int(torch.cuda.memory_allocated(index)),
                 "idle_reserved_bytes": int(torch.cuda.memory_reserved(index)),
                 "idle_free_bytes": int(free), "idle_total_bytes": int(total),
                 "capability": list(torch.cuda.get_device_capability(index))})
print(json.dumps({"python": platform.python_version(), "torch": str(torch.__version__),
                  "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda,
                  "cuda_available": torch.cuda.is_available(), "visible_gpu_count": device_count,
                  "gpus": gpus, "packages": packages}))
'''
    output = run_command([sys.executable, "-c", code], cwd=repo_root).stdout.strip()
    return json.loads(output)


EXPECTED_SOURCE_ARCHIVE_SHA256 = "8ca99e67e579736d3af775680d07a4ac6159feedffbd0e7970176ec602c69d4e"
EXPECTED_SOURCE_COMMIT = "d7cc4bb5f9750c70412154f29ddde93f09cc1ec9"


def unpack_source(repo_commit: str) -> None:
    if repo_commit != EXPECTED_SOURCE_COMMIT:
        raise ValueError(
            f"Provenance mismatch: requested unpack of {repo_commit}, "
            f"but EXPECTED_SOURCE_COMMIT is {EXPECTED_SOURCE_COMMIT}"
        )

    if REPO_ROOT.exists():
        shutil.rmtree(REPO_ROOT)
    REPO_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Check exact dataset source path
    primary_archive = Path("/kaggle/input/tokens-source-ae78085/source.tar.gz")
    if not primary_archive.is_file():
        candidates = list(Path("/kaggle/input").glob("*/source.tar.gz"))
        archive_path = candidates[0] if candidates else None
    else:
        archive_path = primary_archive

    if archive_path is not None and archive_path.is_file():
        actual_hash = sha256(archive_path)
        if actual_hash != EXPECTED_SOURCE_ARCHIVE_SHA256:
            raise RuntimeError(
                f"Source archive SHA256 mismatch at {archive_path}: "
                f"expected {EXPECTED_SOURCE_ARCHIVE_SHA256}, got {actual_hash}"
            )
        print(f"[Provenance] Verified source archive SHA256 ({actual_hash}) at {archive_path}")
        import tarfile
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=REPO_ROOT)
    else:
        # 2. Check if Kaggle automatically expanded the source archive in dataset mount
        extracted_source_dir = Path("/kaggle/input/tokens-source-ae78085")
        if not (extracted_source_dir / "pyproject.toml").is_file():
            candidates = [p for p in Path("/kaggle/input").glob("*tokens-source*") if (p / "pyproject.toml").is_file()]
            if candidates:
                extracted_source_dir = candidates[0]

        if (extracted_source_dir / "pyproject.toml").is_file():
            print(f"[Provenance] Kaggle mounted pre-extracted source tree at {extracted_source_dir}. Staging to {REPO_ROOT}...")
            shutil.copytree(extracted_source_dir, REPO_ROOT, dirs_exist_ok=True)
        else:
            raise FileNotFoundError(
                f"Neither source archive nor extracted source tree found. Checked {primary_archive} and {extracted_source_dir}"
            )

    # Ensure source tree is directly importable
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))


def resolve_dataset_root() -> Path:
    input_root = Path("/kaggle/input")
    manifest_paths = sorted(input_root.rglob("artifact_manifest.json")) if input_root.is_dir() else []
    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("private_dataset_id") == DATASET_ID:
            return manifest_path.parent
    if (DATASET_ROOT / "artifact_manifest.json").is_file():
        return DATASET_ROOT
    raise FileNotFoundError("Could not resolve dataset root for artifact manifest")


def verify_artifacts() -> tuple[dict[str, Any], Path, Path]:
    global DATASET_ROOT
    DATASET_ROOT = resolve_dataset_root()
    artifact_manifest = json.loads((DATASET_ROOT / "artifact_manifest.json").read_text(encoding="utf-8"))
    if artifact_manifest.get("dataset_visibility") != "private" or not artifact_manifest.get("validation_split_only"):
        raise RuntimeError("The Kaggle artifact manifest must describe a private validation-only dataset")

    expected_files = artifact_manifest.get("files", {})
    for name, expected in expected_files.items():
        if name == "dataset-metadata.json":
            continue
        path = DATASET_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Required Kaggle input artifact is missing: {path}")
        if path.stat().st_size != int(expected["size_bytes"]) or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Kaggle input artifact size/hash mismatch: {name}")

    checkpoint_path = DATASET_ROOT / "checkpoint_step_100.pt"
    predictor_dataset_path = DATASET_ROOT / "oracle_guided_predictor.pkl"
    unpack_source(EXPECTED_SOURCE_COMMIT)
    source_predictor = REPO_ROOT / "experiments/checkpoints/oracle_guided_predictor.pkl"
    if sha256(source_predictor) != sha256(predictor_dataset_path):
        raise RuntimeError("Private predictor artifact differs from the predictor pinned in the source commit")
    return artifact_manifest, checkpoint_path, predictor_dataset_path


def install_dependencies() -> None:
    torch_build = json.loads(
        run_command(
            [
                sys.executable,
                "-c",
                "import json, torch; print(json.dumps({'version': str(torch.__version__), 'cuda': torch.version.cuda}))",
            ],
        ).stdout.strip()
    )
    constraint_path = OUTPUT_ROOT / "logs" / "torch-build-constraints.txt"
    constraint_path.parent.mkdir(parents=True, exist_ok=True)
    constraint_path.write_text(f"torch=={torch_build['version']}\n", encoding="utf-8")
    run_command(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(REPO_ROOT)],
        cwd=REPO_ROOT,
        log_path=OUTPUT_ROOT / "logs" / "install_project.log",
    )
    run_command(
        [sys.executable, "-m", "pip", "install", "-c", str(constraint_path), "-r",
         str(REPO_ROOT / "experiments/kaggle/requirements_kaggle.txt")],
        cwd=REPO_ROOT,
        log_path=OUTPUT_ROOT / "logs" / "install_requirements.log",
    )
    torch_after = json.loads(
        run_command(
            [
                sys.executable,
                "-c",
                "import json, torch; print(json.dumps({'version': str(torch.__version__), 'cuda': torch.version.cuda}))",
            ],
        ).stdout.strip()
    )
    if torch_after != torch_build:
        raise RuntimeError(f"Dependency installation changed Kaggle's preinstalled Torch build: {torch_build} -> {torch_after}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    session_started = datetime.now(timezone.utc).isoformat()
    phase_status = {"runtime_decomposition": "not_started"}
    environment: dict[str, Any] | None = None
    sampler = NvidiaSampler()
    sampler.start()

    try:
        artifact_manifest, checkpoint_path, predictor_path = verify_artifacts()
        install_dependencies()
        environment = python_environment(REPO_ROOT)

        if not environment.get("cuda_available") or environment.get("visible_gpu_count", 0) < 2:
            raise RuntimeError(
                f"Two-GPU contract failed: expected >= 2 visible CUDA GPUs, got {environment.get('visible_gpu_count')}"
            )
        for g in environment["gpus"][:2]:
            if "T4" not in g["name"]:
                raise RuntimeError(f"Unexpected GPU accelerator: {g['name']}")

        phase_status["runtime_decomposition"] = "running"
        decomp_output_dir = OUTPUT_ROOT

        command = [
            sys.executable,
            "experiments/profile_gpu_decode_overhead.py",
            "--output-dir", str(decomp_output_dir),
            "--checkpoint", str(checkpoint_path),
            "--predictor", str(predictor_path),
            "--prompt-ids-file", "experiments/checkpoints/quality_benchmark/poc_12_prompt_ids.json",
            "--validation-data", "data/cached_pure_pred_val_60.json",
            "--base-revision", artifact_manifest["phi_revision"],
            "--zip2zip-revision", artifact_manifest["zip2zip_revision"],
            "--tested-commit", EXPECTED_SOURCE_COMMIT,
        ]
        run_command(command, cwd=REPO_ROOT, log_path=OUTPUT_ROOT / "logs" / "runtime_decomposition.log")
        phase_status["runtime_decomposition"] = "complete"

    except Exception as error:
        phase_status["runtime_decomposition"] = "failed"
        (OUTPUT_ROOT / "logs" / "run_error.log").parent.mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "logs" / "run_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Error encountered: {error}", flush=True)

    finally:
        sampler.stop()

    launcher_file_path = Path(__file__).resolve()
    launcher_file_hash = sha256(launcher_file_path) if launcher_file_path.is_file() else None

    finished = datetime.now(timezone.utc).isoformat()
    if environment is not None:
        manifest = {
            "schema": "tokens_kaggle_gpu_environment_v1",
            "repo": "aehng/Tokens",
            "source_commit": EXPECTED_SOURCE_COMMIT,
            "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
            "launcher_file_sha256": launcher_file_hash,
            "launcher_version": LAUNCHER_VERSION,
            "torch_version": environment["torch"],
            "transformers_version": environment["transformers"],
            "cuda_version": environment["cuda_runtime"],
            "python_version": environment["python"],
            "driver_version": sampler.samples[-1].get("driver_version") if sampler.samples else None,
            "visible_gpu_count": environment["visible_gpu_count"],
            "gpus": environment["gpus"],
            "python_packages": environment["packages"],
            "nvidia_smi_utilization": {
                "sample_count": len(sampler.samples),
                "samples": sampler.samples,
                "errors": sampler.errors,
            },
            "session_started_at_utc": session_started,
            "session_finished_at_utc": finished,
            "session_wall_time_s": round(time.monotonic() - start, 3),
            "phase_status": phase_status,
        }
        write_json(OUTPUT_ROOT / "environment_manifest.json", manifest)


if __name__ == "__main__":
    main()
