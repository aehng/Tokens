"""Small DEV-only Phi attribution run with strict A/B0/B1/CF gates."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path


WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


def find_file(filename: str) -> Path:
    matches = sorted(path for path in INPUT.rglob(filename) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"Kaggle input does not contain {filename}")
    return matches[0]


def find_optional_file(filename: str) -> Path | None:
    matches = sorted(path for path in INPUT.rglob(filename) if path.is_file())
    return matches[0] if matches else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


source_sha_file = find_file("SOURCE_ARCHIVE_SHA256.txt")
expected_source_sha = source_sha_file.read_text(encoding="utf-8").strip()
source_root = WORKING / "attribution_source"
source_archive = find_optional_file("proof_source.bin") or find_optional_file("source.tar.gz")
if source_archive is not None:
    actual_source_sha = sha256_file(source_archive)
    if actual_source_sha != expected_source_sha:
        raise RuntimeError("source archive SHA-256 does not match the staged manifest")
    source_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source_archive, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts or not member.isfile():
                raise RuntimeError(f"unsafe source archive member: {member.name}")
        archive.extractall(source_root)
else:
    source_dirs = [path for path in INPUT.rglob("zip2zip") if path.is_dir() and path.parent.name == "src"]
    if not source_dirs:
        raise RuntimeError("Kaggle input contains neither the source archive nor an extracted source tree")
    source_root = source_dirs[0].parent.parent
    actual_source_sha = expected_source_sha

source_file_manifest = json.loads(find_file("SOURCE_FILES_SHA256.json").read_text(encoding="utf-8"))
expected_source_files = set(source_file_manifest)
actual_source_files = {
    path.relative_to(source_root).as_posix()
    for path in source_root.rglob("*")
    if path.is_file()
}
if actual_source_files != expected_source_files:
    raise RuntimeError("extracted source tree file set differs from its hash manifest")
for relative_path, expected_sha in source_file_manifest.items():
    if sha256_file(source_root / relative_path) != expected_sha:
        raise RuntimeError(f"source file hash mismatch: {relative_path}")

sys.path.insert(0, str(source_root))
sys.path.insert(0, str(source_root / "src"))
os.environ["PYTHONPATH"] = (
    str(source_root / "src") + os.pathsep + str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
)
os.environ["GIT_COMMIT"] = find_file("SOURCE_SHA.txt").read_text(encoding="utf-8").strip()
os.environ["SOURCE_ARCHIVE_SHA256"] = actual_source_sha
branch_file = find_optional_file("SOURCE_GIT_BRANCH.txt")
os.environ["SOURCE_GIT_BRANCH"] = branch_file.read_text(encoding="utf-8").strip() if branch_file else "grok/predictive-fidelity-audit"
dirty_file = find_optional_file("SOURCE_GIT_WORKTREE_DIRTY.txt")
os.environ["SOURCE_GIT_WORKTREE_DIRTY"] = dirty_file.read_text(encoding="utf-8").strip() if dirty_file else "true"
status_file = find_optional_file("SOURCE_GIT_STATUS_SHA256.txt")
os.environ["SOURCE_GIT_STATUS_SHA256"] = (
    status_file.read_text(encoding="utf-8").strip()
    if status_file
    else "e4ad10e032faa6242586ffaa0b7cbd8ccc39230ce7a8ee669c744fd92f8ac124"
)

# The repo's `zip2zip` package imports `zip2zip_compression` during package
# initialization. Install it before importing any repo module, while pinning
# Kaggle's preinstalled CUDA-enabled PyTorch build so pip cannot replace it.
cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
import torch


def torch_runtime_signature():
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    device_names = [torch.cuda.get_device_name(index) for index in range(device_count)]
    return {
        "torch_version": str(torch.__version__),
        "cuda_build": str(torch.version.cuda),
        "cuda_available": cuda_available,
        "visible_device_count": device_count,
        "visible_device_names": device_names,
    }


torch_runtime_before = torch_runtime_signature()
constraint_path = Path("/kaggle/temp/phi-attribution-torch-constraints.txt")
constraint_path.parent.mkdir(parents=True, exist_ok=True)
constraint_path.write_text(f"torch=={torch.__version__}\n", encoding="utf-8")
print(
    "Installing zip2zip-compression before repo imports; "
    f"CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}; PyTorch={torch_runtime_before}",
    flush=True,
)
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "--constraint",
    str(constraint_path),
    "zip2zip-compression>=0.3.3",
])
torch_runtime_after = torch_runtime_signature()
if torch_runtime_after != torch_runtime_before:
    raise RuntimeError(
        "Dependency installation changed the preinstalled PyTorch/CUDA runtime: "
        f"before={torch_runtime_before}, after={torch_runtime_after}"
    )
print(f"Dependency install preserved PyTorch/CUDA runtime: {torch_runtime_after}", flush=True)
import zip2zip_compression

print(
    f"zip2zip_compression import resolved to {getattr(zip2zip_compression, '__file__', '<unknown>')}",
    flush=True,
)

from src.zip2zip.predictor_v2.attribution_harness import select_stratified_dev_prompts
from src.zip2zip.predictor_v2.canonical_dataset import load_canonical_dataset


dev_benchmark_path = find_file("dev_phrases_benchmark_48.jsonl")
subtrain_path = find_file("train_phrases_subtrain.jsonl")
val_path = find_file("train_phrases_val.jsonl")
manifest_path = find_file("phrase_dataset_manifest.json")
dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

if dataset_manifest.get("final_split_accessed") is not False:
    raise RuntimeError("refusing dataset with final_split_accessed != False")

print("Starting PHASE 1: ORACLE PER-EXAMPLE H REPRESENTATION TEST", flush=True)
print(f"DEV benchmark phrases: {dataset_manifest['dev_benchmark_example_count']}", flush=True)
print(f"Source commit: {os.environ['GIT_COMMIT']}", flush=True)

# Cleanly uninstall torchao without modifying PEFT internals
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])

from experiments.run_representation_experiment import load_vanilla_phi
from experiments.run_oracle_h_experiment import run_oracle_h_investigation

output_dir = WORKING / "oracle_h_phase1_output"
output_dir.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

t_run_start = time.perf_counter()

# Load pure Vanilla Phi (100% frozen)
model, tokenizer = load_vanilla_phi(device)

# Execute Phase 1: Oracle Per-Example H Representation Test (12 stratified DEV prompts)
phase1_summary = run_oracle_h_investigation(
    model,
    tokenizer,
    dev_benchmark_path,
    device,
    output_dir,
    num_samples=12,
    stage_a_steps=300,
    stage_b_steps=300,
)

summary_json = json.dumps(phase1_summary, indent=2) + "\n"
(WORKING / "oracle_h_phase1_summary.json").write_text(summary_json, encoding="utf-8")

elapsed_seconds = time.perf_counter() - t_run_start
print(f"\nPhase 1 Complete in {elapsed_seconds / 60:.2f} minutes", flush=True)
print(f"Final Phase 1 Verdict: {phase1_summary['final_verdict']}", flush=True)
