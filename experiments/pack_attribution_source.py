"""Build and dry-run the exact source archive the attribution Kaggle job receives.

The archive is created from the working tree with Python's tarfile so the
extracted package can be imported independently of this checkout. It is not
``git archive`` (which rewrites CRLF on this Windows checkout).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

ATTRIBUTION_SOURCE_PATHS: tuple[str, ...] = (
    "experiments/run_phi_attribution_benchmark.py",
    "experiments/load_joint_checkpoint.py",
    "experiments/generation_timing.py",
    "experiments/run_quality_benchmark.py",
    "experiments/pack_attribution_source.py",
    "experiments/mbpp_prompt.py",
    "experiments/benchmark_provenance.py",
    "experiments/runtime_diagnostics.py",
    "experiments/load_oracle_predictor.py",
)

ATTRIBUTION_SOURCE_DIRECTORIES: tuple[str, ...] = (
    "src/zip2zip",
    "src/evaluation",
)

REQUIRED_IMPORTS: tuple[str, ...] = (
    "src.zip2zip.static_codebook",
    "src.zip2zip.predictor_v2.ablation_gates",
    "src.zip2zip.predictor_v2.attribution_harness",
    "src.zip2zip.predictor_v2.forced_oracle",
    "src.zip2zip.predictor_v2.canonical_dataset",
    "src.zip2zip.predictor_v2.attribution_reanalysis",
    "src.zip2zip.model",
    "experiments.run_phi_attribution_benchmark",
    "experiments.load_joint_checkpoint",
)

REQUIRED_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("src.zip2zip.predictor_v2.attribution_harness", "select_stratified_dev_prompts"),
    ("src.zip2zip.predictor_v2.attribution_harness", "STRATIFIED_DEV12_PROMPT_IDS"),
    ("src.zip2zip.predictor_v2.attribution_harness", "COND_B1_UPSTREAM_EPFL_ADAPTER_H_DISABLED"),
    ("src.zip2zip.predictor_v2.attribution_harness", "COND_B2_STEP100_H_DISABLED"),
    ("experiments.run_phi_attribution_benchmark", "select_stratified_dev_prompts"),
    ("experiments.run_phi_attribution_benchmark", "offline_a_b0_token_gate"),
    ("experiments.run_phi_attribution_benchmark", "assert_model_phase_teardown"),
    ("experiments.run_phi_attribution_benchmark", "run_a_b1_matched_prefix_diagnostic"),
    ("experiments.run_phi_attribution_benchmark", "run_attribution_benchmark"),
    ("src.zip2zip.predictor_v2.ablation_gates", "normalize_wrapper_logits"),
    ("src.zip2zip.predictor_v2.ablation_gates", "b0_b1_adapter_isolation_gate"),
    ("src.zip2zip.predictor_v2.ablation_gates", "b1_b2_checkpoint_isolation_gate"),
    ("src.zip2zip.predictor_v2.forced_oracle", "h_vs_base_prefix_pair"),
    ("src.zip2zip.model", "Zip2ZipModel"),
    ("src.zip2zip.static_codebook", "StaticCodebookManager"),
)


def iter_source_files(root: Path = REPO_ROOT) -> List[Path]:
    files: List[Path] = []
    for relative in ATTRIBUTION_SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing attribution source file: {relative}")
        files.append(path)
    for relative in ATTRIBUTION_SOURCE_DIRECTORIES:
        directory = root / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing attribution source directory: {relative}")
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def archive_relative(path: Path, root: Path = REPO_ROOT) -> str:
    return path.relative_to(root).as_posix()


def write_source_archive(destination: Path, root: Path = REPO_ROOT) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = iter_source_files(root)
    with tarfile.open(destination, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for path in files:
            archive.add(path, arcname=archive_relative(path, root))
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {
        "archive": str(destination),
        "sha256": digest,
        "file_count": len(files),
        "files": [archive_relative(path, root) for path in files],
    }


def extract_source_archive(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(destination)
    return destination


def dry_run_extracted_package(extracted_root: Path) -> dict:
    extracted_root = extracted_root.resolve()
    env = os.environ.copy()
    # Do not inherit the developer checkout through PYTHONPATH. The archive
    # must stand on its own exactly as it will after Kaggle extraction.
    env["PYTHONPATH"] = os.pathsep.join([str(extracted_root), str(extracted_root / "src")])
    script = """
import importlib
from pathlib import Path
import sys
failed = []
root = Path.cwd().resolve()
symbols = %r
modules = %r
for name in modules:
    try:
        module = importlib.import_module(name)
        origin = getattr(module, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(root):
            failed.append(f"{name}: imported outside extracted source: {origin}")
    except Exception as exc:
        failed.append(f"{name}: {type(exc).__name__}: {exc}")
for module_name, symbol in symbols:
    try:
        module = importlib.import_module(module_name)
        getattr(module, symbol)
    except Exception as exc:
        failed.append(f"{module_name}.{symbol}: {type(exc).__name__}: {exc}")
if failed:
    raise SystemExit("\\n".join(failed))
print("DRY_RUN_OK")
""" % (list(REQUIRED_SYMBOLS), list(REQUIRED_IMPORTS))
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(extracted_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Extracted attribution package dry-run failed:\n"
            + (completed.stdout or "")
            + (completed.stderr or "")
        )
    runner = extracted_root / "experiments" / "run_phi_attribution_benchmark.py"
    runner_help = subprocess.run(
        [sys.executable, str(runner), "--help"],
        cwd=str(extracted_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if (
        runner_help.returncode != 0
        or "--conditions" not in runner_help.stdout
        or "--logit-only" not in runner_help.stdout
    ):
        raise RuntimeError(
            "Extracted attribution runner failed its CLI import check:\n"
            + (runner_help.stdout or "")
            + (runner_help.stderr or "")
        )
    return {
        "status": "PASS",
        "cwd": str(extracted_root),
        "pythonpath": env["PYTHONPATH"],
        "stdout": completed.stdout.strip(),
        "runner_help_status": "PASS",
        "runner_help_excerpt": " ".join(runner_help.stdout.split())[:700],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("attribution_source.tar.gz"))
    parser.add_argument("--extract-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = write_source_archive(args.output)
    print(f"Wrote {report['archive']} sha256={report['sha256']} files={report['file_count']}")
    if args.dry_run:
        extract_dir = args.extract_dir or (args.output.parent / "extracted_attribution_source")
        extract_source_archive(args.output, extract_dir)
        dry = dry_run_extracted_package(extract_dir)
        print(dry["stdout"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
