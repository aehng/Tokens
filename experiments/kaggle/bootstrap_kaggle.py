"""Kaggle entrypoint. Extracts the pinned source archive, then runs the proof."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path


def _archive() -> Path:
    candidates = [
        Path("/kaggle/working/source.tar.gz"),
        Path(__file__).resolve().parent / "source.tar.gz",
        Path("source.tar.gz"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("source.tar.gz was not uploaded with the kernel")


def main() -> None:
    archive = _archive()
    root = Path("/kaggle/working/tokens_src")
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(root)
    os.chdir(root)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "vllm==0.30.0"]
    )
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(root)])
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sys.path.insert(0, str(root))
    from experiments.kaggle.run_vllm_predictive_proof import main as run_proof

    run_proof()


if __name__ == "__main__":
    main()
