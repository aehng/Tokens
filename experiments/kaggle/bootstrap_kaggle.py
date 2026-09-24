"""Kaggle entrypoint. Mounts the proof source dataset, then runs the proof.

``APPROVED_SOURCE_SHA`` is ``$Format:%H$`` in git. The dataset archive and
the kernel script substitute that marker with the commit that produced the
archive. A commit cannot contain its own hash, so the marker stays in git.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

# Expanded to the archive commit when the Kaggle dataset and kernel script are packed.
APPROVED_SOURCE_SHA = "$Format:%H$"
SOURCE_DATASET_SLUG = "elikearl/tokens-vllm-predictive-source"
# Kaggle extracts an uploaded .tar.gz into dataset files. The same gzip
# bytes are therefore also stored as proof_source.bin.
SOURCE_ARCHIVE_NAMES = ("source.tar.gz", "proof_source.bin")
_SHA_PLACEHOLDER = 'APPROVED_SOURCE_SHA = "$Format:%H$"'


class SourceDatasetError(RuntimeError):
    """The Kaggle input mount does not contain exactly one approved source."""


def expand_approved_source_sha(source: str, commit_sha: str) -> str:
    """Replace the assignment placeholder with the commit that owns this bootstrap."""
    if len(commit_sha) != 40 or any(char not in "0123456789abcdef" for char in commit_sha):
        raise ValueError(f"commit sha {commit_sha!r} is not 40 hex characters")
    lines = source.splitlines(keepends=True)
    hits = [index for index, line in enumerate(lines) if line.strip() == _SHA_PLACEHOLDER]
    if len(hits) != 1:
        raise ValueError("approved source SHA placeholder is missing or repeated")
    index = hits[0]
    ending = "\r\n" if lines[index].endswith("\r\n") else "\n" if lines[index].endswith("\n") else ""
    lines[index] = f'APPROVED_SOURCE_SHA = "{commit_sha}"' + ending
    return "".join(lines)


def find_source_dataset(
    input_root: Path, approved_sha: str = APPROVED_SOURCE_SHA
) -> Path:
    """Return the only input directory whose SOURCE_SHA.txt matches."""
    root = Path(input_root)
    if not root.is_dir():
        raise SourceDatasetError(f"Kaggle input directory does not exist: {root}")
    approved = approved_sha.strip()
    if not approved or approved == "$Format:%H$":
        raise SourceDatasetError("approved source SHA is not set to a commit")
    matches: list[Path] = []
    seen: set[str] = set()
    for sha_file in root.rglob("SOURCE_SHA.txt"):
        recorded = sha_file.read_text(encoding="utf-8").strip()
        if recorded != approved:
            continue
        directory = sha_file.parent.resolve()
        key = str(directory)
        if key in seen:
            continue
        seen.add(key)
        matches.append(directory)
    if len(matches) != 1:
        found = ", ".join(str(path) for path in matches) or "none"
        raise SourceDatasetError(
            f"expected exactly one source dataset with SHA {approved}, found {found}"
        )
    directory = matches[0]
    archive = source_archive(directory)
    if archive is None:
        raise SourceDatasetError(
            f"source.tar.gz is missing from {directory} "
            f"(also checked {', '.join(SOURCE_ARCHIVE_NAMES)})"
        )
    recorded = (directory / "SOURCE_SHA.txt").read_text(encoding="utf-8").strip()
    if recorded != approved:
        raise SourceDatasetError(
            f"SOURCE_SHA.txt in {directory} is {recorded}, expected {approved}"
        )
    return directory


def source_archive(directory: Path) -> Path | None:
    """Return the gzip tar in this mount, if Kaggle left one intact."""
    for name in SOURCE_ARCHIVE_NAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def source_manifest(directory: Path) -> dict[str, str | int]:
    """Identity of the mounted archive, logged before extraction."""
    archive = source_archive(directory)
    if archive is None:
        raise SourceDatasetError(f"source.tar.gz is missing from {directory}")
    sha = (directory / "SOURCE_SHA.txt").read_text(encoding="utf-8").strip()
    payload = archive.read_bytes()
    return {
        "source_dataset_path": str(directory),
        "source_dataset_slug": SOURCE_DATASET_SLUG,
        "source_archive_name": archive.name,
        "source_sha": sha,
        "source_archive_bytes": archive.stat().st_size,
        "source_archive_sha256": hashlib.sha256(payload).hexdigest(),
    }


def log_source_manifest(manifest: dict[str, str | int]) -> None:
    print(f"mounted source dataset path: {manifest['source_dataset_path']}", flush=True)
    print(f"source dataset slug: {manifest['source_dataset_slug']}", flush=True)
    print(f"SOURCE_SHA: {manifest['source_sha']}", flush=True)
    print(
        f"{manifest['source_archive_name']} size: {manifest['source_archive_bytes']}",
        flush=True,
    )
    print(f"archive SHA256: {manifest['source_archive_sha256']}", flush=True)


def main() -> None:
    kaggle_input = Path("/kaggle/input")
    source_dir = find_source_dataset(kaggle_input)
    manifest = source_manifest(source_dir)
    log_source_manifest(manifest)
    archive = source_archive(source_dir)
    if archive is None:
        raise SourceDatasetError(f"source.tar.gz is missing from {source_dir}")
    root = Path("/kaggle/working/tokens_src")
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(root)
    (root / "SOURCE_SHA.txt").write_text(str(manifest["source_sha"]) + "\n", encoding="utf-8")
    (root / "source_provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
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
