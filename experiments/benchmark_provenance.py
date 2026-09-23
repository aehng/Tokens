"""Small, strict helpers for reproducible benchmark runs and result caches."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


TIER1_MANIFEST_SCHEMA = "phi_tier1_manifest_v1"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_tested_commit(repo_root: str | os.PathLike[str]) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit.lower()):
        raise ValueError(f"Expected a full Git commit SHA, got {commit!r}")
    return commit.lower()


def select_prompt_subset(
    all_samples: Sequence[Mapping[str, Any]],
    prompt_ids_file: str | os.PathLike[str],
    required_domain_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Select IDs in file order and fail closed on missing, duplicate, or skewed data."""
    with open(prompt_ids_file, "r", encoding="utf-8") as source:
        prompt_entries = json.load(source)
    if not isinstance(prompt_entries, list) or not prompt_entries:
        raise ValueError("prompt ID file must contain a non-empty JSON list")

    prompt_ids = [
        entry.get("id") if isinstance(entry, Mapping) else entry
        for entry in prompt_entries
    ]
    if any(not isinstance(prompt_id, str) or not prompt_id for prompt_id in prompt_ids):
        raise ValueError("every prompt entry must be a non-empty string ID or object with an ID")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("prompt ID file contains duplicate IDs")

    sample_map: dict[str, Mapping[str, Any]] = {}
    for sample in all_samples:
        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("validation data contains a sample without a string ID")
        if sample_id in sample_map:
            raise ValueError(f"validation data contains duplicate ID {sample_id!r}")
        sample_map[sample_id] = sample

    missing = [prompt_id for prompt_id in prompt_ids if prompt_id not in sample_map]
    if missing:
        raise ValueError(f"prompt IDs are missing from validation data: {missing}")

    selected = [dict(sample_map[prompt_id]) for prompt_id in prompt_ids]
    if required_domain_counts is not None:
        actual: dict[str, int] = {}
        for sample in selected:
            domain = sample.get("domain")
            if not isinstance(domain, str):
                raise ValueError(f"sample {sample['id']!r} has no valid domain")
            actual[domain] = actual.get(domain, 0) + 1
        expected = dict(required_domain_counts)
        if actual != expected:
            raise ValueError(f"prompt domain counts mismatch: expected {expected}, got {actual}")
    return selected


def build_generation_cache_key(
    *,
    condition: Mapping[str, Any],
    prompt_id: str,
    prompt_text: str,
    reference_text: str,
    generation: Mapping[str, Any],
    evaluator_sha256: str,
    environment: Mapping[str, Any],
) -> str:
    """Return an exact per-prompt/config key; K and prompt representation belong in condition."""
    return canonical_sha256(
        {
            "schema": "generation_cache_key_v1",
            "condition": dict(condition),
            "prompt_id": prompt_id,
            "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            "reference_sha256": hashlib.sha256(reference_text.encode("utf-8")).hexdigest(),
            "generation": dict(generation),
            "evaluator_sha256": evaluator_sha256,
            "environment": dict(environment),
        }
    )


def write_json_atomic(path: str | os.PathLike[str], value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, indent=2, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def create_or_verify_run_manifest(
    path: str | os.PathLike[str],
    identity: Mapping[str, Any],
    schema: str = TIER1_MANIFEST_SCHEMA,
) -> tuple[dict[str, Any], bool]:
    """Create a manifest or return an identical one for safe in-run resumption."""
    identity_dict = dict(identity)
    manifest_sha = canonical_sha256(identity_dict)
    manifest_path = Path(path)
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as source:
            existing = json.load(source)
        if (
            existing.get("schema") != schema
            or existing.get("identity_sha256") != manifest_sha
            or existing.get("identity") != identity_dict
        ):
            raise ValueError(
                f"Existing run manifest does not match this configuration: {manifest_path}. "
                "Choose a new run directory; previous results were left untouched."
            )
        return existing, True

    manifest = {
        "schema": schema,
        "identity_sha256": manifest_sha,
        "identity": identity_dict,
        "status": "prepared",
    }
    write_json_atomic(manifest_path, manifest)
    return manifest, False


def validate_generation_cache_record(
    record: Any,
    expected_key: str,
    record_schema: str = "phi_generation_record_v1",
) -> bool:
    return (
        isinstance(record, dict)
        and record.get("generation_cache_key") == expected_key
        and record.get("record_schema") == record_schema
    )
