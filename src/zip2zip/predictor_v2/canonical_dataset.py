"""Strict input contract for the Predictor V2 canonical continuation study.

This module is intentionally separate from ``vanilla_labels``.  That module
contains legacy 60-prompt loading behavior; this one validates the versioned
canonical 900-record artifact and exposes immutable split views.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA = "predictor_v2_canonical_manifest_v1"
RECORD_REQUIRED_FIELDS = frozenset(
    {
        "prompt_id",
        "domain",
        "split",
        "task_prompt_text",
        "rendered_prompt_text",
        "continuation_text",
        "continuation_token_ids",
        "generated_token_count",
        "termination_reason",
        "termination_token_id",
    }
)
SPLITS = ("TRAIN", "DEV", "FINAL")
MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "dataset_sha256",
        "expected_prompt_ids",
        "split_ids",
        "allowed_domains",
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "tokenizer_vocab_size",
        "generation_config",
        "generation_config_sha256",
        "provenance",
        "provenance_sha256",
        "manifest_sha256",
    }
)


class CanonicalDatasetError(ValueError):
    """Raised when a canonical dataset or its provenance is invalid."""


class FinalAccessError(PermissionError):
    """Raised when code tries to open FINAL without a validated permit."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_integrity_hash(manifest: Mapping[str, Any]) -> str:
    return sha256_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def manifest_provenance_hash(manifest: Mapping[str, Any]) -> str:
    keys = (
        "dataset_sha256",
        "expected_prompt_ids",
        "split_ids",
        "allowed_domains",
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "tokenizer_vocab_size",
        "generation_config_sha256",
        "provenance",
    )
    return sha256_json({key: manifest[key] for key in keys})


def create_canonical_manifest(
    dataset_path: str | Path,
    *,
    expected_prompt_ids: Sequence[str],
    split_ids: Mapping[str, Sequence[str]],
    allowed_domains: Sequence[str],
    model_id: str,
    model_revision: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    tokenizer_vocab_size: int,
    generation_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a manifest from independently supplied IDs, splits, and revisions.

    Callers must supply the expected ID/split lists from the prompt inventory,
    not derive them from the completion artifact being verified.
    """
    normalized_splits = {
        split: sorted(str(pid) for pid in split_ids.get(split, ()))
        for split in SPLITS
    }
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "dataset_sha256": sha256_file(dataset_path),
        "expected_prompt_ids": sorted(str(pid) for pid in expected_prompt_ids),
        "split_ids": normalized_splits,
        "allowed_domains": sorted(set(str(d) for d in allowed_domains)),
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "generation_config": dict(generation_config),
        "generation_config_sha256": sha256_json(generation_config),
        "provenance": dict(provenance),
    }
    manifest["provenance_sha256"] = manifest_provenance_hash(manifest)
    manifest["manifest_sha256"] = manifest_integrity_hash(manifest)
    validate_manifest_shape(manifest)
    return manifest


def write_canonical_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    validate_manifest_shape(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(target, flags, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise CanonicalDatasetError(f"{label} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise CanonicalDatasetError(f"{label} must contain non-empty strings")
    if len(set(value)) != len(value):
        raise CanonicalDatasetError(f"{label} contains duplicates")
    return value


def validate_manifest_shape(manifest: Mapping[str, Any]) -> None:
    missing = sorted(MANIFEST_REQUIRED_FIELDS - set(manifest))
    if missing:
        raise CanonicalDatasetError(f"manifest missing required fields: {', '.join(missing)}")
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise CanonicalDatasetError(f"unsupported manifest schema: {manifest['schema']!r}")
    for field in (
        "dataset_sha256",
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "generation_config_sha256",
        "provenance_sha256",
        "manifest_sha256",
    ):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise CanonicalDatasetError(f"manifest field {field} must be a non-empty string")
    expected_ids = _string_list(manifest["expected_prompt_ids"], "expected_prompt_ids")
    split_ids = manifest["split_ids"]
    if not isinstance(split_ids, Mapping) or set(split_ids) != set(SPLITS):
        raise CanonicalDatasetError(f"split_ids must contain exactly {', '.join(SPLITS)}")
    joined: list[str] = []
    for split in SPLITS:
        ids = split_ids[split]
        if not isinstance(ids, list) or any(not isinstance(pid, str) or not pid.strip() for pid in ids):
            raise CanonicalDatasetError(f"split_ids.{split} must be a list of non-empty strings")
        if len(set(ids)) != len(ids):
            raise CanonicalDatasetError(f"split_ids.{split} contains duplicates")
        joined.extend(ids)
    if len(set(joined)) != len(joined):
        raise CanonicalDatasetError("prompt IDs overlap between manifest splits")
    if set(joined) != set(expected_ids):
        raise CanonicalDatasetError("manifest split IDs do not equal expected_prompt_ids")
    domains = _string_list(manifest["allowed_domains"], "allowed_domains")
    if not isinstance(manifest["tokenizer_vocab_size"], int) or isinstance(manifest["tokenizer_vocab_size"], bool) or manifest["tokenizer_vocab_size"] <= 0:
        raise CanonicalDatasetError("tokenizer_vocab_size must be a positive integer")
    config = manifest["generation_config"]
    if not isinstance(config, Mapping):
        raise CanonicalDatasetError("generation_config must be an object")
    max_new_tokens = config.get("max_new_tokens")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
        raise CanonicalDatasetError("generation_config.max_new_tokens must be a positive integer")
    if sha256_json(config) != manifest["generation_config_sha256"]:
        raise CanonicalDatasetError("generation_config_sha256 does not match generation_config")
    if not isinstance(manifest["provenance"], Mapping):
        raise CanonicalDatasetError("provenance must be an object")
    if manifest_provenance_hash(manifest) != manifest["provenance_sha256"]:
        raise CanonicalDatasetError("provenance_sha256 does not match manifest provenance")
    if manifest_integrity_hash(manifest) != manifest["manifest_sha256"]:
        raise CanonicalDatasetError("manifest_sha256 does not match manifest contents")
    # Keep local names referenced so static analyzers also catch accidental empty domains.
    if not domains:
        raise CanonicalDatasetError("allowed_domains must not be empty")


@dataclass(frozen=True)
class CanonicalContinuation:
    prompt_id: str
    domain: str
    split: str
    task_prompt_text: str
    rendered_prompt_text: str
    continuation_text: str
    continuation_token_ids: tuple[int, ...]
    generated_token_count: int
    termination_reason: str
    termination_token_id: int | None
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    generation_config_json: str
    generation_metadata_json: str

    @property
    def generation_config(self) -> dict[str, Any]:
        return json.loads(self.generation_config_json)

    @property
    def generation_metadata(self) -> dict[str, Any]:
        return json.loads(self.generation_metadata_json)

    @property
    def prompt_text(self) -> str:
        """Compatibility spelling used by the existing Predictor V2 feature code."""
        return self.rendered_prompt_text

    def canonical_record(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "domain": self.domain,
            "split": self.split,
            "task_prompt_text": self.task_prompt_text,
            "rendered_prompt_text": self.rendered_prompt_text,
            "continuation_text": self.continuation_text,
            "continuation_token_ids": list(self.continuation_token_ids),
            "generated_token_count": self.generated_token_count,
            "termination_reason": self.termination_reason,
            "termination_token_id": self.termination_token_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "generation_config": self.generation_config,
            "generation_metadata": self.generation_metadata,
        }

    def to_legacy_record(self, tokenizer: Any) -> Any:
        """Create a fresh compatibility record for existing Predictor V2 models."""
        from src.zip2zip.predictor_v2.vanilla_labels import VanillaContinuationRecord

        prompt_ids = tokenizer.encode(self.rendered_prompt_text, add_special_tokens=False)
        config = self.generation_config
        return VanillaContinuationRecord(
            prompt_id=self.prompt_id,
            domain=self.domain,
            prompt_text=self.rendered_prompt_text,
            prompt_token_ids=list(prompt_ids),
            continuation_text=self.continuation_text,
            continuation_token_ids=list(self.continuation_token_ids),
            base_model=self.model_id,
            base_revision=self.model_revision,
            tokenizer_name=self.tokenizer_id,
            generation_config=config,
            dataset_source="canonical_phi_continuations.jsonl",
            task_prompt_text=self.task_prompt_text,
            rendered_prompt_text=self.rendered_prompt_text,
            raw_continuation_text=self.continuation_text,
            generated_token_count=self.generated_token_count,
            termination_reason=self.termination_reason,
            termination_token_id=self.termination_token_id,
        )


def _valid_token_id(value: Any, vocab_size: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < vocab_size:
        raise CanonicalDatasetError(f"{label} must be an integer in [0, {vocab_size})")
    return value


def _stop_ids(config: Mapping[str, Any]) -> set[int]:
    raw: list[Any] = []
    for key in ("eos_token_id", "stop_token_ids"):
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            raw.append(value)
        elif isinstance(value, list):
            raw.extend(value)
    return {v for v in raw if isinstance(v, int) and not isinstance(v, bool)}


def _parse_record(raw: Any, line_number: int, manifest: Mapping[str, Any]) -> CanonicalContinuation:
    if not isinstance(raw, Mapping):
        raise CanonicalDatasetError(f"line {line_number}: record must be a JSON object")
    missing = sorted(RECORD_REQUIRED_FIELDS - set(raw))
    if missing:
        raise CanonicalDatasetError(f"line {line_number}: missing fields: {', '.join(missing)}")
    pid = raw["prompt_id"]
    if not isinstance(pid, str) or not pid.strip():
        raise CanonicalDatasetError(f"line {line_number}: prompt_id must be a non-empty string")
    domain = raw["domain"]
    if not isinstance(domain, str) or domain not in manifest["allowed_domains"]:
        raise CanonicalDatasetError(f"line {line_number}: invalid domain {domain!r}")
    split = raw["split"]
    if isinstance(split, str):
        split = split.upper()
    if split not in SPLITS:
        raise CanonicalDatasetError(f"line {line_number}: split must be TRAIN, DEV, or FINAL")
    for field in ("task_prompt_text", "rendered_prompt_text", "continuation_text"):
        if not isinstance(raw[field], str):
            raise CanonicalDatasetError(f"line {line_number}: {field} must be a string")
    if not raw["task_prompt_text"].strip() or not raw["rendered_prompt_text"].strip():
        raise CanonicalDatasetError(f"line {line_number}: task and rendered prompts must not be empty")
    ids = raw["continuation_token_ids"]
    if not isinstance(ids, list) or not ids:
        raise CanonicalDatasetError(f"line {line_number}: continuation_token_ids must be a non-empty list")
    vocab_size = manifest["tokenizer_vocab_size"]
    token_ids = tuple(_valid_token_id(v, vocab_size, f"line {line_number} continuation_token_ids") for v in ids)
    count = raw["generated_token_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count != len(token_ids):
        raise CanonicalDatasetError(f"line {line_number}: generated_token_count must equal token ID count")
    row_provenance = raw.get("provenance", {})
    if not isinstance(row_provenance, Mapping):
        raise CanonicalDatasetError(f"line {line_number}: provenance must be an object")

    def provenance_value(field: str) -> Any:
        return raw[field] if field in raw else row_provenance.get(field)

    config = provenance_value("generation_config")
    if not isinstance(config, Mapping) or sha256_json(config) != manifest["generation_config_sha256"]:
        raise CanonicalDatasetError(f"line {line_number}: generation_config differs from manifest")
    cap = config["max_new_tokens"]
    if count > cap:
        raise CanonicalDatasetError(f"line {line_number}: generated token count exceeds max_new_tokens")
    reason = raw["termination_reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise CanonicalDatasetError(f"line {line_number}: termination_reason must be a non-empty string")
    term_id = raw["termination_token_id"]
    if term_id is not None:
        term_id = _valid_token_id(term_id, vocab_size, f"line {line_number} termination_token_id")
        if token_ids[-1] != term_id:
            raise CanonicalDatasetError(f"line {line_number}: termination token must be the last generated token")
    reason_key = reason.strip().lower().replace("-", "_")
    if "eos" in reason_key or "stop" in reason_key:
        configured_stops = _stop_ids(config)
        if term_id is None:
            raise CanonicalDatasetError(f"line {line_number}: stop termination requires termination_token_id")
        if configured_stops and term_id not in configured_stops:
            raise CanonicalDatasetError(f"line {line_number}: termination token is not configured as a stop token")
    if any(marker in reason_key for marker in ("length", "max_new_tokens", "token_cap", "cap")) and count != cap:
        raise CanonicalDatasetError(f"line {line_number}: cap termination count must equal max_new_tokens")
    for field in ("model_id", "model_revision", "tokenizer_id", "tokenizer_revision"):
        value = provenance_value(field)
        if not isinstance(value, str) or not value.strip() or value != manifest[field]:
            raise CanonicalDatasetError(f"line {line_number}: {field} differs from manifest")

    expected_record_provenance = manifest["provenance"].get("canonical_provenance", {})
    if isinstance(expected_record_provenance, Mapping):
        for field, expected_value in expected_record_provenance.items():
            if field in row_provenance and row_provenance[field] != expected_value:
                raise CanonicalDatasetError(
                    f"line {line_number}: provenance.{field} differs from canonical provenance"
                )

    metadata = raw.get("generation_metadata")
    if metadata is None:
        # The audited 900-row artifact stores shared generation provenance under
        # `provenance` and row-level audit/health fields beside the continuation.
        # Preserve both in the normalized view while leaving source JSONL bytes
        # untouched.
        excluded = {
            "prompt_id", "domain", "split", "task_prompt_text", "rendered_prompt_text",
            "continuation_text", "raw_continuation_text", "continuation_token_ids",
            "prompt_text", "prompt_token_ids", "provenance",
        }
        metadata = {key: value for key, value in raw.items() if key not in excluded}
        metadata["record_provenance"] = dict(row_provenance)
    if not isinstance(metadata, Mapping):
        raise CanonicalDatasetError(f"line {line_number}: generation_metadata must be an object")

    model_id = provenance_value("model_id")
    model_revision = provenance_value("model_revision")
    tokenizer_id = provenance_value("tokenizer_id")
    tokenizer_revision = provenance_value("tokenizer_revision")
    return CanonicalContinuation(
        prompt_id=pid,
        domain=domain,
        split=split,
        task_prompt_text=raw["task_prompt_text"],
        rendered_prompt_text=raw["rendered_prompt_text"],
        continuation_text=raw["continuation_text"],
        continuation_token_ids=token_ids,
        generated_token_count=count,
        termination_reason=reason,
        termination_token_id=term_id,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        generation_config_json=canonical_json_bytes(config).decode("utf-8"),
        generation_metadata_json=canonical_json_bytes(metadata).decode("utf-8"),
    )


@dataclass(frozen=True)
class FinalAccessPermit:
    dataset_sha256: str
    quality_attribution_gate_sha256: str
    candidate_plan_sha256: str
    candidate_freeze_sha256: str
    architecture_shortlist_sha256: str
    live_integration_gate_sha256: str
    architecture_freeze_sha256: str
    allow_final_eval: bool


@dataclass(frozen=True)
class CanonicalDatasetViews:
    dataset_sha256: str
    manifest_sha256: str
    provenance_sha256: str
    train_split_sha256: str
    dev_split_sha256: str
    final_split_sha256: str
    train: tuple[CanonicalContinuation, ...]
    dev: tuple[CanonicalContinuation, ...]
    _final: tuple[CanonicalContinuation, ...]

    @property
    def train_records(self) -> tuple[CanonicalContinuation, ...]:
        return self.train

    @property
    def dev_records(self) -> tuple[CanonicalContinuation, ...]:
        return self.dev

    @property
    def final_ids(self) -> tuple[str, ...]:
        return tuple(record.prompt_id for record in self._final)

    def open_final(self, permit: FinalAccessPermit | None) -> tuple[CanonicalContinuation, ...]:
        if (
            not isinstance(permit, FinalAccessPermit)
            or not permit.allow_final_eval
            or permit.dataset_sha256 != self.dataset_sha256
            or not permit.quality_attribution_gate_sha256
            or not permit.candidate_plan_sha256
            or not permit.candidate_freeze_sha256
            or not permit.architecture_shortlist_sha256
            or not permit.live_integration_gate_sha256
            or not permit.architecture_freeze_sha256
        ):
            raise FinalAccessError("FINAL requires --allow-final-eval, a passed live integration gate, and frozen candidate/architecture artifacts")
        return self._final


def _split_hash(records: Sequence[CanonicalContinuation]) -> str:
    ordered = [record.canonical_record() for record in sorted(records, key=lambda item: item.prompt_id)]
    return sha256_json(ordered)


def load_canonical_dataset(
    dataset_path: str | Path,
    manifest_path: str | Path,
) -> tuple[CanonicalDatasetViews, dict[str, Any]]:
    """Load and strictly validate the artifact against an independently authored manifest."""
    dataset_path = Path(dataset_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise CanonicalDatasetError("manifest must be a JSON object")
    validate_manifest_shape(manifest)
    actual_file_hash = sha256_file(dataset_path)
    if actual_file_hash != manifest["dataset_sha256"]:
        raise CanonicalDatasetError("dataset_sha256 does not match the JSONL artifact")

    records: list[CanonicalContinuation] = []
    seen: set[str] = set()
    with dataset_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CanonicalDatasetError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            record = _parse_record(raw, line_number, manifest)
            if record.prompt_id in seen:
                raise CanonicalDatasetError(f"duplicate prompt_id: {record.prompt_id}")
            seen.add(record.prompt_id)
            records.append(record)

    expected = set(manifest["expected_prompt_ids"])
    if seen != expected:
        missing = sorted(expected - seen)
        unexpected = sorted(seen - expected)
        raise CanonicalDatasetError(
            f"prompt IDs differ from expected inventory (missing={missing[:5]}, unexpected={unexpected[:5]})"
        )
    split_map = manifest["split_ids"]
    by_split: dict[str, tuple[CanonicalContinuation, ...]] = {}
    for split in SPLITS:
        expected_split = set(split_map[split])
        actual = tuple(r for r in records if r.split == split)
        actual_ids = {r.prompt_id for r in actual}
        if actual_ids != expected_split:
            raise CanonicalDatasetError(f"record splits do not match manifest split_ids.{split}")
        by_split[split] = tuple(sorted(actual, key=lambda item: item.prompt_id))

    views = CanonicalDatasetViews(
        dataset_sha256=actual_file_hash,
        manifest_sha256=manifest["manifest_sha256"],
        provenance_sha256=manifest["provenance_sha256"],
        train_split_sha256=_split_hash(by_split["TRAIN"]),
        dev_split_sha256=_split_hash(by_split["DEV"]),
        final_split_sha256=_split_hash(by_split["FINAL"]),
        train=by_split["TRAIN"],
        dev=by_split["DEV"],
        _final=by_split["FINAL"],
    )
    return views, dict(manifest)


def load_manifest_tokenizer(manifest: Mapping[str, Any], *, local_files_only: bool = True) -> Any:
    """Load the pinned tokenizer and reject a vocabulary-size mismatch."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        manifest["tokenizer_id"],
        revision=manifest["tokenizer_revision"],
        local_files_only=local_files_only,
    )
    actual_size = len(tokenizer)
    if actual_size != manifest["tokenizer_vocab_size"]:
        raise CanonicalDatasetError(
            f"pinned tokenizer vocabulary size mismatch: manifest={manifest['tokenizer_vocab_size']}, loaded={actual_size}"
        )
    return tokenizer


def require_split(records: Sequence[CanonicalContinuation], expected: str, stage: str) -> None:
    if expected not in SPLITS:
        raise ValueError(f"unknown expected split {expected!r}")
    mismatches = [record.prompt_id for record in records if record.split != expected]
    if mismatches:
        raise CanonicalDatasetError(
            f"{stage} accepts {expected} records only; found {len(mismatches)} other-split records"
        )
