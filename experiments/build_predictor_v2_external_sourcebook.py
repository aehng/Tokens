"""Build a resumable Phi-token phrase sourcebook from a clean source sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
from pathlib import Path
from typing import Any, Iterator

from transformers import AutoTokenizer

from src.zip2zip.predictor_v2.canonical_dataset import sha256_file, sha256_json
from src.zip2zip.predictor_v2.external_sourcebook import ExternalExample, SourcebookBuilder


TOKENIZER_ID = "microsoft/Phi-3.5-mini-instruct"
TOKENIZER_REVISION = "2fe192450127e6a83f7441aef6e3ca586c338b77"


def iter_examples(path: str | Path) -> Iterator[ExternalExample]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            yield ExternalExample(
                example_id=str(row["example_id"]),
                prompt=str(row["prompt"]),
                response=str(row["response"]),
                source_domain=str(row.get("source_domain", "unlabeled")),
                retrieval_prompt=str(row.get("retrieval_prompt") or row["prompt"]),
                filter_prompts=(str(row["prompt"]),),
            )


def build_sourcebook(*, sample_jsonl: str, decontamination_report: str, tokenizer_id: str, tokenizer_revision: str, database_path: str, manifest_path: str, chunk_size: int = 128, local_files_only: bool = False) -> dict[str, Any]:
    sample_hash = sha256_file(sample_jsonl)
    decontamination = json.loads(Path(decontamination_report).read_text(encoding="utf-8"))
    expected_report_hash = decontamination.get("report_sha256")
    unhashed_report = {key: value for key, value in decontamination.items() if key != "report_sha256"}
    if expected_report_hash != sha256_json(unhashed_report):
        raise ValueError("decontamination report self-hash does not match")
    expected_sample_hash = decontamination.get("selected_source_jsonl_sha256")
    if expected_sample_hash != sample_hash:
        raise ValueError("clean sample JSONL does not match its contamination report")
    source_config = {
        "sample_jsonl_sha256": sample_hash,
        "decontamination_report_sha256": sha256_file(decontamination_report),
        "dataset": decontamination["dataset"],
        "dataset_revision": decontamination["dataset_revision"],
        "dataset_split": decontamination["split"],
        "canonical_prompt_ids_sha256": decontamination["canonical_prompt_ids_sha256"],
        "filter_scope": decontamination["filter_scope"],
    }
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=tokenizer_revision, local_files_only=local_files_only)
    builder = SourcebookBuilder(database_path, tokenizer=tokenizer, tokenizer_id=tokenizer_id, tokenizer_revision=tokenizer_revision, config={"source": source_config})
    try:
        builder.add_chunk(iter_examples(sample_jsonl), chunk_size=chunk_size)
        summary = builder.summary()
    finally:
        builder.close()
    checkpoint_db = sqlite3.connect(database_path)
    try:
        checkpoint = checkpoint_db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    finally:
        checkpoint_db.close()
    if checkpoint is not None and int(checkpoint[0]) != 0:
        raise RuntimeError(f"sourcebook WAL checkpoint is busy: {checkpoint}")
    manifest = {
        "schema": "predictor_v2_external_sourcebook_manifest_v1",
        "source": source_config,
        "sourcebook_database_sha256": sha256_file(database_path),
        "sourcebook_database_bytes": Path(database_path).stat().st_size,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_special_ids_sha256": sha256_json(sorted(int(item) for item in tokenizer.all_special_ids)),
        "summary": summary,
        "python_version": platform.python_version(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a persistent 2-4 Phi-token phrase sourcebook")
    parser.add_argument("--sample", default="scratch/predictor_v2_external_sourcebook/clean_sample.jsonl")
    parser.add_argument("--decontamination-report", default="scratch/predictor_v2_external_sourcebook/decontamination_report.json")
    parser.add_argument("--tokenizer-id", default=TOKENIZER_ID)
    parser.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    parser.add_argument("--database", default="scratch/predictor_v2_external_sourcebook/sourcebook.sqlite")
    parser.add_argument("--manifest", default="scratch/predictor_v2_external_sourcebook/sourcebook.manifest.json")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    manifest = build_sourcebook(sample_jsonl=args.sample, decontamination_report=args.decontamination_report, tokenizer_id=args.tokenizer_id, tokenizer_revision=args.tokenizer_revision, database_path=args.database, manifest_path=args.manifest, chunk_size=args.chunk_size, local_files_only=args.local_files_only)
    print(json.dumps({"source_prompts": manifest["summary"]["source_prompts"], "unique_phrases_by_phi_token_length": manifest["summary"]["unique_phrases_by_phi_token_length"], "database_bytes": manifest["sourcebook_database_bytes"], "manifest_sha256": manifest["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
