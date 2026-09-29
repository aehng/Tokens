"""Decontaminate and deterministically sample a pinned public chat corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from datasets import load_dataset

from src.zip2zip.predictor_v2.canonical_dataset import sha256_file, sha256_json
from src.zip2zip.predictor_v2.external_sourcebook import (
    ExternalExample,
    PromptContaminationIndex,
    SourcebookError,
    load_canonical_prompt_texts,
    parse_external_example,
)


DEFAULT_DATASET = "HuggingFaceH4/ultrachat_200k"
DEFAULT_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"


def select_examples(
    *, canonical_path: str, output_jsonl: str, report_json: str,
    working_db: str, dataset: str = DEFAULT_DATASET, revision: str = DEFAULT_REVISION,
    split: str = "train_sft", sample_size: int = 50_000, seed: str = "predictor-v2-sourcebook-v1",
    max_examples: int | None = None,
) -> dict[str, Any]:
    if sample_size < 1:
        raise ValueError("sample size must be positive")
    canonical = load_canonical_prompt_texts(canonical_path)
    decontamination = PromptContaminationIndex(canonical)
    target = Path(output_jsonl)
    report_target = Path(report_json)
    stage_path = Path(working_db)
    target.parent.mkdir(parents=True, exist_ok=True)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or report_target.exists():
        raise FileExistsError("sourcebook sample/report already exists; choose fresh output paths")

    db = sqlite3.connect(stage_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS clean (rank BLOB NOT NULL, example_id TEXT PRIMARY KEY, prompt TEXT NOT NULL, retrieval_prompt TEXT NOT NULL, response TEXT NOT NULL, source_domain TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS processed (row_number INTEGER PRIMARY KEY, example_id TEXT NOT NULL, outcome TEXT NOT NULL, canonical_domain TEXT NOT NULL, detail_json TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS selection_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    selection_config = {"dataset": dataset, "revision": revision, "split": split, "seed": seed, "sample_size": sample_size, "max_examples": max_examples, "canonical_prompt_ids_sha256": sha256_json(sorted(item.prompt_id for item in canonical))}
    selection_config_json = json.dumps(selection_config, sort_keys=True)
    selection_config_sha256 = sha256_json(selection_config)
    previous_config = db.execute("SELECT value FROM selection_metadata WHERE key='config_sha256'").fetchone()
    if previous_config and previous_config[0] != selection_config_sha256:
        db.close()
        raise SourcebookError("selection resume database provenance does not match the requested corpus, prompts, split, seed, or sample size")
    db.execute("INSERT OR IGNORE INTO selection_metadata VALUES ('config_sha256',?)", (selection_config_sha256,))
    db.execute("INSERT OR IGNORE INTO selection_metadata VALUES ('config',?)", (selection_config_json,))
    db.commit()
    stream = load_dataset(dataset, split=split, streaming=True, revision=revision)
    for row_number, raw in enumerate(stream, 1):
        if max_examples is not None and row_number > max_examples:
            break
        if db.execute("SELECT 1 FROM processed WHERE row_number=?", (row_number,)).fetchone():
            continue
        item = parse_external_example(raw)
        if item is None:
            db.execute("INSERT INTO processed VALUES (?,?,?,?,?)", (row_number, "", "malformed", "", "{}"))
            if row_number % 2_000 == 0:
                db.commit()
            continue
        source_id = f"{item.example_id}:{row_number}"
        match = None
        matched_user_prompt = item.prompt
        for user_prompt in item.filter_prompts or (item.prompt,):
            match = decontamination.match(user_prompt)
            if match:
                matched_user_prompt = user_prompt
                break
        if match:
            exact = match["method"] == "exact_normalized_sha256"
            bucket = "exact" if exact else "near_duplicate"
            domain = str(match["canonical_domain"])
            detail = {"source_example_id": source_id, "canonical_prompt_id": match["canonical_prompt_id"], "canonical_domain": domain, "reason": match["method"], "word_jaccard": match["word_jaccard"], "char_trigram_jaccard": match["char_trigram_jaccard"], "source_prompt_preview": matched_user_prompt[:120]}
            db.execute("INSERT INTO processed VALUES (?,?,?,?,?)", (row_number, source_id, bucket, domain, json.dumps(detail, ensure_ascii=False, sort_keys=True)))
        else:
            rank = hashlib.sha256(f"{seed}\0{source_id}".encode("utf-8")).digest()
            db.execute("INSERT INTO clean VALUES (?,?,?,?,?,?)", (rank, source_id, item.prompt, item.retrieval_prompt or item.prompt, item.response, item.source_domain))
            db.execute("INSERT INTO processed VALUES (?,?,?,?,?)", (row_number, source_id, "clean", "", "{}"))
        if row_number % 2_000 == 0:
            db.commit()
        if row_number % 5_000 == 0:
            accepted = db.execute("SELECT COUNT(*) FROM clean").fetchone()[0]
            print(f"external source scan: {row_number:,} rows; {accepted:,} prompt-clean candidates", flush=True)
    db.commit()
    outcomes = dict(db.execute("SELECT outcome,COUNT(*) FROM processed GROUP BY outcome"))
    counts = {
        "external_examples_before": sum(int(count) for count in outcomes.values()),
        "malformed_or_missing_prompt_response": int(outcomes.get("malformed", 0)),
        "exact_duplicates_removed": int(outcomes.get("exact", 0)),
        "near_duplicates_removed": int(outcomes.get("near_duplicate", 0)),
        "clean_examples_after_prompt_filter": int(outcomes.get("clean", 0)),
    }
    removed_by_domain: dict[str, dict[str, int]] = {}
    representatives: list[dict[str, Any]] = []
    for domain, outcome, count in db.execute("SELECT canonical_domain,outcome,COUNT(*) FROM processed WHERE outcome IN ('exact','near_duplicate') GROUP BY canonical_domain,outcome"):
        removed_by_domain.setdefault(domain, {"exact": 0, "near_duplicate": 0})[outcome] = int(count)
    for (_domain, _outcome, detail_json) in db.execute("SELECT canonical_domain,outcome,detail_json FROM processed WHERE outcome IN ('exact','near_duplicate') ORDER BY row_number LIMIT 20"):
        representatives.append(json.loads(detail_json))
    selected = db.execute("SELECT example_id,prompt,retrieval_prompt,response,source_domain FROM clean ORDER BY rank,example_id LIMIT ?", (sample_size,)).fetchall()
    if len(selected) < sample_size:
        # Small smoke runs may deliberately cap input before the requested size.
        sample_size_actual = len(selected)
    else:
        sample_size_actual = sample_size
    with target.open("x", encoding="utf-8", newline="\n") as output:
        for example_id, prompt, retrieval_prompt, response, source_domain in selected:
            output.write(json.dumps({"example_id": example_id, "prompt": prompt, "retrieval_prompt": retrieval_prompt, "response": response, "source_domain": source_domain}, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema": "predictor_v2_external_sourcebook_decontamination_v1",
        "dataset": dataset,
        "dataset_revision": revision,
        "split": split,
        "sampling_seed": seed,
        "selection_config_sha256": selection_config_sha256,
        "requested_sample_size": sample_size,
        "selected_sample_size": sample_size_actual,
        "canonical_prompt_count": len(canonical),
        "canonical_prompt_ids_sha256": sha256_json(sorted(item.prompt_id for item in canonical)),
        "counts": counts,
        "removed_by_canonical_domain": removed_by_domain,
        "representative_removals": representatives,
        "selected_source_jsonl_sha256": sha256_file(target),
        "filter_scope": "canonical prompt text only; canonical DEV and FINAL responses are never consulted",
        "heldout_response_overlap": "not_measured_by_design; DEV/FINAL responses are forbidden filtering inputs",
        "contamination_thresholds": {"lexical_word_set_jaccard": 0.84, "fuzzy_char_trigram_jaccard": 0.88, "minhash_seeds": len(range(16)), "minhash_bands": 8},
    }
    report["report_sha256"] = sha256_json(report)
    report_target.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    db.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a pinned external chat corpus against all 900 canonical prompts and deterministically sample clean rows")
    parser.add_argument("--canonical", default="data/canonical_phi_continuations.jsonl")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--split", default="train_sft")
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--seed", default="predictor-v2-sourcebook-v1")
    parser.add_argument("--max-examples", type=int, default=None, help="Small plumbing smoke only")
    parser.add_argument("--working-db", default="scratch/predictor_v2_external_sourcebook/selection.sqlite")
    parser.add_argument("--out-jsonl", default="scratch/predictor_v2_external_sourcebook/clean_sample.jsonl")
    parser.add_argument("--report", default="scratch/predictor_v2_external_sourcebook/decontamination_report.json")
    args = parser.parse_args()
    result = select_examples(canonical_path=args.canonical, output_jsonl=args.out_jsonl, report_json=args.report, working_db=args.working_db, dataset=args.dataset, revision=args.revision, split=args.split, sample_size=args.sample_size, seed=args.seed, max_examples=args.max_examples)
    print(json.dumps({"counts": result["counts"], "selected_sample_size": result["selected_sample_size"], "selected_source_jsonl_sha256": result["selected_source_jsonl_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
