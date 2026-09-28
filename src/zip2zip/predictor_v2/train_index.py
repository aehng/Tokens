"""TRAIN-only association-index construction for canonical Predictor V2 data."""

from __future__ import annotations

import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.zip2zip.predictor_v2.candidate_retrieval import TrainOnlyAssociationIndex
from src.zip2zip.predictor_v2.canonical_dataset import (
    CanonicalContinuation,
    CanonicalDatasetError,
    CanonicalDatasetViews,
    require_split,
    sha256_file,
    sha256_json,
)


def extract_ngrams(
    tokens: Sequence[int], disabled_ids: set[int], min_len: int = 2, max_len: int = 4
) -> Counter[tuple[int, ...]]:
    counts: Counter[tuple[int, ...]] = Counter()
    for length in range(min_len, min(max_len + 1, len(tokens) + 1)):
        for i in range(len(tokens) - length + 1):
            gram = tuple(tokens[i : i + length])
            if not any(token in disabled_ids for token in gram):
                counts[gram] += 1
    return counts


def _disabled_ids(tokenizer: Any) -> set[int]:
    ids = set(getattr(tokenizer, "all_special_ids", ()) or ())
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            ids.add(value)
    # Phi-3.5 reserves this range for special/control IDs used by the study.
    ids.update((0, 1, 2))
    ids.update(range(32000, 32011))
    return ids


def _current_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _index_source_hashes() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[3]
    files = (
        repo / "src/zip2zip/predictor_v2/canonical_dataset.py",
        repo / "src/zip2zip/predictor_v2/candidate_retrieval.py",
        Path(__file__).resolve(),
    )
    return {path.relative_to(repo).as_posix(): sha256_file(path) for path in files}


def build_train_only_index(
    records: Sequence[CanonicalContinuation],
    *,
    tokenizer: Any,
    dataset_sha256: str,
    train_split_sha256: str,
    source_manifest_sha256: str,
    config: Mapping[str, Any] | None = None,
) -> tuple[TrainOnlyAssociationIndex, dict[str, Any]]:
    """Mine associations from TRAIN continuations only and bind their provenance."""
    require_split(records, "TRAIN", "train-only index construction")
    if not records:
        raise CanonicalDatasetError("cannot build a train-only index from an empty TRAIN split")
    config = {
        "min_phrase_length": 2,
        "max_subtokens": 4,
        "min_cooccurrence": 2,
        "top_candidates_per_token": 64,
        **dict(config or {}),
    }
    min_len = int(config["min_phrase_length"])
    max_subtokens = int(config["max_subtokens"])
    min_cooccurrence = int(config["min_cooccurrence"])
    top_candidates_per_token = int(config["top_candidates_per_token"])
    if min_len < 2 or max_subtokens < min_len or min_cooccurrence < 1 or top_candidates_per_token < 1:
        raise ValueError("invalid train-index configuration")

    disabled_ids = _disabled_ids(tokenizer)
    global_counts: Counter[tuple[int, ...]] = Counter()
    domain_counts: dict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
    prompt_to_phrase: dict[int, Counter[tuple[int, ...]]] = defaultdict(Counter)
    prompt_token_totals: Counter[int] = Counter()
    lexical_docs: list[dict[str, Any]] = []

    for record in records:
        prompt_ids = tokenizer.encode(record.rendered_prompt_text, add_special_tokens=False)
        continuation_ngrams = extract_ngrams(
            record.continuation_token_ids, disabled_ids, min_len, max_subtokens
        )
        prompt_tokens = set(prompt_ids) - disabled_ids
        for prompt_token in prompt_tokens:
            prompt_token_totals[prompt_token] += 1
            for phrase, count in continuation_ngrams.items():
                prompt_to_phrase[prompt_token][phrase] += count
        for phrase, count in continuation_ngrams.items():
            global_counts[phrase] += count
            domain_counts[record.domain][phrase] += count
        lexical_docs.append(
            {
                "prompt_id": record.prompt_id,
                "prompt_text": record.rendered_prompt_text,
                "domain": record.domain,
                "continuation_phrases": dict(continuation_ngrams.most_common(32)),
            }
        )

    index = TrainOnlyAssociationIndex(
        disabled_ids=disabled_ids,
        train_prompt_lexical_docs=lexical_docs,
        train_prompt_ids=sorted(record.prompt_id for record in records),
        total_train_continuations=len(records),
        max_subtokens=max_subtokens,
    )
    index.precomputed_global_static = [
        (phrase, math.log1p(count) * (len(phrase) - 1) * 0.25)
        for phrase, count in sorted(
            global_counts.items(), key=lambda item: item[1] * (len(item[0]) - 1), reverse=True
        )[:512]
    ]
    for domain, counts in domain_counts.items():
        ordered = sorted(counts.items(), key=lambda item: item[1] * (len(item[0]) - 1), reverse=True)
        index.precomputed_domain_static[domain] = [phrase for phrase, _ in ordered[:512]]
    for prompt_token, phrases in prompt_to_phrase.items():
        denominator = prompt_token_totals.get(prompt_token, 1) + 25.0
        candidates = [
            (phrase, (count / denominator) * (len(phrase) - 1) * 3.0)
            for phrase, count in phrases.items()
            if count >= min_cooccurrence
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        index.token_associations[prompt_token] = candidates[:top_candidates_per_token]

    provenance = {
        "schema": "predictor_v2_train_index_provenance_v1",
        "source_dataset_sha256": dataset_sha256,
        "train_split_sha256": train_split_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "train_prompt_ids_sha256": sha256_json(index.train_prompt_ids),
        "config": config,
        "candidate_strategies": [
            "baseline",
            "expanded_associations",
            "suffix_conditioned",
            "sparse_lexical",
        ],
        "code_commit": _current_commit(),
        "code_file_sha256": _index_source_hashes(),
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    index.provenance = provenance
    summary = {
        **provenance,
        "provenance_sha256": sha256_json(provenance),
        "train_records": len(records),
        "indexed_tokens_count": len(index.token_associations),
        "global_static_count": len(index.precomputed_global_static),
        "domain_static_counts": {key: len(value) for key, value in index.precomputed_domain_static.items()},
        "train_prompt_ids": index.train_prompt_ids,
    }
    return index, summary


def build_index_from_views(
    views: CanonicalDatasetViews,
    *,
    tokenizer: Any,
    config: Mapping[str, Any] | None = None,
) -> tuple[TrainOnlyAssociationIndex, dict[str, Any]]:
    """Convenience wrapper that deliberately passes only the immutable TRAIN view."""
    return build_train_only_index(
        views.train,
        tokenizer=tokenizer,
        dataset_sha256=views.dataset_sha256,
        train_split_sha256=views.train_split_sha256,
        source_manifest_sha256=views.manifest_sha256,
        config=config,
    )


def validate_train_index(
    index: TrainOnlyAssociationIndex,
    views: CanonicalDatasetViews,
    *,
    index_path: str | None = None,
    expected_index_sha256: str | None = None,
    expected_provenance_sha256: str | None = None,
) -> None:
    provenance = getattr(index, "provenance", None)
    if not isinstance(provenance, dict):
        raise CanonicalDatasetError("index has no canonical TRAIN-only provenance")
    expected = {
        "source_dataset_sha256": views.dataset_sha256,
        "train_split_sha256": views.train_split_sha256,
        "source_manifest_sha256": views.manifest_sha256,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise CanonicalDatasetError(f"index provenance mismatch for {key}")
    if expected_provenance_sha256 and sha256_json(provenance) != expected_provenance_sha256:
        raise CanonicalDatasetError("index provenance hash does not match candidate freeze")
    if expected_index_sha256:
        if not index_path or sha256_file(index_path) != expected_index_sha256:
            raise CanonicalDatasetError("TRAIN index artifact hash does not match candidate freeze")
    expected_ids = sorted(record.prompt_id for record in views.train)
    if sorted(index.train_prompt_ids) != expected_ids:
        raise CanonicalDatasetError("index prompt IDs do not exactly match TRAIN")
    if set(index.train_prompt_ids) & (set(record.prompt_id for record in views.dev) | set(views.final_ids)):
        raise CanonicalDatasetError("index contains DEV or FINAL prompt IDs")
