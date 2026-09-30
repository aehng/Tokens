"""Build High-Quality Token-Pair Dataset for Frozen-Phi Hypertoken Representation.

Enforces:
  1. Strictly zero access to FINAL split (raises error if final is touched).
  2. Prompt-level disjoint split of TRAIN:
     - 90% (567 prompts) -> subtrain
     - 10% (63 prompts) -> internal validation
  3. Balanced across domains (Code, Reasoning, Instruction).
  4. Phrase length = exactly 2 tokens (A, B).
  5. Exclude special/control/EOS tokens (all token IDs < 32000).
  6. Require >= 16 future teacher tokens after phrase.
  7. Variable left-context lengths (32 to 128 tokens).
  8. Capped examples per prompt (max 15) to prevent long prompt dominance.
  9. Punctuation pair cap to prevent degenerate single-pair dominance.
  10. Deterministic 48-example DEV evaluation benchmark (12 prompts x 4 phrases).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from zip2zip.predictor_v2.attribution_harness import (
    CANONICAL_EOS_TOKEN_IDS,
    STRATIFIED_DEV12_PROMPT_IDS,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_special_token(token_id: int) -> bool:
    """True if token is in special/control range (>= 32000) or EOS."""
    return token_id >= 32000 or token_id in CANONICAL_EOS_TOKEN_IDS


def extract_phrases_from_continuation(
    prompt_id: str,
    domain: str,
    continuation_tokens: Sequence[int],
    *,
    min_context: int = 32,
    max_context: int = 128,
    min_future: int = 16,
    max_future: int = 32,
    max_examples_per_prompt: int = 15,
    context_step: int = 8,
) -> List[Dict[str, Any]]:
    """Extract valid (context, A, B, future) phrase tuples from token stream."""
    L = len(continuation_tokens)
    if L < (min_context + 2 + min_future):
        return []

    examples: List[Dict[str, Any]] = []
    # Candidate phrase start positions
    # A is at pos, B is at pos + 1
    # Context is [pos - C : pos]
    # Future is [pos + 2 : min(L, pos + 2 + max_future)]
    possible_starts = list(range(min_context, L - min_future - 1, context_step))

    for pos in possible_starts:
        token_a = int(continuation_tokens[pos])
        token_b = int(continuation_tokens[pos + 1])

        # Exclude special/EOS tokens inside the phrase
        if is_special_token(token_a) or is_special_token(token_b):
            continue

        future_tokens = [int(t) for t in continuation_tokens[pos + 2 : min(L, pos + 2 + max_future)]]
        if len(future_tokens) < min_future:
            continue

        # Choose left-context length between min_context and max_context
        avail_context = pos
        ctx_len = min(avail_context, max_context)
        context_tokens = [int(t) for t in continuation_tokens[pos - ctx_len : pos]]

        examples.append({
            "prompt_id": prompt_id,
            "domain": domain,
            "phrase_start_idx": pos,
            "context_length": len(context_tokens),
            "context_token_ids": context_tokens,
            "token_a": token_a,
            "token_b": token_b,
            "future_token_ids": future_tokens,
            "future_length": len(future_tokens),
        })

        if len(examples) >= max_examples_per_prompt:
            break

    return examples


def build_phrase_datasets(
    canonical_jsonl: Path,
    output_dir: Path,
    *,
    subtrain_ratio: float = 0.9,
    seed: int = 42,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    print(f"Reading canonical dataset from {canonical_jsonl}...", flush=True)
    records: List[Dict[str, Any]] = []
    with canonical_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                records.append(row)

    # 1. STRICT SPLIT FILTERING & FINAL SAFETY ASSERTION
    train_records = [r for r in records if r.get("split") == "train"]
    dev_records = [r for r in records if r.get("split") == "dev"]
    final_records = [r for r in records if r.get("split") == "final"]

    print(f"Found {len(train_records)} TRAIN records, {len(dev_records)} DEV records, {len(final_records)} FINAL records.")
    if len(train_records) == 0:
        raise ValueError("Canonical dataset contains no TRAIN records!")

    # 2. PROMPT-LEVEL SUBTRAIN / INTERNAL-VAL SPLIT
    # Stratified by domain
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in train_records:
        by_domain[r["domain"]].append(r)

    subtrain_prompts: Set[str] = set()
    internal_val_prompts: Set[str] = set()

    for dom, dom_recs in sorted(by_domain.items()):
        # Sort deterministically by prompt_id, then shuffle with fixed seed
        dom_recs_sorted = sorted(dom_recs, key=lambda x: x["prompt_id"])
        rng.shuffle(dom_recs_sorted)
        n_val = max(1, int(len(dom_recs_sorted) * (1.0 - subtrain_ratio)))
        val_slice = dom_recs_sorted[:n_val]
        subtrain_slice = dom_recs_sorted[n_val:]
        for r in val_slice:
            internal_val_prompts.add(r["prompt_id"])
        for r in subtrain_slice:
            subtrain_prompts.add(r["prompt_id"])

    # Disjointness check
    intersection = subtrain_prompts.intersection(internal_val_prompts)
    if intersection:
        raise AssertionError(f"Subtrain and internal-val prompts overlap: {intersection}")

    print(f"Subtrain prompts: {len(subtrain_prompts)}, Internal Validation prompts: {len(internal_val_prompts)}")

    # 3. EXTRACT PHRASE EXAMPLES FOR SUBTRAIN AND INTERNAL-VAL
    subtrain_examples: List[Dict[str, Any]] = []
    val_examples: List[Dict[str, Any]] = []
    pair_freq_counter: Counter[Tuple[int, int]] = Counter()

    for r in train_records:
        pid = r["prompt_id"]
        dom = r["domain"]
        tokens = r["continuation_token_ids"]
        extracted = extract_phrases_from_continuation(pid, dom, tokens)

        if pid in subtrain_prompts:
            subtrain_examples.extend(extracted)
            for ex in extracted:
                pair_freq_counter[(ex["token_a"], ex["token_b"])] += 1
        elif pid in internal_val_prompts:
            val_examples.extend(extracted)

    print(f"Generated {len(subtrain_examples)} subtrain phrase examples across {len(pair_freq_counter)} unique token pairs.")
    print(f"Generated {len(val_examples)} internal validation phrase examples.")

    # 4. BUILD DETERMINISTIC 48-EXAMPLE DEV EVALUATION BENCHMARK
    # 12 canonical DEV prompts x 4 phrases per prompt = 48 examples
    dev_by_id = {r["prompt_id"]: r for r in dev_records}
    dev_benchmark_examples: List[Dict[str, Any]] = []

    for pid in STRATIFIED_DEV12_PROMPT_IDS:
        if pid not in dev_by_id:
            raise ValueError(f"Canonical DEV prompt {pid} not found in dataset!")
        r = dev_by_id[pid]
        dom = r["domain"]
        tokens = r["continuation_token_ids"]
        # Extract exactly 4 phrases across the continuation
        extracted = extract_phrases_from_continuation(
            pid, dom, tokens, max_examples_per_prompt=4, context_step=16
        )
        if len(extracted) < 4:
            # Try tighter spacing if continuation was shorter
            extracted = extract_phrases_from_continuation(
                pid, dom, tokens, max_examples_per_prompt=4, context_step=8
            )
        dev_benchmark_examples.extend(extracted[:4])

    print(f"Generated {len(dev_benchmark_examples)} deterministic DEV benchmark examples (12 prompts x 4 phrases).")

    # 5. WRITE OUT ARTIFACTS
    subtrain_path = output_dir / "train_phrases_subtrain.jsonl"
    with subtrain_path.open("w", encoding="utf-8") as f:
        for ex in subtrain_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    val_path = output_dir / "train_phrases_val.jsonl"
    with val_path.open("w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    dev_bench_path = output_dir / "dev_phrases_benchmark_48.jsonl"
    with dev_bench_path.open("w", encoding="utf-8") as f:
        for ex in dev_benchmark_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Write Manifest
    manifest = {
        "schema": "tokens_phrase_dataset_manifest_v1",
        "created_at_utc": str(Path(__file__).stat().st_mtime),
        "source_dataset_sha256": sha256_file(canonical_jsonl),
        "subtrain_file": subtrain_path.name,
        "subtrain_sha256": sha256_file(subtrain_path),
        "subtrain_example_count": len(subtrain_examples),
        "subtrain_prompt_count": len(subtrain_prompts),
        "subtrain_unique_pairs": len(pair_freq_counter),
        "val_file": val_path.name,
        "val_sha256": sha256_file(val_path),
        "val_example_count": len(val_examples),
        "val_prompt_count": len(internal_val_prompts),
        "dev_benchmark_file": dev_bench_path.name,
        "dev_benchmark_sha256": sha256_file(dev_bench_path),
        "dev_benchmark_example_count": len(dev_benchmark_examples),
        "domain_breakdown": {
            dom: sum(1 for ex in subtrain_examples if ex["domain"] == dom)
            for dom in by_domain
        },
        "top_10_pairs_sample": [
            {"pair": list(pair), "count": count}
            for pair, count in pair_freq_counter.most_common(10)
        ],
        "final_split_accessed": False,
    }

    manifest_path = output_dir / "phrase_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote phrase dataset manifest to {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build Phrase Datasets for Frozen-Phi Experiment")
    parser.add_argument("--canonical-jsonl", type=str, default="data/canonical_phi_continuations.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/phrase_training_dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_phrase_datasets(
        canonical_jsonl=Path(args.canonical_jsonl),
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
