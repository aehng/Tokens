"""Compare Phi-only, external-only, and hybrid candidate pools on DEV."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.zip2zip.predictor_v2.candidate_retrieval import ConfigurableCandidateGenerator, RetrievalStrategy, TrainOnlyAssociationIndex
from src.zip2zip.predictor_v2.canonical_dataset import load_canonical_dataset, load_manifest_tokenizer, sha256_file, sha256_json
from src.zip2zip.predictor_v2.external_sourcebook import ExternalSourcebookCandidateGenerator, make_hybrid_pool
from src.zip2zip.predictor_v2.oracle_candidate_pool import CandidatePoolOracle
from src.zip2zip.predictor_v2.oracle_global import GlobalOccurrenceOracle
from src.zip2zip.predictor_v2.train_index import validate_train_index


POOL_SIZES = (256, 512, 1024)
K = 32
SYSTEMS = ("phi_only", "external_only", "hybrid")
DOMAINS = ("code", "reasoning", "instruction")


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50_ms": None, "p90_ms": None, "p99_ms": None}
    return {f"p{p}_ms": float(np.percentile(values, p)) for p in (50, 90, 99)}


def _capture_interval(lower: int, upper: int, global_lower: int, global_upper: int) -> dict[str, float | None]:
    if global_upper <= 0:
        return {"lower": None, "upper": None, "interpretation": "no_global_opportunity"}
    return {"lower": lower / global_upper, "upper": min(1.0, upper / global_lower) if global_lower > 0 else 1.0, "interpretation": "bound_derived"}


def _pool_memory_bytes(pool: dict[tuple[int, ...], dict[str, Any]]) -> int:
    size = sys.getsizeof(pool)
    for phrase, data in pool.items():
        size += sys.getsizeof(phrase) + sys.getsizeof(data) + sys.getsizeof(data.get("sources", set()))
    return size


def _pool_diagnostics(pool: dict[tuple[int, ...], dict[str, Any]], *, latency: float, generated_count: int, rejections: dict[tuple[int, ...], str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    ordered = sorted(pool.items(), key=lambda item: (item[1]["weight"], -len(item[0]), item[0]), reverse=True)
    result = {
        "generation_latency_ms": latency,
        "generated_candidate_count": generated_count,
        "quality_rejections": rejections,
        "rank_by_phrase": {phrase: rank for rank, (phrase, _) in enumerate(ordered, 1)},
        "sources_by_phrase": {phrase: set(data["sources"]) for phrase, data in ordered},
        "pool_source_counts": {source: sum(source in data["sources"] for data in pool.values()) for source in sorted({source for data in pool.values() for source in data["sources"]})},
        "pool_phrase_length_counts": {str(length): sum(len(phrase) == length for phrase in pool) for length in (2, 3, 4)},
    }
    result.update(extra or {})
    return result


def run_benchmark(*, dataset_path: str, manifest_path: str, train_index_path: str, sourcebook_path: str, sourcebook_manifest_path: str, out_json: str, out_md: str, pool_sizes: Sequence[int] = POOL_SIZES, max_dev_prompts: int | None = None, time_limit_seconds: float = 1.0) -> dict[str, Any]:
    for path in (out_json, out_md):
        if Path(path).exists():
            raise FileExistsError(f"benchmark output already exists: {path}")
    if any(size not in POOL_SIZES for size in pool_sizes):
        raise ValueError(f"pool sizes must be chosen from {POOL_SIZES}")
    source_manifest = json.loads(Path(sourcebook_manifest_path).read_text(encoding="utf-8"))
    expected_source_manifest_hash = source_manifest.get("manifest_sha256")
    if expected_source_manifest_hash != sha256_json({key: value for key, value in source_manifest.items() if key != "manifest_sha256"}):
        raise ValueError("sourcebook manifest self-hash does not match")
    actual_sourcebook_hash = sha256_file(sourcebook_path)
    if actual_sourcebook_hash != source_manifest.get("sourcebook_database_sha256"):
        raise ValueError("sourcebook database hash does not match its manifest")
    views, dataset_manifest = load_canonical_dataset(dataset_path, manifest_path)
    index = TrainOnlyAssociationIndex.load(train_index_path)
    validate_train_index(index, views)
    tokenizer = load_manifest_tokenizer(dataset_manifest)
    phi_generator = ConfigurableCandidateGenerator(index, tokenizer)
    rss_before_sourcebook = _rss_bytes()
    external_generator = ExternalSourcebookCandidateGenerator(sourcebook_path, tokenizer)
    rss_after_sourcebook_open = _rss_bytes()
    records = list(views.dev)
    if max_dev_prompts is not None:
        if max_dev_prompts < 1:
            raise ValueError("max DEV prompt count must be positive")
        records = records[:max_dev_prompts]
    is_full_dev = len(records) == len(views.dev)
    global_oracle = GlobalOccurrenceOracle(min_len=2, max_len=4, time_limit_seconds=time_limit_seconds)
    candidate_oracle = CandidatePoolOracle(tokenizer=tokenizer, time_limit_seconds=time_limit_seconds)
    totals = {name: {size: {"candidate_lower": 0, "candidate_upper": 0, "global_lower": 0, "global_upper": 0, "prompts": 0} for size in pool_sizes} for name in SYSTEMS}
    domain_totals: dict[str, dict[str, dict[int, dict[str, int]]]] = defaultdict(lambda: {name: {size: {"candidate_lower": 0, "candidate_upper": 0, "global_lower": 0, "global_upper": 0, "prompts": 0} for size in pool_sizes} for name in SYSTEMS})
    timings: dict[str, dict[int, list[float]]] = {name: {size: [] for size in pool_sizes} for name in SYSTEMS}
    memory_samples: dict[str, dict[int, list[int]]] = {name: {size: [] for size in pool_sizes} for name in SYSTEMS}
    pool_counts: dict[str, dict[int, Counter[int]]] = {name: {size: Counter() for size in pool_sizes} for name in SYSTEMS}
    misses: dict[str, dict[int, Counter[str]]] = {name: {size: Counter() for size in pool_sizes} for name in SYSTEMS}
    global_exact = 0
    candidate_exact = {name: {size: 0 for size in pool_sizes} for name in SYSTEMS}
    candidate_solves = {name: {size: 0 for size in pool_sizes} for name in SYSTEMS}
    run_started = time.perf_counter()
    try:
        for prompt_number, record in enumerate(records, 1):
            prompt_ids = tokenizer.encode(record.rendered_prompt_text, add_special_tokens=False)
            global_result = global_oracle.solve(record.continuation_token_ids, k=K, tokenizer=tokenizer)
            global_exact += int(global_result.is_exact)
            global_lower = int(global_result.steps_saved_lower_bound)
            global_upper = int(global_result.steps_saved_upper_bound)
            legacy = record.to_legacy_record(tokenizer)
            for size in pool_sizes:
                diagnostics_by_system: dict[str, dict[str, Any]] = {}
                phi_diagnostics: dict[str, Any] = {}
                phi_pool = phi_generator.generate_candidate_pool(prompt_ids, record.rendered_prompt_text, domain=record.domain, strategy=RetrievalStrategy.EXPANDED_ASSOCIATIONS, target_pool_size=size, diagnostics=phi_diagnostics)
                diagnostics_by_system["phi_only"] = phi_diagnostics
                ext_diagnostics: dict[str, Any] = {}
                ext_pool = external_generator.generate_candidate_pool(prompt_ids, record.rendered_prompt_text, domain=record.domain, target_pool_size=size, diagnostics=ext_diagnostics)
                diagnostics_by_system["external_only"] = ext_diagnostics
                started = time.perf_counter()
                hybrid_all = make_hybrid_pool(ext_pool, phi_pool, len(ext_pool) + len(phi_pool))
                hybrid_latency = (time.perf_counter() - started) * 1000.0
                hybrid = dict(list(hybrid_all.items())[:size])
                hybrid_diagnostics = _pool_diagnostics(
                    hybrid_all,
                    latency=float(phi_diagnostics["generation_latency_ms"]) + float(ext_diagnostics["generation_latency_ms"]) + hybrid_latency,
                    generated_count=len(hybrid_all),
                    rejections={**phi_diagnostics["quality_rejections"], **ext_diagnostics["quality_rejections"]},
                    extra={"retrieved_source_examples": ext_diagnostics.get("retrieved_source_examples", 0), "source_index_bytes": external_generator.index_bytes},
                )
                hybrid_diagnostics["pool_source_counts"] = {source: sum(source in data["sources"] for data in hybrid.values()) for source in sorted({source for data in hybrid.values() for source in data["sources"]})}
                hybrid_diagnostics["pool_phrase_length_counts"] = {str(length): sum(len(phrase) == length for phrase in hybrid) for length in (2, 3, 4)}
                diagnostics_by_system["hybrid"] = hybrid_diagnostics
                pools = {"phi_only": phi_pool, "external_only": ext_pool, "hybrid": hybrid}
                for name, pool in pools.items():
                    diag = diagnostics_by_system[name]
                    timings[name][size].append(float(diag["generation_latency_ms"]))
                    memory_samples[name][size].append(_pool_memory_bytes(pool))
                    pool_counts[name][size].update({int(length): int(count) for length, count in diag["pool_phrase_length_counts"].items()})
                    pool_oracle_records = phi_generator.build_candidate_records(legacy, strategy=RetrievalStrategy.EXPANDED_ASSOCIATIONS, target_pool_size=size, candidate_pool=pool) if name == "phi_only" else external_generator.build_candidate_records(legacy, target_pool_size=size, candidate_pool=pool)
                    result = candidate_oracle.solve(pool_oracle_records, record.continuation_token_ids, k=K, global_oracle_steps=global_lower)
                    candidate_exact[name][size] += int(result.is_exact)
                    candidate_solves[name][size] += 1
                    target = totals[name][size]
                    target["candidate_lower"] += int(result.steps_saved_lower_bound)
                    target["candidate_upper"] += int(result.steps_saved_upper_bound)
                    target["global_lower"] += global_lower
                    target["global_upper"] += global_upper
                    target["prompts"] += 1
                    domain = str(record.domain).casefold()
                    domain_target = domain_totals[domain][name][size]
                    domain_target["candidate_lower"] += int(result.steps_saved_lower_bound)
                    domain_target["candidate_upper"] += int(result.steps_saved_upper_bound)
                    domain_target["global_lower"] += global_lower
                    domain_target["global_upper"] += global_upper
                    domain_target["prompts"] += 1
                    rank_by_phrase = diag["rank_by_phrase"]
                    for phrase in global_result.selected_phrases:
                        rank = rank_by_phrase.get(tuple(phrase))
                        if rank is None:
                            reason = diag["quality_rejections"].get(tuple(phrase))
                            if reason:
                                misses[name][size][f"quality_filtered:{reason}"] += 1
                            else:
                                misses[name][size]["not_generated_from_retrieved_prompts"] += 1
                        elif rank > size:
                            misses[name][size]["generated_but_truncated_by_pool_limit"] += 1
                        elif rank > 32:
                            misses[name][size]["present_but_ranked_below_top32"] += 1
                        else:
                            misses[name][size]["global_incumbent_ranked_top32"] += 1
            if prompt_number % 10 == 0 or prompt_number == len(records):
                print(f"DEV sourcebook comparison: {prompt_number}/{len(records)} prompts", flush=True)
    finally:
        external_generator.close()

    systems: dict[str, Any] = {}
    strategy_results: dict[str, dict[str, Any]] = {}
    for name in SYSTEMS:
        by_size: dict[str, Any] = {}
        for size in pool_sizes:
            agg = totals[name][size]
            interval = _capture_interval(agg["candidate_lower"], agg["candidate_upper"], agg["global_lower"], agg["global_upper"])
            domains: dict[str, Any] = {}
            for domain in DOMAINS:
                values = domain_totals[domain][name][size]
                domains[domain] = {
                    "prompts": values["prompts"],
                    "capture_interval": _capture_interval(values["candidate_lower"], values["candidate_upper"], values["global_lower"], values["global_upper"]),
                    "candidate_steps_saved_lower_upper": [values["candidate_lower"], values["candidate_upper"]],
                    "global_steps_saved_lower_upper": [values["global_lower"], values["global_upper"]],
                }
            by_size[str(size)] = {
                "k": K,
                "candidate_oracle_capture_interval": interval,
                "candidate_steps_saved_lower_upper": [agg["candidate_lower"], agg["candidate_upper"]],
                "global_steps_saved_lower_upper": [agg["global_lower"], agg["global_upper"]],
                "domains": domains,
                "retrieval_latency": _percentiles(timings[name][size]),
                "mean_candidate_pool_python_bytes_estimate": int(statistics.mean(memory_samples[name][size])) if memory_samples[name][size] else 0,
                "pool_phrase_length_counts": {str(length): pool_counts[name][size][length] for length in (2, 3, 4)},
                "missed_opportunity_categories": dict(sorted(misses[name][size].items())),
                "oracle_exact_solves": candidate_exact[name][size],
                "oracle_solve_count": candidate_solves[name][size],
            }
        systems[name] = by_size
        strategy_name = {"phi_only": "expanded_associations", "external_only": "external_sourcebook", "hybrid": "hybrid_sourcebook"}[name]
        strategy_results[strategy_name] = {
            pool: {
                **metrics,
                "candidate_config": {
                    "strategy": strategy_name,
                    "pool_size": int(pool),
                    "implementation": "ConfigurableCandidateGenerator+external_sourcebook" if name == "hybrid" else ("ExternalSourcebookCandidateGenerator" if name == "external_only" else "ConfigurableCandidateGenerator"),
                    "sourcebook_database_sha256": actual_sourcebook_hash if name != "phi_only" else None,
                    "sourcebook_manifest_sha256": source_manifest.get("manifest_sha256") if name != "phi_only" else None,
                    "train_index_sha256": sha256_file(train_index_path),
                    "fidelity_mode": "external responses propose phrases; no answer steering",
                },
            }
            for pool, metrics in by_size.items()
        }
    bundle = {
        "schema": "predictor_v2_external_sourcebook_dev_benchmark_v1",
        "scope": "DEV",
        "is_full_dev": is_full_dev,
        "dev_prompts_evaluated": len(records),
        "dataset_sha256": views.dataset_sha256,
        "dev_split_sha256": views.dev_split_sha256,
        "train_split_sha256": views.train_split_sha256,
        "pool_sizes": [int(size) for size in pool_sizes],
        "canonical_counts": {"TRAIN": len(views.train), "DEV": len(views.dev), "FINAL": len(views.final_ids)},
        "final_accessed": False,
        "baseline": {"strategy": "expanded_associations", "provisional_reference_k32_1024_capture_interval": [0.406, 0.732], "reference_is_historical": True},
        "systems": systems,
        "strategy_results": strategy_results,
        "sourcebook": {
            "manifest_sha256": source_manifest.get("manifest_sha256"),
            "database_sha256": actual_sourcebook_hash,
            "database_bytes": Path(sourcebook_path).stat().st_size,
            "source_examples": external_generator.example_count,
            "unique_phrases_by_phi_token_length": source_manifest.get("summary", {}).get("unique_phrases_by_phi_token_length"),
            "retrieval_index_terms": external_generator.term_count,
            "process_rss_before_open_bytes": rss_before_sourcebook,
            "process_rss_after_open_bytes": rss_after_sourcebook_open,
            "process_rss_delta_bytes": (rss_after_sourcebook_open - rss_before_sourcebook) if rss_before_sourcebook is not None and rss_after_sourcebook_open is not None else None,
            "index_memory_note": "RSS delta includes Python/tokenizer/SQLite connection and is an approximate process-level measure; on-disk database bytes are exact.",
            "filter_scope": source_manifest["source"].get("filter_scope"),
            "heldout_response_overlap": "not_measured_by_design",
        },
        "train_index_sha256": sha256_file(train_index_path),
        "train_index_provenance_sha256": sha256_json(index.provenance),
        "global_oracle": {"k": K, "exact_solves": global_exact, "solve_count": len(records), "time_limit_seconds": time_limit_seconds},
        "runtime_seconds": time.perf_counter() - run_started,
        "decision_scope": "offline candidate opportunity only; does not establish H emission, continuation health, task quality, or live decode-step savings",
    }
    bundle["benchmark_sha256"] = sha256_json(bundle)
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Predictor V2 External Sourcebook DEV Comparison", "", f"Prompts: {len(records)} ({'full DEV' if is_full_dev else 'DEV smoke'}) · K={K} · FINAL accessed: no", "", "| Candidate source | Pool | Capture interval | p50/p90/p99 retrieval (ms) | Mean pool bytes | Code | Reasoning | Instruction |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in SYSTEMS:
        for size in pool_sizes:
            result = systems[name][str(size)]
            interval = result["candidate_oracle_capture_interval"]
            capture = "n/a" if interval["lower"] is None else f"{interval['lower']:.3f}–{interval['upper']:.3f}"
            latency = result["retrieval_latency"]
            lat = f"{latency['p50_ms']:.3f}/{latency['p90_ms']:.3f}/{latency['p99_ms']:.3f}"
            domain_values = []
            for domain in DOMAINS:
                value = result["domains"][domain]["capture_interval"]
                domain_values.append("n/a" if value["lower"] is None else f"{value['lower']:.3f}–{value['upper']:.3f}")
            lines.append(f"| {name} | {size} | {capture} | {lat} | {result['mean_candidate_pool_python_bytes_estimate']} | {domain_values[0]} | {domain_values[1]} | {domain_values[2]} |")
    lines += ["", "The `phi_only` row is the live rerun of the provisional `expanded_associations` baseline. Capture values are bounded oracle results, not point estimates. Prompt decontamination never reads canonical continuations; external-response overlap with held-out completions is not measured. Offline capture is not evidence of live task-quality parity, H-token emission, continuation health, or decode-step savings.", ""]
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="DEV-only sourcebook candidate/oracle comparison")
    parser.add_argument("--dataset", default="data/canonical_phi_continuations.jsonl")
    parser.add_argument("--manifest", default="data/canonical_phi_continuations.manifest.json")
    parser.add_argument("--train-index", default="experiments/checkpoints/train_only_association_index.pkl")
    parser.add_argument("--sourcebook", default="scratch/predictor_v2_external_sourcebook/sourcebook.sqlite")
    parser.add_argument("--sourcebook-manifest", default="scratch/predictor_v2_external_sourcebook/sourcebook.manifest.json")
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=list(POOL_SIZES))
    parser.add_argument("--max-dev-prompts", type=int, default=None, help="Plumbing smoke only; cannot be frozen")
    parser.add_argument("--time-limit-seconds", type=float, default=1.0)
    parser.add_argument("--out-json", default="scratch/predictor_v2_external_sourcebook/dev_comparison.json")
    parser.add_argument("--out-md", default="scratch/predictor_v2_external_sourcebook/dev_comparison.md")
    args = parser.parse_args()
    run_benchmark(dataset_path=args.dataset, manifest_path=args.manifest, train_index_path=args.train_index, sourcebook_path=args.sourcebook, sourcebook_manifest_path=args.sourcebook_manifest, out_json=args.out_json, out_md=args.out_md, pool_sizes=args.pool_sizes, max_dev_prompts=args.max_dev_prompts, time_limit_seconds=args.time_limit_seconds)


if __name__ == "__main__":
    main()
