"""DEV-only candidate recall benchmark for the canonical Predictor V2 contract."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.zip2zip.predictor_v2.candidate_retrieval import (
    ConfigurableCandidateGenerator,
    RetrievalStrategy,
    TrainOnlyAssociationIndex,
)
from src.zip2zip.predictor_v2.canonical_dataset import (
    load_canonical_dataset,
    load_manifest_tokenizer,
    sha256_file,
    sha256_json,
)
from src.zip2zip.predictor_v2.experiment_protocol import (
    make_experiment_manifest,
    predictor_v2_source_hashes,
    write_json_exclusive,
)
from src.zip2zip.predictor_v2.oracle_candidate_pool import CandidatePoolOracle
from src.zip2zip.predictor_v2.oracle_global import GlobalOccurrenceOracle
from src.zip2zip.predictor_v2.train_index import validate_train_index


K_VALUES = (8, 16, 32)
POOL_SIZES = (256, 512, 1024, 2048)
LATENCY_TARGET_P50_MS = 5.0
LATENCY_STRETCH_P50_MS = 10.0


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50_ms": None, "p90_ms": None, "p99_ms": None}
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def _capture_interval(candidate_lower: int, candidate_upper: int, global_lower: int, global_upper: int) -> dict[str, float | None]:
    if global_upper <= 0:
        return {"lower": None, "upper": None, "interpretation": "no_global_opportunity"}
    lower = candidate_lower / global_upper
    upper = min(1.0, candidate_upper / global_lower) if global_lower > 0 else 1.0
    return {"lower": lower, "upper": upper, "interpretation": "bound_derived"}


def _prompt_length_bin(length: int) -> str:
    if length <= 256:
        return "0-256"
    if length <= 512:
        return "257-512"
    if length <= 1024:
        return "513-1024"
    return "1025+"


def _memory_estimate(pool: dict[tuple[int, ...], dict[str, Any]]) -> int:
    size = sys.getsizeof(pool)
    for phrase, payload in pool.items():
        size += sys.getsizeof(phrase) + sys.getsizeof(payload)
        size += sys.getsizeof(payload.get("sources", set()))
        size += sum(sys.getsizeof(item) for item in payload.get("sources", ()))
    return size


def _candidate_configuration(
    strategy: RetrievalStrategy,
    pool_size: int,
    max_subtokens: int,
    train_index_provenance_sha256: str,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "strategy": strategy.value,
        "pool_size": pool_size,
        "max_subtokens": max_subtokens,
        "prompt_ngrams": {"minimum_length": 2, "base_weight": 8.0},
        "background_bank": {"limit": min(128, pool_size // 2), "weight_scale": 0.25},
        "filters": ["disabled_token_ids", "whitespace_only", "punctuation_only", "trailing_space_or_tab"],
        "truncation_order": "weight_desc_then_length_desc_then_token_tuple_desc",
        "implementation": "ConfigurableCandidateGenerator",
        "train_index_provenance_sha256": train_index_provenance_sha256,
    }
    strategy_config = {
        RetrievalStrategy.BASELINE: {"association_hops": 1, "associations_per_prompt_token": 24},
        RetrievalStrategy.EXPANDED_ASSOCIATIONS: {
            "one_hop_per_prompt_token": 48,
            "two_hop_seed_candidates": 32,
            "two_hop_associations_per_seed_token": 12,
            "two_hop_weight_attenuation": 0.15,
        },
        RetrievalStrategy.SUFFIX_CONDITIONED: {
            "suffix_window_tokens": 16,
            "suffix_prompt_ngram_weight_multiplier": 2.5,
            "suffix_association_weight_multiplier": 2.0,
            "suffix_associations_per_token": 32,
            "non_suffix_associations_per_token": 16,
        },
        RetrievalStrategy.SPARSE_LEXICAL: {
            "association_hops": 1,
            "associations_per_prompt_token": 24,
            "retrieval": "BM25 over TRAIN prompt documents",
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
            "nearest_train_prompts": 5,
            "bm25_score_cap": 10.0,
            "lexical_weight_scale": 2.0,
        },
    }
    config["strategy_parameters"] = strategy_config[strategy]
    return config


def _checkpoint_code_hashes() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[1]
    return predictor_v2_source_hashes(repo)


def _save_resume_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    persisted = dict(state)
    persisted["state_sha256"] = sha256_json(state)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temp:
            temp_name = temp.name
            json.dump(persisted, temp, indent=2, sort_keys=True)
            temp.write("\n")
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def run_benchmark(
    *,
    dataset_path: str,
    manifest_path: str,
    index_path: str,
    out_json: str,
    out_md: str,
    pool_sizes: Sequence[int] = POOL_SIZES,
    max_dev_prompts: int | None = None,
    time_limit_seconds: float = 10.0,
    resume_state_path: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    output_json_path = Path(out_json)
    output_md_path = Path(out_md)
    if (output_json_path.exists() or output_md_path.exists()) and not resume:
        raise FileExistsError("candidate benchmark outputs already exist; choose new paths to preserve prior evidence")
    if any(pool_size not in POOL_SIZES for pool_size in pool_sizes):
        raise ValueError(f"pool sizes must be selected from {POOL_SIZES}")
    startup_started = time.perf_counter()
    phase_started = time.perf_counter()
    views, data_manifest = load_canonical_dataset(dataset_path, manifest_path)
    dataset_validation_load_ms = (time.perf_counter() - phase_started) * 1000.0
    phase_started = time.perf_counter()
    index = TrainOnlyAssociationIndex.load(index_path)
    index_load_ms = (time.perf_counter() - phase_started) * 1000.0
    phase_started = time.perf_counter()
    validate_train_index(index, views)
    index_leakage_validation_ms = (time.perf_counter() - phase_started) * 1000.0
    index_provenance_sha256 = sha256_json(index.provenance)
    index_sha256 = sha256_file(index_path)
    phase_started = time.perf_counter()
    tokenizer = load_manifest_tokenizer(data_manifest)
    tokenizer_load_ms = (time.perf_counter() - phase_started) * 1000.0
    phase_started = time.perf_counter()
    generator = ConfigurableCandidateGenerator(index, tokenizer)
    generator_init_ms = (time.perf_counter() - phase_started) * 1000.0
    startup_cost = {
        "dataset_validation_and_load_ms": dataset_validation_load_ms,
        "index_load_ms": index_load_ms,
        "index_leakage_validation_ms": index_leakage_validation_ms,
        "tokenizer_load_ms": tokenizer_load_ms,
        "candidate_generator_init_ms": generator_init_ms,
        "total_until_generation_ready_ms": (time.perf_counter() - startup_started) * 1000.0,
        "measurement": "first invocation startup; excludes model loading because this is an offline candidate-generation benchmark",
    }
    records = list(views.dev)
    if max_dev_prompts is not None:
        if max_dev_prompts < 1:
            raise ValueError("--max-dev-prompts must be positive")
        records = records[:max_dev_prompts]
    full_dev = len(records) == len(views.dev)

    state_path = Path(resume_state_path or f"{out_json}.resume.json")
    run_config = {
        "dataset_sha256": views.dataset_sha256,
        "manifest_sha256": views.manifest_sha256,
        "train_split_sha256": views.train_split_sha256,
        "dev_split_sha256": views.dev_split_sha256,
        "train_index_sha256": sha256_file(index_path),
        "train_index_provenance_sha256": sha256_json(index.provenance),
        "pool_sizes": list(pool_sizes),
        "max_dev_prompts": max_dev_prompts,
        "dev_prompts_evaluated": len(records),
        "time_limit_seconds": time_limit_seconds,
        "k_values": list(K_VALUES),
        "strategies": [strategy.value for strategy in RetrievalStrategy],
        "code_sha256": _checkpoint_code_hashes(),
    }
    run_config_sha256 = sha256_json(run_config)
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"resume state does not exist: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema") != "predictor_v2_candidate_recall_resume_v1":
            raise ValueError("unsupported candidate benchmark resume-state schema")
        state_sha256 = state.pop("state_sha256", None)
        if state_sha256 != sha256_json(state):
            raise ValueError("candidate benchmark resume state checksum mismatch")
        if state.get("run_config_sha256") != run_config_sha256 or state.get("run_config") != run_config:
            raise ValueError("candidate benchmark resume state does not match dataset, index, code, or run configuration")
        completed = state.get("completed_combinations")
        if not isinstance(completed, dict):
            raise ValueError("candidate benchmark resume state has invalid completed combinations")
    else:
        if state_path.exists():
            raise FileExistsError(f"candidate benchmark resume state already exists; pass --resume to continue: {state_path}")
        state = {
            "schema": "predictor_v2_candidate_recall_resume_v1",
            "run_config": run_config,
            "run_config_sha256": run_config_sha256,
            "startup_cost": startup_cost,
            "completed_combinations": {},
        }
        _save_resume_state(state_path, state)
        completed = state["completed_combinations"]
    if resume and isinstance(state.get("startup_cost"), dict):
        startup_cost = state["startup_cost"]

    allowed_keys = {
        strategy.value: {str(size) for size in pool_sizes} for strategy in RetrievalStrategy
    }
    for strategy_name, by_size in completed.items():
        if strategy_name not in allowed_keys or not isinstance(by_size, dict) or not set(by_size).issubset(allowed_keys[strategy_name]):
            raise ValueError("candidate benchmark resume state contains unexpected strategy or pool-size results")

    all_combinations_complete = all(
        set(completed.get(strategy.value, {})) == allowed_keys[strategy.value]
        for strategy in RetrievalStrategy
    )
    final_bundle = state.get("final_bundle")
    if final_bundle is not None:
        if not all_combinations_complete or state.get("final_bundle_sha256") != sha256_json(final_bundle):
            raise ValueError("candidate benchmark final bundle is incomplete or has a checksum mismatch")
        if final_bundle.get("run_config_sha256") != run_config_sha256:
            raise ValueError("candidate benchmark final bundle does not match the requested resume configuration")
        if output_json_path.exists():
            existing_bundle = json.loads(output_json_path.read_text(encoding="utf-8"))
            if sha256_json(existing_bundle) != state["final_bundle_sha256"]:
                raise ValueError("existing candidate benchmark JSON does not match the checkpointed final bundle")
        if output_md_path.exists():
            if output_md_path.read_text(encoding="utf-8") != markdown_text(final_bundle):
                raise ValueError("existing candidate benchmark Markdown does not match its checkpointed JSON result")
        else:
            write_markdown(final_bundle, out_md)
        if not output_json_path.exists():
            write_json_exclusive(out_json, final_bundle)
        return final_bundle
    if output_json_path.exists():
        raise ValueError("candidate benchmark JSON exists but no checkpointed final bundle can validate it")

    global_oracle = GlobalOccurrenceOracle(min_len=2, max_len=index.max_subtokens, time_limit_seconds=time_limit_seconds)
    pool_oracle = CandidatePoolOracle(tokenizer=tokenizer, time_limit_seconds=time_limit_seconds)
    global_results: dict[tuple[str, int], Any] = {}
    if not all_combinations_complete:
        for record in records:
            for k in K_VALUES:
                global_results[(record.prompt_id, k)] = global_oracle.solve(
                    record.continuation_token_ids, k=k, tokenizer=tokenizer
                )

    strategies = list(RetrievalStrategy)
    results: dict[str, dict[str, Any]] = {
        strategy.value: dict(completed.get(strategy.value, {})) for strategy in strategies
    }
    rss_before = _rss_bytes()

    for strategy in strategies:
        for pool_size in pool_sizes:
            pool_key = str(pool_size)
            if pool_key in completed.get(strategy.value, {}):
                results[strategy.value][pool_key] = completed[strategy.value][pool_key]
                continue
            domain_accum: dict[tuple[int, str], dict[str, Any]] = defaultdict(
                lambda: {
                    "candidate_lower": 0,
                    "candidate_upper": 0,
                    "global_lower": 0,
                    "global_upper": 0,
                    "prompts": 0,
                    "unretrieved_global_incumbent_phrases": 0,
                    "unretrieved_by_length": Counter(),
                    "unweighted_missed_incumbent_phrase_weight": 0,
                }
            )
            candidate_lower = {k: 0 for k in K_VALUES}
            candidate_upper = {k: 0 for k in K_VALUES}
            global_lower = {k: 0 for k in K_VALUES}
            global_upper = {k: 0 for k in K_VALUES}
            exact_counts = {
                "global": 0,
                "global_solves": 0,
                "candidate_pool": 0,
                "candidate_pool_solves": 0,
            }
            global_exact_by_k = {k: 0 for k in K_VALUES}
            candidate_exact_by_k = {k: 0 for k in K_VALUES}
            latencies: list[float] = []
            latency_by_bin: dict[str, list[float]] = defaultdict(list)
            actual_pool_sizes: list[int] = []
            pool_memory_estimates: list[int] = []
            missed = {
                "unretrieved_global_incumbent_phrases": 0,
                "unretrieved_by_length": Counter(),
                "unweighted_missed_incumbent_phrase_weight": 0,
                "residual_gap_lower_steps": 0,
                "residual_gap_upper_steps": 0,
            }
            retrieval_health = {
                "global_incumbent_phrase_count": 0,
                "present_in_pool": 0,
                "present_rank_le_32": 0,
                "present_rank_33_to_128": 0,
                "present_rank_129_to_pool": 0,
                "generated_but_truncated": 0,
                "quality_filtered": Counter(),
                "not_generated_with_prompt_word_overlap": 0,
                "not_generated_without_observed_prompt_signal": 0,
                "rank_values": [],
                "pool_phrase_length_counts": Counter(),
                "pool_source_incidence_counts": Counter(),
                "quality_rejection_counts": Counter(),
                "pool_candidates": 0,
                "pool_slots": 0,
                "dead_candidates": 0,
                "background_only_dead_candidates": 0,
                "rank_examples": defaultdict(list),
                "domain_gap_examples": defaultdict(list),
            }

            for record in records:
                prompt_ids = tokenizer.encode(record.rendered_prompt_text, add_special_tokens=False)
                pool_diagnostics: dict[str, Any] = {}
                pool = generator.generate_candidate_pool(
                    prompt_ids,
                    record.rendered_prompt_text,
                    domain=record.domain,
                    strategy=strategy,
                    target_pool_size=pool_size,
                    diagnostics=pool_diagnostics,
                )
                latency_ms = float(pool_diagnostics["generation_latency_ms"])
                latencies.append(latency_ms)
                latency_by_bin[_prompt_length_bin(len(prompt_ids))].append(latency_ms)
                actual_pool_sizes.append(len(pool))
                pool_memory_estimates.append(_memory_estimate(pool))
                candidate_records = generator.build_candidate_records(
                    record.to_legacy_record(tokenizer),
                    strategy=strategy,
                    target_pool_size=pool_size,
                    candidate_pool=pool,
                )
                pool_phrases = set(pool)
                continuation_phrases = {
                    tuple(record.continuation_token_ids[i : i + length])
                    for length in range(2, min(generator.max_subtokens + 1, len(record.continuation_token_ids) + 1))
                    for i in range(len(record.continuation_token_ids) - length + 1)
                }
                retrieval_health["pool_candidates"] += len(pool)
                retrieval_health["pool_slots"] += pool_size
                retrieval_health["dead_candidates"] += len(pool_phrases - continuation_phrases)
                retrieval_health["pool_phrase_length_counts"].update(
                    pool_diagnostics["pool_phrase_length_counts"]
                )
                retrieval_health["pool_source_incidence_counts"].update(
                    pool_diagnostics["pool_source_counts"]
                )
                retrieval_health["quality_rejection_counts"].update(
                    Counter(pool_diagnostics["quality_rejections"].values())
                )
                sources_by_phrase = pool_diagnostics["sources_by_phrase"]
                background_only_dead = [
                    phrase for phrase in pool_phrases - continuation_phrases
                    if sources_by_phrase.get(phrase) == {"background_bank"}
                ]
                retrieval_health["background_only_dead_candidates"] += len(background_only_dead)

                def add_rank_example(category: str, phrase: tuple[int, ...], rank: int | None, reason: str | None = None) -> None:
                    examples = retrieval_health["rank_examples"][category]
                    if len(examples) >= 4:
                        return
                    phrase_text = tokenizer.decode(list(phrase))
                    examples.append(
                        {
                            "prompt_id": record.prompt_id,
                            "domain": record.domain,
                            "prompt_excerpt": record.task_prompt_text[:180],
                            "useful_phrase": phrase_text,
                            "candidate_rank": rank,
                            "quality_filter_reason": reason,
                            "candidate_sources": sorted(sources_by_phrase.get(phrase, set())),
                        }
                    )

                if any(k == 32 for k in K_VALUES):
                    global_incumbents = global_results[(record.prompt_id, 32)].selected_phrases
                    retrieval_health["global_incumbent_phrase_count"] += len(global_incumbents)
                    prompt_words = set(re.findall(r"\b\w+\b", record.task_prompt_text.lower()))
                    rank_by_phrase = pool_diagnostics["rank_by_phrase"]
                    quality_rejections = pool_diagnostics["quality_rejections"]
                    for phrase in global_incumbents:
                        rank = rank_by_phrase.get(phrase)
                        if phrase in pool_phrases:
                            retrieval_health["present_in_pool"] += 1
                            retrieval_health["rank_values"].append(rank)
                            if rank <= 32:
                                retrieval_health["present_rank_le_32"] += 1
                            elif rank <= 128:
                                retrieval_health["present_rank_33_to_128"] += 1
                                add_rank_example("present_rank_33_to_128", phrase, rank)
                            else:
                                retrieval_health["present_rank_129_to_pool"] += 1
                                add_rank_example("present_rank_129_to_pool", phrase, rank)
                        elif rank is not None:
                            retrieval_health["generated_but_truncated"] += 1
                            add_rank_example("generated_but_truncated", phrase, rank)
                        elif phrase in quality_rejections:
                            reason = quality_rejections[phrase]
                            retrieval_health["quality_filtered"][reason] += 1
                            add_rank_example("quality_filtered", phrase, None, reason)
                        else:
                            phrase_text = tokenizer.decode(list(phrase))
                            phrase_words = set(re.findall(r"\b\w+\b", phrase_text.lower()))
                            if phrase_words & prompt_words:
                                retrieval_health["not_generated_with_prompt_word_overlap"] += 1
                                add_rank_example("prompt_related_but_not_generated", phrase, None)
                            else:
                                retrieval_health["not_generated_without_observed_prompt_signal"] += 1
                                add_rank_example("no_observed_prompt_signal", phrase, None)

                for k in K_VALUES:
                    global_result = global_results[(record.prompt_id, k)]
                    candidate_result = pool_oracle.solve(
                        candidate_records,
                        record.continuation_token_ids,
                        k=k,
                        global_oracle_steps=global_result.steps_saved_lower_bound,
                    )
                    gl = int(global_result.steps_saved_lower_bound)
                    gu = int(global_result.steps_saved_upper_bound)
                    cl = int(candidate_result.steps_saved_lower_bound)
                    cu = int(candidate_result.steps_saved_upper_bound)
                    candidate_lower[k] += cl
                    candidate_upper[k] += cu
                    global_lower[k] += gl
                    global_upper[k] += gu
                    exact_counts["global_solves"] += 1
                    exact_counts["candidate_pool_solves"] += 1
                    exact_counts["global"] += int(global_result.is_exact)
                    exact_counts["candidate_pool"] += int(candidate_result.is_exact)
                    global_exact_by_k[k] += int(global_result.is_exact)
                    candidate_exact_by_k[k] += int(candidate_result.is_exact)

                    acc = domain_accum[(k, record.domain)]
                    acc["candidate_lower"] += cl
                    acc["candidate_upper"] += cu
                    acc["global_lower"] += gl
                    acc["global_upper"] += gu
                    acc["prompts"] += 1

                    if k == 32:
                        missing = [phrase for phrase in global_result.selected_phrases if phrase not in pool_phrases]
                        missed["unretrieved_global_incumbent_phrases"] += len(missing)
                        acc["unretrieved_global_incumbent_phrases"] += len(missing)
                        for phrase in missing:
                            missed["unretrieved_by_length"][str(len(phrase))] += 1
                            acc["unretrieved_by_length"][str(len(phrase))] += 1
                            weight = max(0, len(phrase) - 1)
                            missed["unweighted_missed_incumbent_phrase_weight"] += weight
                            acc["unweighted_missed_incumbent_phrase_weight"] += weight
                        missed["residual_gap_lower_steps"] += max(0, gl - cu)
                        missed["residual_gap_upper_steps"] += max(0, gu - cl)
                        lower_gap = max(0, gl - cl)
                        if lower_gap:
                            example_phrase = missing[0] if missing else (global_result.selected_phrases[0] if global_result.selected_phrases else ())
                            retrieval_health["domain_gap_examples"][record.domain].append(
                                {
                                    "prompt_id": record.prompt_id,
                                    "domain": record.domain,
                                    "candidate_minus_global_lower_bound_gap_steps": lower_gap,
                                    "global_oracle_incumbent_phrase_not_retrieved": tokenizer.decode(list(example_phrase)) if example_phrase else None,
                                }
                            )

            metric_by_k: dict[str, Any] = {}
            for k in K_VALUES:
                interval = _capture_interval(candidate_lower[k], candidate_upper[k], global_lower[k], global_upper[k])
                metric_by_k[str(k)] = {
                    "candidate_pool_oracle_steps": {
                        "lower_bound": candidate_lower[k],
                        "upper_bound": candidate_upper[k],
                    },
                    "global_oracle_steps": {
                        "lower_bound": global_lower[k],
                        "upper_bound": global_upper[k],
                        "all_prompts_exact": global_exact_by_k[k] == len(records),
                    },
                    "candidate_pool_oracle_all_prompts_exact": candidate_exact_by_k[k] == len(records),
                    "capture_interval": interval,
                    "capture_ratio": interval["lower"] if interval["lower"] == interval["upper"] else None,
                }
            latency_profiles = {
                "overall": _percentiles(latencies),
                "by_prompt_length_tokens": {
                    key: _percentiles(value) for key, value in sorted(latency_by_bin.items())
                },
                "measurement": "CPU candidate-pool API on actual DEV prompts; excludes oracle solve time",
                "p50_targets_ms": {
                    "working_target": LATENCY_TARGET_P50_MS,
                    "stretch_target": LATENCY_STRETCH_P50_MS,
                    "interpret_as_historical_guidance": True,
                },
            }
            domain_capture_k32 = {
                domain: _capture_interval(
                    data["candidate_lower"], data["candidate_upper"],
                    data["global_lower"], data["global_upper"],
                )
                for (k, domain), data in domain_accum.items()
                if k == 32
            }
            ranked_useful = retrieval_health["rank_values"]
            total_useful = retrieval_health["global_incumbent_phrase_count"]
            present_useful = retrieval_health["present_in_pool"]
            rank_percentiles = {
                f"p{percentile}_rank": float(np.percentile(ranked_useful, percentile))
                for percentile in (50, 90, 99)
            } if ranked_useful else {"p50_rank": None, "p90_rank": None, "p99_rank": None}
            missed_categories = {
                "A_no_observed_prompt_conditioning": {
                    "count": retrieval_health["not_generated_without_observed_prompt_signal"],
                    "interpretation": "heuristic only: no useful phrase word overlaps the task prompt and no implemented generator source produced it; this cannot prove the phrase was impossible to predict",
                    "examples": retrieval_health["rank_examples"]["no_observed_prompt_signal"],
                },
                "B_prompt_related_or_generated_but_absent": {
                    "count": retrieval_health["not_generated_with_prompt_word_overlap"] + retrieval_health["generated_but_truncated"],
                    "prompt_related_not_generated": retrieval_health["not_generated_with_prompt_word_overlap"],
                    "generated_but_truncated_by_pool_size": retrieval_health["generated_but_truncated"],
                    "examples": (
                        retrieval_health["rank_examples"]["prompt_related_but_not_generated"]
                        + retrieval_health["rank_examples"]["generated_but_truncated"]
                    )[:6],
                },
                "C_present_but_beyond_rank_32": {
                    "count": retrieval_health["present_rank_33_to_128"] + retrieval_health["present_rank_129_to_pool"],
                    "present_rank_33_to_128": retrieval_health["present_rank_33_to_128"],
                    "present_rank_129_to_pool": retrieval_health["present_rank_129_to_pool"],
                    "examples": (
                        retrieval_health["rank_examples"]["present_rank_33_to_128"]
                        + retrieval_health["rank_examples"]["present_rank_129_to_pool"]
                    )[:6],
                },
                "D_filtered_by_generic_quality_rules": {
                    "count": sum(retrieval_health["quality_filtered"].values()),
                    "filter_reasons": dict(retrieval_health["quality_filtered"]),
                    "examples": retrieval_health["rank_examples"]["quality_filtered"],
                },
                "E_generic_background_bank_clutter": {
                    "dead_background_only_candidates": retrieval_health["background_only_dead_candidates"],
                    "interpretation": "pool entries sourced only from the TRAIN global background bank that do not occur in this DEV continuation; descriptive clutter measure, not a proven cause of missed opportunity",
                },
                "F_domain_specific_retrieval_failure": {
                    "domain_capture_intervals": domain_capture_k32,
                    "lowest_lower_bound_capture_domain": min(
                        domain_capture_k32,
                        key=lambda domain: domain_capture_k32[domain]["lower"]
                        if domain_capture_k32[domain]["lower"] is not None else 1.0,
                    ) if domain_capture_k32 else None,
                    "largest_gap_examples": [
                        example
                        for domain in sorted(retrieval_health["domain_gap_examples"])
                        for example in sorted(
                            retrieval_health["domain_gap_examples"][domain],
                            key=lambda item: item["candidate_minus_global_lower_bound_gap_steps"],
                            reverse=True,
                        )[:2]
                    ][:6],
                },
                "G_other_oracle_bound_uncertainty": {
                    "non_exact_global_solves": exact_counts["global_solves"] - exact_counts["global"],
                    "non_exact_candidate_pool_solves": exact_counts["candidate_pool_solves"] - exact_counts["candidate_pool"],
                    "residual_gap_steps_interval": {
                        "lower_bound": missed["residual_gap_lower_steps"],
                        "upper_bound": missed["residual_gap_upper_steps"],
                    },
                },
            }
            retrieval_health_summary = {
                "useful_phrase_definition": "selected phrase in the DEV global-oracle incumbent at K=32; non-OPTIMAL incumbents are lower-bound solutions, not asserted global optima",
                "global_oracle_incumbent_phrase_count": total_useful,
                "useful_phrase_present_in_pool": present_useful,
                "useful_phrase_absent_from_pool": total_useful - present_useful,
                "useful_phrase_rank_disposition": {
                    "present_rank_1_to_32": retrieval_health["present_rank_le_32"],
                    "present_rank_33_to_128": retrieval_health["present_rank_33_to_128"],
                    "present_rank_129_to_pool": retrieval_health["present_rank_129_to_pool"],
                    "generated_but_truncated_by_pool_size": retrieval_health["generated_but_truncated"],
                    "filtered_by_quality_rule": sum(retrieval_health["quality_filtered"].values()),
                    "not_generated_with_prompt_word_overlap": retrieval_health["not_generated_with_prompt_word_overlap"],
                    "not_generated_without_observed_prompt_signal": retrieval_health["not_generated_without_observed_prompt_signal"],
                },
                "rank_percentiles_for_useful_phrases_present_in_pool": rank_percentiles,
                "quality_filter_rejection_counts": dict(retrieval_health["quality_rejection_counts"]),
                "candidate_phrase_length_distribution": dict(retrieval_health["pool_phrase_length_counts"]),
                "candidate_source_incidence_counts": dict(retrieval_health["pool_source_incidence_counts"]),
                "mean_pool_candidates": retrieval_health["pool_candidates"] / max(1, len(records)),
                "mean_pool_fill_fraction": retrieval_health["pool_candidates"] / max(1, retrieval_health["pool_slots"]),
                "dead_candidates": {
                    "count_across_prompt_pools": retrieval_health["dead_candidates"],
                    "fraction_of_pool_entries": retrieval_health["dead_candidates"] / max(1, retrieval_health["pool_candidates"]),
                    "background_only_dead_count": retrieval_health["background_only_dead_candidates"],
                },
                "missed_opportunity_categories": missed_categories,
            }
            result_entry = {
                "candidate_config": {
                    **_candidate_configuration(
                        strategy, pool_size, generator.max_subtokens, index_provenance_sha256
                    ),
                },
                "dev_prompts": len(records),
                "mean_actual_pool_size": float(statistics.mean(actual_pool_sizes)) if actual_pool_sizes else 0.0,
                "oracle_metrics_by_k": metric_by_k,
                "dev_domain_breakdown_by_k": {str(k): {} for k in K_VALUES},
                "candidate_generation_latency": latency_profiles,
                "retrieval_health": retrieval_health_summary,
                "memory": {
                    "index_file_bytes": Path(index_path).stat().st_size,
                    "index_serialized_bytes": len(pickle.dumps(index, protocol=pickle.HIGHEST_PROTOCOL)),
                    "mean_candidate_pool_python_bytes_estimate": float(statistics.mean(pool_memory_estimates)) if pool_memory_estimates else 0.0,
                    "process_rss_before_bytes": rss_before,
                    "process_rss_after_bytes": _rss_bytes(),
                    "candidate_pool_estimate_is_shallow": True,
                },
                "missed_opportunity_categories_k32": {
                    "unretrieved_global_incumbent_phrases": missed["unretrieved_global_incumbent_phrases"],
                    "unretrieved_phrase_count_by_length": dict(missed["unretrieved_by_length"]),
                    "unweighted_missed_incumbent_phrase_weight": missed["unweighted_missed_incumbent_phrase_weight"],
                    "residual_gap_steps_interval": {
                        "lower_bound": missed["residual_gap_lower_steps"],
                        "upper_bound": missed["residual_gap_upper_steps"],
                    },
                    "interpretation": "phrase counts use selected global-oracle incumbent phrases, which may not be optimal; phrase weights are descriptive and can overlap; residual gap is the oracle-bound measure",
                },
                "oracle_solver_counts": exact_counts,
            }
            for (k, domain), data in domain_accum.items():
                result_entry["dev_domain_breakdown_by_k"][str(k)][domain] = {
                    "prompts": data["prompts"],
                    "candidate_pool_oracle_steps": {
                        "lower_bound": data["candidate_lower"],
                        "upper_bound": data["candidate_upper"],
                    },
                    "global_oracle_steps": {
                        "lower_bound": data["global_lower"],
                        "upper_bound": data["global_upper"],
                    },
                    "capture_interval": _capture_interval(
                        data["candidate_lower"], data["candidate_upper"], data["global_lower"], data["global_upper"]
                    ),
                    "unretrieved_global_incumbent_phrases": data["unretrieved_global_incumbent_phrases"],
                    "unretrieved_phrase_count_by_length": dict(data["unretrieved_by_length"]),
                    "unweighted_missed_incumbent_phrase_weight": data["unweighted_missed_incumbent_phrase_weight"],
                }
            results[strategy.value][pool_key] = result_entry
            completed.setdefault(strategy.value, {})[pool_key] = result_entry
            _save_resume_state(state_path, state)

    run_manifest = make_experiment_manifest(
        stage="candidate_recall",
        views=views,
        model_revision=data_manifest["model_revision"],
        tokenizer_revision=data_manifest["tokenizer_revision"],
        candidate_config={
            strategy.value: {
                str(size): results[strategy.value][str(size)]["candidate_config"]
                for size in pool_sizes
            }
            for strategy in strategies
        },
        output_artifacts={"json": out_json, "markdown": out_md},
    )
    bundle = {
        "schema": "predictor_v2_candidate_recall_v1",
        "scope": "DEV" if full_dev else "DEV_SMOKE",
        "is_full_dev": full_dev,
        "dataset_sha256": views.dataset_sha256,
        "manifest_sha256": views.manifest_sha256,
        "train_split_sha256": views.train_split_sha256,
        "dev_split_sha256": views.dev_split_sha256,
        "train_index_sha256": index_sha256,
        "train_index_provenance_sha256": index_provenance_sha256,
        "startup_cost": startup_cost,
        "dev_prompts_evaluated": len(records),
        "dev_prompts_available": len(views.dev),
        "run_config_sha256": run_config_sha256,
        "pool_sizes": list(pool_sizes),
        "strategy_results": results,
        "gate_state": {
            "candidate_generator_frozen": False,
            "offline_architecture_shortlist_complete": False,
            "live_integration_gate_passed": False,
            "architecture_frozen": False,
            "final_evaluated": False,
            "candidate_generator_status": "offline_dev_evidence_only; post-live freeze is still required",
        },
        "run_manifest": run_manifest,
        "global_oracle_interpretation": "lower/upper bounds are preserved; only report exact ceilings when every solve is OPTIMAL",
    }
    state["final_bundle"] = bundle
    state["final_bundle_sha256"] = sha256_json(bundle)
    _save_resume_state(state_path, state)
    write_markdown(bundle, out_md)
    write_json_exclusive(out_json, bundle)
    return bundle


def markdown_text(bundle: dict[str, Any]) -> str:
    lines = [
        "# Canonical Predictor V2 Candidate Recall (DEV)",
        "",
        f"- Scope: `{bundle['scope']}` ({bundle['dev_prompts_evaluated']} / {bundle['dev_prompts_available']} DEV prompts)",
        f"- Dataset SHA-256: `{bundle['dataset_sha256']}`",
        f"- DEV split SHA-256: `{bundle['dev_split_sha256']}`",
        f"- Startup to generation-ready (first invocation): {bundle.get('startup_cost', {}).get('total_until_generation_ready_ms', 'n/a'):.3f} ms" if isinstance(bundle.get("startup_cost", {}).get("total_until_generation_ready_ms"), (int, float)) else "- Startup to generation-ready: not recorded",
        "- Candidate generator freeze: **not frozen**; offline DEV evidence must be combined with a small live integration check before freeze.",
        "- Global oracle values retain solver lower/upper bounds; no unproven exact ceiling is inferred.",
        "- Phrase-occurrence diagnosis uses each K=32 global-oracle incumbent; it does not prove phrases impossible to predict from prompt context.",
        "",
        "| Strategy | Pool | K=32 capture interval | Global steps LB–UB | Pool steps LB–UB | CPU p50/p90/p99 ms | Useful phrases in pool / total | Rank ≤32 | Absent | Dead pool % | Index bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    strategy_results = bundle["strategy_results"]
    for strategy in (item.value for item in RetrievalStrategy):
        by_size = strategy_results[strategy]
        for size in sorted(by_size, key=int):
            metrics = by_size[size]
            k32 = metrics["oracle_metrics_by_k"]["32"]
            interval = k32["capture_interval"]
            capture = "n/a" if interval["lower"] is None else f"{interval['lower']:.3f}–{interval['upper']:.3f}"
            latency = metrics["candidate_generation_latency"]["overall"]
            p50 = "n/a" if latency["p50_ms"] is None else f"{latency['p50_ms']:.3f}"
            p90 = "n/a" if latency["p90_ms"] is None else f"{latency['p90_ms']:.3f}"
            p99 = "n/a" if latency["p99_ms"] is None else f"{latency['p99_ms']:.3f}"
            health = metrics["retrieval_health"]
            disposition = health["useful_phrase_rank_disposition"]
            global_steps = k32["global_oracle_steps"]
            candidate_steps = k32["candidate_pool_oracle_steps"]
            global_range = f"{global_steps['lower_bound']}–{global_steps['upper_bound']}"
            candidate_range = f"{candidate_steps['lower_bound']}–{candidate_steps['upper_bound']}"
            useful = f"{health['useful_phrase_present_in_pool']}/{health['global_oracle_incumbent_phrase_count']}"
            dead = health["dead_candidates"]["fraction_of_pool_entries"] * 100.0
            lines.append(
                f"| {strategy} | {size} | {capture} | {global_range} | {candidate_range} | {p50}/{p90}/{p99} | {useful} | {disposition['present_rank_1_to_32']} | {health['useful_phrase_absent_from_pool']} | {dead:.1f}% | {metrics['memory']['index_file_bytes']} |"
            )
    lines += ["", "## Domain capture at K=32", "", "| Strategy | Pool | Code | Reasoning | Instruction |", "|---|---:|---:|---:|---:|"]
    for strategy in (item.value for item in RetrievalStrategy):
        for size in sorted(strategy_results[strategy], key=int):
            domains = strategy_results[strategy][size]["dev_domain_breakdown_by_k"]["32"]
            cells = []
            for domain in ("code", "reasoning", "instruction"):
                domain_result = domains.get(domain)
                if domain_result is None:
                    cells.append("n/a")
                    continue
                interval = domain_result["capture_interval"]
                cells.append("n/a" if interval["lower"] is None else f"{interval['lower']:.3f}–{interval['upper']:.3f}")
            lines.append(f"| {strategy} | {size} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(lines) + "\n"


def write_markdown(bundle: dict[str, Any], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_text(encoding="utf-8") == markdown_text(bundle):
            return
        raise FileExistsError(f"candidate benchmark Markdown already exists with different content: {target}")
    target.write_text(markdown_text(bundle), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark candidate recall on DEV only")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=list(POOL_SIZES))
    parser.add_argument("--max-dev-prompts", type=int, default=None, help="Smoke only; result cannot be frozen")
    parser.add_argument("--time-limit-seconds", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true", help="Continue from matching per-combination checkpoint state")
    parser.add_argument("--resume-state", default=None, help="Checkpoint path; defaults to <out-json>.resume.json")
    parser.add_argument("--out-json", default="experiments/results/predictor_v2_candidate_recall.json")
    parser.add_argument("--out-md", default="experiments/results/PREDICTOR_V2_CANDIDATE_RECALL.md")
    args = parser.parse_args()
    run_benchmark(
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        index_path=args.index,
        out_json=args.out_json,
        out_md=args.out_md,
        pool_sizes=args.pool_sizes,
        max_dev_prompts=args.max_dev_prompts,
        time_limit_seconds=args.time_limit_seconds,
        resume_state_path=args.resume_state,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
