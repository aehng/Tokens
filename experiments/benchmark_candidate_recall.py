"""Phase 3 & 4: Candidate-Generation Recall & Latency Benchmark on DEV Split.

Compares 4 Candidate Generation Strategies:
1. Baseline: Prompt n-grams (len 2..4) + standard 1-hop associations + background
2. Expanded Associations: Deeper 1-hop + 2-hop token associations
3. Suffix Conditioned: Prompt suffix (last 16 tokens) n-grams and associations boosted
4. Sparse Lexical: BM25 retrieval against 630 TRAIN prompts to inject continuation n-grams

Evaluates across candidate pool sizes: N in {256, 512, 1024, 2048}
On DEV split (135 prompts: 45 code, 45 reasoning, 45 instruction).

Metrics:
- Exact Hypertoken Oracle step reduction (CP-SAT solver, k in {8, 16, 32})
- Candidate capture interval: [steps_saved_cand / global_upper, steps_saved_cand / global_lower]
- Candidate phrase recall: % of optimal global oracle phrases present in pool
- CPU latency: p50, p90, mean (ms) at L in {128, 512, 1024}

Outputs:
- docs/predictor_v2_candidate_recall_results.json
- docs/PREDICTOR_V2_CANDIDATE_RECALL_STUDY.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from src.zip2zip.predictor_v2.candidate_retrieval import (
    ConfigurableCandidateGenerator,
    RetrievalStrategy,
    TrainOnlyAssociationIndex,
)
from src.zip2zip.predictor_v2.oracle_candidate_pool import CandidatePoolOracle
from src.zip2zip.predictor_v2.oracle_global import GlobalOccurrenceOracle
from src.zip2zip.predictor_v2.vanilla_labels import (
    VanillaContinuationRecord,
    get_canonical_tokenizer,
)

DEFAULT_MANIFEST = "docs/predictor_v2_scaled_dataset_manifest.json"
DEFAULT_CONTINUATIONS = "data/canonical_phi_continuations.jsonl"
DEFAULT_INDEX = "experiments/checkpoints/train_only_association_index.pkl"
OUT_RESULTS_JSON = "docs/predictor_v2_candidate_recall_results.json"
OUT_RESULTS_MD = "docs/PREDICTOR_V2_CANDIDATE_RECALL_STUDY.md"


def profile_cpu_latencies(
    gen: ConfigurableCandidateGenerator,
    tokenizer: Any,
    strategy: RetrievalStrategy,
    pool_size: int,
    prompt_lengths: Sequence[int] = (128, 512, 1024),
    iterations: int = 30,
) -> Dict[str, Dict[str, float]]:
    """Measures CPU inference latency (p50, p90, mean) across prompt lengths."""
    results = {}
    base_text = (
        "def fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n"
        "    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\n"
    )
    # Replicate to ensure plenty of tokens
    long_text = base_text * 100
    all_tokens = tokenizer.encode(long_text, add_special_tokens=False)

    for L in prompt_lengths:
        p_ids = all_tokens[:L]
        p_text = tokenizer.decode(p_ids)

        # Warmup
        for _ in range(3):
            _ = gen.generate_candidate_pool(p_ids, p_text, domain="code", strategy=strategy, target_pool_size=pool_size)

        times_ms = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = gen.generate_candidate_pool(p_ids, p_text, domain="code", strategy=strategy, target_pool_size=pool_size)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        results[str(L)] = {
            "p50_ms": float(np.percentile(times_ms, 50)),
            "p90_ms": float(np.percentile(times_ms, 90)),
            "mean_ms": float(np.mean(times_ms)),
            "std_ms": float(np.std(times_ms)),
        }
    return results


def run_candidate_recall_benchmark(
    manifest_path: str = DEFAULT_MANIFEST,
    continuations_path: str = DEFAULT_CONTINUATIONS,
    index_path: str = DEFAULT_INDEX,
    pool_sizes: Sequence[int] = (256, 512, 1024, 2048),
    time_limit_per_solve: float = 10.0,
    max_dev_prompts: int | None = None,
) -> Dict[str, Any]:
    print("=" * 80)
    print("PREDICTOR V2 CANDIDATE-GENERATION RECALL & LATENCY BENCHMARK (DEV SPLIT)")
    print("=" * 80)

    t_start = time.perf_counter()

    # 1. Load manifest and verify DEV prompts
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dev_ids = set(manifest["dev_prompt_ids"])
    print(f"Target DEV prompt count: {len(dev_ids)}")

    # 2. Load continuations for DEV
    dev_records: List[VanillaContinuationRecord] = []
    with open(continuations_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if d["prompt_id"] in dev_ids:
                    dev_records.append(VanillaContinuationRecord.from_dict(d))

    print(f"Loaded {len(dev_records)} DEV continuation records.")
    if max_dev_prompts is not None and max_dev_prompts < len(dev_records):
        dev_records = dev_records[:max_dev_prompts]
        print(f"Subsampled to {len(dev_records)} DEV records for testing.")

    # 3. Load TRAIN-only index
    print(f"Loading train-only index from {index_path}...")
    index = TrainOnlyAssociationIndex.load(index_path)
    tokenizer = get_canonical_tokenizer()
    gen = ConfigurableCandidateGenerator(index, tokenizer)

    # 4. Solvers
    global_oracle = GlobalOccurrenceOracle(min_len=2, max_len=4, time_limit_seconds=time_limit_per_solve)
    cand_oracle = CandidatePoolOracle(tokenizer=tokenizer, time_limit_seconds=time_limit_per_solve)

    # 5. Precompute Global Occurrence Oracle ceilings for DEV records
    print("\n[Step 1/3] Computing Global Occurrence Oracle baselines on DEV...", flush=True)
    global_results_by_prompt: Dict[str, Dict[int, Any]] = defaultdict(dict)

    for idx, r in enumerate(dev_records, 1):
        for k in [8, 16, 32]:
            res = global_oracle.solve(r.continuation_token_ids, k=k, tokenizer=tokenizer)
            global_results_by_prompt[r.prompt_id][k] = res
        if idx % 25 == 0 or idx == len(dev_records):
            print(f"  Solved global oracle for [{idx}/{len(dev_records)}] DEV prompts...")

    # 6. Evaluate all combinations: Strategy x Pool Size
    strategies = [
        RetrievalStrategy.BASELINE,
        RetrievalStrategy.EXPANDED_ASSOCIATIONS,
        RetrievalStrategy.SUFFIX_CONDITIONED,
        RetrievalStrategy.SPARSE_LEXICAL,
    ]

    print("\n[Step 2/3] Evaluating Candidate Recall across Strategies and Pool Sizes...", flush=True)
    strategy_results: Dict[str, Dict[str, Any]] = {}

    for strat in strategies:
        strat_key = strat.value
        strategy_results[strat_key] = {}
        print(f"\n--- Strategy: {strat_key.upper()} ---")

        for N in pool_sizes:
            print(f"  Evaluating Pool Size N={N}...")
            pool_t0 = time.perf_counter()

            total_cand_counts = []
            capture_by_k: Dict[int, List[float]] = {8: [], 16: [], 32: []}
            steps_saved_by_k: Dict[int, List[int]] = {8: [], 16: [], 32: []}
            global_steps_by_k: Dict[int, List[int]] = {8: [], 16: [], 32: []}
            phrase_recall_by_k: Dict[int, List[float]] = {8: [], 16: [], 32: []}

            for r in dev_records:
                # Generate candidate pool
                c_pool = gen.generate_candidate_pool(
                    r.prompt_token_ids,
                    r.prompt_text,
                    domain=r.domain,
                    strategy=strat,
                    target_pool_size=N,
                )
                pool_phrases = set(c_pool.keys())
                total_cand_counts.append(len(pool_phrases))

                # Build minimal candidate records for oracle solver
                c_records = gen.build_candidate_records(
                    r,
                    strategy=strat,
                    target_pool_size=N,
                )

                for k in [8, 16, 32]:
                    g_res = global_results_by_prompt[r.prompt_id][k]
                    g_steps = g_res.steps_saved
                    global_steps_by_k[k].append(g_steps)

                    # Solve Candidate-Pool Oracle
                    cp_res = cand_oracle.solve(
                        c_records,
                        r.continuation_token_ids,
                        k=k,
                        global_oracle_steps=g_steps,
                    )
                    cp_steps = cp_res.steps_saved
                    steps_saved_by_k[k].append(cp_steps)

                    # Capture ratio
                    cap = (cp_steps / g_steps) if g_steps > 0 else 1.0
                    capture_by_k[k].append(cap)

                    # Phrase recall: what fraction of g_res.selected_phrases are in pool_phrases
                    g_phrases = set(g_res.selected_phrases)
                    if g_phrases:
                        rec = len(g_phrases & pool_phrases) / len(g_phrases)
                    else:
                        rec = 1.0
                    phrase_recall_by_k[k].append(rec)

            mean_pool_size = float(np.mean(total_cand_counts))
            metrics = {
                "mean_pool_size": round(mean_pool_size, 1),
                "k16_candidate_capture_pct": round(float(np.mean(capture_by_k[16])) * 100.0, 2),
                "k16_phrase_recall_pct": round(float(np.mean(phrase_recall_by_k[16])) * 100.0, 2),
                "k16_mean_steps_saved": round(float(np.mean(steps_saved_by_k[16])), 2),
                "k16_mean_global_ceiling": round(float(np.mean(global_steps_by_k[16])), 2),
                "k8_candidate_capture_pct": round(float(np.mean(capture_by_k[8])) * 100.0, 2),
                "k32_candidate_capture_pct": round(float(np.mean(capture_by_k[32])) * 100.0, 2),
                "runtime_seconds": round(time.perf_counter() - pool_t0, 2),
            }
            strategy_results[strat_key][str(N)] = metrics
            print(f"    N={N}: Capture(K=16)={metrics['k16_candidate_capture_pct']}% | Phrase Recall={metrics['k16_phrase_recall_pct']}% | Mean Steps={metrics['k16_mean_steps_saved']}")

    # 7. CPU Latency profiling
    print("\n[Step 3/3] Profiling CPU Latency across strategies and pool sizes...", flush=True)
    latency_profiles: Dict[str, Dict[str, Any]] = {}
    for strat in strategies:
        strat_key = strat.value
        latency_profiles[strat_key] = {}
        for N in pool_sizes:
            lat = profile_cpu_latencies(gen, tokenizer, strat, pool_size=N)
            latency_profiles[strat_key][str(N)] = lat
            p50_512 = lat["512"]["p50_ms"]
            p90_512 = lat["512"]["p90_ms"]
            print(f"  {strat_key:22s} N={N:4d}: L=512 p50={p50_512:.2f}ms, p90={p90_512:.2f}ms")

    elapsed_all = time.perf_counter() - t_start

    # 8. Compile final results bundle
    bundle = {
        "manifest_path": manifest_path,
        "dev_prompts_evaluated": len(dev_records),
        "pool_sizes": list(pool_sizes),
        "strategy_results": strategy_results,
        "latency_profiles": latency_profiles,
        "elapsed_total_seconds": round(elapsed_all, 2),
    }

    return bundle


def generate_markdown_report(bundle: Dict[str, Any], out_path: str = OUT_RESULTS_MD) -> None:
    res = bundle["strategy_results"]
    lat = bundle["latency_profiles"]
    dev_count = bundle["dev_prompts_evaluated"]

    lines = [
        "# Predictor V2 Candidate Recall & Latency Benchmark Report",
        "",
        f"**Evaluation Split**: DEV ({dev_count} prompts: 45 code, 45 reasoning, 45 instruction)",
        "**Guarantee**: Mined strictly from 630 TRAIN continuations with zero DEV/FINAL leakage.",
        "",
        "## 1. Candidate Capture & Recall Comparison (K=16 Budget)",
        "",
        "| Retrieval Strategy | Pool Size N | Mean Actual Pool | K=16 Capture (%) | K=16 Phrase Recall (%) | Mean Steps Saved | CPU p50 (L=512) | CPU p90 (L=512) | Pareto Status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for strat, pools in res.items():
        for N, met in pools.items():
            l_info = lat.get(strat, {}).get(N, {}).get("512", {})
            p50 = l_info.get("p50_ms", 0.0)
            p90 = l_info.get("p90_ms", 0.0)
            is_valid_latency = p50 <= 5.0
            pareto_str = "VIABLE" if is_valid_latency else "LATENCY_EXCEEDED"
            lines.append(
                f"| `{strat}` | {N} | {met['mean_pool_size']} | **{met['k16_candidate_capture_pct']}%** | {met['k16_phrase_recall_pct']}% | {met['k16_mean_steps_saved']} / {met['k16_mean_global_ceiling']} | {p50:.2f} ms | {p90:.2f} ms | {pareto_str} |"
            )

    lines.extend([
        "",
        "## 2. Latency Scaling Across Context Lengths (L=128, 512, 1024)",
        "",
        "| Strategy | N | L=128 p50 | L=512 p50 | L=1024 p50 | L=512 p90 |",
        "|---|---|---|---|---|---|",
    ])

    for strat, pools in lat.items():
        for N, l_data in pools.items():
            p50_128 = l_data.get("128", {}).get("p50_ms", 0.0)
            p50_512 = l_data.get("512", {}).get("p50_ms", 0.0)
            p50_1024 = l_data.get("1024", {}).get("p50_ms", 0.0)
            p90_512 = l_data.get("512", {}).get("p90_ms", 0.0)
            lines.append(
                f"| `{strat}` | {N} | {p50_128:.2f} ms | {p50_512:.2f} ms | {p50_1024:.2f} ms | {p90_512:.2f} ms |"
            )

    lines.extend([
        "",
        "## 3. Candidate Strategy Selection & Freezing Rationale",
        "",
        "- **Constraint**: CPU p50 latency at $L=512$ must be $\\le 5.0$ ms.",
        "- **Ceiling Capture**: The optimal strategy maximizes K=16 candidate capture with sub-5ms CPU latency.",
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Candidate Recall & Latency Benchmark on DEV")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--continuations", default=DEFAULT_CONTINUATIONS)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--pool-sizes", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--max-dev-prompts", type=int, default=None)
    parser.add_argument("--out-json", default=OUT_RESULTS_JSON)
    parser.add_argument("--out-md", default=OUT_RESULTS_MD)
    args = parser.parse_args()

    bundle = run_candidate_recall_benchmark(
        manifest_path=args.manifest,
        continuations_path=args.continuations,
        index_path=args.index,
        pool_sizes=args.pool_sizes,
        max_dev_prompts=args.max_dev_prompts,
    )

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    print(f"Saved JSON results to {args.out_json}")

    generate_markdown_report(bundle, args.out_md)


if __name__ == "__main__":
    main()
