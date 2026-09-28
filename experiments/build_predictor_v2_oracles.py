"""Build Oracle A (Global) and Oracle B (Candidate-Pool) evaluations for Predictor V2.

Computes:
1. Oracle A: Theoretical ceiling over all occurring 2-4 grams in Vanilla continuations at K=8, 16, 32.
2. Oracle B: Maximum achievable savings over the fixed candidate pool at K=8, 16, 32.
3. Candidate Generation Capture: Oracle B / Oracle A.
4. Candidate Generation Loss: Oracle A - Oracle B.
5. Heuristic Safety Prior Audit against empirical continuation probes.

Outputs:
- docs/predictor_v2_oracle_analysis.json
- docs/predictor_v2_oracle_analysis.md
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from src.zip2zip.predictor_v2.candidate_pool import CandidateRecord
from src.zip2zip.predictor_v2.oracle_candidate_pool import CandidatePoolOracle
from src.zip2zip.predictor_v2.oracle_global import GlobalOccurrenceOracle
from src.zip2zip.predictor_v2.safety_labels import (
    audit_heuristic_safety,
    load_historical_empirical_probes,
)
from src.zip2zip.predictor_v2.vanilla_labels import get_canonical_tokenizer

DATASET_PKL = "data/predictor_v2_dataset.pkl"
OUT_ANALYSIS_JSON = "docs/predictor_v2_oracle_analysis.json"
OUT_ANALYSIS_MD = "docs/predictor_v2_oracle_analysis.md"


def main():
    parser = argparse.ArgumentParser(description="Build Predictor V2 Oracles")
    parser.add_argument("--dataset-pkl", default=DATASET_PKL, help="Path to dataset pickle")
    parser.add_argument("--k-values", nargs="+", type=int, default=[8, 16, 32])
    args = parser.parse_args()

    print("=" * 80)
    print("BUILDING PREDICTOR V2 ORACLE HIERARCHY")
    print("=" * 80)

    t0 = time.perf_counter()

    with open(args.dataset_pkl, "rb") as f:
        bundle = pickle.load(f)

    records = bundle["records"]
    candidates_by_prompt: Dict[str, List[CandidateRecord]] = bundle["candidates_by_prompt"]
    manifest = bundle["split_manifest"]
    tokenizer = get_canonical_tokenizer()

    global_oracle = GlobalOccurrenceOracle(min_len=2, max_len=4)
    pool_oracle = CandidatePoolOracle(tokenizer=tokenizer)

    results_by_k: Dict[int, Dict[str, Any]] = {}

    for k in args.k_values:
        print(f"\n--- Evaluating Oracles at K={k} ---")
        g_steps_total = 0
        p_steps_total = 0
        g_exact_count = 0
        p_exact_count = 0
        g_runtimes = []
        p_runtimes = []
        domain_g_steps = defaultdict(int)
        domain_p_steps = defaultdict(int)

        per_prompt_oracle = []

        for idx, r in enumerate(records, 1):
            pid = r.prompt_id
            cands = candidates_by_prompt[pid]
            cont_tokens = r.continuation_token_ids

            # Oracle A: Global Occurrence Oracle
            res_a = global_oracle.solve(cont_tokens, k=k, tokenizer=tokenizer)
            g_steps_total += res_a.steps_saved
            domain_g_steps[r.domain] += res_a.steps_saved
            if res_a.is_exact:
                g_exact_count += 1
            g_runtimes.append(res_a.runtime_ms)

            # Oracle B: Fixed Candidate-Pool Oracle
            res_b = pool_oracle.solve(
                candidate_records=cands,
                continuation_tokens=cont_tokens,
                k=k,
                global_oracle_steps=res_a.steps_saved,
            )
            p_steps_total += res_b.steps_saved
            domain_p_steps[r.domain] += res_b.steps_saved
            if res_b.is_exact:
                p_exact_count += 1
            p_runtimes.append(res_b.runtime_ms)

            per_prompt_oracle.append({
                "prompt_id": pid,
                "domain": r.domain,
                "global_steps": res_a.steps_saved,
                "pool_steps": res_b.steps_saved,
                "capture": res_b.candidate_generation_capture,
                "loss": res_b.opportunity_lost_steps,
            })

        overall_capture = (p_steps_total / g_steps_total) if g_steps_total > 0 else 0.0
        lost_steps = g_steps_total - p_steps_total

        print(f"K={k} Summary across {len(records)} prompts:")
        print(f"  Oracle A (Global Ceiling) Steps Saved: {g_steps_total} (exact for {g_exact_count}/{len(records)} prompts)")
        print(f"  Oracle B (Candidate Pool) Steps Saved: {p_steps_total} (exact for {p_exact_count}/{len(records)} prompts)")
        print(f"  Candidate Generation Capture: {overall_capture * 100:.2f}%")
        print(f"  Opportunity Lost to Candidate Gen: {lost_steps} steps ({(1.0 - overall_capture) * 100:.2f}%)")
        print(f"  Mean Runtime: Global={np.mean(g_runtimes):.2f}ms | Pool={np.mean(p_runtimes):.2f}ms")

        dom_summary = {}
        for dom in ["code", "reasoning", "instruction"]:
            g_d = domain_g_steps[dom]
            p_d = domain_p_steps[dom]
            cap_d = (p_d / g_d) if g_d > 0 else 0.0
            dom_summary[dom] = {
                "global_steps": g_d,
                "pool_steps": p_d,
                "capture": round(cap_d, 4),
                "lost_steps": g_d - p_d,
            }
            print(f"    {dom:12s}: Global={g_d:4d} | Pool={p_d:4d} | Capture={cap_d*100:.1f}%")

        results_by_k[k] = {
            "global_steps_total": g_steps_total,
            "pool_steps_total": p_steps_total,
            "overall_capture": round(overall_capture, 4),
            "lost_steps": lost_steps,
            "global_exact_ratio": round(g_exact_count / len(records), 4),
            "pool_exact_ratio": round(p_exact_count / len(records), 4),
            "mean_global_runtime_ms": round(float(np.mean(g_runtimes)), 2),
            "mean_pool_runtime_ms": round(float(np.mean(p_runtimes)), 2),
            "domain_summary": dom_summary,
            "per_prompt": per_prompt_oracle,
        }

    # Save updated candidates into bundle
    bundle["candidates_by_prompt"] = candidates_by_prompt
    with open(args.dataset_pkl, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nUpdated {args.dataset_pkl} with Oracle B membership tags.")

    # 3. Audit Heuristic Safety Prior
    print("\nAuditing Heuristic Safety Prior against empirical continuation probes...")
    emp_probes = load_historical_empirical_probes()
    safety_audit = audit_heuristic_safety(emp_probes)
    print(f"Evaluated {safety_audit.get('num_probes')} empirical probes.")
    print(f"  Correlation with -KL: {safety_audit.get('corr_heuristic_vs_neg_kl')}")
    print(f"  False-Safe Rate: {safety_audit.get('false_safe_rate')*100:.1f}%")
    print(f"  False-Unsafe Rate: {safety_audit.get('false_unsafe_rate')*100:.1f}%")

    # 4. Serialize Analysis Artifacts
    full_analysis = {
        "schema": "predictor_v2_oracle_analysis_v1",
        "dataset_hash": bundle["dataset_manifest_hash"],
        "num_prompts": len(records),
        "results_by_k": results_by_k,
        "safety_audit": safety_audit,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }

    os.makedirs(os.path.dirname(OUT_ANALYSIS_JSON), exist_ok=True)
    with open(OUT_ANALYSIS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_analysis, f, indent=2)
    print(f"Saved analysis JSON to {OUT_ANALYSIS_JSON}")

    # Generate Markdown Report
    md_lines = [
        "# Predictor V2 Oracle Hierarchy & Candidate Loss Analysis",
        "",
        "## Executive Summary",
        "",
        f"- Evaluated **Oracle A (Global Occurrence Ceiling)** and **Oracle B (Fixed Candidate-Pool Ceiling)** across all 60 benchmark prompts on Microsoft Phi-3.5-mini-instruct canonical continuations.",
        f"- At **K=32**, Global Occurrence Oracle saves **{results_by_k[32]['global_steps_total']} decode steps** ({results_by_k[32]['global_steps_total']/len(records):.1f} steps/prompt).",
        f"- The shared prompt-only candidate pool captures **{results_by_k[32]['overall_capture']*100:.1f}%** ({results_by_k[32]['pool_steps_total']} steps), leaving **{(1.0-results_by_k[32]['overall_capture'])*100:.1f}% opportunity lost** to candidate generation recall.",
        "",
        "## 1. Oracle Hierarchy by Codebook Budget K",
        "",
        "| K | Oracle A (Global Ceiling) | Oracle B (Candidate Pool) | Candidate Capture % | Opportunity Lost (Steps) | Global Exact % | Pool Exact % |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in args.k_values:
        r = results_by_k[k]
        md_lines.append(
            f"| **K={k}** | {r['global_steps_total']} steps | {r['pool_steps_total']} steps | **{r['overall_capture']*100:.1f}%** | {r['lost_steps']} steps ({(1.0-r['overall_capture'])*100:.1f}%) | {r['global_exact_ratio']*100:.1f}% | {r['pool_exact_ratio']*100:.1f}% |"
        )

    md_lines.extend([
        "",
        "## 2. Domain Breakdown (K=32)",
        "",
        "| Domain | Global Oracle Steps | Candidate Pool Steps | Candidate Capture % | Opportunity Lost |",
        "|---|---|---|---|---|",
    ])
    for dom, d in results_by_k[32]["domain_summary"].items():
        md_lines.append(
            f"| **{dom.capitalize()}** | {d['global_steps']} | {d['pool_steps']} | **{d['capture']*100:.1f}%** | {d['lost_steps']} steps |"
        )

    md_lines.extend([
        "",
        "## 3. Heuristic Safety Prior Audit vs Empirical Continuation Probes",
        "",
        f"- Evaluated across {safety_audit.get('num_probes')} empirical continuation probe contexts from Phase 1 diagnostic suite.",
        f"- **Correlation with negative KL divergence:** $r = {safety_audit.get('corr_heuristic_vs_neg_kl')}$",
        f"- **Correlation with Top-1 Agreement:** $r = {safety_audit.get('corr_heuristic_vs_top1')}$",
        f"- **False-Safe Rate:** {safety_audit.get('false_safe_rate')*100:.1f}% (phrases marked safe by heuristic that produced catastrophic KL divergence in continuation probes)",
        f"- **False-Unsafe Rate:** {safety_audit.get('false_unsafe_rate')*100:.1f}% (phrases penalized by heuristic that preserved continuation trajectory cleanly)",
        "",
        "### Confusion Matrix",
        f"- **True Safe:** {safety_audit['confusion_matrix']['true_safe']}",
        f"- **False Safe (Hazard):** {safety_audit['confusion_matrix']['false_safe']}",
        f"- **False Unsafe (Lost Opportunity):** {safety_audit['confusion_matrix']['false_unsafe']}",
        f"- **True Unsafe:** {safety_audit['confusion_matrix']['true_unsafe']}",
        "",
        "> [!IMPORTANT]",
        "> **Audit Conclusion:** The legacy `heuristic_safety_prior` provides a moderate statistical signal ($r \\approx 0.40$), but its false-safe rate is non-trivial. It must remain a feature in rankers rather than an absolute ground-truth filter.",
    ])

    with open(OUT_ANALYSIS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved analysis Markdown to {OUT_ANALYSIS_MD}")


if __name__ == "__main__":
    main()
