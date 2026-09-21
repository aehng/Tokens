"""Training Data Coverage & Density Audit (Caution 3).

Audits response coverage under strict prompt-only predictive codebook selection (K=32).

For each training sample:
1. Run CappedPredictorPolicy on prompt ONLY.
2. Segment prompt with the resulting codebook.
3. Segment response with the SAME prompt-selected codebook.
4. Record:
   - Whether response contains >= 1 hypertoken (coverage flag)
   - Number of hypertokens in response
   - Target hypertoken density: (hypertokens) / (compressed response tokens)
   - Tokens saved in prompt and response
   - Category breakdown of matched phrases (content vs structural/numeric)
5. Report aggregated statistics across all samples and by domain.
"""

import os
import sys
import json
import pickle
import time
from collections import defaultdict
from typing import Dict, List, Any

from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from zip2zip.predictor_policy import CappedPredictorPolicy, is_structural_or_numeric
from src.evaluation.offline_segmenter import segment_tokens_dp

DATA_PATH = "data/train.jsonl"
PREDICTOR_PATH = "experiments/checkpoints/cached_predictor.pkl"
OUTPUT_PATH = "experiments/checkpoints/training_data_audit_results.json"
SAMPLE_LIMIT = 1000  # Thorough 1,000-sample audit


def run_audit(sample_limit: int = SAMPLE_LIMIT) -> Dict[str, Any]:
    print(f"\n{'='*80}")
    print(f"TRAINING DATA COVERAGE & DENSITY AUDIT (K=32, Sample Limit: {sample_limit})")
    print(f"{'='*80}\n")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

    print(f"Loading predictor from {PREDICTOR_PATH}...")
    with open(PREDICTOR_PATH, "rb") as f:
        predictor_raw = pickle.load(f)

    # If predictor_raw is FastPredictor, get its index; else it is index
    p_index = getattr(predictor_raw, "index", predictor_raw)
    policy = CappedPredictorPolicy(p_index, tokenizer, budget=32, max_structural_slots=8)

    print(f"Loading samples from {DATA_PATH}...")
    samples = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_limit:
                break
            samples.append(json.loads(line.strip()))

    print(f"Auditing {len(samples)} samples across domains...\n")

    records: List[Dict[str, Any]] = []
    domain_stats = defaultdict(lambda: {
        "count": 0,
        "covered_resp_ge1": 0,
        "covered_resp_ge2": 0,
        "total_prompt_tokens": 0,
        "saved_prompt_tokens": 0,
        "total_resp_tokens": 0,
        "saved_resp_tokens": 0,
        "total_resp_hypers": 0,
        "total_compressed_resp_tokens": 0,
        "content_hypers": 0,
        "structural_hypers": 0,
    })

    t0 = time.time()
    for idx, s in enumerate(samples):
        dom = s.get("domain", "general")
        prompt_text = s["prompt"]
        resp_text = s["response"]

        p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        r_ids = tokenizer.encode(resp_text, add_special_tokens=False)

        if not p_ids or not r_ids:
            continue

        # 1. Select codebook using PROMPT ONLY
        codebook, meta = policy.select_codebook(p_ids)
        phrases_set = set(codebook.keys())

        # 2. Segment prompt
        p_comp_len, p_tiles, _ = segment_tokens_dp(p_ids, phrases_set)
        p_saved = len(p_ids) - p_comp_len

        # 3. Segment response with the SAME codebook
        r_comp_len, r_tiles, _ = segment_tokens_dp(r_ids, phrases_set)
        r_saved = len(r_ids) - r_comp_len

        # Response hypertoken count
        resp_hypers = [tuple(t) for t in r_tiles if len(t) > 1 and tuple(t) in phrases_set]
        num_resp_hypers = len(resp_hypers)
        has_ge1 = num_resp_hypers >= 1
        has_ge2 = num_resp_hypers >= 2

        density = num_resp_hypers / r_comp_len if r_comp_len > 0 else 0.0

        n_content = sum(1 for h in resp_hypers if not is_structural_or_numeric(h, tokenizer))
        n_struct = num_resp_hypers - n_content

        rec = {
            "id": s.get("id", f"sample_{idx}"),
            "domain": dom,
            "prompt_len": len(p_ids),
            "prompt_saved": p_saved,
            "resp_len": len(r_ids),
            "resp_compressed_len": r_comp_len,
            "resp_saved": r_saved,
            "resp_hypers_count": num_resp_hypers,
            "has_resp_hyper": has_ge1,
            "hyper_density": round(density, 4),
            "content_hypers": n_content,
            "structural_hypers": n_struct,
        }
        records.append(rec)

        ds = domain_stats[dom]
        ds["count"] += 1
        if has_ge1:
            ds["covered_resp_ge1"] += 1
        if has_ge2:
            ds["covered_resp_ge2"] += 1
        ds["total_prompt_tokens"] += len(p_ids)
        ds["saved_prompt_tokens"] += p_saved
        ds["total_resp_tokens"] += len(r_ids)
        ds["saved_resp_tokens"] += r_saved
        ds["total_resp_hypers"] += num_resp_hypers
        ds["total_compressed_resp_tokens"] += r_comp_len
        ds["content_hypers"] += n_content
        ds["structural_hypers"] += n_struct

        if (idx + 1) % 250 == 0:
            print(f"  Processed {idx + 1}/{len(samples)} samples ({time.time() - t0:.1f}s)...")

    # Overall aggregates
    tot_samples = len(records)
    tot_covered_ge1 = sum(1 for r in records if r["has_resp_hyper"])
    tot_covered_ge2 = sum(1 for r in records if r["resp_hypers_count"] >= 2)
    tot_resp_base = sum(r["resp_len"] for r in records)
    tot_resp_saved = sum(r["resp_saved"] for r in records)
    tot_resp_comp = sum(r["resp_compressed_len"] for r in records)
    tot_hypers = sum(r["resp_hypers_count"] for r in records)
    tot_content = sum(r["content_hypers"] for r in records)
    tot_struct = sum(r["structural_hypers"] for r in records)

    overall_coverage_pct = (tot_covered_ge1 / tot_samples) * 100.0 if tot_samples else 0.0
    overall_ge2_pct = (tot_covered_ge2 / tot_samples) * 100.0 if tot_samples else 0.0
    overall_density = (tot_hypers / tot_resp_comp) * 100.0 if tot_resp_comp else 0.0
    micro_resp_comp_pct = (tot_resp_saved / tot_resp_base) * 100.0 if tot_resp_base else 0.0

    print(f"\n{'='*80}")
    print("TRAINING DATA AUDIT SUMMARY:")
    print(f"  Total Samples Audited:           {tot_samples:,}")
    print(f"  Samples with >= 1 Response Hyper: {tot_covered_ge1:,} ({overall_coverage_pct:.1f}%)")
    print(f"  Samples with >= 2 Response Hypers:{tot_covered_ge2:,} ({overall_ge2_pct:.1f}%)")
    print(f"  Target Hypertoken Density:       {overall_density:.2f}% of compressed response tokens")
    print(f"  Micro Response Compression:      {micro_resp_comp_pct:.2f}% tokens saved")
    print(f"  Avg Response Tokens Saved/Sample: {tot_resp_saved / tot_samples:.2f} tokens")
    print(f"  Category Ratio (Emitted Hypers): {tot_content:,} content ({tot_content/(tot_hypers or 1)*100:.1f}%) | "
          f"{tot_struct:,} structural/numeric ({tot_struct/(tot_hypers or 1)*100:.1f}%)")
    print(f"{'='*80}")

    print("\nBREAKDOWN BY DOMAIN:")
    print(f"{'Domain':<15} {'Samples':>8} {'Covered >=1':>12} {'Covered >=2':>12} {'Density%':>10} {'MicroComp%':>12}")
    print("-" * 75)
    domain_summary = {}
    for dom, ds in sorted(domain_stats.items()):
        c = ds["count"]
        cov1 = (ds["covered_resp_ge1"] / c) * 100.0 if c else 0.0
        cov2 = (ds["covered_resp_ge2"] / c) * 100.0 if c else 0.0
        dens = (ds["total_resp_hypers"] / ds["total_compressed_resp_tokens"]) * 100.0 if ds["total_compressed_resp_tokens"] else 0.0
        mcomp = (ds["saved_resp_tokens"] / ds["total_resp_tokens"]) * 100.0 if ds["total_resp_tokens"] else 0.0
        print(f"{dom:<15} {c:>8d} {cov1:>11.1f}% {cov2:>11.1f}% {dens:>9.2f}% {mcomp:>11.2f}%")
        domain_summary[dom] = {
            "samples": c,
            "covered_ge1_pct": round(cov1, 2),
            "covered_ge2_pct": round(cov2, 2),
            "density_pct": round(dens, 2),
            "micro_compression_pct": round(mcomp, 2),
            "content_hypers": ds["content_hypers"],
            "structural_hypers": ds["structural_hypers"],
        }
    print(f"{'='*80}\n")

    result = {
        "overall": {
            "samples_audited": tot_samples,
            "covered_resp_ge1_count": tot_covered_ge1,
            "covered_resp_ge1_pct": round(overall_coverage_pct, 2),
            "covered_resp_ge2_count": tot_covered_ge2,
            "covered_resp_ge2_pct": round(overall_ge2_pct, 2),
            "target_hypertoken_density_pct": round(overall_density, 2),
            "micro_response_compression_pct": round(micro_resp_comp_pct, 2),
            "avg_tokens_saved_per_response": round(tot_resp_saved / (tot_samples or 1), 2),
            "content_hypers_total": tot_content,
            "structural_hypers_total": tot_struct,
        },
        "domain_breakdown": domain_summary,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved audit report to {OUTPUT_PATH}")

    return result


if __name__ == "__main__":
    run_audit(SAMPLE_LIMIT)
