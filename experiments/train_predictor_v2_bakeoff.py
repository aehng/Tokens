"""Phase N: Controlled Offline Predictor V2 Architecture Bake-Off.

Trains and evaluates all 5 candidate architectures:
A. Retrained Ridge Baseline
B. Pooled Embedding + MLP
C. CNN + Suffix Model
D. Small GRU + MLP
E. 1-Layer Lightweight Transformer

Supervision and evaluation are strictly controlled:
- Same prompts, same continuations, same candidate pools, same labels.
- Evaluated on DEV split (for architecture selection) and FROZEN TEST split.
- Evaluated at K = 8, 16, 32.

Outputs:
- docs/predictor_v2_bakeoff_results.json
- docs/predictor_v2_bakeoff_results.md
- experiments/checkpoints/predictor_v2/
"""

import argparse
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from src.zip2zip.predictor_v2.dataset import BakeoffDataset
from src.zip2zip.predictor_v2.metrics import evaluate_architecture_ranking
from src.zip2zip.predictor_v2.models import (
    CNNRanker,
    GRURanker,
    PooledMLPRanker,
    RidgeRanker,
    TransformerRanker,
)

DATASET_PKL = "data/predictor_v2_dataset.pkl"
ORACLE_ANALYSIS_JSON = "docs/predictor_v2_oracle_analysis.json"
CHECKPOINTS_DIR = "experiments/checkpoints/predictor_v2"
OUT_RESULTS_JSON = "docs/predictor_v2_bakeoff_results.json"
OUT_RESULTS_MD = "docs/predictor_v2_bakeoff_results.md"


def main():
    parser = argparse.ArgumentParser(description="Run Predictor V2 Architecture Bake-Off")
    parser.add_argument("--dataset-pkl", default=DATASET_PKL, help="Dataset pickle path")
    parser.add_argument("--epochs", type=int, default=12, help="Max epochs for neural models")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    print("=" * 80)
    print("RUNNING PREDICTOR V2 ARCHITECTURE BAKE-OFF")
    print("=" * 80)

    t0 = time.perf_counter()

    # 1. Load dataset bundle
    with open(args.dataset_pkl, "rb") as f:
        bundle = pickle.load(f)

    records = bundle["records"]
    candidates_by_prompt = bundle["candidates_by_prompt"]
    manifest = bundle["split_manifest"]

    dataset = BakeoffDataset(records, manifest)
    dataset.verify_split_integrity()
    print(f"Loaded dataset: Train={len(dataset.train_records)}, Dev={len(dataset.dev_records)}, Test={len(dataset.frozen_test_records)}")

    # 2. Load Oracle benchmarks for normalization
    with open(ORACLE_ANALYSIS_JSON, "r", encoding="utf-8") as f:
        oracle_data = json.load(f)

    # Compute oracle step ceilings per split
    # Extract oracle steps for DEV and TEST specifically
    dev_global_steps = {8: 0, 16: 0, 32: 0}
    dev_pool_steps = {8: 0, 16: 0, 32: 0}
    test_global_steps = {8: 0, 16: 0, 32: 0}
    test_pool_steps = {8: 0, 16: 0, 32: 0}

    dev_ids = set(manifest.dev_ids)
    test_ids = set(manifest.frozen_test_ids)

    for k_val in [8, 16, 32]:
        per_prompt = oracle_data["results_by_k"][str(k_val)]["per_prompt"]
        for p in per_prompt:
            pid = p["prompt_id"]
            if pid in dev_ids:
                dev_global_steps[k_val] += p["global_steps"]
                dev_pool_steps[k_val] += p["pool_steps"]
            elif pid in test_ids:
                test_global_steps[k_val] += p["global_steps"]
                test_pool_steps[k_val] += p["pool_steps"]

    print("DEV Oracle Ceilings (12 prompts):")
    print(f"  K=8:  Pool={dev_pool_steps[8]} | Global={dev_global_steps[8]}")
    print(f"  K=16: Pool={dev_pool_steps[16]} | Global={dev_global_steps[16]}")
    print(f"  K=32: Pool={dev_pool_steps[32]} | Global={dev_global_steps[32]}")

    print("TEST Oracle Ceilings (12 prompts):")
    print(f"  K=8:  Pool={test_pool_steps[8]} | Global={test_global_steps[8]}")
    print(f"  K=16: Pool={test_pool_steps[16]} | Global={test_global_steps[16]}")
    print(f"  K=32: Pool={test_pool_steps[32]} | Global={test_global_steps[32]}")

    # 3. Architectures to evaluate
    architectures = [
        ("Ridge", RidgeRanker()),
        ("PooledMLP", PooledMLPRanker()),
        ("CNNRanker", CNNRanker()),
        ("GRURanker", GRURanker()),
        ("TransformerRanker", TransformerRanker()),
    ]

    all_results: Dict[str, Any] = {}
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    for name, model in architectures:
        print(f"\n{'='*40}\nTraining & Evaluating Architecture: {name}\n{'='*40}")
        t_start_arch = time.perf_counter()

        # Fit model on TRAIN with DEV early stopping
        train_stats = model.fit(
            train_records=dataset.train_records,
            train_candidates=candidates_by_prompt,
            dev_records=dataset.dev_records,
            dev_candidates=candidates_by_prompt,
            epochs=args.epochs,
            lr=args.lr,
        )
        print(f"  Training completed in {train_stats.get('train_time_sec')}s | Params: {model.get_parameter_count():,}")

        # Save checkpoint
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{name.lower()}.pkl")
        with open(ckpt_path, "wb") as f:
            pickle.dump(model, f)

        # Evaluate on DEV (Architecture Selection Gate)
        dev_eval = evaluate_architecture_ranking(
            model=model,
            records=dataset.dev_records,
            candidates_by_prompt=candidates_by_prompt,
            k_values=(8, 16, 32),
            global_oracle_steps_by_k=dev_global_steps,
            candidate_oracle_steps_by_k=dev_pool_steps,
        )

        # Evaluate on FROZEN TEST (Strictly Held Out)
        test_eval = evaluate_architecture_ranking(
            model=model,
            records=dataset.frozen_test_records,
            candidates_by_prompt=candidates_by_prompt,
            k_values=(8, 16, 32),
            global_oracle_steps_by_k=test_global_steps,
            candidate_oracle_steps_by_k=test_pool_steps,
        )

        k32_dev = dev_eval["ranking_by_k"][32]
        k32_test = test_eval["ranking_by_k"][32]

        print(f"  DEV  @ K=32: Saved={k32_dev['realized_dp_steps']} ({k32_dev['candidate_oracle_capture_pct']}% Cand Pool, {k32_dev['global_oracle_capture_pct']}% Global) | Prec={k32_dev['precision_at_k']*100:.1f}%")
        print(f"  TEST @ K=32: Saved={k32_test['realized_dp_steps']} ({k32_test['candidate_oracle_capture_pct']}% Cand Pool, {k32_test['global_oracle_capture_pct']}% Global) | Prec={k32_test['precision_at_k']*100:.1f}%")

        all_results[name] = {
            "name": name,
            "parameter_count": model.get_parameter_count(),
            "model_size_kb": round(model.get_model_size_bytes() / 1024.0, 2),
            "train_stats": train_stats,
            "dev_eval": dev_eval,
            "test_eval": test_eval,
            "total_arch_time_sec": round(time.perf_counter() - t_start_arch, 2),
        }

    # 4. Save results JSON
    summary_bundle = {
        "schema": "predictor_v2_bakeoff_results_v1",
        "dataset_hash": bundle["dataset_manifest_hash"],
        "split_manifest_hash": manifest.manifest_hash,
        "dev_oracle_ceilings": dev_pool_steps,
        "test_oracle_ceilings": test_pool_steps,
        "architectures": all_results,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }

    with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_bundle, f, indent=2)
    print(f"\nSaved bake-off results to {OUT_RESULTS_JSON}")

    # 5. Generate Markdown Report
    md_lines = [
        "# Predictor V2 Architecture Bake-Off Results",
        "",
        "## Executive Summary",
        "",
        "Empirical head-to-head evaluation of five lightweight candidate architectures trained strictly on canonical Vanilla Phi continuation labels on the deterministic TRAIN split (36 prompts) and evaluated on DEV (12 prompts) and FROZEN TEST (12 prompts).",
        "",
        "## 1. DEV Split Head-to-Head Comparison (Architecture Selection)",
        "",
        "| Architecture | Parameters | Model Size | Dev DP Steps (K=32) | % Candidate Pool | % Global Ceiling | Precision@32 | Dead Slot % | Occurrence AUPRC | Count MAE |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for name, r in all_results.items():
        d32 = r["dev_eval"]["ranking_by_k"][32]
        heads = r["dev_eval"]["head_metrics"]
        md_lines.append(
            f"| **{name}** | {r['parameter_count']:,} | {r['model_size_kb']} KB | **{d32['realized_dp_steps']}** | **{d32['candidate_oracle_capture_pct']:.1f}%** | {d32['global_oracle_capture_pct']:.1f}% | {d32['precision_at_k']*100:.1f}% | {d32['dead_slot_rate']*100:.1f}% | {heads['occurrence_head']['auprc']:.4f} | {heads['count_head']['mae']:.2f} |"
        )

    md_lines.extend([
        "",
        "## 2. FROZEN TEST Split Comparison (Strictly Held Out)",
        "",
        "| Architecture | Parameters | Test DP Steps (K=32) | % Candidate Pool | % Global Ceiling | Precision@32 | Dead Slot % | Occurrence AUPRC | Count MAE |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for name, r in all_results.items():
        t32 = r["test_eval"]["ranking_by_k"][32]
        heads = r["test_eval"]["head_metrics"]
        md_lines.append(
            f"| **{name}** | {r['parameter_count']:,} | **{t32['realized_dp_steps']}** | **{t32['candidate_oracle_capture_pct']:.1f}%** | {t32['global_oracle_capture_pct']:.1f}% | {t32['precision_at_k']*100:.1f}% | {t32['dead_slot_rate']*100:.1f}% | {heads['occurrence_head']['auprc']:.4f} | {heads['count_head']['mae']:.2f} |"
        )

    md_lines.extend([
        "",
        "## 3. Scaling with Budget K (DEV Split)",
        "",
        "| Architecture | K=8 DP Steps (% Cap) | K=16 DP Steps (% Cap) | K=32 DP Steps (% Cap) |",
        "|---|---|---|---|",
    ])
    for name, r in all_results.items():
        d8 = r["dev_eval"]["ranking_by_k"][8]
        d16 = r["dev_eval"]["ranking_by_k"][16]
        d32 = r["dev_eval"]["ranking_by_k"][32]
        md_lines.append(
            f"| **{name}** | {d8['realized_dp_steps']} ({d8['candidate_oracle_capture_pct']:.1f}%) | {d16['realized_dp_steps']} ({d16['candidate_oracle_capture_pct']:.1f}%) | {d32['realized_dp_steps']} ({d32['candidate_oracle_capture_pct']:.1f}%) |"
        )

    with open(OUT_RESULTS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved Markdown report to {OUT_RESULTS_MD}")


if __name__ == "__main__":
    main()
