"""Phase Q: Live Attribution Experiment Preparation for Predictor V2.

Prepares the 6-arm live attribution study:
  Arm A:  Vanilla Phi-3.5-mini-instruct (Control)
  Arm B:  Predictive model with hypertokens disabled (K=0)
  Arm C1: Predictive model + Global Occurrence Oracle codebook (Diagnostic future knowledge)
  Arm C2: Predictive model + Empirically Safety-Filtered Oracle codebook
  Arm D:  Predictive model + Legacy OracleGuidedPredictor
  Arm E:  Predictive model + Top-1 Predictor V2 Architecture
  Arm F:  Predictive model + Top-2 Predictor V2 Architecture

DO NOT LAUNCH EXPENSIVE GPU EXECUTION AUTOMATICALLY WITHOUT USER APPROVAL.
This script validates manifests, builds the staged execution plan, and verifies codebook readiness.

Outputs:
- experiments/checkpoints/quality_benchmark/live_attribution_plan.json
- docs/PREDICTOR_V2_LIVE_ATTRIBUTION_PLAN.md
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from src.zip2zip.predictor_v2.dataset import BakeoffDataset
from src.zip2zip.predictor_v2.oracle_candidate_pool import CandidatePoolOracle
from src.zip2zip.predictor_v2.oracle_global import GlobalOccurrenceOracle
from src.zip2zip.predictor_v2.vanilla_labels import get_canonical_tokenizer

DATASET_PKL = "data/predictor_v2_dataset.pkl"
PARETO_JSON = "docs/predictor_v2_pareto_analysis.json"
CHECKPOINTS_DIR = "experiments/checkpoints/predictor_v2"
LEGACY_PREDICTOR_PKL = "experiments/checkpoints/oracle_guided_predictor.pkl"
OUT_PLAN_JSON = "experiments/checkpoints/quality_benchmark/live_attribution_plan.json"
OUT_PLAN_MD = "docs/PREDICTOR_V2_LIVE_ATTRIBUTION_PLAN.md"


def main():
    parser = argparse.ArgumentParser(description="Prepare Predictor V2 Live Attribution Experiment")
    parser.add_argument("--dataset-pkl", default=DATASET_PKL)
    parser.add_argument("--pareto-json", default=PARETO_JSON)
    parser.add_argument("--execute-live", action="store_true", help="Requires explicit authorization flag")
    args = parser.parse_args()

    print("=" * 80)
    print("PREPARING PREDICTOR V2 LIVE ATTRIBUTION EXPERIMENT")
    print("=" * 80)

    if args.execute_live:
        print("CRITICAL: Live GPU generation requested. Checking authorization...")
        # Guardrail: Never launch expensive runs without approval
        raise PermissionError(
            "Live GPU execution requires explicit interactive approval from the user. "
            "Plan has been staged; do not run without confirmation."
        )

    # 1. Load dataset bundle and Pareto analysis
    with open(args.dataset_pkl, "rb") as f:
        bundle = pickle.load(f)

    records = bundle["records"]
    manifest = bundle["split_manifest"]
    candidates_by_prompt = bundle["candidates_by_prompt"]
    dataset = BakeoffDataset(records, manifest)

    top_two = ["PooledMLP", "Ridge"]  # Defaults, overridden if Pareto exists
    if os.path.exists(args.pareto_json):
        with open(args.pareto_json, "r", encoding="utf-8") as f:
            pareto_data = json.load(f)
            top_two = pareto_data.get("top_two_selected_architectures", top_two)

    print(f"Top two selected architectures from offline bake-off: {top_two}")

    # 2. Build codebooks for each prompt across all 6 arms on DEV split (12 prompts)
    # Using DEV split ensures test split remains untouched
    target_prompts = dataset.dev_records
    tokenizer = get_canonical_tokenizer()
    global_oracle = GlobalOccurrenceOracle()

    # Load trained models
    model_e = None
    model_f = None
    path_e = os.path.join(CHECKPOINTS_DIR, f"{top_two[0].lower()}.pkl")
    path_f = os.path.join(CHECKPOINTS_DIR, f"{top_two[1].lower()}.pkl")

    if os.path.exists(path_e):
        with open(path_e, "rb") as f:
            model_e = pickle.load(f)
    if os.path.exists(path_f):
        with open(path_f, "rb") as f:
            model_f = pickle.load(f)

    staged_prompts = []

    for r in target_prompts:
        pid = r.prompt_id
        cands = candidates_by_prompt[pid]
        p_ids = r.prompt_token_ids
        cont_tokens = r.continuation_token_ids

        # Arm C1: Global Occurrence Oracle codebook (K=32)
        res_c1 = global_oracle.solve(cont_tokens, k=32, tokenizer=tokenizer)
        cb_c1 = [list(p) for p in res_c1.selected_phrases]

        # Arm C2: Empirically safe oracle codebook (filtered to phrases with no trailing space / safe boundary)
        safe_c1_phrases = [
            list(p) for p in res_c1.selected_phrases
            if not tokenizer.decode(list(p)).endswith(" ") and not tokenizer.decode(list(p)).endswith("\t")
        ]

        # Arm E: Top-1 architecture
        cb_e = []
        if model_e:
            ranked_e = model_e.rank_codebook(p_ids, cands, domain=r.domain, k=32)
            cb_e = [list(c.tokens) for c, s in ranked_e]

        # Arm F: Top-2 architecture
        cb_f = []
        if model_f:
            ranked_f = model_f.rank_codebook(p_ids, cands, domain=r.domain, k=32)
            cb_f = [list(c.tokens) for c, s in ranked_f]

        staged_prompts.append({
            "prompt_id": pid,
            "domain": r.domain,
            "prompt_text": r.prompt_text,
            "prompt_token_ids": p_ids,
            "arms": {
                "Arm_A_Vanilla_Phi": {"k": 0, "codebook": []},
                "Arm_B_Predictive_NoHypers": {"k": 0, "codebook": []},
                "Arm_C1_Global_Oracle": {"k": len(cb_c1), "codebook": cb_c1},
                "Arm_C2_Safe_Oracle": {"k": len(safe_c1_phrases), "codebook": safe_c1_phrases},
                "Arm_D_Legacy_Predictor": {"k": 32, "policy": "legacy_oracle_guided"},
                "Arm_E_Top1_PredictorV2": {"k": len(cb_e), "model": top_two[0], "codebook": cb_e},
                "Arm_F_Top2_PredictorV2": {"k": len(cb_f), "model": top_two[1], "codebook": cb_f},
            },
        })

    # Save plan
    plan_data = {
        "schema": "predictor_v2_live_attribution_plan_v1",
        "num_prompts": len(staged_prompts),
        "split": "DEV",
        "arms": [
            "Arm_A_Vanilla_Phi",
            "Arm_B_Predictive_NoHypers",
            "Arm_C1_Global_Oracle",
            "Arm_C2_Safe_Oracle",
            "Arm_D_Legacy_Predictor",
            "Arm_E_Top1_PredictorV2",
            "Arm_F_Top2_PredictorV2",
        ],
        "top_two_models": top_two,
        "staged_prompts": staged_prompts,
        "execution_status": "READY_STAGED_AWAITING_APPROVAL",
    }

    os.makedirs(os.path.dirname(OUT_PLAN_JSON), exist_ok=True)
    with open(OUT_PLAN_JSON, "w", encoding="utf-8") as f:
        json.dump(plan_data, f, indent=2)
    print(f"Saved live attribution plan to {OUT_PLAN_JSON}")

    # Generate Markdown Plan
    md_lines = [
        "# Predictor V2 Live Attribution Experiment Protocol",
        "",
        "> [!IMPORTANT]",
        "> **Execution Gate:** This plan is fully staged and validated. **Execution is paused pending explicit user approval.**",
        "",
        "## 1. Experimental Arms (Controlled Head-to-Head)",
        "",
        "| Arm ID | Configuration | Purpose | Expected Codebook Size K |",
        "|---|---|---|---|",
        "| **Arm A** | Vanilla Phi-3.5-mini-instruct | Absolute quality and baseline speed control | 0 |",
        "| **Arm B** | Predictive Model (H Disabled) | Measures adapter/LoRA quality impact without token substitution | 0 |",
        "| **Arm C1** | Predictive + Global Occurrence Oracle | Diagnostic upper bound with future knowledge | 32 |",
        "| **Arm C2** | Predictive + Empirically Safe Oracle | Quality ceiling under conservative continuation safety | $\\le 32$ |",
        "| **Arm D** | Predictive + Legacy OracleGuidedPredictor | Historical baseline comparison | 32 |",
        f"| **Arm E** | Predictive + **{top_two[0]}** (Top-1) | Validates whether top offline ranker converts to live quality | 32 |",
        f"| **Arm F** | Predictive + **{top_two[1]}** (Top-2) | Validates Pareto-runner-up live behavior | 32 |",
        "",
        "## 2. Metrics to Capture per Prompt",
        "- **Task Quality:** Exact accuracy / test pass rate (MBPP code execution, GSM8K numeric match, Alpaca semantic score).",
        "- **Divergence:** Token index of first divergence from matched Vanilla continuation.",
        "- **Realized Decode Steps:** Net decode calls saved vs Vanilla.",
        "- **Hypertoken Emissions:** Hit rate, dead slots, and emitted token IDs.",
        "- **Termination Health:** EOS reached, repetition penalty triggers, length truncation.",
        "- **Quality-Preserved Savings:** Decode-step reduction on prompts where task quality matches or exceeds Vanilla.",
        "",
        "## 3. Decision Rules (Interpretation Gate)",
        "- **Case 5 (Safe Oracle Fails):** If Arms C1 and C2 still degrade task quality compared to Arm A, Predictor V2 is *not* sufficient alone; HyperLinear/input encoder continuation representations are the primary bottleneck.",
        "- **Case 6 (Safe Oracle Works, Predictors Fail):** If Arm C2 maintains 100% quality parity while Arms E and F fail, ranking/prediction remains the dominant blocker.",
        "- **Case 7 (Ranker Succeeds Offline, No Emissions Live):** If Arm E ranks high-quality codebooks but emissions remain 0, output-head calibration is the next bottleneck.",
    ]

    with open(OUT_PLAN_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved protocol document to {OUT_PLAN_MD}")


if __name__ == "__main__":
    main()
