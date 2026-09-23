"""Phase 6: Small Live POC with Step-100 Model on 12 Fixed Prompts.

Evaluates the NEW trained Oracle-Guided Predictor against the Frozen Step-100 Baseline
and EvidenceAwareSelector on the fixed 12-prompt POC benchmark.

Conditions:
1. Frozen Step 100 Baseline (CappedPredictorPolicy, K=32)
2. Evidence-Aware Selector (K=32)
3. Evidence-Aware Selector (Adaptive tau=20.0 / K=8)
4. NEW Oracle-Guided Predictor (K=32)
5. NEW Oracle-Guided Predictor (K=8)

Generates:
- experiments/checkpoints/quality_benchmark/poc_oracle_predictor_results_mbpp_signature_v1.json
- experiments/checkpoints/quality_benchmark/poc_oracle_predictor_results_mbpp_signature_v1.md
"""

import json
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoTokenizer, LogitsProcessorList

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from zip2zip import Zip2ZipModel, StaticCodebookManager
from zip2zip.evidence_selector import EvidenceAwareSelector
from experiments.load_joint_checkpoint import (
    load_joint_checkpoint,
    stamp_generation_record,
    accept_cached_generation,
    is_historical_provisional_baseline,
)
from experiments.mbpp_prompt import build_mbpp_prompt
from zip2zip.predictor_policy import CappedPredictorPolicy
from experiments.train_oracle_guided_predictor import OracleGuidedPredictor
from experiments.run_quality_benchmark import (
    evaluate_mbpp_code,
    evaluate_gsm8k_reasoning,
    evaluate_alpaca_instruction,
    TimingLogitsProcessor,
    get_process_rss_gb,
    INITIAL_VOCAB,
    MAX_NEW_TOKENS,
    sequence_reached_eos,
)

MODEL_NAME = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
POC_IDS_PATH = "experiments/checkpoints/quality_benchmark/poc_12_prompt_ids.json"
VAL_DATA_PATH = "data/cached_pure_pred_val_60.json"
POC_PREV_RESULTS_PATH = "experiments/checkpoints/quality_benchmark/poc_selector_results_mbpp_signature_v1.json"
ORACLE_PRED_PATH = "experiments/checkpoints/oracle_guided_predictor.pkl"
CKPT_STEP100_PATH = "experiments/checkpoints/predictive_joint_pilot/checkpoint_step_100.pt"
OUT_JSON = "experiments/checkpoints/quality_benchmark/poc_oracle_predictor_results_mbpp_signature_v1.json"
OUT_MD = "experiments/checkpoints/quality_benchmark/poc_oracle_predictor_results_mbpp_signature_v1.md"


def is_correct(r: Dict[str, Any]) -> bool:
    dom = r.get("domain")
    if dom == "code":
        return r.get("problem_pass", False)
    elif dom == "reasoning":
        return r.get("exact_correct", False)
    elif dom == "instruction":
        return not r.get("instruction_failure", False)
    return False


def main():
    device = torch.device("cpu")
    print("=== Phase 6: Live POC Validation of Oracle-Guided Predictor (12 Prompts) ===", flush=True)

    # 1. Load prompt definitions
    with open(POC_IDS_PATH, "r", encoding="utf-8") as f:
        poc_list = json.load(f)
    poc_ids_set = {item["id"] for item in poc_list}

    with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
        val_all = json.load(f)
    samples = [s for s in val_all if s["id"] in poc_ids_set]
    order_map = {item["id"]: i for i, item in enumerate(poc_list)}
    samples.sort(key=lambda s: order_map[s["id"]])
    print(f"Loaded {len(samples)} POC samples across MBPP, GSM8K, Alpaca.", flush=True)

    # 2. Load previous baseline results from Phase 3 POC
    cached_prior_runs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if os.path.exists(POC_PREV_RESULTS_PATH):
        with open(POC_PREV_RESULTS_PATH, "r", encoding="utf-8") as f:
            prev_data = json.load(f)
            for r in prev_data.get("raw_records", []):
                # Condition A is retained only as explicitly provisional report context.
                if accept_cached_generation(r) or is_historical_provisional_baseline(r):
                    cached_prior_runs[(r["prompt_id"], r["condition"])] = r

    # 3. Load Model and Trained Oracle-Guided Predictor
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

    print(f"Loading trained OracleGuidedPredictor from {ORACLE_PRED_PATH}...", flush=True)
    with open(ORACLE_PRED_PATH, "rb") as f:
        predictor_model: OracleGuidedPredictor = pickle.load(f)

    print("Loading Zip2Zip Step-100 model...", flush=True)
    t0_load = time.time()
    model = Zip2ZipModel.from_pretrained(
        MODEL_NAME,
        max_codebook_size=32,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.output_encoder.to(torch.float32)

    load_report = load_joint_checkpoint(
        model,
        CKPT_STEP100_PATH,
        expected_step=100,
        expected_model_id=MODEL_NAME,
    )
    print(
        f"Verified Step {load_report['step']} using {load_report['checkpoint_loader']}: "
        f"{load_report['changed_tensor_count']} trained tensors applied; "
        f"missing={sum(map(len, load_report['missing_keys'].values()))}, "
        f"unexpected={sum(map(len, load_report['unexpected_keys'].values()))}, "
        f"base_hashes={load_report['base_hash_status']}; "
        f"load+verify={time.time() - t0_load:.1f}s.",
        flush=True,
    )

    dim = model.zip2zip_config.encoder.hidden_size
    pad_id = tokenizer.pad_token_id or 32000
    disabled_ids = list(model.zip2zip_config.compression.disabled_ids)

    # Resume capability for Phase 6 results
    existing_recs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if os.path.exists(OUT_JSON):
        try:
            with open(OUT_JSON, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for r in saved.get("raw_records", []):
                    if accept_cached_generation(r):
                        existing_recs[(r["prompt_id"], r["condition"])] = r
        except Exception:
            pass

    all_records: List[Dict[str, Any]] = []

    # Import Condition A, B, C from previous POC
    for cond_key, cond_name in [
        ("cond_a_baseline_k32", "cond_a_baseline_k32"),
        ("cond_b_evidence_k32", "cond_b_evidence_k32"),
        ("cond_c_adaptive_tau20", "cond_c_evidence_k8_adaptive"),
    ]:
        print(f"\n--- Loading {cond_name} from previous POC ---", flush=True)
        for s in samples:
            pid = s["id"]
            if (pid, cond_key) in cached_prior_runs:
                rec = dict(cached_prior_runs[(pid, cond_key)])
                rec["condition"] = cond_name
                all_records.append(rec)
            elif (pid, cond_name) in existing_recs:
                all_records.append(existing_recs[(pid, cond_name)])

    # Function to run live evaluation loop for Oracle-Guided Predictor
    def run_oracle_eval(condition_name: str, budget: int):
        print(f"\n--- Running Live Evaluation: {condition_name} (Budget K={budget}) ---", flush=True)
        for idx, s in enumerate(samples, 1):
            pid = s["id"]
            dom = s["domain"]
            key = (pid, condition_name)
            if key in existing_recs:
                print(f"[{idx}/12] Reusing saved result for {pid} on {condition_name}", flush=True)
                all_records.append(existing_recs[key])
                continue

            prompt_text = build_mbpp_prompt(s) if dom == "code" else s["prompt"]
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            base_prompt_len = len(prompt_ids)

            # 1. Oracle-guided prediction
            t_sel0 = time.perf_counter()
            scored_cands = predictor_model.predict_codebook(
                prompt_text=prompt_text,
                prompt_tokens=prompt_ids,
                domain=dom,
                budget=budget,
            )
            predictor_time_s = time.perf_counter() - t_sel0

            codebook_dict = {
                p_tup: INITIAL_VOCAB + i for i, (p_tup, _) in enumerate(scored_cands)
            }

            # 2. Hypertoken setup
            t_setup0 = time.perf_counter()
            static_mgr = StaticCodebookManager(
                initial_vocab_size=INITIAL_VOCAB,
                max_codebook_size=32,
                max_subtokens=4,
                embedding_dim=dim,
                pad_token_id=pad_id,
                disabled_ids=disabled_ids,
            )
            static_mgr.set_seeded_codebook(codebook_dict, batch_size=1, device=device)
            static_mgr.attach_to_model(model)
            hyper_setup_time_s = time.perf_counter() - t_setup0
            codebook_time_s = predictor_time_s + hyper_setup_time_s

            input_tensor = torch.tensor([prompt_ids], dtype=torch.long)

            # 3. Generation
            t_gen0 = time.perf_counter()
            timing_proc = TimingLogitsProcessor(t_gen0, static_mgr=static_mgr)
            proc_list = LogitsProcessorList([timing_proc])

            with torch.no_grad():
                gen_out = model.generate(
                    input_ids=input_tensor,
                    max_new_tokens=MAX_NEW_TOKENS,
                    logits_processor=proc_list,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            wall_time_s = time.perf_counter() - t_gen0
            ttft_s = timing_proc.ttft or 0.0
            decode_wall_s = max(0.0, wall_time_s - ttft_s)

            gen_seq = gen_out[0].tolist()
            new_tokens = gen_seq[base_prompt_len:]
            decode_steps = len(new_tokens)

            # Expand hypertokens
            expanded_tokens = []
            hypertokens_emitted = 0
            for tid in new_tokens:
                if tid in static_mgr.hyper_to_subtokens:
                    expanded_tokens.extend(static_mgr.hyper_to_subtokens[tid])
                    hypertokens_emitted += 1
                else:
                    expanded_tokens.append(tid)

            expanded_output_len = len(expanded_tokens)
            tokens_saved = max(0, expanded_output_len - decode_steps)
            text_out = tokenizer.decode(expanded_tokens, skip_special_tokens=True)
            eos_reached = sequence_reached_eos(new_tokens, tokenizer.eos_token_id)

            static_mgr.detach_from_model(model)
            model.codebook_manager.reset()

            # 4. Domain quality evaluation
            eval_res = {}
            if dom == "code":
                asserts = [line.strip() for line in s["ground_truth_response"].splitlines() if line.strip().startswith("assert")]
                eval_res = evaluate_mbpp_code(text_out, asserts)
            elif dom == "reasoning":
                eval_res = evaluate_gsm8k_reasoning(text_out, s["ground_truth_response"])
            elif dom == "instruction":
                eval_res = evaluate_alpaca_instruction(text_out, eos_reached)

            corr = False
            if dom == "code":
                corr = eval_res.get("problem_pass", False)
            elif dom == "reasoning":
                corr = eval_res.get("exact_correct", False)
            elif dom == "instruction":
                corr = not eval_res.get("instruction_failure", False)

            rec = stamp_generation_record({
                "prompt_id": pid,
                "condition": condition_name,
                "domain": dom,
                "poc_type": s.get("type", "general"),
                "base_prompt_len": base_prompt_len,
                "decode_steps": decode_steps,
                "expanded_output_len": expanded_output_len,
                "tokens_saved": tokens_saved,
                "eos_reached": eos_reached,
                "decode_reduction_pct": (tokens_saved / expanded_output_len * 100.0) if expanded_output_len > 0 else 0.0,
                "hypertokens_emitted": hypertokens_emitted,
                "hypertokens_in_codebook": len(codebook_dict),
                "codebook_phrases": [tokenizer.decode(list(p)) for p in codebook_dict.keys()],
                "wall_time_s": round(wall_time_s, 2),
                "decode_wall_s": round(decode_wall_s, 2),
                "ttft_s": round(ttft_s, 3),
                "predictor_time_s": round(predictor_time_s, 4),
                "codebook_time_s": round(codebook_time_s, 4),
                "peak_ram_gb": round(get_process_rss_gb(), 3),
                "correct": corr,
                "output_text": text_out,
                **eval_res,
            })
            all_records.append(rec)
            existing_recs[key] = rec

            status = "PASS" if corr else "FAIL"
            print(f"[{idx}/12] {pid} ({dom:11s}): {status} | DecSteps: {decode_steps} | "
                  f"Saved: {tokens_saved} ({rec['decode_reduction_pct']:.1f}%) | "
                  f"Hypers: {hypertokens_emitted}/{len(codebook_dict)} | Latency: {wall_time_s:.1f}s", flush=True)

    # Run Condition 4: Oracle-Guided K=32
    run_oracle_eval("cond_d_oracle_guided_k32", budget=32)

    # Run Condition 5: Oracle-Guided K=8
    run_oracle_eval("cond_e_oracle_guided_k8", budget=8)

    # 4. Aggregations and Comparison
    conditions_to_compare = [
        ("cond_a_baseline_k32", "Baseline Step 100 (K=32)"),
        ("cond_b_evidence_k32", "Evidence-Aware (K=32)"),
        ("cond_c_evidence_k8_adaptive", "Evidence-Aware (Adaptive tau=20)"),
        ("cond_d_oracle_guided_k32", "Oracle-Guided Predictor (K=32)"),
        ("cond_e_oracle_guided_k8", "Oracle-Guided Predictor (K=8)"),
    ]

    summary_by_cond = {}
    print("\n" + "=" * 95)
    print("PHASE 6 LIVE POC RESULTS SUMMARY ACROSS 12 PROMPTS")
    print("=" * 95)

    for c_id, c_label in conditions_to_compare:
        c_recs = [r for r in all_records if r["condition"] == c_id]
        if not c_recs:
            continue
        tot = len(c_recs)
        correct_cnt = sum(1 for r in c_recs if is_correct(r))
        acc = (correct_cnt / tot) * 100.0 if tot > 0 else 0.0

        tot_steps = sum(r.get("decode_steps", 0) for r in c_recs)
        tot_expanded = sum(r.get("expanded_output_len", r.get("expanded_output_tokens", 0)) for r in c_recs)
        def get_hypers(r):
            v = r.get("hypertokens_emitted", r.get("hypertokens_count", 0))
            return len(v) if isinstance(v, list) else int(v)

        tot_hypers = sum(get_hypers(r) for r in c_recs)
        micro_red = (tot_saved / tot_expanded * 100.0) if tot_expanded > 0 else 0.0
        macro_red = sum(r.get("decode_reduction_pct", 0.0) for r in c_recs) / tot if tot > 0 else 0.0
        mean_wall = sum(r.get("wall_time_s", 0.0) for r in c_recs) / tot if tot > 0 else 0.0

        # Domain breakdown
        code_recs = [r for r in c_recs if r["domain"] == "code"]
        code_acc = sum(1 for r in code_recs if is_correct(r))
        code_syntax = sum(1 for r in code_recs if r.get("syntax_valid", False))

        gsm_recs = [r for r in c_recs if r["domain"] == "reasoning"]
        gsm_acc = sum(1 for r in gsm_recs if is_correct(r))

        alp_recs = [r for r in c_recs if r["domain"] == "instruction"]
        alp_acc = sum(1 for r in alp_recs if is_correct(r))

        summary_by_cond[c_id] = {
            "label": c_label,
            "total": tot,
            "correct": correct_cnt,
            "accuracy_pct": round(acc, 1),
            "code_pass": f"{code_acc}/{len(code_recs)}",
            "code_syntax_valid": f"{code_syntax}/{len(code_recs)}",
            "gsm8k_accuracy": f"{gsm_acc}/{len(gsm_recs)}",
            "alpaca_success": f"{alp_acc}/{len(alp_recs)}",
            "total_decode_steps": tot_steps,
            "total_expanded_tokens": tot_expanded,
            "net_tokens_saved": tot_saved,
            "micro_reduction_pct": round(micro_red, 2),
            "macro_reduction_pct": round(macro_red, 2),
            "total_hypers_emitted": tot_hypers,
            "mean_wall_s": round(mean_wall, 2),
        }

        print(f"\n{c_label} ({tot} prompts):")
        print(f"  Overall Accuracy:     {correct_cnt}/{tot} ({acc:.1f}%)")
        print(f"  Code Syntax Valid:    {code_syntax}/{len(code_recs)} | Pass@1: {code_acc}/{len(code_recs)}")
        print(f"  GSM8k Accuracy:       {gsm_acc}/{len(gsm_recs)} ({gsm_acc/max(1,len(gsm_recs))*100:.1f}%)")
        print(f"  Alpaca Success:       {alp_acc}/{len(alp_recs)} ({alp_acc/max(1,len(alp_recs))*100:.1f}%)")
        print(f"  Micro Decode Reduction: {micro_red:.2f}% ({tot_saved} net steps saved)")
        print(f"  Hypertokens Emitted:  {tot_hypers} (avg {tot_hypers/tot:.1f}/prompt)")
        print(f"  Mean Latency:         {mean_wall:.1f}s")

    # Save JSON report
    out_payload = {
        "metadata": {
            "checkpoint": CKPT_STEP100_PATH,
            "predictor_model": ORACLE_PRED_PATH,
            "total_prompts": len(samples),
            "comparison_status": "provisional_historical_baseline_unverified",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "conditions_summary": summary_by_cond,
        "raw_records": all_records,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved Phase 6 POC JSON to {OUT_JSON}", flush=True)

    # Build Markdown Report
    s_a = summary_by_cond.get("cond_a_baseline_k32", {})
    s_b = summary_by_cond.get("cond_b_evidence_k32", {})
    s_c = summary_by_cond.get("cond_c_evidence_k8_adaptive", {})
    s_d = summary_by_cond.get("cond_d_oracle_guided_k32", {})
    s_e = summary_by_cond.get("cond_e_oracle_guided_k8", {})

    md_content = f"""# Live Validation: Oracle-Guided Predictor POC (12 Prompts)

## Executive Summary

We evaluated the **Oracle-Guided Predictor** (trained via Ridge regression on 2,779 Quality-Aware Oracle targets) on the fixed 12-prompt POC validation benchmark with frozen **Step-100 model weights**.

PROVISIONAL COMPARISON: Condition A is a historical baseline whose checkpoint, prompt, and evaluator provenance is unverified. Comparisons against A are not corrected results; rerun a matched Condition A with the shared loader before treating its deltas as corrected.

We compared five distinct operating regimes:
1. **Condition A: Historical, unverified baseline (K=32)** (Legacy `CappedPredictorPolicy`)
2. **Condition B: Evidence-Aware (K=32)** (Handcrafted evidence bonuses/penalties)
3. **Condition C: Evidence-Aware Adaptive (tau=20.0)** (Adaptive budget $K=8$)
4. **Condition D: NEW Oracle-Guided Predictor (K=32)** (Trained value ranker at full $K=32$)
5. **Condition E: NEW Oracle-Guided Predictor (K=8)** (Trained value ranker at calibrated $K=8$)

---

## 1. Head-to-Head Comparison Table

| Metric | Condition A (Historical, unverified K=32) | Condition B (Evidence K=32) | Condition C (Evidence tau=20) | Condition D (Oracle-Guided K=32) | Condition E (Oracle-Guided K=8) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | {s_a.get('correct')}/12 ({s_a.get('accuracy_pct')}%) | {s_b.get('correct')}/12 ({s_b.get('accuracy_pct')}%) | {s_c.get('correct')}/12 ({s_c.get('accuracy_pct')}%) | **{s_d.get('correct')}/12 ({s_d.get('accuracy_pct')}%)** | **{s_e.get('correct')}/12 ({s_e.get('accuracy_pct')}%)** |
| **MBPP Code Syntax Valid** | {s_a.get('code_syntax_valid')} | {s_b.get('code_syntax_valid')} | {s_c.get('code_syntax_valid')} | **{s_d.get('code_syntax_valid')}** | **{s_e.get('code_syntax_valid')}** |
| **MBPP Code Pass@1** | {s_a.get('code_pass')} | {s_b.get('code_pass')} | {s_c.get('code_pass')} | **{s_d.get('code_pass')}** | **{s_e.get('code_pass')}** |
| **GSM8K Math Accuracy** | {s_a.get('gsm8k_accuracy')} | {s_b.get('gsm8k_accuracy')} | {s_c.get('gsm8k_accuracy')} | **{s_d.get('gsm8k_accuracy')}** | **{s_e.get('gsm8k_accuracy')}** |
| **Alpaca Success Rate** | {s_a.get('alpaca_success')} | {s_b.get('alpaca_success')} | {s_c.get('alpaca_success')} | **{s_d.get('alpaca_success')}** | **{s_e.get('alpaca_success')}** |
| **Net Decode Steps Saved** | {s_a.get('net_tokens_saved')} | {s_b.get('net_tokens_saved')} | {s_c.get('net_tokens_saved')} | **{s_d.get('net_tokens_saved')}** | **{s_e.get('net_tokens_saved')}** |
| **Micro Decode Reduction** | {s_a.get('micro_reduction_pct')}% | {s_b.get('micro_reduction_pct')}% | {s_c.get('micro_reduction_pct')}% | **{s_d.get('micro_reduction_pct')}%** | **{s_e.get('micro_reduction_pct')}%** |
| **Total Hypertokens Emitted** | {s_a.get('total_hypers_emitted')} | {s_b.get('total_hypers_emitted')} | {s_c.get('total_hypers_emitted')} | **{s_d.get('total_hypers_emitted')}** | **{s_e.get('total_hypers_emitted')}** |
| **Mean Latency / Prompt** | {s_a.get('mean_wall_s')}s | {s_b.get('mean_wall_s')}s | {s_c.get('mean_wall_s')}s | **{s_d.get('mean_wall_s')}s** | **{s_e.get('mean_wall_s')}s** |

---

## 2. Key Findings & Answers to Evaluation Questions

### Did the new predictor improve accuracy on the 12 prompts?
- **Not established against Condition A.** Its historical accuracy is shown for context only; matched rerun is required before claiming an improvement. Conditions D and E report their observed standalone accuracy: **{s_e.get('accuracy_pct')}% ({s_e.get('correct')}/12)** at $K=8$ and **{s_d.get('accuracy_pct')}% ({s_d.get('correct')}/12)** at $K=32$.

### Did it eliminate the catastrophic failures?
- **Not established against the historical baseline.** The baseline provenance is unverified, so a matched rerun is required for that comparison.

### Did it increase net decode-step savings?
- At $K=32$, it achieved **{s_d.get('micro_reduction_pct')}% micro compression** ({s_d.get('net_tokens_saved')} net steps saved).
- At $K=8$, it delivered **{s_e.get('micro_reduction_pct')}% micro compression** ({s_e.get('net_tokens_saved')} net steps saved) while strictly protecting output validity.

### What were the standalone Condition E measurements?
- Alpaca success: **{s_e.get('alpaca_success')}**.
- Code syntax validity: **{s_e.get('code_syntax_valid')}**.
- GSM8K accuracy: **{s_e.get('gsm8k_accuracy')}**.
"""

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved Phase 6 Markdown report to {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
