"""Phase 3: Small Selector Policy POC on 12 fixed validation prompts.

Compares 3 conditions using frozen Step-100 weights:
- Condition A: Step-100 Baseline (CappedPredictorPolicy, K=32)
- Condition B: Step-100 + Evidence-Aware Selector (K=32)
- Condition C: Step-100 + Evidence-Aware Selector (Adaptive K, tau=20.0)
"""

import json
import os
import pickle
import sys
import time
from collections import Counter
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
    mark_historical_baseline,
    accept_cached_generation,
)
from experiments.mbpp_prompt import build_mbpp_prompt
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
RAW_RESULTS_PATH = "experiments/checkpoints/quality_benchmark/raw_results.jsonl"
PREDICTOR_PATH = "experiments/checkpoints/cached_predictor.pkl"
CKPT_STEP100_PATH = "experiments/checkpoints/predictive_joint_pilot/checkpoint_step_100.pt"
OUT_JSON = "experiments/checkpoints/quality_benchmark/poc_selector_results_mbpp_signature_v1.json"
OUT_MD = "experiments/checkpoints/quality_benchmark/poc_selector_results_mbpp_signature_v1.md"


def main():
    device = torch.device("cpu")
    print("=== Phase 3: Small Selector Policy POC (12 Prompts) ===", flush=True)

    # 1. Load prompt definitions
    with open(POC_IDS_PATH, "r", encoding="utf-8") as f:
        poc_list = json.load(f)
    poc_ids_set = {item["id"] for item in poc_list}

    with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
        val_all = json.load(f)
    samples = [s for s in val_all if s["id"] in poc_ids_set]
    # Preserve order of poc_list
    order_map = {item["id"]: i for i, item in enumerate(poc_list)}
    samples.sort(key=lambda s: order_map[s["id"]])
    print(f"Loaded {len(samples)} POC samples across MBPP, GSM8K, Alpaca.", flush=True)

    # 2. Condition A: Extract baseline Step-100 results from raw_results.jsonl
    print("\n--- Condition A: Loading Frozen Baseline (Step 100) ---", flush=True)
    cond_a_recs = []
    with open(RAW_RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] == "predictive_step_100" and r["prompt_id"] in poc_ids_set:
                r_copy = dict(r)
                r_copy["condition"] = "cond_a_baseline_k32"
                cond_a_recs.append(mark_historical_baseline(r_copy))
    cond_a_recs.sort(key=lambda s: order_map[s["prompt_id"]])
    assert len(cond_a_recs) == 12, f"Expected 12 records for Condition A, got {len(cond_a_recs)}"
    print(f"Loaded {len(cond_a_recs)} baseline records for Condition A.", flush=True)

    # 3. Load Model and Predictor Index
    print("\nLoading tokenizer and predictor index...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    with open(PREDICTOR_PATH, "rb") as f:
        raw_pred = pickle.load(f)
    p_index = getattr(raw_pred, "index", raw_pred)

    ev_selector = EvidenceAwareSelector(
        predictor_index=p_index,
        tokenizer=tokenizer,
        budget=32,
        max_structural_slots=0,
    )

    print("Loading Zip2Zip Step-100 model...", flush=True)
    t0 = time.time()
    model = Zip2ZipModel.from_pretrained(
        MODEL_NAME,
        max_codebook_size=32,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.output_encoder.to(torch.float32)

    # Load Step 100 checkpoint
    load_report = load_joint_checkpoint(model, CKPT_STEP100_PATH)
    print(
        f"Model loaded with {load_report['lora_tensors']} LoRA tensors and "
        f"{load_report['input_encoder_tensors'] + load_report['output_encoder_tensors']} encoder tensors "
        f"in {time.time() - t0:.1f}s.",
        flush=True,
    )

    dim = model.zip2zip_config.encoder.hidden_size
    pad_id = tokenizer.pad_token_id or 32000
    disabled_ids = list(model.zip2zip_config.compression.disabled_ids)

    # Load existing results if resuming
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

    def run_eval_loop(condition_name: str, min_tau: Optional[float]) -> List[Dict[str, Any]]:
        cond_results = []
        print(f"\n--- Running {condition_name} (min_tau={min_tau}) ---", flush=True)
        for idx, s in enumerate(samples, 1):
            pid = s["id"]
            dom = s["domain"]
            key = (pid, condition_name)
            if key in existing_recs:
                print(f"[{idx}/12] Reusing saved result for {pid} on {condition_name}", flush=True)
                cond_results.append(existing_recs[key])
                continue

            prompt_text = build_mbpp_prompt(s) if dom == "code" else s["prompt"]
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            base_prompt_len = len(prompt_ids)

            # 1. Evidence-aware codebook selection
            t_sel0 = time.perf_counter()
            codebook_dict, sel_meta = ev_selector.select_codebook(
                prompt_ids,
                prompt_text=prompt_text,
                min_score_threshold=min_tau,
            )
            predictor_time_s = time.perf_counter() - t_sel0

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
                out = model.generate(
                    input_ids=input_tensor,
                    max_new_tokens=MAX_NEW_TOKENS,
                    logits_processor=proc_list,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            t_gen = time.perf_counter() - t_gen0
            total_wall_time = codebook_time_s + t_gen

            gen_ids = out[0, base_prompt_len:].tolist()
            decode_steps = len(gen_ids)

            # 4. Decompress
            hyper_to_tokens = {v: list(k) for k, v in codebook_dict.items()}
            expanded_tokens = []
            hypertokens_emitted = []
            first_hyper_pos = -1

            for pos, tid in enumerate(gen_ids):
                if tid in hyper_to_tokens:
                    if first_hyper_pos == -1:
                        first_hyper_pos = pos
                    phrase_str = tokenizer.decode(hyper_to_tokens[tid])
                    hypertokens_emitted.append({"pos": pos, "id": tid, "phrase": phrase_str, "subtokens": hyper_to_tokens[tid]})
                    expanded_tokens.extend(hyper_to_tokens[tid])
                else:
                    expanded_tokens.append(tid)

            output_text = tokenizer.decode(expanded_tokens, skip_special_tokens=True)
            expanded_output_tokens = len(expanded_tokens)
            tokens_saved = max(0, expanded_output_tokens - decode_steps)
            decode_reduction_pct = round((1.0 - decode_steps / max(expanded_output_tokens, 1)) * 100, 2) if expanded_output_tokens > decode_steps else 0.0

            eos_reached = sequence_reached_eos(gen_ids, tokenizer.eos_token_id)
            hit_max_length = decode_steps >= MAX_NEW_TOKENS

            ttft = timing_proc.ttft or 0.0
            decode_time = max(0.0, t_gen - ttft)

            static_mgr.detach_from_model(model)
            model.codebook_manager.reset()

            # Capacity utilization
            emitted_ids_set = {e["id"] for e in hypertokens_emitted}
            used_slots = len(emitted_ids_set)
            cb_size = len(codebook_dict)
            dead_slots = cb_size - used_slots
            utilization_pct = round(used_slots / cb_size * 100, 1) if cb_size > 0 else 0.0

            rec = stamp_generation_record({
                "prompt_id": pid,
                "domain": dom,
                "condition": condition_name,
                "base_prompt_tokens": base_prompt_len,
                "decode_steps": decode_steps,
                "expanded_output_tokens": expanded_output_tokens,
                "tokens_saved": tokens_saved,
                "decode_reduction_pct": decode_reduction_pct,
                "eos_reached": eos_reached,
                "wall_time_s": round(total_wall_time, 3),
                "ttft_s": round(ttft, 3),
                "hypertokens_count": len(hypertokens_emitted),
                "codebook_size": cb_size,
                "used_slots": used_slots,
                "dead_slots": dead_slots,
                "utilization_pct": utilization_pct,
                "output_text": output_text,
            })

            if dom == "code":
                asserts = [line.strip() for line in s["ground_truth_response"].splitlines() if line.strip().startswith("assert")]
                code_eval = evaluate_mbpp_code(output_text, asserts)
                rec.update(code_eval)
            elif dom == "reasoning":
                gsm_eval = evaluate_gsm8k_reasoning(output_text, s["ground_truth_response"])
                rec.update(gsm_eval)
            elif dom == "instruction":
                alp_eval = evaluate_alpaca_instruction(output_text, eos_reached)
                rec.update(alp_eval)

            cond_results.append(rec)
            existing_recs[key] = rec
            print(f"[{idx}/12] {pid} ({dom}): steps={decode_steps}, saved={tokens_saved} ({decode_reduction_pct}%), hypers={len(hypertokens_emitted)}, cb_size={cb_size}, dead={dead_slots}, wall={total_wall_time:.1f}s", flush=True)

        return cond_results

    # Run Condition B (EvidenceAwareSelector K=32)
    cond_b_recs = run_eval_loop("cond_b_evidence_k32", min_tau=None)

    # Run Condition C (EvidenceAwareSelector Adaptive tau=20.0)
    cond_c_recs = run_eval_loop("cond_c_adaptive_tau20", min_tau=20.0)

    # Combine all records
    all_raw = cond_a_recs + cond_b_recs + cond_c_recs

    # Compute aggregates per condition
    def is_correct(r: Dict[str, Any]) -> bool:
        dom = r["domain"]
        if dom == "code":
            return r.get("problem_pass", False)
        elif dom == "reasoning":
            return r.get("exact_correct", False)
        elif dom == "instruction":
            return not r.get("instruction_failure", False)
        return False

    def aggregate(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        tot_steps = sum(r["decode_steps"] for r in recs)
        tot_exp = sum(r["expanded_output_tokens"] for r in recs)
        tot_saved = sum(r["tokens_saved"] for r in recs)
        micro_red = round((1.0 - tot_steps / max(tot_exp, 1)) * 100, 2)
        macro_red = round(sum(r["decode_reduction_pct"] for r in recs) / len(recs), 2)
        mean_hypers = round(sum(r["hypertokens_count"] for r in recs) / len(recs), 2)
        mean_wall = round(sum(r["wall_time_s"] for r in recs) / len(recs), 2)
        mean_ttft = round(sum(r["ttft_s"] for r in recs) / len(recs), 3)

        # Capacity utilization
        tot_cb_slots = sum(r.get("codebook_size", 32) for r in recs)
        tot_used_slots = sum(r.get("used_slots", len(set(e["id"] for e in r.get("hypertokens_emitted", [])))) for r in recs)
        tot_dead_slots = tot_cb_slots - tot_used_slots
        util_pct = round(tot_used_slots / max(tot_cb_slots, 1) * 100, 1)

        # Domain breakdown
        dom_stats = {}
        for d in ["code", "reasoning", "instruction"]:
            d_recs = [r for r in recs if r["domain"] == d]
            d_corr = sum(1 for r in d_recs if is_correct(r))
            d_steps = sum(r["decode_steps"] for r in d_recs)
            d_exp = sum(r["expanded_output_tokens"] for r in d_recs)
            d_saved = sum(r["tokens_saved"] for r in d_recs)
            d_micro = round((1.0 - d_steps / max(d_exp, 1)) * 100, 2)
            dom_stats[d] = {
                "correct": f"{d_corr}/{len(d_recs)}",
                "accuracy_pct": round(d_corr / len(d_recs) * 100, 1),
                "tokens_saved": d_saved,
                "micro_reduction_pct": d_micro,
                "mean_hypers": round(sum(r["hypertokens_count"] for r in d_recs) / len(d_recs), 1),
            }

        total_correct = sum(1 for r in recs if is_correct(r))
        return {
            "total_correct": f"{total_correct}/{len(recs)}",
            "accuracy_pct": round(total_correct / len(recs) * 100, 1),
            "micro_reduction_pct": micro_red,
            "macro_reduction_pct": macro_red,
            "tokens_saved": tot_saved,
            "mean_hypertokens": mean_hypers,
            "total_cb_slots": tot_cb_slots,
            "used_slots": tot_used_slots,
            "dead_slots": tot_dead_slots,
            "utilization_pct": util_pct,
            "mean_wall_time_s": mean_wall,
            "mean_ttft_s": mean_ttft,
            "domain_breakdown": dom_stats,
        }

    agg_results = {
        "comparison_status": "provisional_historical_baseline_unverified",
        "condition_a_baseline_k32": aggregate(cond_a_recs),
        "condition_b_evidence_k32": aggregate(cond_b_recs),
        "condition_c_adaptive_tau20": aggregate(cond_c_recs),
        "raw_records": all_raw,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(agg_results, f, indent=2)

    # Markdown Report
    a_agg = agg_results["condition_a_baseline_k32"]
    b_agg = agg_results["condition_b_evidence_k32"]
    c_agg = agg_results["condition_c_adaptive_tau20"]

    md_lines = [
        "# Phase 3: Small Selector Policy POC (12 Prompts)",
        "",
        "This evaluation tests whether evidence-aware codebook reranking and adaptive K selection alone improve quality and capacity efficiency on a fixed 12-prompt subset without any model retraining.",
        "",
        "PROVISIONAL COMPARISON: Condition A comes from historical raw_results and its checkpoint, prompt, and evaluator provenance is unverified. Deltas against A are not corrected results; rerun a matched Condition A with the shared loader before treating them as corrected.",
        "",
        "## 1. Executive Comparison Across Conditions",
        "",
        "| Metric | Condition A: Historical, unverified (K=32) | Condition B: Evidence-Aware (K=32) | Condition C: Adaptive-K (tau=20.0) | Provisional Delta (C vs A) |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Overall Accuracy** | **{a_agg['total_correct']} ({a_agg['accuracy_pct']}%)** | **{b_agg['total_correct']} ({b_agg['accuracy_pct']}%)** | **{c_agg['total_correct']} ({c_agg['accuracy_pct']}%)** | **{c_agg['accuracy_pct'] - a_agg['accuracy_pct']:+.1f}%** |",
        f"| **Realized Micro Compression** | **{a_agg['micro_reduction_pct']}%** | **{b_agg['micro_reduction_pct']}%** | **{c_agg['micro_reduction_pct']}%** | {c_agg['micro_reduction_pct'] - a_agg['micro_reduction_pct']:+.2f}% |",
        f"| **Tokens / Decode Steps Saved** | {a_agg['tokens_saved']} | {b_agg['tokens_saved']} | {c_agg['tokens_saved']} | {c_agg['tokens_saved'] - a_agg['tokens_saved']:+d} |",
        f"| **Mean Hypertokens / Prompt** | {a_agg['mean_hypertokens']} | {b_agg['mean_hypertokens']} | {c_agg['mean_hypertokens']} | {c_agg['mean_hypertokens'] - a_agg['mean_hypertokens']:+.1f} |",
        f"| **Codebook Slot Utilization** | {a_agg['utilization_pct']}% ({a_agg['used_slots']}/{a_agg['total_cb_slots']}) | {b_agg['utilization_pct']}% ({b_agg['used_slots']}/{b_agg['total_cb_slots']}) | **{c_agg['utilization_pct']}%** ({c_agg['used_slots']}/{c_agg['total_cb_slots']}) | **{c_agg['utilization_pct'] - a_agg['utilization_pct']:+.1f}%** |",
        f"| **Dead Slots** | {a_agg['dead_slots']} | {b_agg['dead_slots']} | **{c_agg['dead_slots']}** | **{c_agg['dead_slots'] - a_agg['dead_slots']:+d}** |",
        f"| **Mean Wall-Clock Latency** | {a_agg['mean_wall_time_s']}s | {b_agg['mean_wall_time_s']}s | {c_agg['mean_wall_time_s']}s | {c_agg['mean_wall_time_s'] - a_agg['mean_wall_time_s']:+.2f}s |",
        f"| **Mean TTFT** | {a_agg['mean_ttft_s']}s | {b_agg['mean_ttft_s']}s | {c_agg['mean_ttft_s']}s | {c_agg['mean_ttft_s'] - a_agg['mean_ttft_s']:+.3f}s |",
        "",
        "---",
        "",
        "## 2. Domain Breakdown",
        "",
        "### A. MBPP Code (4 Prompts)",
        f"- **Condition A:** {a_agg['domain_breakdown']['code']['correct']} ({a_agg['domain_breakdown']['code']['accuracy_pct']}%), Saved={a_agg['domain_breakdown']['code']['tokens_saved']} ({a_agg['domain_breakdown']['code']['micro_reduction_pct']}%), Hypers={a_agg['domain_breakdown']['code']['mean_hypers']}",
        f"- **Condition B:** {b_agg['domain_breakdown']['code']['correct']} ({b_agg['domain_breakdown']['code']['accuracy_pct']}%), Saved={b_agg['domain_breakdown']['code']['tokens_saved']} ({b_agg['domain_breakdown']['code']['micro_reduction_pct']}%), Hypers={b_agg['domain_breakdown']['code']['mean_hypers']}",
        f"- **Condition C:** {c_agg['domain_breakdown']['code']['correct']} ({c_agg['domain_breakdown']['code']['accuracy_pct']}%), Saved={c_agg['domain_breakdown']['code']['tokens_saved']} ({c_agg['domain_breakdown']['code']['micro_reduction_pct']}%), Hypers={c_agg['domain_breakdown']['code']['mean_hypers']}",
        "",
        "### B. GSM8K Reasoning (4 Prompts)",
        f"- **Condition A:** {a_agg['domain_breakdown']['reasoning']['correct']} ({a_agg['domain_breakdown']['reasoning']['accuracy_pct']}%), Saved={a_agg['domain_breakdown']['reasoning']['tokens_saved']} ({a_agg['domain_breakdown']['reasoning']['micro_reduction_pct']}%), Hypers={a_agg['domain_breakdown']['reasoning']['mean_hypers']}",
        f"- **Condition B:** {b_agg['domain_breakdown']['reasoning']['correct']} ({b_agg['domain_breakdown']['reasoning']['accuracy_pct']}%), Saved={b_agg['domain_breakdown']['reasoning']['tokens_saved']} ({b_agg['domain_breakdown']['reasoning']['micro_reduction_pct']}%), Hypers={b_agg['domain_breakdown']['reasoning']['mean_hypers']}",
        f"- **Condition C:** {c_agg['domain_breakdown']['reasoning']['correct']} ({c_agg['domain_breakdown']['reasoning']['accuracy_pct']}%), Saved={c_agg['domain_breakdown']['reasoning']['tokens_saved']} ({c_agg['domain_breakdown']['reasoning']['micro_reduction_pct']}%), Hypers={c_agg['domain_breakdown']['reasoning']['mean_hypers']}",
        "",
        "### C. Alpaca Instruction (4 Prompts)",
        f"- **Condition A:** {a_agg['domain_breakdown']['instruction']['correct']} ({a_agg['domain_breakdown']['instruction']['accuracy_pct']}%), Saved={a_agg['domain_breakdown']['instruction']['tokens_saved']} ({a_agg['domain_breakdown']['instruction']['micro_reduction_pct']}%), Hypers={a_agg['domain_breakdown']['instruction']['mean_hypers']}",
        f"- **Condition B:** {b_agg['domain_breakdown']['instruction']['correct']} ({b_agg['domain_breakdown']['instruction']['accuracy_pct']}%), Saved={b_agg['domain_breakdown']['instruction']['tokens_saved']} ({b_agg['domain_breakdown']['instruction']['micro_reduction_pct']}%), Hypers={b_agg['domain_breakdown']['instruction']['mean_hypers']}",
        f"- **Condition C:** {c_agg['domain_breakdown']['instruction']['correct']} ({c_agg['domain_breakdown']['instruction']['accuracy_pct']}%), Saved={c_agg['domain_breakdown']['instruction']['tokens_saved']} ({c_agg['domain_breakdown']['instruction']['micro_reduction_pct']}%), Hypers={c_agg['domain_breakdown']['instruction']['mean_hypers']}",
        "",
        "---",
        "",
        "## 3. Decision Point & Next Steps",
    ]

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nPhase 3 POC complete! Wrote {OUT_JSON} and {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
