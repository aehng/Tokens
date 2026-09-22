"""Reusable K-Sweep and Adaptive-K Optimization Benchmark.

Evaluates fixed codebook budgets K in [4, 8, 16, 24, 32] and adaptive thresholds
tau in [10, 15, 20, 25] with EvidenceAwareSelector on any prompt evaluation subset.
Uses codebook caching to eliminate duplicate model forward passes.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoTokenizer, LogitsProcessorList

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from zip2zip import Zip2ZipModel, StaticCodebookManager
from zip2zip.evidence_selector import EvidenceAwareSelector
from experiments.load_joint_checkpoint import load_joint_checkpoint, stamp_generation_record, accept_cached_generation, CHECKPOINT_LOADER_ID
from experiments.run_quality_benchmark import (
    evaluate_mbpp_code,
    evaluate_gsm8k_reasoning,
    evaluate_alpaca_instruction,
    TimingLogitsProcessor,
    INITIAL_VOCAB,
    MAX_NEW_TOKENS,
)

MODEL_NAME = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
TOKENIZER_NAME = "microsoft/Phi-3.5-mini-instruct"
VAL_DATA_PATH = "data/cached_pure_pred_val_60.json"
PREDICTOR_PATH = "experiments/checkpoints/cached_predictor.pkl"
CKPT_STEP100_PATH = "experiments/checkpoints/predictive_joint_pilot/checkpoint_step_100.pt"
POC_RESULTS_PATH = "experiments/checkpoints/quality_benchmark/poc_selector_results_joint_nested_v1.json"


def codebook_hash(codebook_dict: Dict[Tuple[int, ...], int]) -> str:
    """Deterministic hash of codebook contents."""
    items = sorted(codebook_dict.items())
    repr_str = str(items)
    return hashlib.sha256(repr_str.encode("utf-8")).hexdigest()[:16]


def is_correct(r: Dict[str, Any]) -> bool:
    dom = r.get("domain")
    if dom == "code":
        return r.get("problem_pass", False)
    elif dom == "reasoning":
        return r.get("exact_correct", False)
    elif dom == "instruction":
        return not r.get("instruction_failure", False)
    return False


def run_sweep(
    prompts_path: str,
    k_values: List[int],
    tau_values: List[float],
    out_json: str,
    out_md: str,
    device_str: str = "cpu",
):
    device = torch.device(device_str)
    print("=== Predictive Hypertoken Optimization: K-Sweep & Adaptive K ===", flush=True)

    # 1. Load prompts
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_meta = json.load(f)
    p_ids = [p["id"] if isinstance(p, dict) else p for p in prompts_meta]

    with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
        val_all = json.load(f)
    samples_map = {s["id"]: s for s in val_all}
    samples = [samples_map[pid] for pid in p_ids if pid in samples_map]
    print(f"Loaded {len(samples)} evaluation prompts.", flush=True)

    # 2. Setup tokenizer and predictor
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    with open(PREDICTOR_PATH, "rb") as f:
        raw_pred = pickle.load(f)
    p_index = getattr(raw_pred, "index", raw_pred)

    ev_selector = EvidenceAwareSelector(
        predictor_index=p_index,
        tokenizer=tokenizer,
        budget=32,
        max_structural_slots=0,
    )

    # 3. Cache of codebook generations: (prompt_id, cb_hash) -> generation output dict
    gen_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # Seed cache from existing Phase 3 results if available
    if os.path.exists(POC_RESULTS_PATH):
        try:
            with open(POC_RESULTS_PATH, "r", encoding="utf-8") as f:
                poc = json.load(f)
            for r in poc.get("raw_records", []):
                pid = r["prompt_id"]
                s = samples_map.get(pid)
                if not s or not accept_cached_generation(r):
                    continue
                p_text = s["prompt"]
                p_ids_tok = tokenizer.encode(p_text, add_special_tokens=False)
                if r.get("condition") == "cond_b_evidence_k32":
                    cb, _ = ev_selector.select_codebook(p_ids_tok, prompt_text=p_text, budget=32)
                    chash = codebook_hash(cb)
                    gen_cache[(pid, chash)] = r
                elif r.get("condition") == "cond_c_adaptive_tau20":
                    cb, _ = ev_selector.select_codebook(p_ids_tok, prompt_text=p_text, min_score_threshold=20.0)
                    chash = codebook_hash(cb)
                    gen_cache[(pid, chash)] = r
            print(f"Pre-seeded cache with {len(gen_cache)} generations from Phase 3 POC.", flush=True)
        except Exception as e:
            print(f"Note: Could not pre-seed cache ({e})", flush=True)

    # Load incremental results if resuming sweep
    sweep_data: Dict[str, Any] = {}
    if os.path.exists(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                sweep_data = json.load(f)
                cached_runs = sweep_data.get("cached_generations", {})
                for k_str, val in cached_runs.items():
                    if not accept_cached_generation(val):
                        continue
                    pid, chash = k_str.split("::")
                    gen_cache[(pid, chash)] = val
            print(f"Loaded existing sweep data with {len(cached_runs)} cached generations.", flush=True)
        except Exception:
            pass

    # 4. Load Model
    print("Loading Zip2Zip Step-100 model...", flush=True)
    t0 = time.time()
    model = Zip2ZipModel.from_pretrained(
        MODEL_NAME,
        max_codebook_size=32,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.output_encoder.to(torch.float32)

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

    # Function to execute generation with caching
    def get_generation_for_codebook(
        sample: Dict[str, Any],
        codebook_dict: Dict[Tuple[int, ...], int],
        sel_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        pid = sample["id"]
        dom = sample["domain"]
        chash = codebook_hash(codebook_dict)
        cache_key = (pid, chash)

        if cache_key in gen_cache:
            res = dict(gen_cache[cache_key])
            res["codebook_size"] = len(codebook_dict)
            return res

        prompt_text = sample["prompt"]
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        base_prompt_len = len(prompt_ids)

        # Hypertoken setup
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

        input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
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
        total_wall_time = (sel_meta.get("latency_ms", 0) / 1000.0) + hyper_setup_time_s + t_gen

        gen_ids = out[0, base_prompt_len:].tolist()
        decode_steps = len(gen_ids)

        # Decompress
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

        eos_reached = (len(gen_ids) > 0 and gen_ids[-1] == tokenizer.eos_token_id) or (decode_steps < MAX_NEW_TOKENS)
        ttft = timing_proc.ttft or 0.0

        static_mgr.detach_from_model(model)
        model.codebook_manager.reset()

        emitted_ids_set = {e["id"] for e in hypertokens_emitted}
        used_slots = len(emitted_ids_set)
        cb_size = len(codebook_dict)
        dead_slots = cb_size - used_slots
        utilization_pct = round(used_slots / cb_size * 100, 1) if cb_size > 0 else 0.0

        res = stamp_generation_record({
            "prompt_id": pid,
            "domain": dom,
            "codebook_hash": chash,
            "base_prompt_tokens": base_prompt_len,
            "decode_steps": decode_steps,
            "expanded_output_tokens": expanded_output_tokens,
            "tokens_saved": tokens_saved,
            "decode_reduction_pct": decode_reduction_pct,
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
            asserts = [line.strip() for line in sample["ground_truth_response"].splitlines() if line.strip().startswith("assert")]
            res.update(evaluate_mbpp_code(output_text, asserts))
        elif dom == "reasoning":
            res.update(evaluate_gsm8k_reasoning(output_text, sample["ground_truth_response"]))
        elif dom == "instruction":
            res.update(evaluate_alpaca_instruction(output_text, eos_reached))

        gen_cache[cache_key] = res
        print(f"   [GEN] {pid} (cb_size={cb_size}): steps={decode_steps}, saved={tokens_saved}, dead={dead_slots}, wall={total_wall_time:.1f}s", flush=True)
        return res

    # 5. Run Sweep across all configurations
    configs_results: Dict[str, Any] = {}

    # Define all configurations
    all_configs = []
    for k in k_values:
        all_configs.append((f"fixed_k_{k}", k, None))
    for tau in tau_values:
        all_configs.append((f"adaptive_tau_{tau:.1f}", 32, tau))

    for cfg_name, target_k, min_tau in all_configs:
        print(f"\n>>> Running Config: {cfg_name} (target_k={target_k}, min_tau={min_tau}) <<<", flush=True)
        cfg_records = []
        for s in samples:
            pid = s["id"]
            prompt_text = s["prompt"]
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

            cb, sel_meta = ev_selector.select_codebook(
                prompt_ids,
                prompt_text=prompt_text,
                budget=target_k,
                min_score_threshold=min_tau,
            )

            rec = get_generation_for_codebook(s, cb, sel_meta)
            rec_with_cfg = dict(rec)
            rec_with_cfg["config"] = cfg_name
            cfg_records.append(rec_with_cfg)

        # Aggregate for this config
        tot_steps = sum(r["decode_steps"] for r in cfg_records)
        tot_exp = sum(r["expanded_output_tokens"] for r in cfg_records)
        tot_saved = sum(r["tokens_saved"] for r in cfg_records)
        micro_red = round((1.0 - tot_steps / max(tot_exp, 1)) * 100, 2)
        macro_red = round(sum(r["decode_reduction_pct"] for r in cfg_records) / len(cfg_records), 2)
        tot_correct = sum(1 for r in cfg_records if is_correct(r))
        acc_pct = round(tot_correct / len(cfg_records) * 100, 1)

        tot_cb_slots = sum(r["codebook_size"] for r in cfg_records)
        tot_used_slots = sum(r["used_slots"] for r in cfg_records)
        tot_dead_slots = tot_cb_slots - tot_used_slots
        util_pct = round(tot_used_slots / max(tot_cb_slots, 1) * 100, 1)

        configs_results[cfg_name] = {
            "config": cfg_name,
            "target_k": target_k,
            "min_tau": min_tau,
            "total_correct": f"{tot_correct}/{len(cfg_records)}",
            "accuracy_pct": acc_pct,
            "micro_reduction_pct": micro_red,
            "macro_reduction_pct": macro_red,
            "tokens_saved": tot_saved,
            "mean_hypertokens": round(sum(r["hypertokens_count"] for r in cfg_records) / len(cfg_records), 2),
            "mean_codebook_size": round(tot_cb_slots / len(cfg_records), 1),
            "total_cb_slots": tot_cb_slots,
            "used_slots": tot_used_slots,
            "dead_slots": tot_dead_slots,
            "utilization_pct": util_pct,
            "mean_wall_time_s": round(sum(r["wall_time_s"] for r in cfg_records) / len(cfg_records), 2),
            "mean_ttft_s": round(sum(r["ttft_s"] for r in cfg_records) / len(cfg_records), 3),
            "records": cfg_records,
        }

        print(f"Summary {cfg_name}: Acc={tot_correct}/{len(cfg_records)} ({acc_pct}%), MicroComp={micro_red}%, DeadSlots={tot_dead_slots}, Util={util_pct}%", flush=True)

        # Save intermediate checkpoint
        inter_save = {
            "checkpoint_loader": CHECKPOINT_LOADER_ID,
            "configs": {k: {k2: v2 for k2, v2 in v.items() if k2 != "records"} for k, v in configs_results.items()},
            "cached_generations": {f"{k[0]}::{k[1]}": v for k, v in gen_cache.items()},
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(inter_save, f, indent=2)

    # 6. Final Outputs
    final_output = {
        "checkpoint_loader": CHECKPOINT_LOADER_ID,
        "summary_table": {k: {k2: v2 for k2, v2 in v.items() if k2 != "records"} for k, v in configs_results.items()},
        "configs_detailed": configs_results,
        "cached_generations": {f"{k[0]}::{k[1]}": v for k, v in gen_cache.items()},
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    # Generate Markdown Report
    md_lines = [
        "# Phase 4: K-Sweep & Adaptive-K Pareto Analysis",
        "",
        "This experiment sweeps fixed codebook sizes $K \\in [4, 8, 16, 24, 32]$ and adaptive acceptance thresholds $\\tau \\in [10.0, 15.0, 20.0, 25.0]$ using the EvidenceAwareSelector on the fixed 12-prompt evaluation subset.",
        "The policy rankings below are computed from this run's measurements. Historical narrative claims about diminishing returns or adaptive-policy dominance are not carried forward as corrected findings.",
        "",
        "## 1. Full Policy Comparison",
        "",
        "| Policy / Config | Budget / Tau | Accuracy (Score) | Micro Reduction % | Tokens Saved | Mean Hypers | Mean Allocated K | Dead Slots | Slot Util % | Mean Wall Time |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cfg_name, res in configs_results.items():
        b_str = f"K={res['target_k']}" if res['min_tau'] is None else f"tau={res['min_tau']}"
        md_lines.append(
            f"| **{cfg_name}** | {b_str} | **{res['total_correct']} ({res['accuracy_pct']}%)** | "
            f"**{res['micro_reduction_pct']}%** | {res['tokens_saved']} | {res['mean_hypertokens']} | "
            f"{res['mean_codebook_size']} | **{res['dead_slots']}** | {res['utilization_pct']}% | {res['mean_wall_time_s']}s |"
        )

    # Find best fixed K and best adaptive policy
    fixed_configs = [res for k, res in configs_results.items() if res['min_tau'] is None]
    adaptive_configs = [res for k, res in configs_results.items() if res['min_tau'] is not None]

    # Best fixed: highest accuracy, tie break on micro reduction
    best_fixed = max(fixed_configs, key=lambda x: (x['accuracy_pct'], x['micro_reduction_pct']))
    best_adaptive = max(adaptive_configs, key=lambda x: (x['accuracy_pct'], x['micro_reduction_pct']))

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Pareto Frontier & Policy Recommendations",
        "",
        f"### A. Best Fixed-K Policy: **{best_fixed['config']}**",
        f"- **Accuracy:** {best_fixed['total_correct']} ({best_fixed['accuracy_pct']}%)",
        f"- **Micro Decode Reduction:** {best_fixed['micro_reduction_pct']}% ({best_fixed['tokens_saved']} tokens saved)",
        f"- **Dead Slots:** {best_fixed['dead_slots']} (Capacity Utilization: {best_fixed['utilization_pct']}%)",
        "",
        f"### B. Best Adaptive-K Policy: **{best_adaptive['config']}**",
        f"- **Accuracy:** {best_adaptive['total_correct']} ({best_adaptive['accuracy_pct']}%)",
        f"- **Micro Decode Reduction:** {best_adaptive['micro_reduction_pct']}% ({best_adaptive['tokens_saved']} tokens saved)",
        f"- **Mean Allocated K:** {best_adaptive['mean_codebook_size']} slots",
        f"- **Dead Slots:** {best_adaptive['dead_slots']} (Capacity Utilization: {best_adaptive['utilization_pct']}%)",
        "",
        "### Interpretation:",
        "The best fixed-K and adaptive configurations above are selected from this run's measured accuracy and micro-reduction values. No unconditional dominance or diminishing-returns conclusion is asserted.",
    ])

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nK-Sweep complete! Saved {out_json} and {out_md}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run K-Sweep and Adaptive-K Optimization")
    parser.add_argument("--prompts_file", type=str, default="experiments/checkpoints/quality_benchmark/poc_12_prompt_ids.json")
    parser.add_argument("--k_values", nargs="+", type=int, default=[4, 8, 16, 24, 32])
    parser.add_argument("--tau_values", nargs="+", type=float, default=[10.0, 15.0, 20.0, 25.0])
    parser.add_argument("--out_json", type=str, default="experiments/checkpoints/quality_benchmark/k_sweep_results_joint_nested_v1.json")
    parser.add_argument("--out_md", type=str, default="experiments/checkpoints/quality_benchmark/k_sweep_results_joint_nested_v1.md")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_sweep(
        prompts_path=args.prompts_file,
        k_values=args.k_values,
        tau_values=args.tau_values,
        out_json=args.out_json,
        out_md=args.out_md,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
