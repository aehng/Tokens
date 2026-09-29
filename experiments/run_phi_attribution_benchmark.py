"""Executable Benchmark Runner for Phi Quality/Speed Attribution Experiments.

Supports running Conditions:
  A: Vanilla Phi-3.5-mini-instruct
  B: Predictive Checkpoint (H-disabled, K=0)
  C: Oracle Hindsight (K=32, derived from canonical Vanilla continuation)
  D: Real Current Predictor (Phi-only candidate retrieval + PooledMLP ranker, K=32)

Enforces:
  - Hard constraint against accessing FINAL split.
  - Stratified prompt selection (Code, Reasoning, Instruction).
  - Exact provenance and machine-readable output schemas.
  - Robust resumption without duplicate generation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.zip2zip.model import Zip2ZipModel
from src.zip2zip.static_codebook import StaticCodebookManager
from src.zip2zip.predictor_v2.attribution_harness import (
    ALL_CONDITIONS,
    ATTRIBUTION_RECORD_SCHEMA,
    CANONICAL_EOS_TOKEN_IDS,
    CANONICAL_MODEL_ID,
    CANONICAL_MODEL_REVISION,
    CANONICAL_ZIP2ZIP_ID,
    CANONICAL_ZIP2ZIP_REVISION,
    COND_A_VANILLA,
    COND_B_H_DISABLED,
    COND_C_ORACLE,
    COND_D_REAL_PREDICTOR,
    INITIAL_VOCAB_SIZE,
    MAX_NEW_TOKENS,
    PAD_TOKEN_ID,
    AttributionError,
    AttributionRecord,
    build_canonical_prompt_text,
    build_codebook_dict,
    check_split_safety,
    compute_attribution_summary,
    derive_oracle_codebook_phrases,
    evaluate_output_quality,
    find_first_divergence,
)
from src.zip2zip.predictor_v2.canonical_dataset import load_canonical_dataset, sha256_file
from src.zip2zip.predictor_v2.candidate_retrieval import (
    ConfigurableCandidateGenerator,
    RetrievalStrategy,
    TrainOnlyAssociationIndex,
)
from experiments.load_joint_checkpoint import load_joint_checkpoint
from experiments.run_quality_benchmark import TimingLogitsProcessor, synchronize_device


def current_git_commit() -> str:
    try:
        import subprocess
        commit = os.environ.get("GIT_COMMIT")
        if commit:
            return commit
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def runtime_metadata(device: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "device": device,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return info


def select_stratified_dev_prompts(
    dev_records: Sequence[Any],
    limit: Optional[int] = None,
) -> List[Any]:
    """Select a balanced, stratified subset across Code, Reasoning, and Instruction."""
    by_domain: Dict[str, List[Any]] = {"code": [], "reasoning": [], "instruction": []}
    for r in dev_records:
        dom = getattr(r, "domain", None) or r.get("domain")
        if dom in by_domain:
            by_domain[dom].append(r)

    if limit is None or limit >= len(dev_records):
        ordered = []
        for d in ("code", "reasoning", "instruction"):
            ordered.extend(by_domain[d])
        return ordered

    per_domain = limit // 3
    remainder = limit % 3
    counts = {
        "code": per_domain + (1 if remainder > 0 else 0),
        "reasoning": per_domain + (1 if remainder > 1 else 0),
        "instruction": per_domain,
    }

    selected = []
    for d in ("code", "reasoning", "instruction"):
        selected.extend(by_domain[d][: counts[d]])
    return selected


def load_vanilla_model_and_tokenizer(device: str) -> Tuple[Any, Any]:
    print(f"Loading Vanilla Phi-3.5 from {CANONICAL_MODEL_ID} (rev: {CANONICAL_MODEL_REVISION[:8]})...", flush=True)
    device_obj = torch.device(device)
    dtype = torch.float16 if device_obj.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        CANONICAL_MODEL_ID,
        revision=CANONICAL_MODEL_REVISION,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        CANONICAL_MODEL_ID,
        revision=CANONICAL_MODEL_REVISION,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device_obj)
    model.eval()
    return model, tokenizer


def load_predictive_bundle(
    checkpoint_path: Path,
    device: str,
) -> Tuple[Zip2ZipModel, Any, Dict[str, Any]]:
    print(f"Loading Zip2Zip Predictive Bundle with checkpoint {checkpoint_path}...", flush=True)
    device_obj = torch.device(device)
    dtype = torch.float16 if device_obj.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        CANONICAL_MODEL_ID,
        revision=CANONICAL_MODEL_REVISION,
        trust_remote_code=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        CANONICAL_MODEL_ID,
        revision=CANONICAL_MODEL_REVISION,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = Zip2ZipModel.from_pretrained(
        CANONICAL_ZIP2ZIP_ID,
        base_model=base_model,
        revision=CANONICAL_ZIP2ZIP_REVISION,
        torch_dtype=dtype,
    ).to(device_obj)

    report = load_joint_checkpoint(
        model,
        str(checkpoint_path),
        expected_step=100,
        expected_model_id=CANONICAL_ZIP2ZIP_ID,
    )
    print(f"Loaded checkpoint step {report.get('step')} into Zip2ZipModel.", flush=True)
    model.eval()
    return model, tokenizer, report


def generate_single_prompt(
    model: Any,
    tokenizer: Any,
    input_ids: List[int],
    device: torch.device,
    static_mgr: Optional[StaticCodebookManager] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Tuple[List[int], float, float, int]:
    """Execute forward generation loop and measure accurate TTFT, decode time, and steps."""
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    synchronize_device(device)
    t_start = time.perf_counter()

    timing_proc = TimingLogitsProcessor(t_start, static_mgr=static_mgr)
    proc_list = LogitsProcessorList([timing_proc])

    with torch.no_grad():
        out = model.generate(
            input_ids=input_tensor,
            max_new_tokens=max_new_tokens,
            logits_processor=proc_list,
            do_sample=False,
            pad_token_id=PAD_TOKEN_ID,
            eos_token_id=list(CANONICAL_EOS_TOKEN_IDS),
        )

    synchronize_device(device)
    t_end = time.perf_counter()
    total_gen_time = t_end - t_start
    ttft = timing_proc.ttft if timing_proc.ttft is not None else total_gen_time
    decode_time = max(0.0, total_gen_time - ttft)

    gen_ids = out[0, len(input_ids) :].tolist()
    steps = len(gen_ids)
    return gen_ids, ttft, decode_time, steps


def run_attribution_benchmark(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_attribution_records.jsonl"
    summary_path = output_dir / "attribution_summary.json"
    report_path = output_dir / "attribution_report.md"

    git_commit = current_git_commit()
    runtime = runtime_metadata(args.device)
    device_obj = torch.device(args.device)

    # 1. Load canonical dataset and split
    check_split_safety(args.split)
    views, _ = load_canonical_dataset(args.canonical_dataset, args.canonical_manifest)
    if args.split.upper() == "DEV":
        split_records = getattr(views, "dev", getattr(views, "dev_records", None))
    elif args.split.upper() == "TRAIN":
        split_records = getattr(views, "train", getattr(views, "train_records", None))
    else:
        raise AttributionError(f"Unsupported split: {args.split}")

    # Stratified selection
    target_records = select_stratified_dev_prompts(split_records, limit=args.prompt_limit)
    print(f"Selected {len(target_records)} prompts from {args.split} split.")

    # Load existing completions for resume
    completed: Set[Tuple[str, str]] = set()
    existing_records: List[Dict[str, Any]] = []
    if raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        pair = (rec.get("prompt_id"), rec.get("condition"))
                        if pair[0] and pair[1]:
                            completed.add(pair)
                            existing_records.append(rec)
                    except json.JSONDecodeError:
                        pass
        print(f"Resumed: loaded {len(completed)} existing prompt-condition records from {raw_path}")

    # Build reference lookup for Vanilla continuations (from canonical dataset)
    vanilla_continuation_map: Dict[str, List[int]] = {
        r.prompt_id: list(r.continuation_token_ids) for r in split_records
    }

    # Load ground-truth references if available
    gt_map: Dict[str, str] = {}
    gt_path = getattr(args, "ground_truth", None)
    if not gt_path and args.canonical_dataset:
        candidate_gt = Path(args.canonical_dataset).parent / "dev_ground_truth.json"
        if candidate_gt.is_file():
            gt_path = str(candidate_gt)
    if gt_path and Path(gt_path).is_file():
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_map = json.load(f)
        print(f"Loaded ground truth references for {len(gt_map)} prompts.")

    # Initialize predictor and candidate generator if Condition D is active
    cand_index = None
    ranker_model = None
    if COND_D_REAL_PREDICTOR in args.conditions:
        print("Initializing Phi-only Candidate Retrieval Index and PooledMLP ranker...", flush=True)
        idx_path = Path(args.association_index)
        cand_index = TrainOnlyAssociationIndex.load(str(idx_path))
        with open(args.predictor_model, "rb") as pf:
            ranker_model = pickle.load(pf)

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_sha = sha256_file(checkpoint_path) if checkpoint_path.is_file() else ""

    # Open raw file in append mode
    raw_file = raw_path.open("a", encoding="utf-8", buffering=1)

    # 2. RUN CONDITION A: VANILLA (if selected and needed)
    if COND_A_VANILLA in args.conditions:
        pending_a = [r for r in target_records if (r.prompt_id, COND_A_VANILLA) not in completed]
        if pending_a:
            v_model, v_tok = load_vanilla_model_and_tokenizer(args.device)
            for idx, sample in enumerate(pending_a, 1):
                pid = sample.prompt_id
                dom = sample.domain
                print(f"[A: Vanilla {idx}/{len(pending_a)}] Prompt {pid} ({dom})...", flush=True)

                prompt_text = build_canonical_prompt_text(asdict(sample))
                input_ids = v_tok.encode(prompt_text, add_special_tokens=False)

                t_start = time.perf_counter()
                gen_ids, ttft, dec_time, steps = generate_single_prompt(
                    v_model, v_tok, input_ids, device_obj, static_mgr=None, max_new_tokens=args.max_new_tokens
                )
                total_lat = time.perf_counter() - t_start

                eos_hit = bool(gen_ids and gen_ids[-1] in CANONICAL_EOS_TOKEN_IDS)
                term_reason = "eos" if eos_hit else "length"
                term_token = gen_ids[-1] if gen_ids else None

                output_text = v_tok.decode(gen_ids, skip_special_tokens=True)
                sample_dict = asdict(sample)
                if pid in gt_map:
                    sample_dict["reference_response"] = gt_map[pid]
                sample_dict["eos_reached"] = eos_hit
                pass_status, scores = evaluate_output_quality(dom, output_text, sample_dict)

                rec = AttributionRecord(
                    prompt_id=pid,
                    domain=dom,
                    split=sample.split,
                    condition=COND_A_VANILLA,
                    created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    git_commit=git_commit,
                    runtime=runtime,
                    raw_output=output_text,
                    expanded_output=output_text,
                    generated_token_ids=gen_ids,
                    expanded_token_ids=gen_ids,
                    generated_token_count=len(gen_ids),
                    expanded_token_count=len(gen_ids),
                    termination_reason=term_reason,
                    termination_token_id=term_token,
                    eos_reached=eos_hit,
                    truncated=(not eos_hit and len(gen_ids) >= args.max_new_tokens),
                    task_quality_scores=scores,
                    quality_gate_pass=pass_status,
                    ttft_s=round(ttft, 4),
                    decode_time_s=round(dec_time, 4),
                    tpot_s=round(dec_time / max(steps, 1), 6) if steps > 0 else None,
                    total_latency_s=round(total_lat, 4),
                    time_to_eos_s=round(total_lat, 4),
                    transformer_decode_calls=steps,
                    raw_model_steps=steps,
                    words_per_sec=round(len(output_text.split()) / max(total_lat, 1e-4), 2),
                )
                rec_dict = rec.to_dict()
                raw_file.write(json.dumps(rec_dict, ensure_ascii=False) + "\n")
                completed.add((pid, COND_A_VANILLA))
                existing_records.append(rec_dict)

            del v_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 3. RUN CONDITIONS B, C, D: PREDICTIVE SUITE
    pred_conditions = [c for c in (COND_B_H_DISABLED, COND_C_ORACLE, COND_D_REAL_PREDICTOR) if c in args.conditions]
    has_pending_pred = any((r.prompt_id, c) not in completed for r in target_records for c in pred_conditions)

    if pred_conditions and has_pending_pred:
        p_model, p_tok, load_rep = load_predictive_bundle(checkpoint_path, args.device)
        dim = p_model.zip2zip_config.embedding_dim or 3072

        for cond in pred_conditions:
            pending = [r for r in target_records if (r.prompt_id, cond) not in completed]
            if not pending:
                continue

            print(f"\n--- Running Condition {cond} on {len(pending)} prompts ---", flush=True)

            for idx, sample in enumerate(pending, 1):
                pid = sample.prompt_id
                dom = sample.domain
                print(f"[{cond} {idx}/{len(pending)}] Prompt {pid} ({dom})...", flush=True)

                prompt_text = build_canonical_prompt_text(asdict(sample))
                input_ids = p_tok.encode(prompt_text, add_special_tokens=False)

                t_setup_start = time.perf_counter()
                selected_phrases: List[Tuple[int, ...]] = []
                cands_supplied = 0

                if cond == COND_B_H_DISABLED:
                    codebook_dict = {}
                elif cond == COND_C_ORACLE:
                    # Derived from canonical Vanilla continuation
                    cont_tokens = vanilla_continuation_map.get(pid, [])
                    selected_phrases = derive_oracle_codebook_phrases(
                        cont_tokens, k=args.k, min_len=2, max_len=4
                    )
                    codebook_dict = build_codebook_dict(selected_phrases, INITIAL_VOCAB_SIZE)
                    cands_supplied = len(selected_phrases)
                elif cond == COND_D_REAL_PREDICTOR:
                    # Phi-only candidate retrieval + PooledMLP ranker
                    cand_gen = ConfigurableCandidateGenerator(cand_index, tokenizer=p_tok)
                    legacy_rec = sample.to_legacy_record(p_tok)
                    cand_records = cand_gen.build_candidate_records(
                        legacy_rec,
                        strategy=RetrievalStrategy.EXPANDED_ASSOCIATIONS,
                        target_pool_size=args.pool_size,
                    )
                    cands_supplied = len(cand_records)
                    ranked = ranker_model.rank_codebook(input_ids, cand_records, domain=dom, k=args.k)
                    selected_phrases = [tuple(c.tokens) for c, _ in ranked]
                    codebook_dict = build_codebook_dict(selected_phrases, INITIAL_VOCAB_SIZE)

                static_mgr = StaticCodebookManager(
                    initial_vocab_size=INITIAL_VOCAB_SIZE,
                    max_codebook_size=32,
                    max_subtokens=4,
                    embedding_dim=dim,
                    pad_token_id=PAD_TOKEN_ID,
                )
                static_mgr.set_seeded_codebook(codebook_dict, batch_size=1, device=device_obj)
                static_mgr.attach_to_model(p_model)
                setup_time_s = time.perf_counter() - t_setup_start

                t_gen_start = time.perf_counter()
                gen_ids, ttft, dec_time, steps = generate_single_prompt(
                    p_model, p_tok, input_ids, device_obj, static_mgr=static_mgr, max_new_tokens=args.max_new_tokens
                )
                gen_time_s = time.perf_counter() - t_gen_start
                total_lat = setup_time_s + gen_time_s

                # Hypertoken expansion
                hyper_to_subtokens = {v: list(k) for k, v in codebook_dict.items()}
                expanded_tokens: List[int] = []
                h_emissions: List[Dict[str, Any]] = []
                emitted_hyper_ids: Set[int] = set()
                first_h_pos = None

                for pos, tid in enumerate(gen_ids):
                    if tid in hyper_to_subtokens:
                        if first_h_pos is None:
                            first_h_pos = pos
                        phrase_toks = hyper_to_subtokens[tid]
                        phrase_text = p_tok.decode(phrase_toks)
                        h_emissions.append({"pos": pos, "id": tid, "phrase": phrase_text, "subtokens": phrase_toks})
                        emitted_hyper_ids.add(tid)
                        expanded_tokens.extend(phrase_toks)
                    else:
                        expanded_tokens.append(tid)

                raw_output = p_tok.decode(gen_ids, skip_special_tokens=True)
                expanded_output = p_tok.decode(expanded_tokens, skip_special_tokens=True)

                eos_hit = bool(gen_ids and gen_ids[-1] in CANONICAL_EOS_TOKEN_IDS)
                term_reason = "eos" if eos_hit else "length"
                term_token = gen_ids[-1] if gen_ids else None

                sample_dict = asdict(sample)
                if pid in gt_map:
                    sample_dict["reference_response"] = gt_map[pid]
                sample_dict["eos_reached"] = eos_hit
                pass_status, scores = evaluate_output_quality(dom, expanded_output, sample_dict)

                # Diagnostics
                vanilla_ids = vanilla_continuation_map.get(pid, [])
                vanilla_str = p_tok.decode(vanilla_ids)
                phrase_occurs = [
                    bool(p_tok.decode(list(p)) in vanilla_str) for p in selected_phrases
                ]
                dead_slots = [
                    codebook_dict[p] for p in selected_phrases if codebook_dict[p] not in emitted_hyper_ids
                ]
                utilization = (
                    (len(emitted_hyper_ids) / len(selected_phrases) * 100.0)
                    if selected_phrases
                    else 0.0
                )
                div_pos = find_first_divergence(expanded_tokens, vanilla_ids)
                tokens_after_h = (len(gen_ids) - 1 - h_emissions[-1]["pos"]) if h_emissions else None

                peak_vram = torch.cuda.max_memory_allocated(device_obj) if device_obj.type == "cuda" else None

                rec = AttributionRecord(
                    prompt_id=pid,
                    domain=dom,
                    split=sample.split,
                    condition=cond,
                    created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    checkpoint_name=checkpoint_path.name,
                    checkpoint_sha256=checkpoint_sha,
                    git_commit=git_commit,
                    runtime=runtime,
                    raw_output=raw_output,
                    expanded_output=expanded_output,
                    generated_token_ids=gen_ids,
                    expanded_token_ids=expanded_tokens,
                    generated_token_count=len(gen_ids),
                    expanded_token_count=len(expanded_tokens),
                    termination_reason=term_reason,
                    termination_token_id=term_token,
                    eos_reached=eos_hit,
                    truncated=(not eos_hit and len(gen_ids) >= args.max_new_tokens),
                    task_quality_scores=scores,
                    quality_gate_pass=pass_status,
                    candidates_supplied=cands_supplied,
                    selected_h_slots=[list(p) for p in selected_phrases],
                    phrase_length_per_slot=[len(p) for p in selected_phrases],
                    phrase_occurs_in_vanilla=phrase_occurs,
                    h_emissions=h_emissions,
                    dead_h_slots=dead_slots,
                    h_utilization_pct=round(utilization, 2),
                    first_h_emission_pos=first_h_pos,
                    first_divergence_from_vanilla_pos=div_pos,
                    continuation_tokens_after_last_h=tokens_after_h,
                    ttft_s=round(ttft, 4),
                    decode_time_s=round(dec_time, 4),
                    tpot_s=round(dec_time / max(steps, 1), 6) if steps > 0 else None,
                    total_latency_s=round(total_lat, 4),
                    time_to_eos_s=round(total_lat, 4),
                    transformer_decode_calls=steps,
                    raw_model_steps=steps,
                    words_per_sec=round(len(expanded_output.split()) / max(total_lat, 1e-4), 2),
                    setup_time_s=round(setup_time_s, 4),
                    peak_vram_bytes=peak_vram,
                )
                rec_dict = rec.to_dict()
                raw_file.write(json.dumps(rec_dict, ensure_ascii=False) + "\n")
                completed.add((pid, cond))
                existing_records.append(rec_dict)

        del p_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_file.close()

    # 4. Generate Machine-Readable Summary and Markdown Report
    summary = compute_attribution_summary(existing_records)
    summary["evaluated_prompts_count"] = len({r["prompt_id"] for r in existing_records})
    summary["split"] = args.split
    summary["git_commit"] = git_commit
    summary["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()

    with summary_path.open("w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2, ensure_ascii=False)
    print(f"\nWrote machine-readable summary to {summary_path}")

    # Generate Markdown Report
    md_lines = [
        f"# Phi Quality/Speed Attribution Benchmark Report",
        f"",
        f"- **Generated at**: `{summary['completed_at_utc']}`",
        f"- **Git Commit**: `{git_commit}`",
        f"- **Split**: `{args.split}` (Total Prompts: `{summary['evaluated_prompts_count']}`)",
        f"- **Records Generated**: `{len(existing_records)}`",
        f"",
        f"---",
        f"",
        f"## 1. Executive Summary & Causal Diagnosis",
        f"",
        f"- **Primary Bottleneck**: `{summary.get('causal_diagnosis', {}).get('primary_bottleneck', 'unclear')}`",
        f"- **Diagnosis**: {summary.get('causal_diagnosis', {}).get('detailed_diagnosis', 'N/A')}",
        f"",
        f"---",
        f"",
        f"## 2. Condition Overview",
        f"",
        f"| Condition | Prompts | Agg Quality | MBPP Pass | GSM8K Pass | Alpaca Pass | Mean Latency (s) | Total Steps | Compression | H Emissions |",
        f"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for c in ALL_CONDITIONS:
        s = summary["conditions"].get(c)
        if not s:
            continue
        dq = s.get("domain_quality", {})
        code_p = f"{dq.get('code', {}).get('rate', 0.0) * 100:.1f}%" if 'code' in dq else "N/A"
        gsm_p = f"{dq.get('reasoning', {}).get('rate', 0.0) * 100:.1f}%" if 'reasoning' in dq else "N/A"
        alp_p = f"{dq.get('instruction', {}).get('rate', 0.0) * 100:.1f}%" if 'instruction' in dq else "N/A"
        md_lines.append(
            f"| **{c}** | {s['record_count']} | **{s['aggregate_quality_rate']*100:.1f}%** | "
            f"{code_p} | {gsm_p} | {alp_p} | {s['mean_total_latency_s']:.3f} | "
            f"{s['total_decode_calls']} | {s['realized_compression_pct']:.1f}% | {s['total_h_emissions']} |"
        )

    md_lines.extend([
        f"",
        f"---",
        f"",
        f"## 3. Comparison vs Vanilla (<= 3% Quality Gate)",
        f"",
        f"| Condition | Abs Quality Diff | Rel Quality Drop | Meets <=3% Gate? | Speedup vs Vanilla | Faster? | Steps Saved |",
        f"|---|---:|---:|:---:|---:|:---:|---:|",
    ])

    for c in (COND_B_H_DISABLED, COND_C_ORACLE, COND_D_REAL_PREDICTOR):
        comp = summary["comparison_vs_vanilla"].get(c)
        if not comp:
            continue
        gate_icon = "PASS" if comp["meets_3pct_quality_gate"] else "**FAIL**"
        faster_icon = "YES" if comp["is_faster_than_vanilla"] else "NO"
        md_lines.append(
            f"| **{c}** | {comp['absolute_quality_diff']*100:+.1f}% | {comp['relative_quality_drop_pct']:.2f}% | "
            f"{gate_icon} | {comp['speedup_pct']:+.1f}% | {faster_icon} | {comp['decode_steps_saved']} |"
        )

    md_lines.append("")
    with report_path.open("w", encoding="utf-8") as rf:
        rf.write("\n".join(md_lines) + "\n")
    print(f"Wrote human-readable report to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled Phi quality/speed attribution experiment.")
    parser.add_argument("--canonical-dataset", default="data/canonical_phi_continuations.jsonl")
    parser.add_argument("--canonical-manifest", default="data/canonical_phi_continuations.manifest.json")
    parser.add_argument("--ground-truth", default=None, help="Optional path to DEV ground truth references JSON")
    parser.add_argument("--association-index", default="experiments/checkpoints/train_only_association_index.pkl")
    parser.add_argument("--checkpoint", default="experiments/checkpoints/predictive_joint_pilot/checkpoint_step_100.pt")
    parser.add_argument("--predictor-model", default="experiments/checkpoints/predictor_v2/pooledmlp.pkl")
    parser.add_argument("--conditions", nargs="+", default=list(ALL_CONDITIONS))
    parser.add_argument("--split", default="DEV", choices=["TRAIN", "DEV"], help="Split to run. FINAL is forbidden.")
    parser.add_argument("--prompt-limit", type=int, default=None, help="Number of prompts to evaluate.")
    parser.add_argument("--output-dir", required=True, help="Directory to store raw results and summaries.")
    parser.add_argument("--device", default="cpu", help="Device to use, e.g. cpu or cuda:0")
    parser.add_argument("--k", type=int, default=32, help="Hypertoken budget K.")
    parser.add_argument("--pool-size", type=int, default=1024, help="Candidate pool size.")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    run_attribution_benchmark(args)


if __name__ == "__main__":
    main()
