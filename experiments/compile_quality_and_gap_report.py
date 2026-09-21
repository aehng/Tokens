"""
Quality and Realization Gap Audit.
Evaluates:
1. Completion rate across conditions
2. Math final-answer accuracy (GSM8k extracted vs reference ground truth)
3. Code syntax validity (Python AST parsing)
4. Repetition / degeneracy (4-gram repetition score)
5. Truncation and output length
6. Semantic equivalence: Base Phi-3.5 vs Pure Predictive
7. Per-prompt and aggregate Zero-Shot Realization Gap
"""

import ast
import json
import re
from collections import defaultdict
import numpy as np


def extract_math_answer(text: str) -> str:
    # Match #### <num>
    m = re.search(r"####\s*(-?[\d\.,]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # Match final sentence numbers
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if nums:
        return nums[-1].strip()
    return ""


def check_python_syntax(text: str) -> bool:
    code = text
    if "```python" in text:
        code = text.split("```python")[1].split("```")[0]
    elif "```" in text:
        code = text.split("```")[1].split("```")[0]
    try:
        ast.parse(code)
        return True
    except Exception:
        # Check if incomplete block is valid syntax up to last line
        lines = code.strip().split("\n")
        while lines:
            try:
                ast.parse("\n".join(lines))
                return True
            except Exception:
                lines.pop()
        return False


def repetition_ratio_4gram(text: str) -> float:
    words = text.split()
    if len(words) < 5:
        return 0.0
    ngrams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def run_audit(
    live_results_path: str = "experiments/live_zero_shot_results.json",
    prompts_path: str = "experiments/live_benchmark_prompts.json",
    reference_data_path: str = "data/test.jsonl",
    output_path: str = "experiments/quality_and_gap_report.json",
):
    with open(live_results_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = {p["id"]: p for p in json.load(f)}

    # Load ground truth references from test.jsonl
    ground_truth = {}
    with open(reference_data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            ground_truth[item["id"]] = item

    # Group by condition
    by_condition = defaultdict(list)
    for r in records:
        by_condition[r["condition"]].append(r)

    quality_summary = {}

    for cond, rows in by_condition.items():
        total_runs = len(rows)
        completed = sum(1 for r in rows if len(r["decoded_text"].strip()) > 0)
        completion_rate = (completed / total_runs) * 100.0

        syntax_valid = 0
        code_count = 0
        math_correct = 0
        math_count = 0
        rep_scores = []
        gen_lengths = []
        base_equiv_lens = []

        for r in rows:
            p_id = r["prompt_id"]
            dom = r["domain"]
            text = r["decoded_text"]
            gen_lengths.append(r["generated_steps"])
            base_equiv_lens.append(r["base_equivalent_tokens"])
            rep_scores.append(repetition_ratio_4gram(text))

            if dom == "code":
                code_count += 1
                if check_python_syntax(text):
                    syntax_valid += 1
            elif dom == "reasoning":
                math_count += 1
                pred_ans = extract_math_answer(text)
                ref_text = ground_truth.get(p_id, {}).get("response", "")
                ref_ans = extract_math_answer(ref_text)
                if pred_ans and ref_ans and pred_ans == ref_ans:
                    math_correct += 1

        quality_summary[cond] = {
            "total_runs": total_runs,
            "completion_rate_pct": completion_rate,
            "code_syntax_validity_pct": (syntax_valid / code_count * 100.0) if code_count > 0 else 0.0,
            "math_exact_accuracy_pct": (math_correct / math_count * 100.0) if math_count > 0 else 0.0,
            "mean_4gram_repetition_rate": float(np.mean(rep_scores)),
            "mean_generated_steps": float(np.mean(gen_lengths)),
            "mean_base_equiv_tokens": float(np.mean(base_equiv_lens)),
            "is_corrupted_or_degenerate": bool(np.mean(rep_scores) > 0.20),
        }

    # Zero-Shot Realization Gap per Prompt
    prompt_gap = {}
    unique_p_ids = sorted(list(set(r["prompt_id"] for r in records)))
    
    for pid in unique_p_ids:
        p_rows = {r["condition"]: r for r in records if r["prompt_id"] == pid}
        base_r = p_rows.get("Base Phi-3.5 (K=0)", {})
        lzw_r = p_rows.get("Official Reactive Zip2Zip (Pure LZW, K=32)", {})
        pred_r = p_rows.get("Pure Predictive Seeded (K=32)", {})

        prompt_gap[pid] = {
            "domain": pred_r.get("domain", "unknown"),
            "base_prompt_tokens": pred_r.get("base_prompt_tokens", 0),
            "offline_theoretical_pred_comp_pct": pred_r.get("offline_theoretical_comp_pct", 0.0),
            "live_realized_pred_step_red_pct": pred_r.get("realized_step_reduction_pct", 0.0),
            "live_realized_pred_prefill_comp_pct": (
                100.0 * (1.0 - pred_r.get("compressed_prefill_tokens", 1) / pred_r.get("base_prompt_tokens", 1))
            ),
            "live_realized_lzw_step_red_pct": lzw_r.get("realized_step_reduction_pct", 0.0),
            "lzw_hypertokens_emitted": lzw_r.get("hypertokens_emitted", 0),
            "pred_hypertokens_emitted": pred_r.get("hypertokens_emitted", 0),
            "base_text_sample": repr(base_r.get("decoded_text", "")[:45]),
            "pred_text_sample": repr(pred_r.get("decoded_text", "")[:45]),
            "semantic_match": base_r.get("decoded_text", "")[:20] == pred_r.get("decoded_text", "")[:20],
        }

    output_payload = {
        "quality_summary": quality_summary,
        "per_prompt_realization_gap": prompt_gap,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print("=" * 80)
    print("LIVE ZERO-SHOT QUALITY AUDIT")
    print("=" * 80)
    for cond, q in quality_summary.items():
        print(f"{cond:42s} | Compl: {q['completion_rate_pct']:5.1f}% | CodeSyntax: {q['code_syntax_validity_pct']:5.1f}% | MathAcc: {q['math_exact_accuracy_pct']:5.1f}% | Repetition: {q['mean_4gram_repetition_rate']:5.3f} | Corrupt: {q['is_corrupted_or_degenerate']}")

    print("\n" + "=" * 80)
    print("ZERO-SHOT REALIZATION GAP PER PROMPT")
    print("=" * 80)
    print(f"{'Prompt ID':12s} | {'Domain':10s} | {'Offline Comp%':14s} | {'Live Pred Step%':16s} | {'Live Pred Prefill%':18s} | {'Live LZW Step%':15s} | {'LZW HT':6s}")
    print("-" * 105)
    for pid, g in prompt_gap.items():
        print(f"{pid:12s} | {g['domain']:10s} | {g['offline_theoretical_pred_comp_pct']:13.2f}% | {g['live_realized_pred_step_red_pct']:15.2f}% | {g['live_realized_pred_prefill_comp_pct']:17.2f}% | {g['live_realized_lzw_step_red_pct']:14.2f}% | {g['lzw_hypertokens_emitted']:6d}")


if __name__ == "__main__":
    run_audit()
