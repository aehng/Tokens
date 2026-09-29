"""Reanalyze the committed 45-prompt Stage 1 DEV records without model inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys_path = str(REPO_ROOT)

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from src.zip2zip.predictor_v2.attribution_reanalysis import analyze_records


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if record.get("split") != "DEV":
                raise ValueError(f"{path}:{line_number} is not DEV; FINAL and other splits are refused")
            records.append(record)
    return records


def _pct(value: float) -> str:
    return f"{value:.2f}%"


def _opportunity_row(label: str, row: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {row['prompt_count']} | {row['base_token_count']} | "
        f"{row['oracle_compressed_step_count']} | {row['tokens_saved']} | "
        f"{_pct(row['compression_pct'])} | {row['mean_codebook_size']:.2f} | "
        f"{row['optimal_h_substitutions_total']} | "
        f"`{row['codebook_phrase_length_distribution_total']}` |"
    )


def _markdown(report: Mapping[str, Any]) -> str:
    agg = report["aggregate"]
    lines = [
        "# Stage 1 Phi Quality and Compression: Offline Reanalysis",
        "",
        "**Status labels:** MEASURED = directly computed from pinned source/records; INFERENCE = supported interpretation; HYPOTHESIS = unverified cause; NOT TESTED = no valid controlled result.",
        "",
        "This report reads only the 45 DEV prompt records in `docs/stage1_attribution_records.jsonl`. It does not load the canonical 900-row dataset, run a model, or access FINAL.",
        "",
        "## 1. Pinned provenance and prior C correction",
        "",
        f"- **MEASURED:** Stage 1 source commit `{report['methodology']['stage1_source_commit']}`; 45 prompts and 180 records; Phi/tokenizer revision `2fe192450127e6a83f7441aef6e3ca586c338b77`; Step-100 checkpoint SHA-256 `2c3606c075ac96dff1f607043f58241251d837f2340ae950d3dc309e9820fd44`.",
        "- **MEASURED:** C's exact supplied codebook is present in each record's `selected_h_slots`. The slots replay the original Vanilla-continuation phrase rule against A's observed sequence on all 45 prompts. C phrases were selected from canonical Vanilla-generated continuations, not the reference answer.",
        "- **MEASURED:** C was **ORACLE CODEBOOK LIVE**: B/C/D share the wrapped Step-100 checkpoint; C received hindsight phrases but used ordinary greedy generation, with the model free to emit H or base tokens.",
        "- **MEASURED:** prior 5.39% = `(16,712 C expanded tokens - 15,811 C decode calls) / 16,712 = 5.39%`. The 704 count is H emission events; they saved 901 base-token steps because H phrases span 2–4 tokens.",
        "- **INFERENCE:** 5.39% is live realized compression on C's own output. It is not the ORACLE CEILING and does not measure all DP-compressible opportunity.",
        "",
        "## 2. ORACLE CEILING: Oracle V2 on each realized sequence",
        "",
        "Oracle V2 uses K=32, beam width 4, candidate limit 80, and the product constraints of valid base IDs 0–32010 excluding 0, 1, 2, and 32000–32010. DP segments the full realized sequence, including terminal IDs as ordinary uncompressed base tokens. Values are near-optimal under the repository's Oracle V2 beam search, not a proof of global optimality.",
        "",
        "| Sequence | Prompts | Base tokens | DP steps | Tokens saved | Compression | Mean used phrases | Optimal H substitutions | Codebook lengths |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        _opportunity_row("A, lengths 2–3", agg["oracle_a_ceiling"]["k32_len2_3"]),
        _opportunity_row("A, lengths 2–4", agg["oracle_a_ceiling"]["k32_len2_4"]),
        _opportunity_row("B, lengths 2–3", agg["oracle_b_ceiling"]["k32_len2_3"]),
        _opportunity_row("B, lengths 2–4", agg["oracle_b_ceiling"]["k32_len2_4"]),
        _opportunity_row("C, lengths 2–3", agg["oracle_c_ceiling"]["k32_len2_3"]),
        _opportunity_row("C, lengths 2–4", agg["oracle_c_ceiling"]["k32_len2_4"]),
        _opportunity_row("D, lengths 2–3", agg["oracle_d_ceiling"]["k32_len2_3"]),
        _opportunity_row("D, lengths 2–4", agg["oracle_d_ceiling"]["k32_len2_4"]),
        "",
        "## 3. A-derived C codebook across A/B/C outputs",
        "",
        "This applies C's exact supplied phrases to A, B, and C sequences. Occurrences count overlapping n-gram matches; DP savings use a non-overlapping optimal tiling.",
        "",
        "| Output sequence | Prompts | Base tokens | DP steps | Tokens saved | Compression | Phrase types occurring | Occurrences | Optimal substitutions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in ("A_vanilla", "B_h_disabled", "C_oracle_codebook_live"):
        row = agg["a_derived_c_codebook_cross_output"][condition]
        label = {"A_vanilla": "A", "B_h_disabled": "B", "C_oracle_codebook_live": "C"}[condition]
        lines.append(
            f"| {label} | {row['prompt_count']} | {row['base_token_count']} | {row['oracle_compressed_step_count']} | "
            f"{row['tokens_saved']} | {_pct(row['compression_pct'])} | {row['supplied_phrases_that_occur_total']} | "
            f"{row['supplied_phrase_occurrences_overlapping_total']} | {row['optimal_h_substitutions_total']} |"
        )
    a_occ = agg["a_derived_c_codebook_cross_output"]["A_vanilla"]
    lines.extend([
        "",
        f"- **MEASURED:** Against A's reference distribution, phrase-type occurrence change is B `{agg['a_derived_c_codebook_cross_output']['B_h_disabled']['supplied_phrases_that_occur_total'] - a_occ['supplied_phrases_that_occur_total']:+d}` and C `{agg['a_derived_c_codebook_cross_output']['C_oracle_codebook_live']['supplied_phrases_that_occur_total'] - a_occ['supplied_phrases_that_occur_total']:+d}`; occurrence-count change is B `{agg['a_derived_c_codebook_cross_output']['B_h_disabled']['supplied_phrase_occurrences_overlapping_total'] - a_occ['supplied_phrase_occurrences_overlapping_total']:+d}` and C `{agg['a_derived_c_codebook_cross_output']['C_oracle_codebook_live']['supplied_phrase_occurrences_overlapping_total'] - a_occ['supplied_phrase_occurrences_overlapping_total']:+d}`.",
        "- **INFERENCE:** A-derived C phrases retain only a portion of their A output opportunity on B and C. B versus A combines the Zip2Zip wrapper and Step-100 checkpoint; C versus B also includes active H effects, so those deltas do not assign the distribution shift to one component.",
        "",
        "## 4. C exact supplied codebook opportunity versus live realization",
        "",
        _opportunity_table(agg["c_exact_supplied_codebook_opportunity"], "ORACLE CODEBOOK LIVE: C supplied codebook on C output"),
        "",
        _realization_table(agg["c_live_realization"], "C live H realization"),
        f"- **MEASURED:** C opportunity-to-realization ratio = `{_format_ratio(agg['c_opportunity_to_realization_ratio'])}` (live savings / C-codebook DP opportunity).",
        f"- **MEASURED:** C's supplied codebook captures `{_format_ratio(agg['c_exact_supplied_codebook_opportunity']['tokens_saved'] / agg['oracle_c_ceiling']['k32_len2_4']['tokens_saved'] if agg['oracle_c_ceiling']['k32_len2_4']['tokens_saved'] else None)}` of its own output's 2–4 ORACLE CEILING opportunity.",
        "",
        "## 5. D exact supplied codebook opportunity versus live realization",
        "",
        _opportunity_table(agg["d_exact_supplied_codebook_opportunity"], "D supplied codebook on D output"),
        "",
        _realization_table(agg["d_live_realization"], "D live H realization"),
        f"- **MEASURED:** D opportunity-to-realization ratio = `{_format_ratio(agg['d_opportunity_to_realization_ratio'])}`.",
        f"- **MEASURED:** D's supplied codebook captures `{_format_ratio(agg['d_exact_supplied_codebook_opportunity']['tokens_saved'] / agg['oracle_d_ceiling']['k32_len2_4']['tokens_saved'] if agg['oracle_d_ceiling']['k32_len2_4']['tokens_saved'] else None)}` of its own output's 2–4 ORACLE CEILING opportunity.",
        "",
        "## 6. Quality evidence and causal limits",
        "",
        "| Condition | Passes / 45 | Aggregate quality | Code | Reasoning | Instruction |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for condition in ("A_vanilla", "B_h_disabled", "C_oracle", "D_real_predictor"):
        q = agg["live_condition_quality"][condition]
        domain = q["by_domain"]
        lines.append(
            f"| {condition} | {q['passes']} / {q['prompts']} | {_pct(q['quality_rate'] * 100)} | "
            f"{domain['code']['passes']}/{domain['code']['prompts']} | "
            f"{domain['reasoning']['passes']}/{domain['reasoning']['prompts']} | "
            f"{domain['instruction']['passes']}/{domain['instruction']['prompts']} |"
        )
    lines.extend([
        "",
        "| Paired comparison | Pass→pass | Pass→fail | Fail→pass | Fail→fail | Quality rate delta |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for key, label in (("A_vanilla_to_B_h_disabled", "A → B"), ("B_h_disabled_to_C_oracle", "B → C"), ("C_oracle_to_D_real_predictor", "C → D")):
        p = agg["paired_quality_comparisons"][key]
        lines.append(
            f"| {label} | {p['pass_to_pass']} | {p['pass_to_fail']} | {p['fail_to_pass']} | {p['fail_to_fail']} | "
            f"{p['aggregate_quality_rate_delta_pp']:+.2f} pp |"
        )
    lines.extend([
        "",
        "- **MEASURED:** A and B differ by 10 task passes on this sample, and B versus C has paired gains/losses shown above. The aggregate equality of B/C rates alone is not evidence that H mechanics are harmless.",
        "- **INFERENCE:** A versus B cannot isolate LoRA because Stage 1 A is native Transformers while B also activates the Zip2Zip wrapper, embeddings/head, and position path. The Stage 1 comparison does not contain B0.",
        "- **NOT TESTED:** A/B0 token equality; B0/B1 isolated weight effect; CF forced representation/state safety; direct Vanilla/B0/B1 base-logit fidelity; P-VANILLA versus P-LORA; CUDA timing comparability.",
        "- **NOT TESTED:** LoRA-target predictor training labels do not exist in the Stage 1 DEV records. Scaling label generation is deferred until the B0/B1 ablation confirms whether Step-100 is a viable serving target.",
        "",
        "## 7. Gates and next diagnostic",
        "",
        "| Gate | Result | Evidence / blocker |",
        "|---|---|---|",
        "| 1 — ceiling, opportunity, realization are separate | **PASS** | Oracle V2 sequence ceiling, exact C/D supplied-codebook DP opportunity, and realized H savings are separately reported below and per prompt in JSON. |",
        "| 2 — architecture fidelity A vs B0 | **NOT TESTED** | No B0 records exist. |",
        "| 3 — LoRA fidelity B0 vs B1 | **NOT TESTED** | Current B combines wrapper and Step-100 weights. |",
        "| 4 — H representation CF | **NOT TESTED** | No forced H continuation was run. |",
        "| 5 — predictor target source | **NOT TESTED** | No verified B1 TRAIN continuations; do not scale while Step-100 viability is unresolved. |",
        "| 6 — root cause | **OPEN** | Existing B quality loss is measured, but it cannot yet be causally assigned to wrapper versus checkpoint. |",
        "",
        "- **INFERENCE — recommended next step:** run the smallest matched A/B0/B1/C/D live subset, starting with 4 Code, 4 Reasoning, and 4 Instruction prompts. Reuse existing A/B1/C/D records for these same prompt IDs; add B0, plus CF only after its CPU checks pass. Capture base-logit fidelity in the same run. Decide whether Step-100 remains a viable target before generating any LoRA-target TRAIN labels.",
        "- **MEASURED:** The archived Stage 1 report states 76.8 minutes; its Kaggle kernel log records 4,076.95 seconds (67.95 minutes) for the benchmark body. The earlier experiment tracker reports approximately 1.65/5.0 cumulative hours; Kaggle's weekly quota is a separate counter. This offline reanalysis adds 0 GPU hours; FINAL remains untouched.",
        "- **NOT TESTED:** No new GPU run or real training was launched in this reanalysis.",
        "",
        "## Per-prompt machine-readable results",
        "",
        "See `experiments/results/phi_quality_compression_reanalysis.json`; it contains the exact K=32 codebooks, all per-prompt sequence counts, DP tilings, occurrence counts, live savings, realization ratios, paired quality outcomes, and source constraints.",
        "",
        "---",
        "",
        "### Section summary",
        "",
        "1. **Provenance:** C used Vanilla-derived hindsight phrases but was free to choose H; 5.39% was realized compression. 2. **Ceiling:** Oracle V2 computed K=32 near-optimal DP ceilings on A and B outputs. 3. **Coverage:** A-derived phrases were tested on A/B/C. 4–5. **Realization:** C/D exact stored codebooks were DP-segmented against their own output and compared with emitted H savings. 6–7. **Causality:** wrapper, LoRA, forced-H state, and predictor target-source causes remain open pending controlled comparisons.",
        "",
    ])
    return "\n".join(lines)


def _opportunity_table(row: Mapping[str, Any], title: str) -> str:
    return (
        f"| Opportunity: {title} | Value |\n|---|---:|\n"
        f"| Prompts | {row['prompt_count']} |\n"
        f"| Base-equivalent tokens | {row['base_token_count']} |\n"
        f"| DP compressed steps | {row['oracle_compressed_step_count']} |\n"
        f"| Available tokens saved | {row['tokens_saved']} |\n"
        f"| Available compression | {_pct(row['compression_pct'])} |\n"
        f"| Mean codebook size | {row['mean_codebook_size']:.2f} |\n"
        f"| Supplied phrase types that occur | {row['supplied_phrases_that_occur_total']} |\n"
        f"| Overlapping phrase occurrences | {row['supplied_phrase_occurrences_overlapping_total']} |\n"
        f"| Optimal H substitutions | {row['optimal_h_substitutions_total']} |\n"
        f"| Codebook phrase lengths | `{row['codebook_phrase_length_distribution_total']}` |"
    )


def _realization_table(row: Mapping[str, Any], title: str) -> str:
    return (
        f"| Live realization: {title} | Value |\n|---|---:|\n"
        f"| Base-equivalent tokens | {row['base_equivalent_tokens']} |\n"
        f"| Transformer decode calls | {row['model_decode_calls']} |\n"
        f"| H emissions | {row['h_emissions']} |\n"
        f"| Realized tokens saved | {row['realized_tokens_saved']} |\n"
        f"| Realized compression | {_pct(row['realized_compression_pct'])} |\n"
        f"| Mean H span length | {row['mean_h_span_length']:.3f} |\n"
        f"| Emitted H phrase lengths | `{row['realized_h_phrase_length_distribution_total']}` |"
    )


def _format_ratio(value: float | None) -> str:
    return "undefined (zero opportunity)" if value is None else f"{value:.4f} ({value * 100:.2f}%)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/stage1_attribution_records.jsonl"))
    parser.add_argument("--json-output", type=Path, default=Path("experiments/results/phi_quality_compression_reanalysis.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("experiments/results/phi_quality_compression_reanalysis.md"))
    args = parser.parse_args()
    report = analyze_records(_load_records(args.input))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {len(report['per_prompt'])} DEV prompt analyses to {args.json_output} and {args.markdown_output}")


if __name__ == "__main__":
    main()
