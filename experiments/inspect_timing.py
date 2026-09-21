import json

with open("experiments/repeated_timing_results.json") as f:
    d = json.load(f)

print(f"Summary: mean_prefill_delta_ms = {d['summary']['mean_prefill_delta_ms']:.2f}")
print("Trials:")
for t in d["trials"]:
    print(f"{t['prompt_id']} rep {t['repeat']}: base_pf={t['base_prefill_ms']:.1f}ms, pred_pf={t['pred_prefill_ms']:.1f}ms, delta_pf={t['delta_prefill_ms']:.1f}ms, order={t['order']}, base_len={t['base_prompt_tokens']}, comp_len={t['compressed_prompt_tokens']}")
