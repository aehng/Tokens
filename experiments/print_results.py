import json

with open('experiments/comprehensive_suite_results.json', 'r') as f:
    d = json.load(f)

print("=== 1. CAPACITY RESULTS (TEST N=7,512, CLUSTERS=6,994) ===")
print(f"{'K':<5} | {'Regime':<17} | {'Response Comp %':<26} | {'Prompt Comp %':<26} | {'Total Comp %':<26} | {'Util %':<6}")
print("-" * 115)
for k in ['16', '32', '64', '128', '256', '512']:
    for reg in ['lzw_corr', 'domain_static', 'prompt_predictor', 'oracle']:
        r = d['capacity_results'][k][reg]['response_comp_pct']
        p = d['capacity_results'][k][reg]['prompt_comp_pct']
        t = d['capacity_results'][k][reg]['total_comp_pct']
        u = d['capacity_results'][k][reg]['codebook_utilization']
        r_str = f"{r['mean']:.2f}% [{r['ci_95'][0]:.2f}, {r['ci_95'][1]:.2f}]"
        p_str = f"{p['mean']:.2f}% [{p['ci_95'][0]:.2f}, {p['ci_95'][1]:.2f}]"
        t_str = f"{t['mean']:.2f}% [{t['ci_95'][0]:.2f}, {t['ci_95'][1]:.2f}]"
        print(f"{k:<5} | {reg:<17} | {r_str:<26} | {p_str:<26} | {t_str:<26} | {u*100.0:<6.1f}")
    print("-" * 115)

print("\n=== 2. HYBRID FINALISTS AT K=128 (TEST N=7,512) ===")
print(f"{'Hybrid Configuration':<25} | {'Dom':<4} {'Pred':<5} {'LZW':<4} | {'Response Comp %':<26} | {'Prompt Comp %':<26} | {'Total Comp %':<26} | {'Util %':<6}")
print("-" * 125)
for name, s in d['hybrid_finalists'].items():
    cfg = s['config']
    r = s['response_comp_pct']
    p = s['prompt_comp_pct']
    t = s['total_comp_pct']
    u = s['codebook_utilization']
    r_str = f"{r['mean']:.2f}% [{r['ci_95'][0]:.2f}, {r['ci_95'][1]:.2f}]"
    p_str = f"{p['mean']:.2f}% [{p['ci_95'][0]:.2f}, {p['ci_95'][1]:.2f}]"
    t_str = f"{t['mean']:.2f}% [{t['ci_95'][0]:.2f}, {t['ci_95'][1]:.2f}]"
    print(f"{name:<25} | {cfg[0]:<4} {cfg[1]:<5} {cfg[2]:<4} | {r_str:<26} | {p_str:<26} | {t_str:<26} | {u*100.0:<6.1f}")

print("\n=== 3. CONTROLLED PREFIX SCALING ON SAME RESPONSES (K=64) ===")
print(f"{'Cut Length':<10} | {'Sample N':<8} | {'LZW Resp %':<12} | {'Pred Resp %':<12} | {'Dom Resp %':<12} | {'Hyb Resp %':<12} | {'Delta (Pred - LZW)':<18}")
print("-" * 95)
for cut in ['16', '32', '64', '128', '256', '512', '1024', '2048', '4096']:
    if cut in d['controlled_prefix_scaling']:
        s = d['controlled_prefix_scaling'][cut]
        print(f"{cut:<10} | {s['sample_count_N']:<8} | {s['lzw_resp']:<12.2f} | {s['pred_resp']:<12.2f} | {s['dom_resp']:<12.2f} | {s['hyb_resp']:<12.2f} | {s['delta_pred_minus_lzw']:<+18.2f}")

print("\n=== 4. DOMAIN STUDY (K=128) ===")
for dom, s in d['domain_study']['broad_domains'].items():
    res = s['k128']
    print(f"Broad Domain: {dom:<23} (N={s['sample_count']}) | LZW={res['lzw_resp']:.2f}%, DomStatic={res['dom_resp']:.2f}%, Pred={res['pred_resp']:.2f}%, Oracle={res['oracle_resp']:.2f}%")

print("\n=== 5. PROMPT HEURISTIC TASKS (K=128) ===")
for task, s in d['domain_study']['heuristic_tasks'].items():
    res = s['k128']
    print(f"Task: {task:<30} (N={s['sample_count']}) | LZW={res['lzw_resp']:.2f}%, DomStatic={res['dom_resp']:.2f}%, Pred={res['pred_resp']:.2f}%, Oracle={res['oracle_resp']:.2f}%")
