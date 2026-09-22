"""Phase 1: Continuation-Equivalence Diagnostic Suite.

Tests whether consuming a hypertoken preserves the next-token distribution
and continuation trajectory compared to consuming the original constituent base tokens.

For a context C and phrase P = [t_1, ..., t_n] (2 <= n <= 4):
  Path A (Base):  Context + [t_1, ..., t_n]
  Path B (Hyper): Context + [H(P)]

Measures:
  1. KL divergence D_KL(P_base || P_hyper) over base vocabulary
  2. Top-1 agreement (argmax matches)
  3. Top-5 token overlap
  4. Top-10 token overlap
  5. Hidden-state cosine similarity at last layer
  6. Probability assigned to base argmax token
  7. Multi-step greedy continuation match (next 5 tokens)
Controls for position handling using base_token_end RoPE semantic positions.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
from transformers import AutoTokenizer, LogitsProcessorList

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from zip2zip import Zip2ZipModel, StaticCodebookManager

INITIAL_VOCAB = 32011
MAX_SUBTOKENS = 4


def run_continuation_probe(
    model: Zip2ZipModel,
    tokenizer: Any,
    context_text: str,
    phrase_text: str,
    category: str = "general",
    max_next_tokens: int = 5,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compare continuation distribution after base tokens vs after hypertoken."""
    context_ids = tokenizer.encode(context_text, add_special_tokens=False)
    phrase_ids = tokenizer.encode(phrase_text, add_special_tokens=False)

    if not (2 <= len(phrase_ids) <= MAX_SUBTOKENS):
        return {
            "error": f"Phrase '{phrase_text}' has {len(phrase_ids)} tokens (must be between 2 and {MAX_SUBTOKENS})"
        }

    dim = model.zip2zip_config.encoder.hidden_size
    pad_id = tokenizer.pad_token_id or 32000
    disabled_ids = list(model.zip2zip_config.compression.disabled_ids)

    # -------------------------------------------------------------
    # Path A: Context + Base Tokens
    # -------------------------------------------------------------
    base_input_ids = context_ids + phrase_ids
    input_a = torch.tensor([base_input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out_a = model.base_model(input_a, output_hidden_states=True)
        # Logits at the final token position over base vocabulary
        logits_a = out_a.logits[0, -1, :INITIAL_VOCAB].float()
        log_prob_a = F.log_softmax(logits_a, dim=-1)
        prob_a = F.softmax(logits_a, dim=-1)
        hidden_a = out_a.hidden_states[-1][0, -1, :].float()

        # Generate 5 greedy continuation tokens
        gen_a = model.base_model.generate(
            input_ids=input_a,
            max_new_tokens=max_next_tokens,
            do_sample=False,
        )
        cont_tokens_a = gen_a[0, len(base_input_ids):].tolist()
        cont_text_a = tokenizer.decode(cont_tokens_a)

    # -------------------------------------------------------------
    # Path B: Context + Hypertoken (Controlled Semantic RoPE Position)
    # -------------------------------------------------------------
    hyper_id = INITIAL_VOCAB
    seeded_dict = {tuple(phrase_ids): hyper_id}
    hyper_input_ids = context_ids + [hyper_id]

    static_mgr = StaticCodebookManager(
        initial_vocab_size=INITIAL_VOCAB,
        max_codebook_size=1,
        max_subtokens=MAX_SUBTOKENS,
        embedding_dim=dim,
        pad_token_id=pad_id,
        disabled_ids=disabled_ids,
    )
    static_mgr.set_seeded_codebook(seeded_dict, batch_size=1, device=torch.device(device))
    static_mgr.attach_to_model(model)

    input_b = torch.tensor([hyper_input_ids], dtype=torch.long, device=device)

    # RoPE Position: hypertoken lands at the semantic base-token-end position
    # (position of the final constituent base token)
    sem_pos_b = list(range(len(context_ids))) + [len(context_ids) + len(phrase_ids) - 1]
    pos_b_tensor = torch.tensor([sem_pos_b], dtype=torch.long, device=device)

    logits_proc = LogitsProcessorList([static_mgr.get_logits_processor()])

    with torch.no_grad():
        out_b = model.base_model(input_b, position_ids=pos_b_tensor, output_hidden_states=True)
        # Logits at the hypertoken position over base vocabulary
        logits_b = out_b.logits[0, -1, :INITIAL_VOCAB].float()
        log_prob_b = F.log_softmax(logits_b, dim=-1)
        prob_b = F.softmax(logits_b, dim=-1)
        hidden_b = out_b.hidden_states[-1][0, -1, :].float()

        # Generate greedy continuation after hypertoken
        gen_b = model.generate(
            input_ids=input_b,
            max_new_tokens=max_next_tokens,
            logits_processor=logits_proc,
            do_sample=False,
        )
        cont_tokens_b = gen_b[0, len(hyper_input_ids):].tolist()
        expanded_tokens_b = []
        for tid in cont_tokens_b:
            if tid in static_mgr.hyper_to_subtokens:
                expanded_tokens_b.extend(static_mgr.hyper_to_subtokens[tid])
            else:
                expanded_tokens_b.append(tid)
        cont_text_b = tokenizer.decode(expanded_tokens_b)

    static_mgr.detach_from_model(model)

    # -------------------------------------------------------------
    # Metric Computations
    # -------------------------------------------------------------
    # Stable KL divergence: D_KL(P_A || P_B) = sum(P_A * (log P_A - log P_B))
    kl_div = F.kl_div(log_prob_b, log_prob_a, log_target=True, reduction="sum").item()

    # Top-k metrics
    top1_a = torch.argmax(prob_a).item()
    top1_b = torch.argmax(prob_b).item()
    top1_match = (top1_a == top1_b)

    top5_a = set(torch.topk(prob_a, 5).indices.tolist())
    top5_b = set(torch.topk(prob_b, 5).indices.tolist())
    top5_overlap = len(top5_a & top5_b) / 5.0

    top10_a = set(torch.topk(prob_a, 10).indices.tolist())
    top10_b = set(torch.topk(prob_b, 10).indices.tolist())
    top10_overlap = len(top10_a & top10_b) / 10.0

    # Hidden state cosine similarity
    cos_sim = F.cosine_similarity(hidden_a.unsqueeze(0), hidden_b.unsqueeze(0)).item()

    # Probability assigned to base model's top choice
    p_base_top = prob_a[top1_a].item()
    p_hyper_top = prob_b[top1_a].item()

    return {
        "context": context_text,
        "phrase": phrase_text,
        "category": category,
        "phrase_tokens": phrase_ids,
        "phrase_len": len(phrase_ids),
        "kl_divergence": round(kl_div, 4),
        "top1_match": top1_match,
        "top1_base_token": tokenizer.decode([top1_a]),
        "top1_hyper_token": tokenizer.decode([top1_b]),
        "top5_overlap": round(top5_overlap, 3),
        "top10_overlap": round(top10_overlap, 3),
        "hidden_cosine_sim": round(cos_sim, 4),
        "base_p_at_top1": round(p_base_top, 4),
        "hyper_p_at_top1": round(p_hyper_top, 4),
        "continuation_base": repr(cont_text_a),
        "continuation_hyper": repr(cont_text_b),
        "continuation_match": cont_text_a.strip() == cont_text_b.strip(),
    }


def evaluate_continuation_suite(model: Zip2ZipModel, tokenizer: Any, output_path: str = None) -> List[Dict[str, Any]]:
    """Run continuation probe across diverse phrase categories (2-4 subtokens each)."""
    test_cases = [
        # 1. Normal semantic phrases
        ("The primary advantage of this approach is", " that it allows", "semantic"),
        ("In this paper, we propose a novel method for", " large language models", "semantic"),
        ("According to the recent experimental results,", " the proposed system", "semantic"),
        ("The main conclusion of the investigation is,", " in other words", "semantic"),

        # 2. Code phrases
        ("def compute_metrics(eval_preds):\n   ", " return {", "code"),
        ("import torch\nimport torch.nn as nn\n\n", " def helper", "code"),
        ("results = []\n", " for item in", "code"),
        ("class ModelConfig:\n    def __init__(self):\n       ", " self.name", "code"),

        # 3. Structural / Indentation
        ("def process_data(items):\n", "    return", "structure"),
        ("def parse_tree():\n", "    #", "structure"),
        ("class Handler:\n    def reset(self):\n   ", "    self", "structure"),

        # 4. Numbers / Units
        ("The patient's age was verified as", " 25", "numeric"),
        ("The maximum score allowed is", " 100", "numeric"),

        # 5. Repeated phrases
        ("The quick brown fox jumps over", " the lazy dog", "repeated"),
        ("Recent breakthroughs in deep learning and", " neural networks", "repeated"),
    ]

    print(f"\n{'='*80}", flush=True)
    print(f"CONTINUATION EQUIVALENCE DIAGNOSTIC SUITE ({len(test_cases)} probes)", flush=True)
    print(f"{'='*80}", flush=True)

    results = []
    for i, (ctx, phrase, cat) in enumerate(test_cases, 1):
        res = run_continuation_probe(model, tokenizer, ctx, phrase, category=cat)
        results.append(res)

        print(f"Probe {i:2d} [{cat:9s}] '{phrase.strip()}':", flush=True)
        print(f"  KL Div: {res['kl_divergence']:6.2f} | Cos Sim: {res['hidden_cosine_sim']:6.4f} | "
              f"Top-1 Match: {res['top1_match']} | Top-5 Overlap: {res['top5_overlap']:.2f}", flush=True)
        print(f"  Continuation Base:  {res['continuation_base']}", flush=True)
        print(f"  Continuation Hyper: {res['continuation_hyper']}", flush=True)
        print(f"  Match: {res['continuation_match']}", flush=True)
        print("-" * 60, flush=True)

    # Summary
    mean_kl = sum(r["kl_divergence"] for r in results) / len(results)
    mean_cos = sum(r["hidden_cosine_sim"] for r in results) / len(results)
    top1_acc = sum(1 for r in results if r["top1_match"]) / len(results)
    top5_acc = sum(r["top5_overlap"] for r in results) / len(results)
    cont_match = sum(1 for r in results if r["continuation_match"]) / len(results)

    # Category summaries
    categories = sorted(list(set(r["category"] for r in results)))
    cat_summaries = {}
    for cat in categories:
        cat_probes = [r for r in results if r["category"] == cat]
        cat_summaries[cat] = {
            "count": len(cat_probes),
            "mean_kl": round(sum(r["kl_divergence"] for r in cat_probes) / len(cat_probes), 4),
            "mean_cos_sim": round(sum(r["hidden_cosine_sim"] for r in cat_probes) / len(cat_probes), 4),
            "top1_agreement": round(sum(1 for r in cat_probes if r["top1_match"]) / len(cat_probes), 4),
            "top5_overlap": round(sum(r["top5_overlap"] for r in cat_probes) / len(cat_probes), 4),
            "continuation_match_rate": round(sum(1 for r in cat_probes if r["continuation_match"]) / len(cat_probes), 4),
        }

    sem_summary = cat_summaries.get("semantic", {})

    print(f"\nSUMMARY ACROSS ALL {len(results)} PROBES:")
    print(f"  Overall Mean KL Divergence:    {mean_kl:.3f}")
    print(f"  Overall Mean Cosine Sim:       {mean_cos:.4f}")
    print(f"  Overall Top-1 Agreement Rate:  {top1_acc*100:.1f}%")
    print(f"  Overall Top-5 Overlap:         {top5_acc*100:.1f}%")
    print(f"  Overall 5-Step Cont. Match:    {cont_match*100:.1f}%")
    print(f"\nSEMANTIC SUBSET ({sem_summary.get('count', 0)} probes):")
    print(f"  Semantic Mean KL Divergence:   {sem_summary.get('mean_kl', 0.0):.3f}")
    print(f"  Semantic Mean Cosine Sim:      {sem_summary.get('mean_cos_sim', 0.0):.4f}")
    print(f"  Semantic Top-1 Agreement Rate: {sem_summary.get('top1_agreement', 0.0)*100:.1f}%")
    print(f"  Semantic Top-5 Overlap:        {sem_summary.get('top5_overlap', 0.0)*100:.1f}%")
    print(f"  Semantic 5-Step Cont. Match:   {sem_summary.get('continuation_match_rate', 0.0)*100:.1f}%")
    print(f"{'='*80}\n")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "mean_kl": round(mean_kl, 4),
                    "mean_cos_sim": round(mean_cos, 4),
                    "top1_agreement": round(top1_acc, 4),
                    "top5_overlap": round(top5_acc, 4),
                    "continuation_match_rate": round(cont_match, 4),
                    "semantic_mean_kl": sem_summary.get("mean_kl", 0.0),
                    "semantic_mean_cos_sim": sem_summary.get("mean_cos_sim", 0.0),
                    "semantic_top1_agreement": sem_summary.get("top1_agreement", 0.0),
                    "semantic_top5_overlap": sem_summary.get("top5_overlap", 0.0),
                    "semantic_continuation_match_rate": sem_summary.get("continuation_match_rate", 0.0),
                },
                "category_summaries": cat_summaries,
                "probes": results,
            }, f, indent=2)
        print(f"Saved probe report to {output_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate continuation equivalence.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint pt file")
    parser.add_argument("--output", type=str, default="experiments/checkpoints/continuation_equivalence_baseline.json")
    args = parser.parse_args()

    model_id = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
    tok = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    m = Zip2ZipModel.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint weights from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if "lora_state_dict" in ckpt:
            for k, v in ckpt["lora_state_dict"].items():
                m.base_model.load_state_dict({k: v}, strict=False)
        if "input_encoder_state_dict" in ckpt:
            m.input_encoder.load_state_dict(ckpt["input_encoder_state_dict"], strict=False)
        if "output_encoder_state_dict" in ckpt and getattr(m, "output_encoder", None) is not None:
            m.output_encoder.load_state_dict(ckpt["output_encoder_state_dict"], strict=False)
        print("Checkpoint weights loaded successfully.")

    m.eval()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    evaluate_continuation_suite(m, tok, output_path=args.output)
