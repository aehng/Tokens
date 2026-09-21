"""True hypertoken decode with the no_global phrase policy.

Predicted phrases are inserted into the output vocabulary before generation.
At each step the frozen model can select a base token or one of those phrases.
Selecting a phrase emits its whole span in that one step, and the later
base-token steps for that span are not run.

This is not a post-hoc checker. Coverage of an already written answer is
reported separately from hypertokens the model actually selects.
"""
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import torch
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from experiments.phrase_policy import DEFAULT_POLICY, select_phrases
from src.evaluation.offline_segmenter import segment_tokens_dp
from zip2zip import StaticCodebookManager, Zip2ZipModel

INITIAL = 32011
BUDGET = 32
MAX_NEW = 48
PER_DOMAIN = 2
MODEL = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
OUT_PATH = "experiments/true_hypertoken_decode_results.json"


class MaskHyper(LogitsProcessor):
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __call__(self, input_ids, scores):
        if scores.shape[-1] > self.vocab_size:
            scores = scores.clone()
            scores[..., self.vocab_size :] = float("-inf")
        return scores


def _coverage(token_ids, phrases):
    if not token_ids or not phrases:
        return 0, 0.0
    comp, _, _ = segment_tokens_dp(list(token_ids), set(phrases))
    saved = len(token_ids) - comp
    return saved, 100.0 * saved / len(token_ids)


def _seedable(phrases, max_subtokens, disabled):
    kept = []
    for phrase in phrases:
        if not (2 <= len(phrase) <= max_subtokens):
            continue
        if any(tok in disabled or tok >= INITIAL for tok in phrase):
            continue
        kept.append(phrase)
    return kept


def main():
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    with open("experiments/checkpoints/cached_predictor.pkl", "rb") as f:
        predictor = pickle.load(f)
    records = json.load(open("data/cached_pure_pred_val_60.json", encoding="utf-8"))
    chosen = []
    seen = defaultdict(int)
    for rec in records:
        if seen[rec["domain"]] < PER_DOMAIN:
            chosen.append(rec)
            seen[rec["domain"]] += 1

    print("Loading frozen Zip2Zip model (original encoders)...", flush=True)
    model = Zip2ZipModel.from_pretrained(
        MODEL,
        max_codebook_size=BUDGET,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval()
    max_sub = int(model.zip2zip_config.compression.max_subtokens)
    disabled = set(model.zip2zip_config.compression.disabled_ids)
    dim = model.zip2zip_config.encoder.hidden_size
    pad_id = tokenizer.pad_token_id or 32000
    mask = LogitsProcessorList([MaskHyper(INITIAL)])

    rows = []
    print(f"policy={DEFAULT_POLICY} budget={BUDGET} max_new={MAX_NEW} max_subtokens={max_sub}", flush=True)

    for rec in chosen:
        prompt_ids = rec["prompt_token_ids"]
        prompt = torch.tensor([prompt_ids], dtype=torch.long)
        print("\n" + "=" * 70, flush=True)
        print(rec["id"], rec["domain"], flush=True)

        if hasattr(model.codebook_manager, "reset"):
            model.codebook_manager.reset()
        t0 = time.perf_counter()
        with torch.no_grad():
            base_out = model.base_model.generate(
                input_ids=prompt,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                logits_processor=mask,
            )
        base_s = time.perf_counter() - t0
        base_tokens = base_out[0, len(prompt_ids) :].tolist()

        t_sel = time.perf_counter()
        raw_phrases = select_phrases(
            predictor, tokenizer, prompt_ids, DEFAULT_POLICY, budget=BUDGET, domain=rec["domain"]
        )
        phrases = _seedable(raw_phrases, max_sub, disabled)
        select_ms = (time.perf_counter() - t_sel) * 1000.0
        seeded = {phrase: INITIAL + i for i, phrase in enumerate(phrases)}

        mgr = StaticCodebookManager(
            initial_vocab_size=INITIAL,
            max_codebook_size=BUDGET,
            max_subtokens=max_sub,
            embedding_dim=dim,
            pad_token_id=pad_id,
            disabled_ids=list(disabled),
        )
        t_setup = time.perf_counter()
        mgr.set_seeded_codebook(seeded, batch_size=1, device=torch.device("cpu"))
        mgr.attach_to_model(model)
        with torch.no_grad():
            mgr.synthesize_hyper_vectors(model, batch_size=1)
        setup_ms = select_ms + (time.perf_counter() - t_setup) * 1000.0

        t_gen = time.perf_counter()
        with torch.no_grad():
            hyp_out = model.generate(
                input_ids=prompt,
                max_new_tokens=MAX_NEW,
                do_sample=False,
            )
        gen_s = time.perf_counter() - t_gen
        gen_ids = hyp_out[0, len(prompt_ids) :].tolist()
        expanded = mgr.decode_sequence(gen_ids)
        mgr.detach_from_model(model)
        if hasattr(model.codebook_manager, "reset"):
            model.codebook_manager.reset()

        hypers = [t for t in gen_ids if t >= INITIAL]
        covered, cover_pct = _coverage(base_tokens, phrases)
        n = min(len(base_tokens), len(expanded))
        prefix = next((i for i in range(n) if base_tokens[i] != expanded[i]), n)
        exact = base_tokens == expanded
        steps = len(gen_ids)
        expanded_n = len(expanded)
        saved_steps = expanded_n - steps
        realized = 100.0 * saved_steps / expanded_n if expanded_n else 0.0

        base_text = tokenizer.decode(base_tokens, skip_special_tokens=True)
        hyp_text = tokenizer.decode(expanded, skip_special_tokens=True)
        print(f"exposed={len(phrases)} selected={len(hypers)}", flush=True)
        print(f"offline coverage on base answer: {cover_pct:.1f}% ({covered} tokens)", flush=True)
        print(
            f"decode steps={steps} expanded={expanded_n} "
            f"realized_step_reduction={realized:.1f}% exact={exact} prefix={prefix}/{n}",
            flush=True,
        )
        print(f"time base={base_s:.1f}s hyper={gen_s:.1f}s setup={setup_ms:.0f}ms", flush=True)
        print("BASE:", base_text[:180].replace("\n", " / "), flush=True)
        print("HYPER:", hyp_text[:180].replace("\n", " / "), flush=True)
        if hypers:
            for hid in hypers:
                print("  selected:", repr(tokenizer.decode(mgr.decode_hypertoken(hid))), flush=True)

        rows.append({
            "id": rec["id"],
            "domain": rec["domain"],
            "phrases_exposed": len(phrases),
            "hypertokens_selected": len(hypers),
            "offline_coverage_tokens": covered,
            "offline_coverage_pct": cover_pct,
            "decode_steps": steps,
            "expanded_tokens": expanded_n,
            "realized_step_reduction_pct": realized,
            "exact_match": exact,
            "prefix_match": prefix,
            "base_seconds": base_s,
            "hyper_seconds": gen_s,
            "setup_ms": setup_ms,
            "base_text": base_text,
            "hyper_text": hyp_text,
        })

    print("\nAGGREGATE", flush=True)
    exposed = sum(r["phrases_exposed"] for r in rows)
    selected = sum(r["hypertokens_selected"] for r in rows)
    steps = sum(r["decode_steps"] for r in rows)
    expanded = sum(r["expanded_tokens"] for r in rows)
    print(f"phrases exposed={exposed} selected={selected}", flush=True)
    print(f"decode steps={steps} expanded tokens={expanded}", flush=True)
    if expanded:
        print(f"realized step reduction={100.0 * (expanded - steps) / expanded:.2f}%", flush=True)
    print(f"exact matches={sum(r['exact_match'] for r in rows)}/{len(rows)}", flush=True)
    covered = sum(r["offline_coverage_tokens"] for r in rows)
    print(
        f"offline coverage={covered} tokens on the base answers  "
        f"mean base {sum(r['base_seconds'] for r in rows)/len(rows):.1f}s  "
        f"mean hyper {sum(r['hyper_seconds'] for r in rows)/len(rows):.1f}s  "
        f"mean setup {sum(r['setup_ms'] for r in rows)/len(rows):.0f}ms",
        flush=True,
    )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"policy": DEFAULT_POLICY, "rows": rows}, f, indent=2)
    print("wrote", OUT_PATH, flush=True)


if __name__ == "__main__":
    main()
