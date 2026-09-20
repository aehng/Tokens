"""Calibration and Zero-Shot Seeded Generation Diagnostic.

Tests whether epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1 can emit zero-shot
pre-seeded hypertokens and compares logit distributions across:
1. Base generation (all hypertokens masked)
2. Official LZW generation (dynamic codebook)
3. Seeded generation (pre-seeded static codebook from Predictor / Top-K)
"""

import sys
import os
import time
import pickle
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from zip2zip import Zip2ZipModel, Zip2ZipTokenizer
from zip2zip.static_codebook import StaticCodebookManager, StaticCodebookLogitsWarper
from experiments.dataset_loader import load_split
from experiments.heldout_predictor_benchmark import StrictHeldOutPhraseBank, StrictPredictor

def run_calibration_diagnostic():
    model_id = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
    print(f"Loading official pretrained model {model_id}...")
    tokenizer = Zip2ZipTokenizer.from_pretrained(model_id)
    model = Zip2ZipModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Load trained PhraseBank
    cache_path = "experiments/strict_phrase_bank.pkl"
    if os.path.exists(cache_path):
        print(f"Loading cached PhraseBank from {cache_path}...")
        with open(cache_path, "rb") as f:
            bank = pickle.load(f)
    else:
        print("Mining PhraseBank from data/train.jsonl...")
        train_samples = load_split("train")
        bank = StrictHeldOutPhraseBank(
            disabled_ids=set(model.zip2zip_config.compression.disabled_ids),
            max_subtokens=model.zip2zip_config.compression.max_subtokens,
        )
        bank.train_on_samples(train_samples, tokenizer.hf_tokenizer)
        with open(cache_path, "wb") as f:
            pickle.dump(bank, f)
        print(f"Saved PhraseBank to {cache_path}")

    predictor = StrictPredictor(
        bank=bank,
        initial_vocab_size=tokenizer.initial_vocab_size,
        max_subtokens=model.zip2zip_config.compression.max_subtokens,
    )

    # Test prompt (Code sample from test.jsonl)
    prompt = "Write a concise Python function to calculate the factorial of an integer n."
    print(f"\n--- PROMPT ---\n{prompt}\n")

    prompt_inputs = tokenizer([prompt], return_tensors="pt")
    input_ids = prompt_inputs["input_ids"]

    # 1. Inspect logits at prompt end under Official LZW
    print("=" * 60)
    print("REGIME 1: Official Zip2Zip (Dynamic LZW)")
    print("=" * 60)
    model.codebook_manager.reset()
    model.codebook_manager.init_codebooks_and_hyper_weight_cache(1)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=40,
            do_sample=False,
        )
    lzw_gen_ids = out[0].tolist()[input_ids.shape[1]:]
    lzw_hypertokens = [t for t in lzw_gen_ids if t >= tokenizer.initial_vocab_size]
    print(f"LZW Steps: {len(lzw_gen_ids)}, Hypertokens emitted: {len(lzw_hypertokens)} ({lzw_hypertokens})")
    lzw_text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"LZW Output:\n{lzw_text}\n")

    # 2. Seeded Static Codebook (Prompt Predictor, K=32)
    print("=" * 60)
    print("REGIME 2: Prompt Predictor Seeded Codebook (K=32)")
    print("=" * 60)
    prompt_ids = tokenizer.hf_tokenizer.encode(prompt, add_special_tokens=False)
    predicted_codebook, pred_latency = predictor.select_prompt_conditioned(prompt_ids, budget=32)
    valid_subtokens = list(predicted_codebook.keys())
    print(f"Predicted {len(valid_subtokens)} phrases (first 5): {[tokenizer.hf_tokenizer.decode(s) for s in valid_subtokens[:5]]}")

    static_manager = StaticCodebookManager.from_config(model.zip2zip_config)
    static_manager.set_seeded_codebook(valid_subtokens, batch_size=1)
    static_manager.attach_to_model(model)

    # Diagnostic: Logit inspection at prefill
    with torch.no_grad():
        fwd_out = model(input_ids)
        logits = fwd_out.logits[0, -1, :]  # shape: (vocab_size + max_codebook_size)
        base_logits = logits[:tokenizer.initial_vocab_size]
        hyper_logits = logits[tokenizer.initial_vocab_size : tokenizer.initial_vocab_size + len(valid_subtokens)]

        print(f"\n--- Logit Calibration Diagnostic at Step 0 ---")
        print(f"Base logits:  min={base_logits.min().item():.2f}, mean={base_logits.mean().item():.2f}, max={base_logits.max().item():.2f}")
        print(f"Hyper logits: min={hyper_logits.min().item():.2f}, mean={hyper_logits.mean().item():.2f}, max={hyper_logits.max().item():.2f}")
        top_base_val, top_base_idx = torch.topk(base_logits, 3)
        print(f"Top 3 Base tokens: {[tokenizer.hf_tokenizer.decode([idx.item()]) for idx in top_base_idx]} (logits: {top_base_val.tolist()})")
        top_hyper_val, top_hyper_idx = torch.topk(hyper_logits, 3)
        print(f"Top 3 Hyper tokens: {[tokenizer.hf_tokenizer.decode(valid_subtokens[idx.item()]) for idx in top_hyper_idx]} (logits: {top_hyper_val.tolist()})")
        
        # Softmax probabilities over unmasked vocabulary
        all_active = torch.cat([base_logits, hyper_logits])
        probs = F.softmax(all_active.float(), dim=-1)
        base_prob = probs[:tokenizer.initial_vocab_size].sum().item()
        hyper_prob = probs[tokenizer.initial_vocab_size:].sum().item()
        print(f"Total Base Prob: {base_prob*100:.2f}%, Total Hyper Prob: {hyper_prob*100:.2f}%")

    # Generate with Seeded Codebook
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=40,
            do_sample=False,
        )
    gen_ids = out[0].tolist()[input_ids.shape[1]:]
    emitted_hyper = [t for t in gen_ids if t >= tokenizer.initial_vocab_size]
    print(f"\nSeeded Steps: {len(gen_ids)}, Hypertokens emitted: {len(emitted_hyper)} ({emitted_hyper})")
    # Decode sequence expanding hypertokens
    decoded_ids = static_manager.decode_sequence(out[0].tolist())
    seeded_text = tokenizer.hf_tokenizer.decode(decoded_ids, skip_special_tokens=True)
    print(f"Seeded Output:\n{seeded_text}\n")

    static_manager.detach_from_model(model)
    print("Calibration diagnostic complete.")

if __name__ == "__main__":
    run_calibration_diagnostic()
