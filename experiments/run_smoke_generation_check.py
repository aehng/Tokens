"""Phase 7: Real-Generation Smoke Check on 3 prompts.

Evaluates 1 code, 1 reasoning, 1 instruction prompt on:
1. Step-0 (baseline zero-shot pure predictive)
2. Step-5 (probe checkpoint)

Measures:
- Hypertokens emitted & which hypertokens
- Decode steps vs expanded base tokens
- Output text & coherence
- Truncation / repetition check
"""

import json
import os
import pickle
import sys
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from zip2zip import Zip2ZipModel, StaticCodebookManager
from zip2zip.predictor_policy import CappedPredictorPolicy
from experiments.load_joint_checkpoint import load_joint_checkpoint

PROMPTS = [
    {
        "domain": "code",
        "name": "python_binary_search",
        "prompt": "Instruction: Write a Python function `binary_search(arr, target)` that returns the index of target in sorted arr, or -1 if not found.\nAnswer:\n",
        "max_new_tokens": 80,
    },
    {
        "domain": "reasoning",
        "name": "gsm8k_math_problem",
        "prompt": "Question: A bookstore sold 45 books on Monday and twice as many on Tuesday. On Wednesday, it sold 15 fewer books than on Tuesday. How many books were sold in total over the three days?\nAnswer:\n",
        "max_new_tokens": 80,
    },
    {
        "domain": "instruction",
        "name": "alpaca_renewable_energy",
        "prompt": "Instruction: Name three renewable energy sources and briefly describe how each generates electricity.\nAnswer:\n",
        "max_new_tokens": 80,
    },
]

PREDICTOR_PATH = "experiments/checkpoints/cached_predictor.pkl"
CHECKPOINT_PATH = "experiments/checkpoints/predictive_joint_pilot/probe_final_step_5.pt"


def generate_with_predictive_codebook(model, tokenizer, policy, prompt_text, max_new_tokens=80):
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    codebook_dict, policy_meta = policy.select_codebook(prompt_ids)

    dim = model.zip2zip_config.encoder.hidden_size
    pad_id = tokenizer.pad_token_id or 32000
    disabled_ids = list(model.zip2zip_config.compression.disabled_ids)

    static_mgr = StaticCodebookManager(
        initial_vocab_size=32011,
        max_codebook_size=32,
        max_subtokens=4,
        embedding_dim=dim,
        pad_token_id=pad_id,
        disabled_ids=disabled_ids,
    )
    static_mgr.set_seeded_codebook(codebook_dict, batch_size=1, device=torch.device("cpu"))
    static_mgr.attach_to_model(model)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    initial_len = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0, initial_len:].tolist()
    decode_steps = len(gen_ids)

    # Expand hypertokens
    hyper_to_tokens = {v: list(k) for k, v in codebook_dict.items()}
    expanded_tokens = []
    hypertokens_emitted = []

    for tok in gen_ids:
        if tok in hyper_to_tokens:
            hypertokens_emitted.append((tok, tokenizer.decode(hyper_to_tokens[tok])))
            expanded_tokens.extend(hyper_to_tokens[tok])
        else:
            expanded_tokens.append(tok)

    output_text = tokenizer.decode(expanded_tokens)

    static_mgr.detach_from_model(model)
    model.codebook_manager.reset()

    return {
        "decode_steps": decode_steps,
        "expanded_tokens": len(expanded_tokens),
        "hypertokens_count": len(hypertokens_emitted),
        "hypertokens_emitted": [h[1] for h in hypertokens_emitted],
        "output_text": output_text,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Real generation smoke check.")
    parser.add_argument("--checkpoint", type=str, default="experiments/checkpoints/predictive_joint_pilot/probe_final_step_5.pt", help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="experiments/checkpoints/real_generation_smoke_check.json", help="Output JSON path")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    out_file = args.output

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    with open(PREDICTOR_PATH, "rb") as f:
        raw_predictor = pickle.load(f)
    p_index = getattr(raw_predictor, "index", raw_predictor)

    policy = CappedPredictorPolicy(
        p_index,
        tokenizer,
        budget=32,
        max_structural_slots=0,
        allow_numeric=True,
        filter_bare_punctuation=True,
    )

    print("\n" + "="*80)
    print("REAL ANSWER SMOKE CHECK: 3 PROMPTS (Code, Reasoning, Instruction)")
    print(f"Evaluating Checkpoint: {checkpoint_path}")
    print("="*80 + "\n")

    # Load Model (single instance to conserve memory)
    print("Loading Zip2Zip Model (fp16 backbone)...")
    model = Zip2ZipModel.from_pretrained(
        "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print("\n--- Running Step-0 Baseline Evaluations ---")
    step0_results = {}
    for item in PROMPTS:
        print(f"Generating Step-0 for [{item['domain'].upper()}] {item['name']}...")
        step0_results[item["name"]] = generate_with_predictive_codebook(
            model, tokenizer, policy, item["prompt"], max_new_tokens=item["max_new_tokens"]
        )

    # Load checkpoint weights into the existing model
    target_step = 5
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Requested predictive checkpoint is missing: {checkpoint_path}")
    print(f"\nLoading weights into model from {checkpoint_path}...")
    load_report = load_joint_checkpoint(
        model,
        checkpoint_path,
        expected_model_id="epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1",
    )
    target_step = load_report["step"]
    print(
        f"Verified Step-{target_step} with {load_report['changed_tensor_count']} changed tensors; "
        f"missing={sum(map(len, load_report['missing_keys'].values()))}, "
        f"unexpected={sum(map(len, load_report['unexpected_keys'].values()))}, "
        f"base_hashes={load_report['base_hash_status']}.\n"
    )

    print(f"--- Running Step-{target_step} Evaluations ---")
    stepN_results = {}
    for item in PROMPTS:
        print(f"Generating Step-{target_step} for [{item['domain'].upper()}] {item['name']}...")
        stepN_results[item["name"]] = generate_with_predictive_codebook(
            model, tokenizer, policy, item["prompt"], max_new_tokens=item["max_new_tokens"]
        )

    results = []
    for item in PROMPTS:
        name = item["name"]
        res0 = step0_results[name]
        resN = stepN_results[name]

        print(f"\n=======================================================")
        print(f"PROMPT: [{item['domain'].upper()}] {name}")
        print(f"=======================================================")
        print("  [Step 0]:")
        print(f"    Decode Steps: {res0['decode_steps']} -> Expanded Tokens: {res0['expanded_tokens']} (Saved: {res0['expanded_tokens'] - res0['decode_steps']})")
        print(f"    Hypertokens: {res0['hypertokens_count']} {res0['hypertokens_emitted']}")
        print(f"    Output Snippet: {repr(res0['output_text'][:160])}")

        print(f"  [Step {target_step}]:")
        print(f"    Decode Steps: {resN['decode_steps']} -> Expanded Tokens: {resN['expanded_tokens']} (Saved: {resN['expanded_tokens'] - resN['decode_steps']})")
        print(f"    Hypertokens: {resN['hypertokens_count']} {resN['hypertokens_emitted']}")
        print(f"    Output Snippet: {repr(resN['output_text'][:160])}")

        results.append({
            "domain": item["domain"],
            "name": name,
            "step_0": res0,
            f"step_{target_step}": resN,
        })

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved real generation smoke results to {out_file}")


if __name__ == "__main__":
    main()
