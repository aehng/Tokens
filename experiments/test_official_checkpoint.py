"""Verification of Official Pretrained Zip2Zip Checkpoint.

Loads epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1 (trained LoRA + trained hyper-encoders).
Runs baseline LZW zip2zip generation to prove it generates sensible, fluent text.
"""

import time
import torch
from zip2zip import Zip2ZipModel, Zip2ZipTokenizer

def test_official_zip2zip_generation():
    model_id = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
    print(f"Loading official pretrained Zip2Zip checkpoint: {model_id}...")
    t0 = time.time()
    tokenizer = Zip2ZipTokenizer.from_pretrained(model_id)
    # Load on CPU with bfloat16 to fit comfortably in RAM
    model = Zip2ZipModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print(f"Model and tokenizer loaded in {time.time() - t0:.1f}s.")

    prompt = "Write a concise Python function to calculate the factorial of an integer n."
    inputs = tokenizer(prompt, return_tensors="pt")

    print("\nRunning official Zip2Zip generation (reactive LZW)...")
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
        )
    t1 = time.perf_counter()

    raw_ids = outputs[0].tolist()
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = raw_ids[prompt_len:]

    # Inspect hypertokens in generated stream
    initial_vocab_size = tokenizer.initial_vocab_size
    hypertokens = [t for t in gen_ids if t >= initial_vocab_size]
    print(f"Total steps generated: {len(gen_ids)}")
    print(f"Hypertokens emitted: {len(hypertokens)} (IDs: {set(hypertokens)})")
    print(f"Generation latency: {(t1 - t0)*1000:.1f}ms ({(len(gen_ids)/(t1 - t0)):.1f} steps/sec)")

    # Decode text
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nDecoded text output:")
    print("-" * 60)
    print(decoded_text)
    print("-" * 60)

    # Colorized token breakdown
    colored = tokenizer.color_decode(outputs)
    print("\nColored token breakdown sample:")
    print(colored[0][:200] + "...")

if __name__ == "__main__":
    test_official_zip2zip_generation()
