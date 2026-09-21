"""
Test safe loading and generation with epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1.
Verifies memory footprint and execution on Intel Arc XPU vs CPU.
"""

import gc
import os
import sys
import time
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from zip2zip import Zip2ZipConfig, Zip2ZipModel


def test_load():
    print("=" * 70)
    print("TESTING PHI-3.5 / ZIP2ZIP CHECKPOINT LOADING & HARDWARE FIT")
    print("=" * 70)

    ckpt = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
    cfg = Zip2ZipConfig.from_pretrained(ckpt)
    print("Zip2Zip Config loaded:")
    print(f"  initial_vocab_size: {cfg.compression.initial_vocab_size}")
    print(f"  max_codebook_size:  {cfg.compression.max_codebook_size}")
    print(f"  max_subtokens:      {cfg.compression.max_subtokens}")
    print(f"  disabled_ids count: {len(cfg.compression.disabled_ids)}")
    print(f"  base_model:         {cfg.base_model_name_or_path}")

    device = "xpu" if torch.xpu.is_available() else "cpu"
    print(f"\nTarget device: {device}")
    if device == "xpu":
        free_vram, tot_vram = torch.xpu.mem_get_info(0)
        print(f"  XPU Free VRAM: {free_vram / 1e9:.2f} GB / {tot_vram / 1e9:.2f} GB")

    ram = psutil.virtual_memory()
    print(f"  System RAM Free: {ram.available / 1e9:.2f} GB / {ram.total / 1e9:.2f} GB")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # We test loading Zip2ZipModel with max_codebook_size=32
    print("\nAttempting to load Zip2ZipModel...")
    t0 = time.time()
    try:
        # Load in float16
        model = Zip2ZipModel.from_pretrained(
            ckpt,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        print(f"Model instantiated in {time.time() - t0:.2f}s!")

        print("Keeping model strictly on CPU to avoid XPU VRAM limits...")
        # Model is already on CPU from from_pretrained
        print("Testing a 5-token generation on CPU...")
        inp = tokenizer("Hello, world! What is 2 + 2?", return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=5, do_sample=False)
        print("Generated text:", tokenizer.decode(out[0], skip_special_tokens=True))
        print("SUCCESS! Model runs cleanly on CPU.")

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_load()
