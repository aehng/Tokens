import os, sys
sys.path.insert(0, os.path.abspath("."))

from src.data.corpus_builder import build_corpus

if __name__ == "__main__":
    stage2_targets = {
        "oasst1": 8000,
        "wildchat": 12000,
        "codealpaca": 8000,
        "gsm8k": 4000,
        "mbpp": 974,
    }
    print("Building Stage 2 Corpus (~30,000 pairs)...")
    counts = build_corpus(stage2_targets, prefix="corpus_stage2")
    print("Stage 2 build finished with split counts:", counts)
