import os
import sys

sys.path.insert(0, os.path.abspath("."))
from src.data.corpus_builder import build_corpus

if __name__ == "__main__":
    stage3_targets = {
        "oasst1": 30000,     # Ingest all available (~25.3k)
        "wildchat": 50000,   # Ingest all available (~45.6k)
        "codealpaca": 25000, # Ingest all available (~20.0k)
        "gsm8k": 10000,      # Ingest all available (~8.8k)
        "mbpp": 1000,        # Ingest all available (~1.0k)
    }
    print("Building Full Stage 3 Corpus (100,000+ raw pairs)...")
    counts = build_corpus(stage3_targets, prefix="corpus_stage3")
    print("Stage 3 build complete with split counts:", counts)
