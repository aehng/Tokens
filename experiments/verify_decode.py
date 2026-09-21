"""Retired. This file used to check phrases after the tokens were already produced.

That does not skip transformer forwards. The decode path that can skip them is
experiments/true_hypertoken_decode.py.
"""
raise RuntimeError(
    "verify_decode.py is not a decode accelerator. "
    "Run experiments/true_hypertoken_decode.py"
)
