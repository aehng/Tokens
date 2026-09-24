"""vLLM general-plugin entry point for the predictive Phi model."""


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "PredictivePhi3ForCausalLM",
        "tokens_vllm.model:PredictivePhi3ForCausalLM",
    )
