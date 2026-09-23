import unittest

import torch
from torch import nn

from experiments.train_predictive_zip2zip import get_base_weight_hashes


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(4, 4, bias=False)
        self.lora_A = nn.Linear(4, 2, bias=False)
        self.lora_B = nn.Linear(2, 4, bias=False)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = TinyAttention()


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.model.layers = nn.ModuleList([TinyLayer()])


class TinyPeftWrapper(nn.Module):
    """PEFT-shaped wrapper whose causal LM is one level below the wrapper."""

    def __init__(self, causal_lm):
        super().__init__()
        self.model = causal_lm

    def get_base_model(self):
        return self.model


class TinyZip2Zip(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_encoder = nn.Linear(4, 4, bias=False)
        self.output_encoder = nn.Linear(4, 4, bias=False)
        causal_lm = TinyCausalLM()
        # Match Zip2Zip's encoder aliases inside its wrapped base model.
        causal_lm.model.hyperencoder_alias = self.input_encoder
        self.base_model = TinyPeftWrapper(causal_lm)


class EmptyBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Linear(4, 4)
        self.input_encoder = nn.Linear(4, 4)
        self.output_encoder = nn.Linear(4, 4)


class EmbeddingOnlyBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.embed_tokens = nn.Embedding(8, 4)
        self.input_encoder = nn.Linear(4, 4)
        self.output_encoder = nn.Linear(4, 4)


class PredictiveTrainingBaseHashTests(unittest.TestCase):
    def test_hashes_wrapped_backbone_tensors_deterministically_and_excludes_aliases(self):
        model = TinyZip2Zip()

        hashes = get_base_weight_hashes(model)

        self.assertEqual(set(hashes), {"embed_tokens", "layer_0_qkv_base"})
        self.assertEqual(hashes, get_base_weight_hashes(model))
        self.assertNotIn("lora_A", hashes)
        self.assertNotIn("hyperencoder_alias", hashes)

        qkv_weight = (
            model.base_model.get_base_model()
            .model.layers[0]
            .self_attn.qkv_proj.base_layer.weight
        )
        with torch.no_grad():
            qkv_weight[0, 0].add_(1)
        self.assertNotEqual(
            hashes["layer_0_qkv_base"],
            get_base_weight_hashes(model)["layer_0_qkv_base"],
        )

    def test_encoder_parameter_alias_is_not_hashed_as_backbone(self):
        model = TinyZip2Zip()
        model.base_model.get_base_model().model.layers[0].self_attn.qkv_proj.base_layer.weight = (
            model.input_encoder.weight
        )

        with self.assertRaisesRegex(RuntimeError, "vacuous frozen-weight checks"):
            get_base_weight_hashes(model)

    def test_empty_backbone_selection_fails_loudly(self):
        with self.assertRaisesRegex(RuntimeError, "vacuous frozen-weight checks"):
            get_base_weight_hashes(EmptyBaseModel())
        with self.assertRaisesRegex(RuntimeError, "vacuous frozen-weight checks"):
            get_base_weight_hashes(EmbeddingOnlyBaseModel())


if __name__ == "__main__":
    unittest.main()
