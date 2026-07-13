"""Specification Sections 14--16: exact prompts and generation config."""

import unittest
from unittest.mock import patch

from chunkrag.mainstudy.generation import LocalGenerator, resolved_generation_config
from chunkrag.mainstudy.prompts import messages, normalize_generated_answer, prompt_version


class Tokenizer:
    eos_token_id = 2
    pad_token_id = None

    def decode(self, token_ids, **kwargs):
        return "generated"


class CapturingModel:
    def __init__(self) -> None:
        self.kwargs = None

    def generate(self, **kwargs):
        import torch

        self.kwargs = kwargs
        suffix = torch.tensor([[2]], dtype=torch.long, device=kwargs["input_ids"].device)
        return torch.cat((kwargs["input_ids"], suffix), dim=1)


class GenerationPromptTests(unittest.TestCase):
    def test_prompt_versions_and_text(self) -> None:
        self.assertEqual(prompt_version("techqa"), "technical-v1")
        self.assertEqual(prompt_version("squad_v2"), "extractive-v1")
        self.assertIn("exactly unanswerable", messages("squad_v2", "Q", "C")[0]["content"])

    def test_normalization_only_removes_leading_label(self) -> None:
        self.assertEqual(normalize_generated_answer(" Answer: Boston. "), "Boston.")
        self.assertEqual(normalize_generated_answer("Explanation: Answer: Boston"), "Explanation: Answer: Boston")

    def test_resolved_generation_config(self) -> None:
        config = resolved_generation_config(64, Tokenizer())
        self.assertFalse(config["do_sample"])
        self.assertEqual(config["pad_token_id"], 2)

    def test_local_generation_passes_explicit_all_ones_attention_mask(self) -> None:
        generator = LocalGenerator("repository", "revision", device="cpu")
        generator.tokenizer = Tokenizer()
        generator.model = CapturingModel()
        with patch("torch.cuda.is_available", return_value=False):
            text, trace = generator.generate([2, 11, 2, 12], 8)
        self.assertEqual(text, "generated")
        self.assertEqual(trace["stopping_reason"], "eos")
        self.assertEqual(
            generator.model.kwargs["attention_mask"].tolist(),
            [[1, 1, 1, 1]],
        )
        self.assertEqual(
            generator.model.kwargs["input_ids"].shape,
            generator.model.kwargs["attention_mask"].shape,
        )


if __name__ == "__main__":
    unittest.main()
