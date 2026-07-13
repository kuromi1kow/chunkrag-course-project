"""Specification Sections 14--16: exact prompts and generation config."""

import unittest

from chunkrag.mainstudy.generation import resolved_generation_config
from chunkrag.mainstudy.prompts import messages, normalize_generated_answer, prompt_version


class Tokenizer:
    eos_token_id = 2
    pad_token_id = None


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


if __name__ == "__main__":
    unittest.main()
