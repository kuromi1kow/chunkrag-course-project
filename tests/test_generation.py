from __future__ import annotations

import unittest
from types import SimpleNamespace

from chunkrag.generation import (
    OpenAICompatibleGenerator,
    compress_answer,
    normalize_qa_response,
)


class FakeChatTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return list(range(sum(len(message["content"].split()) for message in messages) + 1))

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(f"token-{index}" for index in token_ids)


class FakeCompletions:
    def __init__(self) -> None:
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Step one.\nStep two."),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(completion_tokens=6),
        )


class GenerationTests(unittest.TestCase):
    def test_normalize_qa_response_strips_citation_prefix_and_suffix(self) -> None:
        self.assertEqual(
            normalize_qa_response("[3] Over 17.5 million people. [1]"),
            "Over 17.5 million people",
        )

    def test_complete_answer_normalization_preserves_multiple_lines(self) -> None:
        self.assertEqual(
            normalize_qa_response(
                "Final answer: Step one.\nStep two.",
                preserve_multiline=True,
            ),
            "Step one.\nStep two.",
        )

    def test_openai_compatible_generator_honors_complete_answer_style(self) -> None:
        generator = OpenAICompatibleGenerator.__new__(OpenAICompatibleGenerator)
        completions = FakeCompletions()
        generator.model_name = "fake-model"
        generator.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        generator.tokenizer = FakeChatTokenizer()
        generator.max_input_tokens = 1_024
        generator.max_new_tokens = 96
        generator.temperature = 0.0
        generator.last_trace = {}

        answer = generator.answer_with_style(
            "How do I complete the task?",
            "Use the documented two-step process.",
            answer_style="complete",
            max_new_tokens=256,
        )

        self.assertEqual(answer, "Step one.\nStep two.")
        self.assertEqual(completions.last_request["max_tokens"], 256)
        self.assertIn("concise but complete", completions.last_request["messages"][0]["content"])
        self.assertFalse(generator.last_trace["generation_length_capped"])
        self.assertEqual(generator.last_trace["generated_tokens"], 6)

    def test_compress_answer_extracts_quantity_phrase(self) -> None:
        self.assertEqual(
            compress_answer(
                "How many people does the Greater Los Angeles Area have?",
                "Over 17.5 million people",
            ),
            "Over 17.5 million",
        )

    def test_compress_answer_extracts_subject_for_who_question(self) -> None:
        self.assertEqual(
            compress_answer(
                "Who disliked the affiliate program?",
                "Several University of Chicago professors disliked the program",
            ),
            "Several University of Chicago professors",
        )

    def test_compress_answer_extracts_predicate_span_for_what_question(self) -> None:
        self.assertEqual(
            compress_answer(
                "What type of professionals are pharmacists?",
                "Pharmacists are healthcare professionals",
            ),
            "healthcare professionals",
        )


if __name__ == "__main__":
    unittest.main()
