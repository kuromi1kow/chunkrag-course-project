from __future__ import annotations

import unittest

from chunkrag.generation import compress_answer, normalize_qa_response


class GenerationTests(unittest.TestCase):
    def test_normalize_qa_response_strips_citation_prefix_and_suffix(self) -> None:
        self.assertEqual(
            normalize_qa_response("[3] Over 17.5 million people. [1]"),
            "Over 17.5 million people",
        )

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
