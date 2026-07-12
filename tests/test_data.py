from __future__ import annotations

import unittest
from unittest.mock import patch

from datasets import Dataset

from chunkrag.data import load_techqa_documents_and_examples


class TechQADataTests(unittest.TestCase):
    def test_loader_keeps_answerable_rows_and_deduplicates_contexts(self) -> None:
        dataset = Dataset.from_list(
            [
                {
                    "id": "q1",
                    "question": "How is setting A enabled?",
                    "answer": "Run command A.",
                    "is_impossible": False,
                    "contexts": [
                        {"filename": "a.txt", "text": "Title: A guide\n\nRun command A."}
                    ],
                },
                {
                    "id": "q2",
                    "question": "What also uses guide A?",
                    "answer": "Feature B.",
                    "is_impossible": False,
                    "contexts": [
                        {"filename": "a.txt", "text": "Title: A guide\n\nRun command A."}
                    ],
                },
                {
                    "id": "q3",
                    "question": "Unanswerable",
                    "answer": "-",
                    "is_impossible": True,
                    "contexts": [],
                },
            ]
        )

        with patch("chunkrag.data.load_dataset", return_value=dataset) as mocked:
            documents, examples = load_techqa_documents_and_examples(
                split="train",
                max_examples=3,
                seed=7,
                revision="abc123",
            )

        mocked.assert_called_once_with(
            "nvidia/TechQA-RAG-Eval",
            split="train",
            revision="abc123",
        )
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].doc_id, "techqa::a.txt")
        self.assertEqual(documents[0].title, "A guide")
        self.assertEqual({example.example_id for example in examples}, {"q1", "q2"})
        self.assertTrue(all(example.relevant_doc_ids == ["techqa::a.txt"] for example in examples))

    def test_loader_builds_full_answerable_corpus_before_sampling_questions(self) -> None:
        dataset = Dataset.from_list(
            [
                {
                    "id": "q1",
                    "question": "Question one?",
                    "answer": "Answer one.",
                    "is_impossible": False,
                    "contexts": [{"filename": "a.txt", "text": "Title: A\nAnswer one."}],
                },
                {
                    "id": "q2",
                    "question": "Question two?",
                    "answer": "Answer two.",
                    "is_impossible": False,
                    "contexts": [{"filename": "b.txt", "text": "Title: B\nAnswer two."}],
                },
            ]
        )

        with patch("chunkrag.data.load_dataset", return_value=dataset):
            documents, examples = load_techqa_documents_and_examples(
                split="train",
                max_examples=1,
                seed=7,
            )

        self.assertEqual({document.doc_id for document in documents}, {"techqa::a.txt", "techqa::b.txt"})
        self.assertEqual(len(examples), 1)
        self.assertIn(examples[0].relevant_doc_ids[0], {document.doc_id for document in documents})


if __name__ == "__main__":
    unittest.main()
