"""Specification Sections 8--10 and 17: synthetic E0 materialization."""

import unittest

from chunkrag.mainstudy.data import (
    _hotpot_documents, _hotpot_supporting_fact_records, _select, cluster_records,
    materialize_hotpot_rows, normalize_corpus_text, squad_document_id,
)


class DataMaterializationTests(unittest.TestCase):
    def test_normalization_and_document_id(self) -> None:
        self.assertEqual(normalize_corpus_text("e\u0301\r\n"), "é\n")
        self.assertEqual(squad_document_id("T", "C"), squad_document_id("T", "C"))

    def test_hash_selection_respects_cap(self) -> None:
        rows = [{"selection_hash": f"{index:064x}", "question_id": str(index), "selection_rank": -1, "eligibility": {"allocation_key": "a" if index < 3 else "b"}} for index in range(6)]
        selected = _select(rows, 4, 2)
        self.assertEqual([row["selection_rank"] for row in selected], [0, 1, 2, 3])
        self.assertEqual(sum(row["eligibility"]["allocation_key"] == "a" for row in selected), 2)

    def test_cluster_records(self) -> None:
        rows = [{"question_id": "q1", "cluster_id": "c"}, {"question_id": "q2", "cluster_id": "c"}]
        self.assertEqual(cluster_records("d", rows)[0]["size"], 2)

    def test_unavailable_hotpot_sentence_index_is_retained_without_inference(self) -> None:
        row = {
            "id": "malformed", "question": "q", "answer": "a",
            "context": {"title": ["Document"], "sentences": [["zero", "one"]]},
            "supporting_facts": {"title": ["Document"], "sent_id": [902]},
        }
        documents, by_title, provenance = _hotpot_documents(row, "revision", 7)
        facts = _hotpot_supporting_fact_records(
            documents, by_title, provenance, ["Document"], [902],
        )
        self.assertEqual(facts[0]["sentence_index"], 902)
        self.assertEqual(facts[0]["document_index"], 0)
        self.assertIsNone(facts[0]["char_start"])
        self.assertIsNone(facts[0]["char_end"])


if __name__ == "__main__":
    unittest.main()
