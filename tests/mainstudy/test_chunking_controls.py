"""Specification Sections 11--12: span round trips and deterministic controls."""

import unittest

from chunkrag.mainstudy.chunking import TokenizedSource, chunk_records, fixed_cuts, recursive_cuts, semantic_cuts, sentence_cuts
from chunkrag.mainstudy.controls import jitter_cuts


class CharTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text))), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class ChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = ("A" * 180 + ". \n\n") * 3
        self.source = TokenizedSource.build(self.text, CharTokenizer())
        self.document = {"dataset": "squad_v2", "document_id": "doc", "text": self.text}

    def test_fixed_round_trip(self) -> None:
        cuts = fixed_cuts(self.source.tokens)
        records = chunk_records(self.document, self.source, "fixed192", cuts, "tok", "rev")
        self.assertEqual("".join(row["text"] for row in records), self.text)
        self.assertTrue(all(row["token_count"] <= 254 for row in records))

    def test_recursive_and_sentence_are_deterministic(self) -> None:
        self.assertEqual(recursive_cuts(self.source), recursive_cuts(self.source))
        spans = [(0, 184), (184, 368), (368, len(self.text))]
        self.assertEqual(sentence_cuts(self.source, spans), sentence_cuts(self.source, spans))

    def test_semantic_tie_break_is_deterministic(self) -> None:
        spans = [(0, 184), (184, 368), (368, len(self.text))]
        encode = lambda texts: [[1.0, float(index)] for index, _ in enumerate(texts)]
        self.assertEqual(semantic_cuts(self.source, encode, spans), semantic_cuts(self.source, encode, spans))

    def test_jitter_is_reproducible_and_bounded(self) -> None:
        base = fixed_cuts(800)
        left = jitter_cuts(base, seed=1103, policy="fixed192", document_id="doc", final_short=False)
        right = jitter_cuts(base, seed=1103, policy="fixed192", document_id="doc", final_short=False)
        self.assertEqual(left, right)
        lengths = [b - a for a, b in zip(left.cuts, left.cuts[1:])]
        self.assertTrue(all(64 <= length <= 254 for length in lengths))


if __name__ == "__main__":
    unittest.main()
