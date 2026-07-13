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

    def test_single_window_sources_remain_one_final_short_chunk(self) -> None:
        for total in (1, 8, 63, 64, 191, 192):
            text = "x" * total
            source = TokenizedSource.build(text, CharTokenizer())
            expected = [0, total]
            self.assertEqual(fixed_cuts(total), expected)
            self.assertEqual(recursive_cuts(source), expected)
            self.assertEqual(sentence_cuts(source, [(0, total)]), expected)
            self.assertEqual(semantic_cuts(source, lambda _: [[1.0, 0.0]], [(0, total)]), expected)
            records = chunk_records(
                {"dataset": "squad_v2", "document_id": f"short-{total}", "text": text},
                source, "fixed192", expected, "tok", "rev",
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["token_count"], total)
            self.assertEqual(records[0]["final_short"], total < 64)

    def test_zero_token_source_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonempty tokenized source"):
            fixed_cuts(0)

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
