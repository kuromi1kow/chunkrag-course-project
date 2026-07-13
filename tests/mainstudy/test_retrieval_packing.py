"""Specification Sections 13 and 16: RRF, ties, and context packing."""

import unittest

from chunkrag.mainstudy.packing import matched_pack, matched_target, operational_pack
from chunkrag.mainstudy.retrieval import lexical_tokenize, ranked, rerank_candidates, weighted_rrf


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": list(text.encode("utf-8"))}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        text = "\n".join(item["content"] for item in messages) + "\nassistant:"
        return list(text.encode("utf-8"))


def chunks(count=20):
    return [{"chunk_id": f"c{i}", "document_id": f"d{i}", "title": f"T{i}", "text": "x" * 40, "char_start": 0, "char_end": 40} for i in range(count)]


class RetrievalPackingTests(unittest.TestCase):
    def test_lexical_rule(self) -> None:
        self.assertEqual(lexical_tokenize("A_B, Café!"), ["a_b", "café"])

    def test_weighted_rrf_and_rerank_ties(self) -> None:
        dense = ranked({"a": 2.0, "b": 1.0}, 2)
        sparse = ranked({"b": 2.0, "c": 1.0}, 2)
        fused = weighted_rrf(dense, sparse)
        self.assertEqual(fused[0]["chunk_id"], "b")
        reranked = rerank_candidates(fused, {row["chunk_id"]: 1.0 for row in fused})
        self.assertEqual(len(reranked), 3)

    def test_operational_and_matched_pack(self) -> None:
        tokenizer = FakeTokenizer()
        operational = operational_pack(tokenizer, "squad_v2", "q?", chunks(), 1024)
        self.assertLessEqual(len(operational.prompt_token_ids), 1024)
        rendered = ["x" * 900, "y" * 800]
        target = matched_target(tokenizer, "squad_v2", "q?", rendered, 1024)
        packed = matched_pack(tokenizer, "squad_v2", "q?", chunks(), 1024, target)
        self.assertLessEqual(abs(packed.context_tokens - target), 2)


if __name__ == "__main__":
    unittest.main()
