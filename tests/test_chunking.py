from __future__ import annotations

import unittest

import numpy as np

from chunkrag.chunking import build_document_chunks, semantic_chunks, sentence_chunks
from chunkrag.chunking import ChunkingContext
from chunkrag.schemas import Document


class WhitespaceTokenizer:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}
        self.next_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = text.split()
        token_ids: list[int] = []
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.next_id
                self.inverse_vocab[self.next_id] = token
                self.next_id += 1
            token_ids.append(self.vocab[token])
        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self.inverse_vocab[token_id] for token_id in token_ids)


class BoundarySensitiveTokenizer(WhitespaceTokenizer):
    """Mimic a decode/re-encode expansion around punctuation boundaries."""

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return super().decode(token_ids, skip_special_tokens).replace("alpha-beta", "alpha - beta")


class StubSentenceEncoder:
    def encode(self, sentences, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        vectors = np.asarray(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
            ][: len(sentences)],
            dtype=float,
        )
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / norms
        return vectors


class ChunkingTests(unittest.TestCase):
    def test_build_document_chunks_uses_registry_for_fixed_chunker(self) -> None:
        tokenizer = WhitespaceTokenizer()
        document = Document(
            doc_id="doc-1",
            title="Doc",
            text="alpha beta gamma delta",
            dataset="unit",
        )
        chunker_spec = {
            "name": "fixed_2",
            "type": "fixed",
            "chunk_size": 2,
            "chunk_overlap": 0,
        }

        chunks = build_document_chunks(
            document,
            chunker_spec,
            ChunkingContext(tokenizer=tokenizer),
        )

        self.assertEqual([chunk.text for chunk in chunks], ["alpha beta", "gamma delta"])

    def test_semantic_chunks_preserve_expected_boundaries(self) -> None:
        tokenizer = WhitespaceTokenizer()
        document = Document(
            doc_id="doc-2",
            title="Doc",
            text="Alpha one. Alpha two. Beta three.",
            dataset="unit",
        )

        chunks = semantic_chunks(
            document,
            tokenizer,
            chunk_size=20,
            similarity_threshold=0.8,
            embedding_model=StubSentenceEncoder(),
            chunker_name="semantic_test",
            min_chunk_tokens=1,
        )

        self.assertEqual([chunk.text for chunk in chunks], ["Alpha one. Alpha two.", "Beta three."])

    def test_sentence_chunks_split_a_single_oversize_sentence(self) -> None:
        tokenizer = WhitespaceTokenizer()
        document = Document(
            doc_id="doc-3",
            title="Doc",
            text="one two three four five six seven eight nine.",
            dataset="unit",
        )

        chunks = sentence_chunks(
            document,
            tokenizer,
            chunk_size=4,
            chunker_name="sentence_4",
            enforce_token_limit=True,
        )

        self.assertEqual([chunk.token_count for chunk in chunks], [4, 4, 1])
        self.assertEqual(
            " ".join(chunk.text for chunk in chunks),
            "one two three four five six seven eight nine.",
        )

    def test_strict_fixed_chunks_enforce_limit_after_decode_round_trip(self) -> None:
        tokenizer = BoundarySensitiveTokenizer()
        document = Document(
            doc_id="doc-4",
            title="Doc",
            text="alpha-beta gamma delta epsilon zeta eta",
            dataset="unit",
        )

        chunks = build_document_chunks(
            document,
            {
                "name": "fixed_4",
                "type": "fixed",
                "chunk_size": 4,
                "chunk_overlap": 0,
                "enforce_token_limit": True,
            },
            ChunkingContext(tokenizer=tokenizer),
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.token_count <= 4 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
