from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerBase

from chunkrag.schemas import Chunk, Document

try:
    from chonkie import RecursiveChunker as ChonkieRecursiveChunker
    from chonkie import SemanticChunker as ChonkieSemanticChunker
    from chonkie.embeddings import AutoEmbeddings as ChonkieAutoEmbeddings
except ImportError:  # pragma: no cover - optional dependency
    ChonkieRecursiveChunker = None
    ChonkieSemanticChunker = None
    ChonkieAutoEmbeddings = None


_SPACY_PIPELINE = None
_CHONKIE_EMBEDDINGS_CACHE = {}


def get_sentencizer():
    global _SPACY_PIPELINE
    if _SPACY_PIPELINE is None:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        _SPACY_PIPELINE = nlp
    return _SPACY_PIPELINE


def count_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def sentence_split(text: str) -> list[str]:
    doc = get_sentencizer()(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _chunk_from_texts(
    document: Document,
    texts: Iterable[str],
    tokenizer: PreTrainedTokenizerBase,
    chunker_name: str,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, text in enumerate(texts):
        clean_text = text.strip()
        if not clean_text:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{chunker_name}::{document.doc_id}::{index}",
                doc_id=document.doc_id,
                title=document.title,
                dataset=document.dataset,
                text=clean_text,
                token_count=count_tokens(tokenizer, clean_text),
            )
        )
    return chunks


def fixed_token_chunks(
    document: Document,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    chunk_overlap: int,
    chunker_name: str,
) -> list[Chunk]:
    token_ids = tokenizer.encode(document.text, add_special_tokens=False)
    stride = max(1, chunk_size - chunk_overlap)
    texts: list[str] = []
    for start in range(0, len(token_ids), stride):
        window = token_ids[start : start + chunk_size]
        if not window:
            continue
        texts.append(tokenizer.decode(window, skip_special_tokens=True))
        if start + chunk_size >= len(token_ids):
            break
    return _chunk_from_texts(document, texts, tokenizer, chunker_name)


def recursive_chunks(
    document: Document,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    chunk_overlap: int,
    chunker_name: str,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return _chunk_from_texts(document, splitter.split_text(document.text), tokenizer, chunker_name)


def sentence_chunks(
    document: Document,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    chunker_name: str,
) -> list[Chunk]:
    texts: list[str] = []
    current: list[str] = []
    for sentence in sentence_split(document.text):
        candidate = " ".join(current + [sentence])
        if current and count_tokens(tokenizer, candidate) > chunk_size:
            texts.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        texts.append(" ".join(current))
    return _chunk_from_texts(document, texts, tokenizer, chunker_name)


def semantic_chunks(
    document: Document,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    similarity_threshold: float,
    embedding_model: SentenceTransformer,
    chunker_name: str,
    min_chunk_tokens: int | None = None,
) -> list[Chunk]:
    sentences = sentence_split(document.text)
    if len(sentences) <= 1:
        return _chunk_from_texts(document, sentences, tokenizer, chunker_name)

    min_chunk_tokens = min_chunk_tokens or max(32, chunk_size // 2)
    embeddings = embedding_model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    texts: list[str] = []
    current_sentences = [sentences[0]]
    for idx in range(1, len(sentences)):
        candidate = " ".join(current_sentences + [sentences[idx]])
        similarity = float(np.dot(embeddings[idx - 1], embeddings[idx]))
        current_text = " ".join(current_sentences)
        current_token_count = count_tokens(tokenizer, current_text)
        candidate_token_count = count_tokens(tokenizer, candidate)
        similarity_break = similarity < similarity_threshold and current_token_count >= min_chunk_tokens
        size_break = candidate_token_count > chunk_size
        should_split = similarity_break or size_break
        if should_split:
            texts.append(current_text)
            current_sentences = [sentences[idx]]
        else:
            current_sentences.append(sentences[idx])
    if current_sentences:
        texts.append(" ".join(current_sentences))
    return _chunk_from_texts(document, texts, tokenizer, chunker_name)


def get_chonkie_embeddings(model_name: str):
    if ChonkieAutoEmbeddings is None:
        raise ImportError("Chonkie is not installed. Install it with `pip install 'chonkie[semantic]'`.")
    if model_name not in _CHONKIE_EMBEDDINGS_CACHE:
        _CHONKIE_EMBEDDINGS_CACHE[model_name] = ChonkieAutoEmbeddings.get_embeddings(model_name)
    return _CHONKIE_EMBEDDINGS_CACHE[model_name]


def chonkie_recursive_chunks(
    document: Document,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    chunker_name: str,
) -> list[Chunk]:
    if ChonkieRecursiveChunker is None:
        raise ImportError("Chonkie is not installed. Install it with `pip install chonkie`.")
    chunker = ChonkieRecursiveChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        min_characters_per_chunk=24,
    )
    raw_chunks = chunker.chunk(document.text)
    return [
        Chunk(
            chunk_id=f"{chunker_name}::{document.doc_id}::{index}",
            doc_id=document.doc_id,
            title=document.title,
            dataset=document.dataset,
            text=chunk.text.strip(),
            token_count=int(chunk.token_count),
        )
        for index, chunk in enumerate(raw_chunks)
        if chunk.text.strip()
    ]


def chonkie_semantic_chunks(
    document: Document,
    chunk_size: int,
    chunker_name: str,
    embedding_model_name: str,
    threshold: float = 0.7,
    min_sentences_per_chunk: int = 1,
    similarity_window: int = 3,
    skip_window: int = 0,
) -> list[Chunk]:
    if ChonkieSemanticChunker is None:
        raise ImportError("Chonkie semantic support is not installed. Install it with `pip install 'chonkie[semantic]'`.")
    chunker = ChonkieSemanticChunker(
        embedding_model=get_chonkie_embeddings(embedding_model_name),
        threshold=threshold,
        chunk_size=chunk_size,
        similarity_window=similarity_window,
        min_sentences_per_chunk=min_sentences_per_chunk,
        skip_window=skip_window,
    )
    raw_chunks = chunker.chunk(document.text)
    return [
        Chunk(
            chunk_id=f"{chunker_name}::{document.doc_id}::{index}",
            doc_id=document.doc_id,
            title=document.title,
            dataset=document.dataset,
            text=chunk.text.strip(),
            token_count=int(chunk.token_count),
        )
        for index, chunk in enumerate(raw_chunks)
        if chunk.text.strip()
    ]
