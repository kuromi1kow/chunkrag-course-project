# Midway Report

## Title

Comparing Chunking Strategies for Retrieval-Augmented Question Answering

## Problem Definition and Motivation

This project studies a simple but important RAG design choice: how documents should be chunked before embedding and retrieval. In a standard retrieval-augmented QA pipeline, the chunking function determines which pieces of text can be retrieved together. If chunks are too large, the retriever may return noisy context; if they are too small, relevant evidence can be fragmented across multiple chunks. The proposal focused on the question:

Which chunking strategy best supports end-to-end QA accuracy, and does that answer change between relatively local questions and multi-hop questions?

That question is useful because chunking is often treated as an implementation detail even though recent engineering studies report sizable retrieval differences from chunking alone.

## Related Work

Retrieval-augmented generation was popularized by Lewis et al. (2020), while dense passage retrieval for open-domain QA was established by Karpukhin et al. (2020). For evaluation data, I use SQuAD 2.0 (Rajpurkar et al., 2018) and HotpotQA (Yang et al., 2018). Recent chunking-focused work includes the Chroma technical report on chunking for retrieval (Smith and Troynikov, 2024), NVIDIA's chunking analysis for RAG systems (NVIDIA, 2024), Qu et al. (2024) on the cost-effectiveness of semantic chunking, and Shaukat et al. (2026) on broader chunking and embedding sensitivity.

## Current Experimental Setup

The current pipeline holds the embedding model, retriever, and generator fixed while changing only the chunking strategy.

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Generator: `google/flan-t5-base`
- Retriever: FAISS inner-product search over normalized dense embeddings
- Top-k retrieval: 4 chunks
- Device: local Apple MPS

Chunking strategies currently implemented:

1. Fixed token chunking at 128, 256, and 512 tokens.
2. Recursive chunking with `RecursiveCharacterTextSplitter`.
3. Sentence-based chunking with spaCy sentencization.
4. Semantic chunking using sentence embedding similarity plus a minimum chunk-length guard.

Current evaluation subset:

- SQuAD v2: 60 answerable validation questions sampled from a shuffled 500-example candidate pool, spanning 35 pseudo-documents formed by concatenating same-title paragraphs.
- HotpotQA distractor: 30 validation questions with 291 supporting/distractor documents in the induced corpus.

Metrics:

- End-to-end answer quality: Exact Match and token-level F1
- Retrieval quality: Recall@4 and Precision@4
- Efficiency: average chunk size, number of indexed chunks, generation latency

## Preliminary Results

### SQuAD v2

| System | EM | F1 | Recall@4 | Precision@4 |
|---|---:|---:|---:|---:|
| `parametric_only` | 1.7 | 6.7 | - | - |
| `fixed_128` | 61.7 | 72.8 | 95.0 | 78.8 |
| `fixed_256` | 51.7 | 59.9 | 91.7 | 74.6 |
| `fixed_512` | 46.7 | 53.4 | 93.3 | 65.0 |
| `recursive_256` | 65.0 | 75.1 | 91.7 | 73.8 |
| `sentence_256` | 60.0 | 68.4 | 88.3 | 72.9 |
| `semantic_256` | 76.7 | 85.4 | 95.0 | 77.1 |

### HotpotQA

| System | EM | F1 | Recall@4 | Precision@4 |
|---|---:|---:|---:|---:|
| `parametric_only` | 13.3 | 17.9 | - | - |
| `fixed_128` | 36.7 | 43.1 | 100.0 | 47.5 |
| `fixed_256` | 36.7 | 44.8 | 100.0 | 46.7 |
| `fixed_512` | 36.7 | 44.8 | 100.0 | 46.7 |
| `recursive_256` | 36.7 | 49.7 | 100.0 | 46.7 |
| `sentence_256` | 36.7 | 49.7 | 100.0 | 46.7 |
| `semantic_256` | 36.7 | 47.2 | 100.0 | 45.8 |

## Discussion

The initial experiments already show a clear pattern.

- On SQuAD, chunking has a large effect on both EM and F1. Smaller or more structure-aware chunks outperform larger fixed windows.
- On HotpotQA, exact match is stubbornly flat, but recursive and sentence chunking improve token-level F1. That suggests better partial grounding even when the generator still misses the exact full answer.
- The no-retrieval baseline is much worse on both datasets, which confirms that the task is genuinely retrieval-sensitive.

There are two important caveats at this stage.

- The QA generator is still relatively small compared with the originally proposed 7B-class models, so some remaining error is probably generation-limited rather than retrieval-limited.
- The current retrieval metric is coarse, especially for HotpotQA, where Recall@4 saturates quickly on the induced distractor corpus.

## Plan for the Rest of the Project

The next steps are concrete.

1. Scale the current subset experiments upward, ideally on SCC, to improve statistical stability.
2. Swap in a stronger generator, such as Mistral-7B-Instruct or Llama-3-8B-Instruct, while keeping the chunking comparison fixed.
3. Add stronger HotpotQA-specific retrieval diagnostics, such as full supporting-fact coverage instead of only answer-hit recall.
4. Extend the analysis with more qualitative failures and a clearer discussion of when semantic chunking helps versus when it over-fragments context.
5. Polish the final writeup into a paper-style report with complete methods, integrated related work, and a broader discussion section.

## References

Full references are collected in [`reports/references.bib`](/Users/assylkhan/Documents/NLP/reports/references.bib).
