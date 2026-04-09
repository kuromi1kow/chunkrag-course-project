# Comparing Chunking Strategies for Retrieval-Augmented Question Answering

## Abstract

This project studies how document chunking affects retrieval-augmented question answering. I compare fixed-size token chunking, recursive chunking, sentence-based chunking, and semantic chunking in a controlled RAG pipeline with fixed embeddings, dense retrieval, and generation. Experiments on shuffled SQuAD v2 pseudo-documents and HotpotQA distractor examples show that chunking materially changes end-to-end answer quality. On SQuAD, semantic chunking performed best, reaching 76.7 EM and 85.4 F1, while recursive chunking was the strongest non-semantic baseline. On HotpotQA, exact match remained flat across systems, but recursive and sentence chunking improved token-level F1 over fixed windows. An auxiliary comparison with Chonkie showed that recursive chunking is fairly robust across implementations, while semantic chunking is more sensitive to the specific boundary heuristic. The results suggest that chunking matters most when it reduces distractor context without over-fragmenting evidence, and that the best strategy depends on whether the task is local fact extraction or multi-hop reasoning.

## 1. Introduction

Retrieval-augmented generation (RAG) improves factual question answering by retrieving relevant external context before generation. In practice, however, RAG systems do not retrieve whole documents; they retrieve chunks. That makes chunking a central design decision. A chunking policy determines how much context is available at retrieval time, whether evidence remains intact, and how much irrelevant text is carried into the prompt.

Despite that importance, chunking is often treated as a default preprocessing step rather than an experimental variable. Prior engineering studies have reported noticeable retrieval swings from chunking alone, which motivates a systematic comparison in an end-to-end QA setting. This project asks:

Which chunking strategy works best for retrieval-augmented QA, and how does the answer differ between relatively local questions and multi-hop questions?

I answer that question with a controlled empirical study on SQuAD v2 and HotpotQA, holding the embedding model, retriever, and generator fixed while varying only the chunking strategy.

## 2. Related Work

RAG combines neural retrieval with text generation, typically by retrieving passages and conditioning a generator on them (Lewis et al., 2020). Dense retrieval for question answering was established by work such as Dense Passage Retrieval (Karpukhin et al., 2020), which showed that neural retrieval quality can strongly affect downstream QA.

For evaluation, SQuAD 2.0 (Rajpurkar et al., 2018) remains a standard benchmark for short-answer reading comprehension, while HotpotQA (Yang et al., 2018) is useful for testing multi-hop reasoning and evidence composition. Together, they provide a natural contrast between mostly local answer extraction and questions that often require combining information from more than one place.

Recent work has started to focus directly on chunking. Smith and Troynikov (2024) showed that chunking can shift retrieval recall by up to 9% in a token-level retrieval evaluation. NVIDIA's 2024 engineering analysis argued that retrieval quality and answer quality can diverge, and that chunk size therefore needs to be evaluated end to end. Qu et al. (2024) questioned whether semantic chunking consistently justifies its computational overhead, while Shaukat et al. (2026) found that content-aware chunking strategies can outperform naive fixed-size baselines across domains. This project fits into that emerging literature by evaluating chunking in a compact but fully reproducible QA pipeline.

## 3. Experimental Setup

### 3.1 Task

Given a corpus, a question, and a chunking function, the pipeline:

1. Splits documents into chunks.
2. Embeds each chunk.
3. Retrieves the top 4 chunks for a question.
4. Concatenates the retrieved chunks into a prompt.
5. Generates a short answer.

The only experimental variable is the chunking strategy.

### 3.2 Datasets

I use two QA benchmarks.

- SQuAD v2: I restrict to answerable validation examples, shuffle a 500-example candidate pool, then sample 60 questions in a round-robin fashion across titles. Paragraphs sharing a title are concatenated into article-level pseudo-documents, producing 35 documents in the final corpus. This makes chunking meaningful instead of trivial paragraph retrieval.
- HotpotQA distractor: I use 30 validation examples and build the induced document collection from the titles provided in each example's distractor set, yielding 291 documents.

This setup is deliberately modest so the experiments remain reproducible on local hardware, but it is still large enough to show meaningful chunking effects.

### 3.3 Models and Retrieval

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Retriever: FAISS `IndexFlatIP` over normalized dense embeddings
- Generator: `google/flan-t5-base`
- Retrieval depth: 4 chunks
- Runtime device: Apple MPS

This differs from the original proposal's 7B-class generator plan, but it preserves the central experimental comparison while keeping the project fully runnable in the provided environment.

### 3.4 Chunking Strategies

I compare four core chunking families.

1. Fixed-size token chunking with sizes 128, 256, and 512.
2. Recursive chunking using `RecursiveCharacterTextSplitter`.
3. Sentence-based chunking using spaCy sentence segmentation and greedy packing to a token budget.
4. Semantic chunking based on sentence embedding similarity, with a minimum chunk-length guard to avoid pathological micro-chunks.

I also run an auxiliary implementation comparison with Chonkie's recursive and semantic chunkers at the 256-token setting. That comparison is not the main table because it changes the implementation rather than just the strategy family, but it helps separate conceptual chunking effects from library-specific behavior.

### 3.5 Metrics

I report:

- End-to-end QA: Exact Match and token-level F1
- Retrieval: Recall@4 and Precision@4
- Efficiency: average chunk size, number of chunks, average generation latency

For SQuAD, a retrieved chunk counts as relevant if it comes from the gold document or contains a gold answer string. For HotpotQA, the metric is similar but uses the example's supporting documents. This retrieval metric is useful but still coarse, especially for multi-hop reasoning, so it should be interpreted alongside answer quality rather than in isolation.

## 4. Results

### 4.1 SQuAD v2

| System | EM | F1 | Recall@4 | Precision@4 | Avg chunk tokens | # chunks |
|---|---:|---:|---:|---:|---:|---:|
| `parametric_only` | 1.7 | 6.7 | - | - | - | - |
| `fixed_128` | 61.7 | 72.8 | 95.0 | 78.8 | 125.3 | 631 |
| `fixed_256` | 51.7 | 59.9 | 91.7 | 74.6 | 244.0 | 322 |
| `fixed_512` | 46.7 | 53.4 | 93.3 | 65.0 | 472.2 | 164 |
| `recursive_256` | 65.0 | 75.1 | 91.7 | 73.8 | 175.8 | 389 |
| `sentence_256` | 60.0 | 68.4 | 88.3 | 72.9 | 223.9 | 302 |
| `semantic_256` | 76.7 | 85.4 | 95.0 | 77.1 | 144.8 | 467 |

The SQuAD results show a strong chunking effect. All retrieval-based systems greatly outperform the parametric baseline, and the ranking among chunkers is clear. The best fixed-size baseline is `fixed_128`, while `semantic_256` is the best overall system. Larger fixed windows degrade both EM and F1, which suggests that the generator is increasingly distracted by irrelevant context when chunks become too broad.

`recursive_256` is the strongest non-semantic system, indicating that respecting textual structure helps preserve compact, answer-bearing context. `semantic_256` improves further, likely because it keeps locally coherent evidence together while still staying small enough to avoid excessive prompt noise.

### 4.2 HotpotQA

| System | EM | F1 | Recall@4 | Precision@4 | Avg chunk tokens | # chunks |
|---|---:|---:|---:|---:|---:|---:|
| `parametric_only` | 13.3 | 17.9 | - | - | - | - |
| `fixed_128` | 36.7 | 43.1 | 100.0 | 47.5 | 89.6 | 427 |
| `fixed_256` | 36.7 | 44.8 | 100.0 | 46.7 | 117.3 | 310 |
| `fixed_512` | 36.7 | 44.8 | 100.0 | 46.7 | 122.0 | 294 |
| `recursive_256` | 36.7 | 49.7 | 100.0 | 46.7 | 116.3 | 310 |
| `sentence_256` | 36.7 | 49.7 | 100.0 | 46.7 | 115.0 | 310 |
| `semantic_256` | 36.7 | 47.2 | 100.0 | 45.8 | 96.8 | 368 |

HotpotQA tells a different story. Exact Match is flat across all chunkers at 36.7, but token-level F1 differs meaningfully. Recursive and sentence chunking outperform all fixed-size baselines on F1, and semantic chunking lands in between. This suggests that chunking still matters for answer quality, but the bottleneck shifts from pure retrieval to how well the generator can compose or verbalize multi-hop evidence.

The retrieval metric saturates on this small induced distractor corpus, so Hotpot's answer-level differences are more informative than Recall@4 alone. The most plausible interpretation is that structure-aware chunkers help the model recover more of the needed evidence span, even when they do not fully solve the reasoning problem.

### 4.3 Auxiliary Chonkie Comparison

| System | SQuAD EM | SQuAD F1 | HotpotQA F1 | Avg chunk tokens |
|---|---:|---:|---:|---:|
| `recursive_256` | 65.0 | 75.1 | 49.7 | 175.8 |
| `chonkie_recursive_256` | 66.7 | 75.9 | 49.7 | 174.3 |
| `semantic_256` | 76.7 | 85.4 | 47.2 | 144.8 |
| `chonkie_semantic_256` | 65.0 | 77.1 | 49.7 | 101.5 |

The Chonkie comparison adds a useful nuance. Recursive chunking is very stable across implementations: Chonkie's version is nearly identical to the LangChain-based baseline and even gains slightly on SQuAD. Semantic chunking is less stable. Chonkie's semantic splitter produces many more, smaller chunks, which helps HotpotQA F1 but substantially reduces SQuAD accuracy relative to the custom semantic chunker. That suggests semantic chunking is not one single method so much as a family of boundary heuristics whose behavior depends heavily on implementation details.

## 5. Error Analysis

Two patterns stood out.

First, on SQuAD, chunking changed both retrieval quality and answer generation quality. `semantic_256` had only 3 retrieval failures and 11 wrong answers despite retrieved context, compared with 5 retrieval failures and 24 wrong-with-context cases for `fixed_256`. One illustrative example is:

- Question: "What is thought to have happened to the y. pestis that caused the black death?"
- Gold answer: "may no longer exist"
- `recursive_256`: `unanswerable`
- `semantic_256`: `may no longer exist`

That example suggests that the semantic chunker sometimes preserves a short causal statement as a cleaner unit than the recursive alternative.

Second, on HotpotQA, chunking mostly affected answer completeness rather than retrieval recall. For the question:

- "The director of the romantic comedy 'Big Stone Gap' is based in what New York city?"
- Gold answer: "Greenwich Village, New York City"
- `fixed_256`: `virginia`
- `recursive_256`: `New York City`
- `sentence_256`: `New York City`
- `semantic_256`: `Virginia`

Here the structure-aware chunkers recover the correct city mention more often, while fixed and semantic chunking are more likely to latch onto the distractor location from the film description.

## 6. Discussion

The main conclusion is that chunking is not a cosmetic preprocessing choice. It changes both what gets retrieved and how useful that retrieved evidence is to the generator.

Three trends are especially clear.

1. Smaller chunks are better than larger fixed windows on SQuAD.
2. Structure-aware chunkers are consistently strong.
3. The best chunking strategy depends on task complexity.

For relatively local questions like SQuAD, semantic chunking works best once over-fragmentation is controlled. For multi-hop questions like HotpotQA, recursive and sentence chunking are safer choices because they preserve more interpretable local structure without splitting evidence too aggressively.

These results also highlight a broader methodological point: retrieval metrics can saturate while answer quality still changes. That makes end-to-end QA evaluation essential for chunking studies.

The auxiliary Chonkie comparison reinforces another point: recursive chunking seems conceptually robust, while semantic chunking is more implementation-sensitive. When a paper claims gains from "semantic chunking," the exact splitter design may matter nearly as much as the label itself.

## 7. Limitations

This project has several important limitations.

- The experiments use small evaluation subsets.
- SQuAD uses only answerable questions, not the full unanswerable setting of SQuAD 2.0.
- The generator is `flan-t5-base`, not the stronger 7B-class models proposed originally.
- The Hotpot retrieval metric is coarse and does not fully capture supporting-fact coverage.
- I ran the project locally instead of scaling the final sweep to SCC.
- The Chonkie comparison covers only a subset of systems and should be interpreted as an auxiliary implementation study.

None of these invalidate the observed chunking trends, but they do limit how far the conclusions should be generalized.

## 8. Conclusion

This project shows that chunking strategy can substantially affect retrieval-augmented QA. On a more diverse SQuAD setup, semantic chunking delivered the strongest results, while recursive chunking was the best broadly robust alternative. On HotpotQA, recursive and sentence chunking produced the best F1, suggesting that structure-aware segmentation helps when answers depend on combining multiple pieces of evidence. The auxiliary Chonkie run sharpened that conclusion by showing that recursive chunking is fairly stable across libraries, whereas semantic chunking depends much more on the exact implementation.

The next natural step is to rerun the same pipeline with larger subsets and a stronger instruction-tuned generator on SCC. Even in its current form, though, the project supports the proposal's core claim: chunking is a meaningful experimental variable, and the "best" chunker depends on the QA setting.

## References

- Karpukhin, Vladimir, et al. 2020. Dense Passage Retrieval for Open-Domain Question Answering. [arXiv](https://arxiv.org/abs/2004.04906)
- Lewis, Patrick, et al. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. [arXiv](https://arxiv.org/abs/2005.11401)
- NVIDIA. 2024. Finding the Best Chunking Strategy for Accurate AI Responses. [Blog](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/)
- Qu, Renyi, Ruixuan Tu, and Forrest Bao. 2024. Is Semantic Chunking Worth the Computational Cost? [arXiv](https://arxiv.org/abs/2410.13070)
- Rajpurkar, Pranav, Robin Jia, and Percy Liang. 2018. Know What You Don't Know: Unanswerable Questions for SQuAD. [ACL Anthology](https://aclanthology.org/P18-2124/)
- Shaukat, Muhammad Arslan, Muntasir Adnan, and Carlos C. N. Kuhn. 2026. A Systematic Investigation of Document Chunking Strategies and Embedding Sensitivity. [arXiv](https://arxiv.org/abs/2603.06976)
- Smith, Brandon, and Anton Troynikov. 2024. Evaluating Chunking Strategies for Retrieval. [Chroma Technical Report](https://www.trychroma.com/research/evaluating-chunking)
- Yang, Zhilin, et al. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. [EMNLP Anthology](https://aclanthology.org/D18-1259/)
