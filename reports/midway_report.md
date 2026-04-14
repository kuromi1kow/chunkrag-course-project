# Comparing Chunking Strategies for Retrieval-Augmented Question Answering

**Team Members:** Pablo Bello, Assylkhan Geniyat, Jaime Bernal, Batyrkhan Baimukhanov

## Abstract

This empirical project studies how document chunking affects retrieval-augmented question answering. The proposal asked whether chunking strategy changes end-to-end QA accuracy, and whether the best strategy depends on query complexity. To investigate this, we built a controlled RAG pipeline that holds the embedding model, dense retriever, and generator fixed while varying only the chunking function. The current comparison includes fixed-size token chunking, recursive chunking, sentence-based chunking, and semantic chunking, where the semantic splitter acts as a dynamic content-aware chunker rather than a fixed window. Experiments on SQuAD v2 pseudo-documents and HotpotQA distractor examples already show a clear chunking effect. On SQuAD, semantic chunking is the strongest system so far, while on HotpotQA recursive and sentence chunking provide the best token-level F1. A no-retrieval baseline performs much worse on both datasets, confirming that the task is retrieval-sensitive rather than purely parametric. At the same time, the preliminary results suggest that local factoid QA and multi-hop QA reward different chunking behaviors. We also implemented an auxiliary Chonkie comparison and a stronger retrieval stack for final-stage analysis. Because the current results come from small subsets and a lightweight generator, we treat them as preliminary but informative. The main remaining work is to scale the experiments to larger subsets, strengthen the retrieval diagnostics, and rerun the comparison with a stronger instruction-tuned generator on SCC.

## 1. Introduction

Retrieval-augmented generation (RAG) answers questions by retrieving external evidence and passing it to a language model. In a standard pipeline, however, retrieval is done over chunks rather than full documents. That makes chunking a central design decision: the chunking function controls how much context is stored together, how likely a retriever is to recover relevant evidence, and whether the final prompt contains coherent support or fragmented noise.

This project follows the empirical plan proposed in [proposal.pdf](/Users/assylkhan/Downloads/proposal.pdf). The core question is:

Which chunking strategy maximizes end-to-end QA accuracy in a fixed RAG pipeline, and does the answer differ between relatively local questions and more complex multi-hop questions?

Formally, let a corpus be \(D\), a question be \(q\), and a chunking function be \(c : D \rightarrow \{d_1, \ldots, d_n\}\). After chunking, each chunk is embedded and indexed. At query time, the system retrieves the top-\(K\) chunks for \(q\), concatenates them into a prompt, and produces an answer \(\hat{a}\). The optimization target is not retrieval alone but answer quality:

\[
c^* = \arg\max_c \; \mathbb{E}_{(q,a)} \left[ F_1(\hat{a}, a) \right].
\]

This problem is interesting for two reasons. First, chunking is widely treated as an implementation detail even though recent engineering studies suggest it can substantially affect retrieval behavior. Second, retrieval quality and answer quality do not always move together: a system can retrieve a locally relevant chunk but still fail if the evidence is incomplete or badly fragmented. That is exactly why the proposal emphasized both end-to-end QA metrics and retrieval diagnostics.

The problem is also hard in a way that makes simple baselines informative. A no-retrieval baseline asks the generator to answer from parametric knowledge alone, which should fail whenever the answer depends on the supplied corpus. A naive fixed-window strategy can also fail: large chunks may dilute evidence with distractors, while very small chunks may separate evidence that needs to stay together. The current experiments already show both failure modes.

## 2. Related Work

RAG combines neural retrieval with text generation, typically by retrieving passages and conditioning a generator on them (Lewis et al., 2020). Dense Passage Retrieval (Karpukhin et al., 2020) established dense retrieval as a strong backbone for question answering, which makes it a natural fixed retriever for this chunking study.

For datasets, SQuAD 2.0 (Rajpurkar et al., 2018) provides mostly local answer extraction over Wikipedia passages, while HotpotQA (Yang et al., 2018) is useful for testing multi-hop reasoning and evidence composition. This dataset pair mirrors the proposal's plan to compare chunking behavior across different query complexities.

The chunking motivation in the proposal came from recent benchmarking and engineering analyses. Smith and Troynikov (2024) show that retrieval quality can change meaningfully under different chunking strategies, while NVIDIA (2024) emphasizes that the best chunking policy should be judged end to end rather than only by retrieval recall. Qu et al. (2024) further question whether semantic chunking is always worth its extra cost. The current project fits into this line of work by isolating chunking as the only controlled experimental variable in a reproducible QA pipeline.

## 3. Methods

The implemented system follows the proposal closely. For each chunking strategy, the pipeline is:

1. Chunk each document.
2. Embed every chunk with the same encoder.
3. Build the same FAISS dense index.
4. Retrieve the top 4 chunks for each question.
5. Pass the retrieved context to the same generator.

The only variable is the chunking function.

### 3.1 Chunking Strategies

The current implementation includes the four chunking families from the proposal:

1. **Fixed token chunking:** `fixed_128`, `fixed_256`, and `fixed_512`, each with approximately 15% overlap.
2. **Recursive chunking:** `recursive_256`, implemented with LangChain's `RecursiveCharacterTextSplitter`.
3. **Sentence chunking:** `sentence_256`, implemented with spaCy sentence segmentation and greedy packing to a token budget.
4. **Semantic chunking:** `semantic_256`, implemented by embedding consecutive sentences and splitting when cosine similarity drops below a threshold, with a minimum chunk-length guard. In practice, this is our dynamic chunker: chunk length varies with local semantic continuity rather than a fixed token budget.

We also implemented auxiliary `chonkie_recursive_256` and `chonkie_semantic_256` variants. These let us compare not only chunking families, but also whether a result is robust across different library implementations of the same basic idea.

### 3.2 Models and Software

The current milestone fixes the rest of the pipeline as follows:

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Retriever: FAISS inner-product search over normalized dense embeddings
- Generator: `google/flan-t5-base`
- Retrieval depth: top 4 chunks
- Runtime: local Apple MPS

This is the main deviation from the proposal. The proposal planned to use a stronger 7B-class generator such as Mistral-7B or Llama-3-8B. For the midway report, we kept the generator smaller so the full chunking comparison would remain reproducible on local hardware. The current SCC rerun target is `mistralai/Mistral-7B-Instruct-v0.3`, which matches the promised model family better than the earlier Qwen-based deployment experiments. Beyond the dense-retrieval table reported here, the broader codebase now also supports BM25, hybrid dense+sparse retrieval, reranking, supporting-document coverage, and bootstrap confidence intervals. Those components are already implemented, but the main midway analysis keeps the ablation focused on chunking.

### 3.3 Baseline

The proposal required at least one simple baseline. The current baseline is **parametric-only QA**, where the generator answers the question without any retrieved context. This directly tests whether retrieval is actually necessary and provides a clear lower bound for the retrieval-based systems.

## 4. Experiments

### 4.1 Datasets

The current experiments use the two datasets proposed originally.

- **SQuAD v2:** 60 answerable validation questions are sampled from a shuffled 500-example candidate pool. Paragraphs with the same title are concatenated into article-level pseudo-documents to make chunking non-trivial. The resulting corpus has 35 pseudo-documents.
- **HotpotQA distractor:** 30 validation questions are used, with the induced document collection built from the distractor and supporting titles in each example. The resulting corpus has 291 documents.

This setup is intentionally small for the midway stage, but it is large enough to reveal meaningful differences between chunking strategies.

### 4.2 Evaluation

The report follows the evaluation design from the proposal.

- **End-to-end QA:** Exact Match and token-level F1
- **Retrieval:** Recall@4 and Precision@4
- **Efficiency:** average chunk size, number of indexed chunks, and generation latency

The proposal also called for error analysis. We have started organizing failures into three categories:

1. **Retrieval failure:** the relevant chunk is not retrieved.
2. **Context fragmentation:** evidence is split in a way that weakens generation.
3. **Generation failure:** the relevant evidence is retrieved, but the model still answers incorrectly.

### 4.3 Experimental Status

At this point, the core dense-retrieval chunking comparison is implemented and run end to end on both datasets. This means the project is past the setup stage and already has interpretable preliminary results, but larger subsets and stronger generators remain for the final milestone.

## 5. Results

### 5.1 SQuAD v2

| System | EM | F1 | Recall@4 | Precision@4 | Avg chunk tokens | # chunks |
|---|---:|---:|---:|---:|---:|---:|
| `parametric_only` | 1.7 | 6.7 | - | - | - | - |
| `fixed_128` | 61.7 | 72.8 | 95.0 | 78.8 | 125.3 | 631 |
| `fixed_256` | 51.7 | 59.9 | 91.7 | 74.6 | 244.0 | 322 |
| `fixed_512` | 46.7 | 53.4 | 93.3 | 65.0 | 472.2 | 164 |
| `recursive_256` | 65.0 | 75.1 | 91.7 | 73.8 | 175.8 | 389 |
| `sentence_256` | 60.0 | 68.4 | 88.3 | 72.9 | 223.9 | 302 |
| `semantic_256` | 76.7 | 85.4 | 95.0 | 77.1 | 144.8 | 467 |

### 5.2 HotpotQA

| System | EM | F1 | Recall@4 | Precision@4 | Avg chunk tokens | # chunks |
|---|---:|---:|---:|---:|---:|---:|
| `parametric_only` | 13.3 | 17.9 | - | - | - | - |
| `fixed_128` | 36.7 | 43.1 | 100.0 | 47.5 | 89.6 | 427 |
| `fixed_256` | 36.7 | 44.8 | 100.0 | 46.7 | 117.3 | 310 |
| `fixed_512` | 36.7 | 44.8 | 100.0 | 46.7 | 122.0 | 294 |
| `recursive_256` | 36.7 | 49.7 | 100.0 | 46.7 | 116.3 | 310 |
| `sentence_256` | 36.7 | 49.7 | 100.0 | 46.7 | 115.0 | 310 |
| `semantic_256` | 36.7 | 47.2 | 100.0 | 45.8 | 96.8 | 368 |

## 6. Discussion

The midway results already support the proposal's main hypothesis that chunking is a meaningful experimental variable.

First, the no-retrieval baseline is far worse than every retrieval-based system on both datasets. That confirms that the project is measuring a genuinely retrieval-sensitive problem rather than asking a small generator to answer from memorized knowledge alone.

Second, the best chunker appears to depend on question type. On SQuAD, semantic chunking is currently the strongest system, with recursive chunking as the best non-semantic baseline. That pattern is consistent with the proposal's intuition that local factoid questions benefit from compact but semantically coherent evidence. On HotpotQA, exact match is flat but token-level F1 improves under recursive and sentence chunking, which suggests that multi-hop questions benefit from preserving more interpretable local structure.

Third, the simple fixed-window strategy is not reliably strong. Smaller fixed chunks perform better than larger ones on SQuAD, which supports the claim that large windows can introduce distractor context. At the same time, very fine segmentation is not a universal win, because the multi-hop setting seems to reward chunk boundaries that stay closer to sentence and paragraph structure.

An auxiliary Chonkie comparison adds one more useful finding. Recursive chunking is fairly stable across implementations, but semantic chunking is more implementation-sensitive. That matters for the final report because it suggests that "dynamic" or "semantic" chunking should not be treated as a single method; the exact boundary heuristic can materially change the tradeoff between compactness and evidence preservation.

Overall, we view these scores as good midway results rather than final claims. They are strong enough to justify the project's direction, but they still need validation on larger subsets and under stronger generation settings.

The current error profile also matches the proposal's failure taxonomy. Some failures are clearly retrieval failures, especially when a relevant answer-bearing span never appears in the top 4 chunks. Others are context fragmentation errors, where the evidence is partially retrieved but split in an unhelpful way. On HotpotQA in particular, many remaining errors look like generation failures: the system retrieves relevant material, but the small generator does not compose the final answer exactly.

## 7. Project Plan

The remaining work is concrete and aligned with the proposal.

1. **Scale the current experiments.** Increase the evaluation subsets so the chunking comparison is statistically more stable.
2. **Move to SCC and use a stronger generator.** Replace `flan-t5-base` with a stronger instruction-tuned model, closer to the original 7B-class plan in the proposal.
3. **Strengthen retrieval diagnostics.** Add supporting-document coverage and a clearer separation between retrieval failure and generation failure, especially for HotpotQA.
4. **Deepen error analysis.** Add more qualitative examples showing when chunking helps and when it fragments evidence.
5. **Integrate the broader retrieval stack.** Use the already implemented BM25, hybrid, and reranked settings to test whether chunking effects persist under stronger retrieval.
6. **Convert the report into final ACL-ready form.** Integrate the methods and results more tightly, add better figures, and make the discussion more complete.

Relative to the proposal, the project has stayed on the same main trajectory. The largest change is computational rather than conceptual: the current milestone uses a smaller local generator for reproducibility, while the final stage will emphasize stronger-model validation and larger-scale runs.

## References

The bibliography for this project is maintained in [references.bib](/Users/assylkhan/Documents/NLP/reports/references.bib). The core references used in this midway report are:

- Karpukhin et al. 2020. *Dense Passage Retrieval for Open-Domain Question Answering.*
- Lewis et al. 2020. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
- NVIDIA. 2024. *Finding the Best Chunking Strategy for Accurate AI Responses.*
- Qu, Tu, and Bao. 2024. *Is Semantic Chunking Worth the Computational Cost?*
- Rajpurkar, Jia, and Liang. 2018. *Know What You Don't Know: Unanswerable Questions for SQuAD.*
- Smith and Troynikov. 2024. *Evaluating Chunking Strategies for Retrieval.*
- Yang et al. 2018. *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.*
