# EAAI Phase 2 Code and Artifact Audit

## Audit status

- Audit date: 2026-08-09 (UTC baseline freeze: 2026-08-09T18:10:02.423729+00:00).
- Repository branch at freeze: `eswa-transfer-revision`.
- Repository HEAD at freeze: `964a9e0fcfe2893e927eba15deae339cbda807f0`.
- Baseline inventory: `reports/eaai_phase2_baseline_manifest.json`.
- Frozen files: 1,322 files, 101,736,005 bytes.
- Frozen tree digest: `7572ec911f852c9d420c6728f74fedbfbc88f2652e2df4b9b4963a363fda9ac6`.
- Audit scope: the full EAAI manuscript and supplement, experiment configuration and orchestration code, retrieval and generation implementations, retained question-level artifacts, and existing statistical scripts.
- No retrieval, embedding, reranking, or generation inference was run for this audit.

The worktree was already dirty when the freeze was created. The manifest therefore freezes bytes rather than treating Git cleanliness as provenance. Every full Phase 2 run must verify all listed file hashes before execution and must write only below `results/eaai_phase2/` and `artifacts/eaai_phase2/`.

## Existing scientific claim and missing experiment

The current EAAI manuscript is an engineering evaluation rather than a new RAG method. It correctly reports that the retained reranking extension and controlled generation extension are separate experiments. The reranking extension compares BGE hybrid RRF with the same candidate pool followed by an MS MARCO cross-encoder, but it records retrieval outcomes only. The controlled TechQA generation extension uses BGE hybrid retrieval without reranking. Consequently, the retained evidence cannot answer whether the observed TechQA retrieval degradation propagates to token F1 or exact match.

The main scientific gap is therefore real and narrowly defined: no retained TechQA question has paired hybrid and hybrid-plus-reranker generations under the same generator. Candidate scores and per-question reranking latency were also not retained, so an adaptive reranking policy cannot be reconstructed retrospectively.

## Exact reusable implementations

### TechQA loading and corpus construction

`src/chunkrag/data.py::load_techqa_documents_and_examples` loads `nvidia/TechQA-RAG-Eval` at a caller-supplied revision and split. It retains rows satisfying all three conditions: `is_impossible` is false, the stripped answer is non-empty, and at least one context exists. It builds one closed corpus from every eligible row before sampling questions. Context filenames become document identifiers of the form `techqa::{filename}`; conflicting text under one filename raises an error. The current pinned train revision is `0b5bbc84b7f07d6d09d063130e90b716d8d4a32a`.

Direct read-only inspection of the cached Arrow file found 910 total rows, 608 eligible answerable rows with unique IDs, and 496 unique context filenames. Existing runs shuffle eligible rows with a data seed and select the first `max_examples`. Phase 2 must replace that sampling step with a frozen development/test/reserve partition while reusing the same eligibility and corpus rules.

### Chunking

`src/chunkrag/chunking.py` uses the pinned `sentence-transformers/all-MiniLM-L6-v2` tokenizer revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The prospective configurations are:

| Name | Method | Target | Overlap | Limit enforcement |
|---|---|---:|---:|---|
| `fixed_128` | token windows | 128 | 19 | round-trip token count enforced |
| `fixed_254` | token windows | 254 | 38 | round-trip token count enforced |
| `recursive_254` | LangChain recursive splitter | 254 | 38 | oversize fragments split again |
| `sentence_254` | spaCy sentencizer packing | 254 | 0 | oversize sentences split again |

Fixed windows decode the longest prefix whose re-encoded length fits the target. Recursive splitting uses separators `\n\n`, `\n`, `. `, space, and the empty string. Sentence packing appends sentences until the next sentence would exceed the target. Phase 2 should call the existing chunk builders unchanged.

### Dense retrieval and indexing

`src/chunkrag/retrieval.py::DenseRetriever` uses `SentenceTransformer.encode` with normalized embeddings and a FAISS `IndexFlatIP`. It applies the configured query prefix only to queries. BGE is pinned as `BAAI/bge-small-en-v1.5` revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; queries use `Represent this sentence for searching relevant passages: `. Embedding cache keys include the encoder identifier and every chunk ID, document ID, and text byte sequence.

### BM25 and weighted reciprocal-rank fusion

`BM25Retriever` tokenizes with lowercase Unicode `\w+` tokens and uses `rank_bm25.BM25Okapi`. `HybridRetriever` independently requests 20 dense and 20 BM25 results, then calls `mean_reciprocal_rank_fusion`. For a candidate at one-indexed rank `r`, the component contribution is `weight / (60 + r)`. Dense and sparse weights are 0.6 and 0.4. The fused top 20 form the reranker candidate pool, and the final retrieval depth is four.

### Cross-encoder reranking

`RerankRetriever` scores `(question, chunk.text)` pairs with `sentence_transformers.CrossEncoder.predict`, sorts scores descending, and keeps four candidates. The model is `cross-encoder/ms-marco-MiniLM-L6-v2`, revision `c5ee24cb16019beea0893ab7796b1df96625c6b8`, with a batch size of 32 in the retained reranking config. The existing class does not expose component rankings, candidate scores, or reranker-only latency. Phase 2 needs an isolated wrapper that reuses these operations while retaining those fields.

### Context construction and prompts

`src/chunkrag/pipeline.py::SystemRunner._format_context` joins TechQA chunks in final rank order as `[i] {chunk.text}`, separated by blank lines. Titles are included only for HotpotQA, so TechQA does not receive an additional title field.

`src/chunkrag/generation.py::build_local_qa_messages` supplies TechQA's `complete` prompt. Its system message requires a concise but complete grounded answer, no reasoning or citations, and exactly `unanswerable` when unsupported. The user message contains the question, ranked context passages, and `Return only the final answer.`

For causal checkpoints, `QAGenerator` applies the model's native chat template and greedily decodes with `do_sample=False` and `num_beams=1`. The complete-chat input budget is 1,536 tokens. If necessary, `_prepare_causal_qa_prompt` binary-searches for the longest prefix of the ranked context that fits; this can truncate inside a chunk. TechQA allows at most 512 new tokens. Complete answers preserve multiple lines and do not pass through the extractive `compress_answer` heuristic. Raw output, normalized output, prompt lengths, truncation, generated length, and length-cap status are retained.

The pinned generators are:

- Primary feasible Colab checkpoint: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Secondary replication checkpoint: `mistralai/Mistral-7B-Instruct-v0.3`, revision `c170c708c41dac9275d15a8fff4eca08d52bab71`, float16.

### Answer and retrieval metrics

`src/chunkrag/text_utils.py::normalize_answer` lowercases, removes ASCII punctuation and the English articles `a`, `an`, and `the`, then normalizes whitespace. Exact match compares normalized strings. Token F1 uses multiset token overlap and the harmonic mean of token precision and recall; the maximum score over reference answers is retained.

`src/chunkrag/evaluation.py::retrieval_metrics` computes:

- Precision@k from chunks whose document ID is gold or whose text contains a normalized reference-answer token sequence.
- Recall@k and support coverage from the fraction of gold document IDs represented.
- AllHit@k from whether every gold document ID is represented.
- AnsVis@k from whether any retrieved chunk contains a complete normalized reference-answer token sequence.

For TechQA, the long explanatory answer can make AnsVis conservative. The Phase 2 primary endpoint is token F1, not AnsVis or AllHit.

### Existing uncertainty procedures

`scripts/analyze_reviewer_robustness.py::paired_bootstrap_ci` resamples question-level paired differences with replacement and takes percentile 2.5% and 97.5% bounds. `paired_randomization_p` independently flips the sign of each paired difference and reports a two-sided Monte Carlo p-value with the plus-one correction. The existing prospective generation analysis applies Holm adjustment to a frozen 18-test family. Phase 2 is a separate prospective family and must not read, modify, or expand that family.

The existing reranking analysis (`scripts/analyze_ipmc_firm_rerank.py`) pairs hybrid and reranked binary retrieval outcomes within dataset, chunker, and seed. It aggregates seed means descriptively. It performs no reranked generation analysis.

## Retained artifact audit

### Retrieval extension

`results/canonical/retrieval_extension.jsonl.gz` retains question text, references, retrieved chunk/document IDs, retrieval metrics, dataset and model revisions, cell-mean latency, and provenance. It does not retain candidate scores or per-question latency. BM25 was serialized under both embedder labels in the raw matrix and is deduplicated by later analysis.

### Reranking extension

`results/canonical/reranking_extension.jsonl.gz` and `outputs/ipmc_firm_rerank_bge/` contain paired hybrid/reranked top-four outcomes for seeds 13, 21, and 34. TechQA has 200 sampled questions per seed and 496 corpus documents. Existing rows contain final retrieved IDs and binary evidence metrics but no generation, candidate scores, reranker decisions, or per-question latency.

### Generation extension

`results/canonical/generation_extension.jsonl.gz` contains deterministic Qwen and Mistral generations, answer scores, top-four hybrid chunk/document IDs, prompt traces, and cell-mean latency. The data seed is 42. Reranking is absent (`reranking_condition=none`), and candidate pools were not retained. Therefore these rows cannot be transformed into a valid paired reranking-generation experiment.

### Manuscript and supplement

The EAAI source explicitly states that reranking and controlled generation are separate extensions and avoids a causal propagation claim. It also states that candidate scores and per-question latency are unavailable. These statements are accurate for the frozen baseline and must remain byte-identical until the held-out Phase 2 analysis is complete. Manuscript integration, if warranted, must be performed on a copy or in a later explicitly authorized stage.

## Reuse versus new isolated code

Reuse without modification:

- TechQA eligibility and document construction semantics.
- Existing prospective chunk builders and tokenizer limits.
- BGE embedding, BM25 tokenization, FAISS inner-product retrieval, and weighted RRF formula.
- Cross-encoder model, pair construction, sorting direction, candidate depth, and batch size.
- TechQA context formatting, complete-answer prompt, native chat template, truncation rule, greedy decoding, and normalization.
- EM, token F1, AllHit, AnsVis, and retrieval metrics.

New isolated Phase 2 code is required for:

- deterministic disjoint development, held-out test, and reserve partitions over all 608 eligible TechQA IDs;
- one retrieval pass that exposes dense, BM25, fused, and reranked rankings without changing their algorithms;
- pre-reranker-only feature extraction;
- a lightweight gate trained only on development outcomes;
- paired hybrid and reranked generation with checkpoint/resume support;
- per-question timing and complete candidate-score traces;
- the new question-level bootstrap and sign-flip family;
- baseline hash verification and result-directory guards;
- Colab GPU execution and deterministic analysis.

No existing source output needs to be overwritten. New code should live in a dedicated `chunkrag.eaai_phase2` package and new scripts/configurations should carry the `eaai_phase2` name.

## Leakage and validity risks identified before protocol freeze

1. **Outcome leakage into the gate.** Gold answers, gold document IDs, AnsVis, AllHit, generated text, cross-encoder scores, and reranked ranks cannot be gate features. The gate must be evaluated on question IDs disjoint from training.
2. **Post-decision feature leakage.** A feature requiring execution of the cross-encoder defeats selective reranking. Only query, chunk metadata, dense/BM25 rankings, and fused scores available before reranking are eligible.
3. **Correlated chunker rows.** Four configurations share each question. Splitting and inferential resampling must operate by question, never by question--chunker row.
4. **Generator multiplicity.** Using both generators in one primary endpoint would obscure interpretation and add a model dimension. A single prespecified primary generator is statistically cleaner; a second generator can be a labeled secondary replication.
5. **Adaptive threshold tuning.** Searching thresholds on held-out results would invalidate the evaluation. Model hyperparameters and the decision threshold must be frozen before the full run.
6. **Latency interpretation.** Colab timing is hardware- and load-dependent. Invocation rate and reranker-only wall time should be primary efficiency descriptors; timing must not be presented as model-intrinsic.
7. **Repeated stochastic decoding.** Generation is deterministic and has one run per checkpoint. Question-level inference conditions on those fixed generations and does not estimate decoding variability.
8. **Data reuse.** Earlier studies sampled from the same benchmark. The new held-out split is held out from adaptive-policy fitting, not claimed to be a never-observed external benchmark.

## Audit conclusion

The existing code is sufficient to reproduce every scientific component of the proposed direct comparison, but the old artifacts cannot answer it retrospectively. A valid extension requires fresh, paired TechQA generation under hybrid and hybrid-plus-reranker retrieval and a question-disjoint development/test design for any adaptive policy. The cleanest confirmatory endpoint is the held-out question-level mean difference in Qwen token F1 averaged over the four prespecified chunkers. Adaptive gating and Mistral transfer should be secondary analyses. This conclusion is made before any Phase 2 result generation.
