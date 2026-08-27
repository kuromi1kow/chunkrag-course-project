# Prospective Protocol: EAAI Phase 2 TechQA Reranking and Adaptive Invocation

## Protocol identity and freeze rules

- Protocol version: 1.0.
- Protocol date: 2026-08-09.
- Study label: `eaai_phase2`.
- Baseline inventory: `reports/eaai_phase2_baseline_manifest.json`.
- Baseline tree SHA-256: `7572ec911f852c9d420c6728f74fedbfbc88f2652e2df4b9b4963a363fda9ac6`.
- This protocol must be committed before any full development or held-out inference.
- After commit, the protocol text may not be rewritten in response to results. Deviations may only be appended to the dated deviation log at the end.
- Every full command must abort if any baseline hash differs or if this protocol differs from its committed version.
- New question-level results must be written below `results/eaai_phase2/`; models, partitions, manifests, candidate traces, and other supporting artifacts must be written below `artifacts/eaai_phase2/`.
- Existing retrospective, retrieval, reranking, generation, statistical, manuscript, supplement, and prior submission artifacts must remain byte-identical.

## Engineering objective

The extension evaluates whether the retained MS MARCO cross-encoder reranker changes end-to-end answer quality on TechQA, whether retrieval changes propagate to generation, and whether a lightweight policy can avoid reranking when pre-reranker signals predict no benefit.

The extension does not introduce a new retriever, reranker, generator, chunker, theorem, or causal identification strategy. Its methodological contribution is a low-cost, pre-reranker invocation gate evaluated on question-disjoint held-out data.

## Dataset, corpus, and frozen partition

The study uses the answerable rows of `nvidia/TechQA-RAG-Eval`, train split, revision `0b5bbc84b7f07d6d09d063130e90b716d8d4a32a`. Eligibility and corpus construction exactly follow `load_techqa_documents_and_examples`: `is_impossible` must be false, the stripped answer must be non-empty, and at least one context must exist. All eligible rows jointly define the same 496-document closed corpus.

The run must find exactly 608 eligible rows with unique question IDs and exactly 496 unique context filenames; otherwise it aborts before model inference.

Question IDs are assigned without reference to text, labels, retrieval outcomes, or generation outcomes:

1. Compute `sha256("eaai-phase2-techqa-v1\\0" + question_id)` for each eligible ID.
2. Sort by `(digest, question_id)` ascending.
3. Assign positions 1--200 to `development`.
4. Assign positions 201--400 to `heldout_test`.
5. Assign positions 401--608 to `reserve`.

The reserve set is not used in this extension. Partition artifacts are private research artifacts because question IDs derive from a benchmark; distributable packages must retain only counts and cryptographic hashes unless redistribution is permitted.

## Frozen system configurations

### Chunking

The four prespecified configurations are:

- `fixed_128`: 128 MiniLM tokens, overlap 19.
- `fixed_254`: 254 MiniLM tokens, overlap 38.
- `recursive_254`: 254 MiniLM tokens, overlap 38.
- `sentence_254`: 254 MiniLM tokens, no overlap.

All enforce the existing round-trip token limit. The tokenizer is `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

### Hybrid retrieval

- Dense model: `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
- Query prefix: `Represent this sentence for searching relevant passages: `.
- Dense and BM25 candidate depths: 20 each.
- BM25 tokenization: lowercase Unicode `\w+` tokens with `BM25Okapi`.
- Fusion: weighted reciprocal-rank fusion with dense weight 0.6, BM25 weight 0.4, and `rrf_k=60.0`.
- Fused candidate pool: top 20.
- Final context depth: top four.

### Reranking

- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L6-v2`.
- Revision: `c5ee24cb16019beea0893ab7796b1df96625c6b8`.
- Input pairs: `(question, chunk.text)` for the fused top 20.
- Batch size: 32.
- Sort: descending cross-encoder score.
- Final context depth: top four.

### Generation

The primary generator is `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. It uses the existing TechQA complete-answer messages, native chat template, greedy decoding, a 1,536-token complete-chat input limit, prefix context truncation, and at most 512 new tokens. No sampling, beam search, answer compression, or refinement is permitted.

The secondary replication generator is `mistralai/Mistral-7B-Instruct-v0.3`, revision `c170c708c41dac9275d15a8fff4eca08d52bab71`, float16, with the same prompt, context, truncation, and decoding settings. Mistral is run only on the held-out test set after the Qwen-trained gate is frozen. Mistral results are not part of the primary test.

Both retrieval conditions for a question--chunker pair must be generated in the same run environment and from the same corpus and question. Condition execution order alternates deterministically by the parity of `sha256(question_id + "\\0" + chunker)` to reduce systematic order effects. This is not random decoding.

## PRIMARY CONFIRMATORY ANALYSIS

### Question

For held-out TechQA questions, does cross-encoder reranking improve or degrade end-to-end token F1 relative to hybrid RRF retrieval under the primary Qwen generator?

### Sampling and analysis unit

The held-out test contains 200 question IDs. Four chunker observations are repeated measurements within each question and are not treated as independent samples.

For question `q`, let `F1_H(q,c)` be token F1 for hybrid RRF and `F1_R(q,c)` token F1 for hybrid RRF plus reranking for chunker `c`. Define:

`F1_H(q) = mean_c F1_H(q,c)`

`F1_R(q) = mean_c F1_R(q,c)`

`DeltaF1(q) = F1_R(q) - F1_H(q)`

The primary point estimate is `mean_q DeltaF1(q)`. Positive values favor reranking. This aggregation was chosen before full inference to produce one global reranking effect while respecting question-level dependence and avoiding four primary chunker tests.

### Primary endpoint and inference

- Primary endpoint: normalized token F1 as implemented by `chunkrag.text_utils.token_f1` and maximized across references.
- Confidence interval: two-sided 95% percentile paired bootstrap over the 200 question-level differences, 20,000 draws, NumPy `PCG64` seed 20,260,809.
- Test: two-sided paired sign-flip/randomization test over the 200 question-level differences, 100,000 Monte Carlo draws, NumPy `PCG64` seed 20,260,810, with plus-one correction.
- Standardized effect: paired-sample Cohen's `dz = mean(DeltaF1) / sample_sd(DeltaF1)`. It is reported as unavailable if the sample standard deviation is zero.
- Multiplicity: there is one primary test. It is a new prospective family and receives no Holm adjustment. The previous 18-test Holm family remains unchanged.
- Reporting: point estimate, interval, raw two-sided p-value, effect size, number of positive/negative/tied question differences, and the complete paired question-level distribution.

No alternative primary endpoint, aggregation, test, seed, exclusion, or model may replace this analysis after results are observed.

## SECONDARY PRESPECIFIED ANALYSES

These analyses are not allowed to replace or redefine the primary result.

1. **Exact match.** Repeat the question-level chunker aggregation for EM. Report the mean paired difference and 95% paired bootstrap interval without an additional hypothesis test.
2. **Chunker-specific propagation.** Report paired F1 and EM changes separately for each of the four chunkers with descriptive 95% paired bootstrap intervals. Do not select a best chunker or add p-values.
3. **Retrieval-to-generation propagation.** Cross-tabulate changes in AllHit@4 and AnsVis@4 against the sign and magnitude of paired F1 changes. Report counts and descriptive associations, not causal mediation.
4. **Context budget.** Report condition-specific prompt truncation and generated-length capping, and paired F1 changes within truncation strata. These strata are descriptive.
5. **Mistral replication.** On the same 200 held-out questions, compute the same global F1 and EM paired estimates and bootstrap intervals. Do not combine Qwen and Mistral into one primary endpoint and do not add a second confirmatory p-value.
6. **Efficiency.** Report per-question hybrid retrieval time, reranker-only time, total pre-generation retrieval time, generation time, reranker invocation rate, and latency ratios. Timings are hardware-specific engineering measurements.

## LIGHTWEIGHT ADAPTIVE RERANKING METHOD

### Decision point

The gate acts after dense retrieval, BM25 retrieval, and weighted RRF have produced the fused top-20 candidate pool, but before the cross-encoder runs. It decides independently for each question--chunker pair whether to use the hybrid top four or invoke the cross-encoder and use the reranked top four.

Gold answers, gold document IDs, retrieval labels, generated outputs, cross-encoder scores, reranked ranks, and any post-reranker timing are prohibited features.

### Frozen pre-reranker features

The following numeric features are computed before reranking:

1. `query_token_count`: count from the existing lowercase `\w+` tokenizer.
2. `dense_bm25_jaccard_at_20`: set Jaccard overlap of dense and BM25 top-20 chunk IDs.
3. `dense_bm25_jaccard_at_4`: set Jaccard overlap of dense and BM25 top-four chunk IDs.
4. `fused_top1_score`: weighted RRF score of the first fused candidate.
5. `fused_top1_top2_margin`: first minus second fused score.
6. `fused_top4_top5_margin`: fourth minus fifth fused score.
7. `fused_score_entropy`: normalized entropy of non-negative fused top-20 scores after dividing by their sum; zero if fewer than two non-zero values.
8. `fused_top4_mean_dense_rank`: mean dense rank of fused top-four chunks, with missing candidates assigned rank 21.
9. `fused_top4_mean_bm25_rank`: mean BM25 rank of fused top-four chunks, with missing candidates assigned rank 21.
10. `fused_top4_mean_query_overlap`: mean over top-four chunks of `|query_tokens intersect chunk_tokens| / max(1, |query_tokens|)`.
11. `fused_top4_max_query_overlap`: maximum of the same overlap.
12. `fused_top4_mean_chunk_tokens`: mean stored token count of the fused top four.
13. `fused_top4_sd_chunk_tokens`: population standard deviation of those token counts.
14. One-hot indicators for the four prespecified chunker names.

The exact feature order and formulas must be serialized with the trained gate.

### Development target and model

Development provides 800 question--chunker rows from 200 questions. The binary target is one exactly when reranked Qwen F1 is strictly greater than hybrid Qwen F1 for that row; losses and ties receive zero. All rows from one question remain in the same development partition by construction.

The gate is a scikit-learn pipeline with median imputation, standard scaling for numeric features, one-hot chunker encoding with the frozen four-category order, and L2 logistic regression using `C=1.0`, `solver="liblinear"`, `class_weight="balanced"`, `max_iter=1000`, and `random_state=20260809`. The invocation threshold is fixed at probability 0.5. There is no hyperparameter search, cross-validation, feature selection, calibration, or threshold tuning.

If the development labels contain only one class, the prespecified fallback is a constant gate that predicts that class. This event must be reported and no alternative model may be substituted.

The serialized gate, feature schema, development question-partition hash, training-label counts, software versions, and SHA-256 digest must be frozen before any held-out generation is analyzed. The held-out runner records and verifies this digest.

### Held-out adaptive evaluation

For each held-out question--chunker row, the frozen gate selects the already paired hybrid or reranked output. Adaptive F1 and EM are then averaged across chunkers within question, exactly as in the primary analysis.

Report:

- adaptive F1 and EM;
- paired adaptive-minus-always-hybrid differences with descriptive 95% question bootstrap intervals;
- paired adaptive-minus-always-rerank differences with descriptive 95% question bootstrap intervals;
- reranker invocation rate overall and by chunker;
- fraction of the always-rerank F1 change relative to hybrid retained by the adaptive system, reported only when the denominator is non-zero;
- measured reranker overhead avoided, constructed from per-question component timings selected by the frozen gate;
- an unattainable per-row oracle that selects the higher-F1 branch, labeled strictly as a diagnostic upper bound.

No adaptive comparison receives a confirmatory p-value. Negative or null held-out performance is retained and reported.

For Mistral, the Qwen-trained gate is applied without refitting. This is a secondary transfer diagnostic.

## EXPLORATORY ANALYSES

The following are allowed only when labeled exploratory and reported regardless of direction:

- gate coefficients and univariate feature distributions by development benefit label;
- gate discrimination on development data, explicitly in-sample and not a generalization estimate;
- held-out benefit classification accuracy, balanced accuracy, precision, recall, Brier score, and ROC AUC when both held-out classes occur;
- answer-length, query-length, retrieval-agreement, chunker, and truncation strata;
- gain/loss examples selected by deterministic largest absolute paired F1 difference, with benchmark text excluded from distributable anonymous artifacts unless permitted;
- rank changes and top-four set overlap between hybrid and reranked retrieval.

No exploratory result may alter the gate, partition, primary endpoint, statistical family, or manuscript's primary conclusion.

## Execution stages and stopping rules

1. `preflight`: verify all 1,322 frozen baseline files and the committed protocol; verify output isolation and exact pinned configuration.
2. `partition`: load TechQA, validate expected counts, and write private partition IDs plus public count/hash metadata.
3. `development`: run paired Qwen hybrid/reranked retrieval and generation for the 200 development questions and four chunkers. Checkpoint after every question--chunker pair.
4. `fit-gate`: train the frozen logistic gate using development outcomes only; write and hash the gate manifest.
5. `heldout-qwen`: verify the gate digest, then run paired Qwen hybrid/reranked conditions for the 200 held-out questions and four chunkers.
6. `analyze-primary`: execute the frozen primary analysis once and write immutable analysis outputs.
7. `heldout-mistral`: optional secondary replication on the same held-out IDs, using the frozen Qwen-trained gate without refitting.
8. `analyze-secondary`: write secondary and exploratory artifacts without modifying primary outputs.

The runner supports resume by recognizing complete row artifacts whose input and configuration hashes match. It must never overwrite a completed row with a different hash. An explicit failed/incomplete marker is retained after exceptions.

The run must stop before inference if model or dataset revisions cannot be resolved exactly, the baseline differs, the partition counts differ, the protocol is uncommitted or modified, output paths escape the two Phase 2 roots, or an output conflict is found. Runtime out-of-memory errors may be addressed only through batch-size reduction for embedding or cross-encoder scoring; any such change is appended below before resuming. Generator quantization, prompt changes, model substitution, context-depth changes, or reduced held-out samples require a documented protocol deviation and cannot be silently applied.

## Artifact schema and reporting obligations

Every question--chunker--condition row retains:

- protocol, config, baseline, source, partition, and gate digests;
- question ID, private question/reference fields, split, and chunker;
- dense, BM25, fused, and final candidate IDs and scores/ranks;
- all pre-reranker features;
- cross-encoder scores for the reranked condition;
- final chunk/document IDs and retrieval metrics;
- raw and normalized generation, EM, F1, prompt trace, and token counts;
- component wall times and hardware/runtime metadata;
- completion timestamp and row SHA-256.

Question text, answers, context text, labels, and benchmark-derived IDs are private research artifacts and must be excluded from later anonymous source packages unless redistribution is clearly permitted. Aggregate analyses must remain reproducible from the retained controlled archive.

All negative, null, tied, and failed observations remain in the controlled results. Exclusions are limited to pre-inference eligibility rules or technically incomplete rows; incomplete rows are listed, not silently dropped.

## Manuscript integration rule

No Phase 2 claim or adaptive method is integrated into the manuscript until:

- development and held-out partitions are verified disjoint;
- the gate was frozen before held-out analysis;
- all 200 held-out questions have complete paired Qwen outputs for all four chunkers;
- baseline hashes still pass;
- the primary analysis reproduces deterministically from row artifacts;
- all deviations and failures are disclosed.

If the adaptive method does not improve the prespecified held-out engineering trade-off, it may be reported as a negative method evaluation but must not be described as an improvement. The direct reranking-to-generation result is reported regardless of direction.

## Dated deviation log

No deviations recorded as of 2026-08-09. Future entries must be appended below this line and must not edit the protocol above.

### 2026-08-27: Colab backend-import compatibility

The current Google Colab Python 3.13 image failed before retrieval because
Transformers imported an unused preinstalled TensorFlow/JAX backend whose JAX
version expects NumPy 2, while this protocol freezes NumPy 1.26.4. Phase 2 uses
PyTorch exclusively. The execution environment therefore sets `USE_TF=0` and
`USE_FLAX=0` before any Transformers import. A direct import check for
`PreTrainedModel`, `SentenceTransformer`, and `CrossEncoder` passed afterward.
No package pin, model or dataset revision, partition, seed, retrieval method,
generation setting, endpoint, or analysis rule changed. The first failed run
terminated during imports and produced no new retrieval or generation row.
