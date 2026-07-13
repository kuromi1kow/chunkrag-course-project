# Immutable Specification: ChunkRAG Main Study

Protocol version: 1.0  
Protocol date: 2026-07-13  
Status: FROZEN BEFORE PRIMARY DATA MATERIALIZATION  
Protocol identifier: chunkrag-main-v1

## 1. Governance

This document governs every primary and secondary experiment, metric, statistical test,
table, figure, and artifact in the new ACL/EMNLP submission.

Later implementation must conform to this document. A deviation is permitted only when:

1. the specified operation is technically impossible;
2. a dataset or model artifact is unavailable at its pinned revision;
3. validation exposes a factual contradiction in the protocol; or
4. continuing would corrupt or misrepresent the study.

Every deviation must be recorded before the affected outcome is inspected in an
append-only protocol amendment containing the reason, old rule, new rule, affected
experiments, date, and Git commit. Compute inconvenience, unfavorable results, weak
significance, or a desire for a cleaner story are not valid reasons for amendment.

No primary aggregate outcome may be inspected until Experiments E0 and E1 pass their
validation criteria and all confirmatory analysis code passes synthetic-data tests.

This protocol is prospective for the newly regenerated `chunkrag-main-v1` study, but it
is not claimed to be independent of all prior evidence. The archived pilot, reviewer
comments, and exploratory robustness outputs informed the design. Those prior outputs
are permanently excluded from confirmatory estimation, sample replacement, parameter
tuning, and main-study stopping decisions. The paper must call the analyses
"protocol-frozen" rather than falsely calling the whole project preregistered.

## 2. Scientific identity

The paper is a treatment-identification and measurement paper, not a chunker leaderboard.

The central question is:

> Which measured RAG chunking effects are attributable to boundary policy, and which
> are produced by unequal encoder, retrieval, and generator evidence exposure?

The archived pilot is a motivating appendix case. It is not part of the confirmatory
family and does not provide primary performance evidence.

## 3. Research questions

### RQ1: Boundary identification

Under equal generator-consumed evidence-token budgets, do recursive, sentence, or
semantic boundaries outperform deterministic policy-matched randomized boundaries?

### RQ2: Exposure distortion

Does the apparent effect of a structured chunker relative to fixed windows change when
generator-consumed evidence tokens are equalized?

### RQ3: Budget moderation and mechanism

How do input budget, retrieval evidence coverage, and consumed evidence coverage explain
the difference between operational and exposure-matched effects?

### RQ4: Domain transfer

Do the RQ1 and RQ2 patterns transfer to long technical-support answers when answer
quality is measured semantically and the evaluator is validated against blinded human
judgment?

### RQ5: Stack robustness

Do the descriptive retrieval patterns persist when the embedder and retrieval stage are
changed, and when a second 7B generator receives the frozen primary retrieval traces?

RQ1 and RQ2 are confirmatory. RQ3 has one confirmatory budget family and otherwise
mechanistic secondary analyses. RQ4 has a separately corrected confirmatory family only
if evaluator validation passes. RQ5 is secondary robustness evidence.

## 4. Hypotheses

### Primary boundary hypotheses

For each policy in recursive, sentence, and semantic, and each dataset in SQuAD and
HotpotQA:

- H1 alternative: mean answer F1 under the exact structured policy differs from mean
  answer F1 averaged across its five randomized-boundary counterparts under the
  exposure-matched 4,096-token condition.
- H1 null: the paired mean difference is zero.

There are six H1 tests.

The scientific expectation is positive, but every test is two-sided. A negative result
is publishable and will be interpreted as structured boundaries underperforming nearby
random alternatives.

### Primary exposure-distortion hypotheses

For each structured policy and each of SQuAD and HotpotQA:

- H2 estimand:

  [(structured minus fixed) under operational 1,024-token packing]
  minus
  [(structured minus fixed) under exposure-matched 1,024-token packing].

- H2 alternative: the paired difference-in-differences is nonzero.
- H2 null: the paired difference-in-differences is zero.

There are six H2 tests.

H1 and H2 form one 12-test primary family.

### Secondary budget hypotheses

For each structured policy and each of SQuAD and HotpotQA:

- H3 estimand:

  [(structured minus fixed) under operational 1,024-token packing]
  minus
  [(structured minus fixed) under operational 4,096-token packing].

- H3 alternative: the policy effect is moderated by the input budget.

The six H3 tests form one secondary family.

### Secondary corpus-heterogeneity hypotheses

For each structured policy, H4 is the SQuAD tau_boundary minus the HotpotQA
tau_boundary. The three two-sided H4 tests form one secondary Holm-corrected family.
No cross-dataset interaction claim is permitted from separate within-dataset
significance labels.

### TechQA hypotheses

If semantic-evaluator validation passes, H1 and H2 are repeated on TechQA using semantic
answer utility rather than token F1. The three H1 and three H2 TechQA tests form one
six-test family. If validation fails, these tests are not confirmatory and TechQA answer
results are reported as human-subset and exploratory judge analyses only.

## 5. Estimands

All estimands are finite-sample estimands over the frozen questions and pinned system
components. No population-wide causal claim is permitted.

### 5.1 Operational policy effect

For structured policy p, dataset d, budget b, and outcome Y:

tau_operational(p,d,b) =
mean over frozen questions of
[Y(p, operational, b) - Y(fixed, operational, b)].

This estimand includes boundary placement, realized lengths, chunk count, retrieval
competition, evidence density, and prompt truncation. It answers which complete policy
performs better in the specified stack.

### 5.2 Exposure-matched policy effect

tau_matched(p,d,b) =
mean over frozen questions of
[Y(p, matched, b) - Y(fixed, matched, b)].

This equalizes the number of generator-tokenized context tokens consumed for each
question and budget. It does not equalize semantic information, passage count, evidence
density, or ordering.

### 5.3 Policy-local boundary effect

tau_boundary(p,d) =
mean over frozen questions of
[Y(exact policy p, matched, 4096)
 - average over five jitter controls Y(jitter(p,r), matched, 4096)].

The randomized counterparts preserve the parent policy, document coverage, chunk count,
and feasible granularity while destroying the exact selected cut locations.

### 5.4 Exposure distortion

delta_exposure(p,d) =
tau_operational(p,d,1024) - tau_matched(p,d,1024).

### 5.5 Budget moderation

delta_budget(p,d) =
tau_operational(p,d,1024) - tau_operational(p,d,4096).

### 5.6 Gold-evidence gap

gap_gold(system,d,b) =
mean[Y(gold evidence,d,b) - Y(system,d,b)].

This is a diagnostic upper-bound gap, not the performance of a deployable system.

## 6. Causal and interpretive assumptions

The following assumptions are required for within-study treatment language:

1. The source documents and frozen questions are identical across policies.
2. Query text, retriever models, reranker, generator, prompts, and decoding are fixed
   within a planned contrast.
3. Boundary randomization is generated from document ID, policy, cut index, and frozen
   randomization seed without using questions, answers, retrieval scores, or outcomes.
4. Exposure matching uses only tokenizer counts and frozen retrieval order, not gold
   answers or outcomes.
5. Every policy covers the complete source document without gaps or duplicated source
   tokens in the primary non-overlapping design.
6. Question outcomes do not change another question's retrieval index or generation
   result after the corpus is frozen.
7. Infrastructure retries do not change model, prompt, seed, precision, or context.

The study does not assume that equal token counts contain equal information. It does not
identify effects outside the pinned datasets and stack. It does not interpret the
operational effect as a pure boundary effect.

## 7. Pinned external artifacts

### Datasets

| Dataset | Repository/config | Split | Revision |
|---|---|---|---|
| SQuAD 2.0 | squad_v2 | validation | 3ffb306f725f7d2ce8394bc1873b24868140c412 |
| HotpotQA | hotpot_qa / distractor | validation | 1908d6afbbead072334abe2965f91bd2709910ab |
| TechQA | nvidia/TechQA-RAG-Eval | train | 0b5bbc84b7f07d6d09d063130e90b716d8d4a32a |

### Models

| Role | Repository | Revision |
|---|---|---|
| Canonical chunk tokenizer and semantic-boundary encoder | sentence-transformers/all-MiniLM-L6-v2 | 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 |
| Primary dense embedder | BAAI/bge-small-en-v1.5 | 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a |
| Cross-encoder reranker | cross-encoder/ms-marco-MiniLM-L6-v2 | c5ee24cb16019beea0893ab7796b1df96625c6b8 |
| Primary generator | mistralai/Mistral-7B-Instruct-v0.3 | c170c708c41dac9275d15a8fff4eca08d52bab71 |
| Secondary generator and TechQA judge | Qwen/Qwen2.5-7B-Instruct | fe11104b620d588ccc049ff6631dd3ea002e3d98 |

The Qwen checkpoint may judge only Mistral TechQA outputs. It may not judge its own
generated outputs.

## 8. Frozen question selection

Question IDs are frozen mathematically by this section. Literal JSONL manifests are
derived artifacts of Experiment E0, not discretionary selections.

### 8.1 Canonical ordering

For each eligible row, compute:

SHA256("chunkrag-main-v1" + NUL + dataset_name + NUL + example_id)

using UTF-8 bytes. Sort ascending by the hexadecimal digest, breaking an impossible hash
tie by ascending example ID.

The exact dataset_name strings are `squad_v2`, `hotpot_qa`, and `techqa`.

### 8.2 SQuAD eligibility and sample

- Include only rows with at least one non-empty reference answer.
- Select 500 questions in canonical hash order.
- Permit at most 20 selected questions per NFC-normalized title. This cap permits the
  500-question target on the finite validation split while keeping every title below
  4% of the sample.
- Continue through the ordered eligible rows until 500 are selected.
- The cluster identifier is the exact title string.

### 8.3 HotpotQA eligibility and sample

- Include rows with a non-empty answer, at least one supporting fact, and all supporting
  titles present in that row's context.
- Select 500 questions in canonical hash order.
- Define the allocation key as the lexicographically first NFC-normalized supporting
  title.
- Permit at most two selected questions per allocation key.
- The inferential cluster is the connected component of questions sharing any gold
  supporting document ID.

### 8.4 TechQA eligibility and sample

- Exclude is_impossible rows.
- Require a non-empty answer and at least one non-empty context.
- Select 300 questions in canonical hash order.
- Define the allocation key as the lexicographically first NFC-normalized context
  filename.
- Permit at most two selected questions per allocation key.
- The inferential cluster is the connected component of questions sharing any gold
  context filename.

If the pinned data cannot satisfy a requested sample and cap, Experiment E0 must stop.
No cap relaxation or substitute question is permitted without a protocol amendment.

### 8.5 Sample-size sensitivity analysis

The frozen sizes are precision/compute choices, not a claim of guaranteed power. After
E0 fixes cluster sizes but before E1, run an outcome-free sensitivity simulation with
100,000 replicates for contrast standard deviations 15, 20, and 25 F1 points, intraclass
correlations 0.0, 0.1, and 0.2, and true mean effects from 0 to 10 in 0.25-point steps.
Generate Gaussian cluster and residual terms scaled to the requested variance, retain
the actual E0 cluster sizes, and test the mean with the Section 20 cluster-robust
intercept model at conservative two-sided alpha 0.05/12. Use test ID
`design-sensitivity:{dataset}:{sd}:{icc}` and procedure name `power-simulation` under the
master-seed rule in Section 20. Report the smallest grid effect attaining at least 80%
power for each scenario. This simulation cannot change sample size after outcomes exist.

## 9. Corpus construction

Corpora are constructed from the full pinned split before question selection.

### 9.1 SQuAD corpus

Each distinct pair of exact title and exact context paragraph is one document. Paragraphs
are not concatenated by title.

Document ID:

squad::SHA256(title + NUL + context)

The title and context in this formula are the normalized values from Section 9.4.

The relevant document for a question is the document formed from that row's exact title
and context.

### 9.2 HotpotQA corpus

Build the union of every context document in the complete pinned distractor-validation
split. A document is the normalized title and the sentence list from one row, with each
normalized sentence stripped of surrounding Unicode whitespace and joined by one ASCII
space. Original sentence indices and their resulting character spans are retained.

Document ID:

hotpot::SHA256(title + NUL + joined_sentences)

Both formula fields are the normalized values defined above.

Exact duplicate title/text pairs are deduplicated. If the same title occurs with
different text, both documents remain distinct. Gold documents for a question are the
exact row-context documents whose titles occur in supporting facts.

### 9.3 TechQA corpus

Build the union of every non-empty context in every eligible answerable row before
question sampling.

Document ID:

techqa::filename

The filename is Unicode-NFC normalized without case folding or whitespace stripping.

Two contexts with the same filename must have byte-identical normalized text. A conflict
stops E0. Gold documents are the filenames supplied with that question.

### 9.4 Text normalization

Corpus text preserves case, punctuation, and paragraph boundaries. Only CRLF is
normalized to LF and Unicode is normalized to NFC. No corpus-level trimming,
lowercasing, HTML stripping, sentence reordering, or answer-dependent cleaning is
permitted, except the explicitly frozen HotpotQA per-sentence join in Section 9.2.

Question and reference strings are CRLF-to-LF and NFC normalized, then stripped only of
surrounding Unicode whitespace. Dataset example IDs are never normalized. For SQuAD,
each normalized answer start is recomputed as the character length of the normalized
original-context prefix ending at the archived start; the normalized answer end is that
start plus the character length of the normalized archived answer substring. E0 must
verify that every resulting span exactly equals its normalized reference text.

## 10. Corpus and question manifests

E0 must write:

- dataset_manifest.json;
- questions/{dataset}.jsonl;
- corpora/{dataset}.jsonl;
- clusters/{dataset}.jsonl.

The dataset manifest records repository, configuration, split, revision, dataset
fingerprint, row count before and after eligibility filtering, every local cache-file
SHA-256, license string from the pinned card metadata, and the final question/corpus
manifest hashes.

Each question record contains:

- dataset and pinned revision;
- example ID;
- selection hash and rank;
- question;
- all references;
- gold document IDs;
- gold answer spans or supporting facts when available;
- cluster ID;
- eligibility fields.

Each corpus record contains:

- document ID;
- dataset;
- title or filename;
- normalized text;
- source split and revision;
- SHA-256 of normalized text;
- source-row provenance.

Records are sorted by ID. No experiment reads the Hugging Face dataset after E0; all
later stages read the frozen manifests.

## 11. Primary chunker definitions

All primary policies partition the canonical MiniLM tokenization of each document into
contiguous, non-overlapping source spans. Every source token appears exactly once.
Tokenization uses `add_special_tokens=false`, `truncation=false`, and returns source
offset mappings. Chunk character spans include every intervening whitespace character,
so concatenating chunks in order reconstructs the normalized source exactly.

Nominal target: 192 canonical content tokens.  
Minimum non-final chunk: 64 tokens.  
Maximum chunk: 254 tokens.  
Overlap: zero.  
Special tokens are excluded from these counts.

Documents shorter than 64 tokens form one chunk.

### 11.1 Fixed-192

Cut at token positions 192, 384, 576, and so on. If the final chunk has fewer than 64
tokens, merge it into the preceding chunk only when the merged length is at most 254;
otherwise retain the short final chunk and mark final_short=true.

### 11.2 Recursive-192

Starting at the current source position, inspect the next 192 tokens. Choose the latest
separator boundary not before token 128 in this priority order:

1. double newline;
2. newline;
3. sentence-final punctuation followed by whitespace;
4. whitespace.

Priority is strict: choose the latest boundary in the first separator class that has an
eligible boundary; consult the next class only if the higher class has none. A double
newline boundary is after the complete run of two or more `\n` characters; a newline
boundary is after the `\n`. A sentence-final boundary consists of `.`, `!`, or `?`,
followed by zero or more characters from the exact set double quote, apostrophe, right
parenthesis, and right square bracket, then one or more Unicode whitespace characters;
the boundary falls after the complete whitespace run. A whitespace boundary likewise
falls after the complete Unicode whitespace run.

If no eligible separator exists, cut at 192. A cut may extend beyond 192 only to merge a
final segment shorter than 64, with a hard maximum of 254. If that merge would exceed
254, retain the short final chunk and mark it `final_short=true`.

### 11.3 Sentence-192

Use spaCy blank English with the sentencizer component, punctuation characters exactly
period, exclamation mark, and question mark, and no statistical model. Greedily
append complete sentences while total canonical tokens remain at most 192. If the
current chunk is shorter than 96 and adding the next sentence remains at most 254, add
it. A single sentence longer than 254 is split at 192-token boundaries. Final segments
follow the same under-64 merge rule.

### 11.4 Semantic-192

Use the pinned MiniLM encoder, independent of the BGE retrieval encoder. Encode each
sentence obtained from the exact sentencizer in Section 11.3 with normalized vectors.
Before encoding, any sentence longer than 254
canonical tokens is deterministically divided into consecutive 192-token pseudo-sentences;
the last pseudo-sentence retains its remainder. These pseudo-sentences are treated as
ordinary sentences for boundary selection. Beginning at the current position:

- consider sentence boundaries producing a chunk between 128 and 254 tokens;
- choose the boundary with the minimum adjacent-sentence cosine similarity;
- break ties by choosing the boundary closest to 192;
- then choose the earlier boundary;
- if no candidate exists, cut at the latest source token not exceeding 192;
- apply the final under-64 merge rule.

The document-end boundary is not assigned a cosine score because it has no following
sentence. When the remaining source fits within 254 tokens, it becomes the final chunk.

No similarity threshold is tuned or used. This prevents a test-set-tuned semantic
threshold.

### 11.5 Required chunk metadata

Every chunk contains:

- policy and policy version;
- dataset and document ID;
- source character start/end;
- canonical token start/end;
- canonical token count;
- text SHA-256;
- preceding and following separator type;
- parent chunk ordinal;
- final_short flag;
- canonical tokenizer repository and revision.

Round-trip validation must prove exact ordered source coverage.

## 12. Randomized-boundary controls

Five controls are generated for each of fixed, recursive, sentence, and semantic
policies. Primary generation uses the recursive, sentence, and semantic controls. Fixed
controls are retrieval-only exploratory controls.

Frozen control seeds:

1103, 2207, 3301, 4409, 5519.

For a base policy with m chunks over N canonical tokens, let b1 through b(m-1) be its
internal cut points. For control seed r and cut j:

1. Compute SHA256("chunkrag-jitter-v1" + NUL + r + NUL + policy + NUL + document_id
   + NUL + j).
   Here r and the one-based j are encoded as unsigned base-10 ASCII integers.
2. Convert the first eight digest bytes to an unsigned big-endian integer.
3. Map modulo 97 to an integer delta in [-48, 48].
4. If delta is zero, replace it with +1 for even j and -1 for odd j.
5. Process cuts from left to right.
   Set previous_new_cut=0 before processing the first cut. At cut j, define
   remaining_segments=m-j, the number of segments that must follow the new cut.
6. For the next cut, define:

   lower = max(previous_new_cut + 64, N - 254 * remaining_segments)

   upper = min(previous_new_cut + 254, N - 64 * remaining_segments)

7. Set the new cut to clamp(original_cut + delta, lower, upper).

Documents whose base segmentation contains an allowed final_short segment use a relaxed
final minimum of one token for that last segment only. The control retains the same
number of chunks, exact full-document coverage, no overlap, and the 254-token maximum.

Validation requires:

- every control covers the source exactly once;
- every control has the same chunk count as its parent policy;
- every non-final-short length is in [64,254];
- at least 80% of feasible internal boundaries change corpus-wide for each
  policy/seed/dataset;
- control generation is byte-identical on two independent invocations.

Failure of any requirement stops E1.

A boundary is feasible-to-change if, at the moment it is processed, its inclusive
integer interval [lower,upper] contains at least one position unequal to the original
cut. The 80% denominator is exactly the count of such boundaries and the numerator is
the count whose final clamped position differs from the original.

## 13. Retrieval stack

### 13.1 Primary dense retrieval

- Model: pinned BGE-small-en-v1.5.
- Encoder inference dtype float32 and batch size 64.
- Normalize document and query vectors.
- Query prefix: "Represent this sentence for searching relevant passages: "
- Use the model's native maximum length of 512 tokens. No selected question or chunk
  may be truncated by this encoder; E1 records native token counts and stops if the
  requirement is violated.
- Exact CPU `IndexFlatIP` FAISS inner-product index.
- Retrieve top 50 dense candidates.

### 13.2 Sparse retrieval

- BM25Okapi.
- Tokenization: apply Python Unicode `str.lower()`, then extract tokens with
  `re.findall(r"(?u)\b\w+\b", text)`; no stemming or stop-word removal.
- Retrieve top 50 sparse candidates.

### 13.3 Hybrid fusion

- Weighted reciprocal-rank fusion.
- Dense weight 0.6.
- BM25 weight 0.4.
- RRF constant 60.
- Union dense and sparse candidates.
- Sort by fused score descending, then chunk ID ascending.
- Retain top 50.

### 13.4 Reranking

- Pinned cross-encoder/ms-marco-MiniLM-L6-v2.
- Score all top-50 hybrid candidates.
- Inference dtype float32 and batch size 32.
- Tokenize each (question, passage) pair to at most 512 reranker tokens, preserving the
  full question and applying `only_second` truncation to the passage. Persist original
  and retained token counts. A question that alone cannot fit after special tokens stops
  E1; it is not silently truncated or dropped.
- Sort reranker score descending, then fused score descending, then chunk ID ascending.
- Persist all candidate, component, fused, and reranker scores.
- Freeze the top 16 reranked chunks as the retrieval trace.

### 13.5 Retrieval depths

Retrieval metrics are reported at k=4 and k=8 from the same frozen top-16 trace.

Operational generation uses top 4.

Exposure-matched generation may consume the ranked prefix of top 16 until its matched
context-token allocation is exhausted.

No retrieval is rerun for a different generator.

### 13.6 Secondary retrieval matrix

E5 uses exactly two dense encoders: pinned BGE with the query prefix in Section 13.1,
and pinned all-MiniLM-L6-v2 with no query prefix. Both use normalized embeddings and an
exact inner-product index. For each encoder, the three reported stacks are:

1. dense top 50;
2. dense plus the identical BM25 index under the frozen weighted RRF rule;
3. that hybrid top 50 plus the identical pinned cross-encoder reranker.

Each stack freezes top 16 and uses the same score ordering and tie rules as the primary
stack. E5 never tunes fusion weights, depth, or reranker separately by dataset.

## 14. Generators and decoding

### 14.1 Primary generator

Mistral-7B-Instruct-v0.3 at the pinned revision.

- float16;
- generation batch size 1;
- greedy decoding;
- do_sample=false;
- num_beams=1;
- temperature omitted from the local generate call;
- repetition_penalty=1.0, no_repeat_ngram_size=0, length_penalty=1.0, and use_cache=true;
- native eos_token_id and pad_token_id, with pad_token_id set to eos_token_id only if
  the pinned tokenizer declares no pad token;
- native tokenizer and chat template;
- no refinement call;
- no question-conditioned compression;
- native EOS is the only early stopping token; generation otherwise stops at the frozen
  maximum-new-token limit, with no custom stop strings;
- output normalization removes only surrounding whitespace and, at most once, a
  case-insensitive leading label matching
  `^\s*(Answer|Final answer)\s*:\s*`; no other text is altered.

Maximum new tokens:

- SQuAD: 64;
- HotpotQA: 64;
- TechQA: 384.

### 14.2 Secondary generator

Qwen2.5-7B-Instruct at the pinned revision.

It runs only on SQuAD and HotpotQA for the four deterministic policies under matched
4,096-token packing and the 4,096-token gold-evidence condition. Its purpose is
checkpoint robustness, not an additional primary family.

Decoding and output limits match Mistral.

## 15. Frozen prompt versions

### 15.1 Extractive prompt: extractive-v1

System message:

You are an extractive question answering assistant. Use only the provided context.
Copy the shortest answer span supported by the context. Do not explain your reasoning.
If the answer is not fully supported, reply with exactly unanswerable.

User message:

Answer the following question using only the context.

Question: {question}

Context passages:
{context}

Return only the answer text with no explanation.

This prompt is used for SQuAD and HotpotQA, including gold evidence.

### 15.2 Technical prompt: technical-v1

System message:

You are a grounded technical question answering assistant. Use only the provided
context. Give a concise but complete answer containing the information needed to resolve
the question. Do not add citations or unsupported details. If the answer is not
supported, reply with exactly unanswerable.

User message:

Answer the following technical question using only the context.

Question: {question}

Context passages:
{context}

Return only the final answer.

This prompt is used for TechQA, including gold evidence.

No prompt alternative belongs to the confirmatory study.

## 16. Input budgets and context packing

Complete-chat input budgets are 1,024 and 4,096 native generator tokens. Output tokens
are not included in these input limits. Token counts use the native chat template with
`tokenize=true` and `add_generation_prompt=true`; the stored prompt token IDs are the
exact IDs passed to `generate`.

Passage format:

[rank] Title: {title}
Passage: {chunk_text}

SQuAD and TechQA use the same format; title falls back to document ID. Passage order is
retrieval rank.

### 16.1 Operational packing

Take top four reranked chunks. Render all four in rank order. If the chat exceeds the
input budget, retain the longest prefix of the rendered context whose complete chat fits.
The prefix may cut inside the final chunk. No later chunk may replace an earlier chunk.

### 16.2 Exposure-matched packing

Nominal context allocations are:

- 768 context tokens for the 1,024-token input budget;
- 3,072 context tokens for the 4,096-token input budget.

For question q and budget b:

1. Compute the maximum context-token count that fits after the frozen system message,
   question, user boilerplate, chat template, and a 16-token safety margin.
2. Cap it at the nominal context allocation.
3. For every primary system, render the top-16 context without truncation and count its
   native generator tokens.
4. Define M(q,b) as the minimum of the cap and the available rendered-context tokens
   across all 19 generation systems: fixed, three exact structured policies, and fifteen
   structured jitter controls.
5. For every system, retain the longest rendered-context prefix whose native tokenizer
   round trip is at most M(q,b).

Validation requires the consumed context-token count for every system to differ from
M(q,b) by at most two tokens. No system-specific padding is used. M(q,b) is determined
without answers or outcomes.

## 17. Gold-evidence conditions

Gold evidence uses the same prompt and budgets as retrieved evidence.

For Mistral gold evidence, the target context count is the already frozen M(q,b) from
Section 16.2. Pack the ordered gold evidence to the longest native-token prefix at most
M(q,b). If all available gold evidence is shorter than M(q,b), consume all of it and
record the shortfall; never add non-gold filler. Thus gold gaps compare equal context
budgets whenever the annotated evidence is long enough and transparently report when it
is not. Gold units use the passage format in Section 16, receive ranks in their frozen
order, and may be cut only inside the final unit when the target falls inside it.

### SQuAD

Use the exact annotated source paragraph. Center the context window on the annotated
answer span when the paragraph exceeds capacity. If several annotated spans exist,
choose the smallest character start, breaking ties by the longest span and then the
lexicographically smallest answer text. Retain as much symmetric surrounding text as
possible in native generator-token space and break a window-position tie toward earlier
source tokens. The title/header tokens count against M(q,b).

### HotpotQA

Order annotated supporting sentences by their source document's position in the row and
then by sentence index, deduplicating repeated annotations. Then enumerate unused
adjacent sentences by distance 1, 2, and so on; within each distance visit supporting
sentences in the frozen order and visit previous before next, deduplicating candidates.
Render each selected sentence as its own titled passage in this priority order until the
gold target is filled.

### TechQA

This is explicitly answer-informed oracle evidence. Split every gold context document
into fixed-192 spans. Rank spans by normalized token F1 against the reference answer,
using the Section 19 SQuAD/Hotpot token normalization solely for this oracle ranking,
then by filename and source position. Pack in that order. This condition is an upper
bound and may not be described as deployable retrieval.

Gold evidence is generated at both 1,024 and 4,096 input budgets.

## 18. Endpoints

### 18.1 Primary answer endpoint

For SQuAD and HotpotQA: standard normalized token F1, maximum across references, reported
in percentage points.

Exact Match is secondary.

### 18.2 Primary TechQA endpoint

Semantic answer utility:

(correctness score + completeness score) / 4,

where each dimension is 0, 1, or 2 under the frozen evaluator rubric. Groundedness is
reported separately. This endpoint becomes confirmatory only after evaluator validation.

### 18.3 Evidence endpoints

SQuAD:

- answer span fully present in retrieved chunks at k=4 and k=8;
- answer span fully present in consumed context;
- gold document hit.

HotpotQA:

- supporting-document coverage at k=4 and k=8;
- all supporting documents hit;
- fraction of annotated supporting sentences fully represented in consumed context;
- all supporting sentences consumed.

TechQA:

- gold-document coverage at k=4 and k=8;
- all gold documents hit;
- normalized answer sequence visibility, labeled conservative and descriptive;
- gold-document coverage in consumed context.

Evidence coverage is computed from the union of source character intervals, never from
string search, whenever gold spans exist. A SQuAD answer or HotpotQA supporting sentence
is fully represented when every character in its normalized gold interval is covered by
the retrieved or consumed source intervals; coverage may be supplied by two adjacent
chunks. A gold document is hit when at least one ranked chunk from that document is
present, and consumed-document coverage requires at least one source character from the
document to survive packing. TechQA answer-sequence visibility is true only when the
complete normalized reference-token sequence appears contiguously inside one consumed
source span; it is explicitly not treated as correctness.

### 18.4 Ranking endpoints

- Convert every ranked chunk list to a document ranking by retaining only the first
  occurrence of each document ID. All document-level metrics below use this deduplicated
  order; repeated chunks from one document never earn repeated relevance credit.
- MRR of the first gold document;
- nDCG@8 with binary gold-document relevance;
- DocCov@4 and DocCov@8;
- AllHit@4 and AllHit@8.

DocCov is the fraction of that question's distinct gold document IDs present at depth
k. AllHit is its binary indicator that DocCov equals one. Dataset summaries are
unweighted means over frozen questions.

### 18.5 Operational endpoints

- chunk count;
- mean, median, p10, p90, and maximum canonical chunk tokens;
- index-build seconds;
- embedding tokens;
- index bytes;
- retrieval seconds per question;
- reranker seconds per question;
- prompt input tokens;
- generated tokens;
- generation seconds;
- peak allocated GPU memory.

Timing is secondary and is measured only on the canonical SCC A100 environment after
five warm-up questions.

## 19. Answer normalization

SQuAD/Hotpot evaluation normalization:

- lowercase;
- Unicode NFC;
- remove punctuation;
- remove English articles a, an, the;
- collapse whitespace.

No heuristic quantity, person, date, or relation extraction is permitted.

TechQA generated text preserves multiple lines. Only surrounding whitespace and the
optional answer label are removed before semantic evaluation.

## 20. Statistical protocol

Test IDs are `{family}:{dataset}:{policy}`; H4 uses
`H4:squad_v2-minus-hotpot_qa:{policy}`. Dataset order is `squad_v2`, `hotpot_qa`,
`techqa`; policy order is `recursive192`, `sentence192`, `semantic192`. Each stochastic
inference procedure derives its unsigned 64-bit seed from the first eight bytes of
SHA256(`8677` + NUL + test ID + NUL + procedure name) and uses NumPy `PCG64`. Thus every
test is invariant to execution order while retaining 8677 as the master seed.

### 20.1 Significance

- All confirmatory tests are two-sided.
- Familywise alpha is 0.05.
- Report raw and adjusted p-values.
- A claim of statistical significance requires adjusted p < 0.05.
- Unadjusted significance may not appear in the abstract, conclusion, or contribution
  list.

### 20.2 Multiplicity

- Primary family: 12 H1/H2 tests; Holm correction once across all 12.
- Budget family: six H3 tests; Holm correction once across all six.
- TechQA family: six tests; Holm correction once across all six, only after evaluator
  validation.
- Corpus-heterogeneity family: three H4 tests; Holm correction once across all three.
- Equivalence family: six H1 boundary contrasts; Holm correction across the six TOST
  p-values.
- Secondary robustness tables report intervals and effect sizes without null-hypothesis
  claims.

### 20.3 Effect sizes

Every planned contrast reports:

- paired mean difference in endpoint units;
- 95% cluster-bootstrap interval;
- paired rank-biserial correlation;
- number of questions and clusters;
- adjusted and raw p-value for confirmatory tests.

Paired rank-biserial correlation is `(n_positive - n_negative) / n_nonzero` over
question-level paired contrasts; exact zeros are excluded, and the value is defined as
zero if all contrasts are zero.

H4 is the sole exception to the paired rank-biserial requirement because it compares
independent dataset samples. H4 reports the difference in mean boundary effects and
Cliff's delta between the SQuAD and HotpotQA question-level boundary contrasts.

### 20.4 Cluster definitions

- SQuAD: title.
- HotpotQA: connected component induced by shared gold document IDs.
- TechQA: connected component induced by shared gold context filenames.

Clusters are computed from frozen manifests before outcomes.

Validation requires at least 30 clusters per dataset and no single cluster containing
more than 10% of selected questions. Failure stops E0 and requires a protocol amendment
before any model run.

### 20.5 Cluster bootstrap

- 20,000 draws;
- procedure name `cluster-bootstrap` under the master-seed rule above;
- sample clusters with replacement;
- include all questions from sampled clusters;
- compute the unweighted mean over sampled question records;
- for randomized controls, average the five random-control outcomes within question
  before computing the paired difference;
- use percentile 2.5% and 97.5% limits.

### 20.6 Randomization test

- Compute question-level paired contrasts.
- Sum question contributions within each frozen cluster, apply one sign to the whole
  cluster sum, and divide the signed total by the fixed number of questions. This
  preserves the question-mean estimand while respecting cluster dependence.
- Apply independent Rademacher sign flips to cluster contributions.
- Use every sign pattern when there are at most 20 clusters; otherwise use 99,999 Monte
  Carlo draws with procedure name `cluster-sign-flip`, plus one for the observed
  statistic.
- The p-value is the proportion of absolute randomized means at least as large as the
  observed absolute mean.

The sign-flip tests assume cluster-level contrast exchangeability and symmetry under
the sharp zero-effect null. This assumption and the observed cluster-contrast
distributions are reported; the bootstrap intervals remain the primary uncertainty
display if symmetry is visibly poor.

For H4, concatenate the SQuAD and HotpotQA question-level H1 contrasts for one policy
and fit `contrast ~ 1 + I(dataset=SQuAD)`. Prefix cluster IDs by dataset, use a CR1
cluster-robust standard error, and test the dataset coefficient two-sided with a t
distribution having `G_total-2` degrees of freedom. Independently cluster-bootstrap
each dataset in each of 20,000 paired draws, subtract the two bootstrapped means, and
report the percentile 95% interval. Holm-correct the three regression p-values.

### 20.7 Practical equivalence

The smallest effect size of interest for SQuAD/HotpotQA F1 is two percentage points.

For the six H1 contrasts, perform two one-sided tests against [-2,+2] F1 points using
cluster-robust intercept-only regression on the question-level contrasts. Use a CR1
sandwich standard error, a t distribution with G-1 degrees of freedom, where G is the
number of frozen clusters, and the two statistics `(mean+2)/SE` and `(mean-2)/SE`.
The raw TOST p-value is the maximum of the lower-bound and upper-bound one-sided
p-values. Holm-correct these six maximum p-values. Also report 90% cluster-bootstrap
intervals. Practical equivalence may be claimed only when the adjusted TOST p-value is
below 0.05. Failure to reject a zero-effect null is not equivalence.

For TechQA semantic utility, the exploratory equivalence region is [-0.05,+0.05].

### 20.8 Missing and failed outputs

- Empty, unparsable, or permanently failed generation receives answer score zero.
- Infrastructure failure may be retried twice with identical inputs.
- A successful retry replaces no prior valid result and records every attempt.
- If more than 1% of a condition remains failed, the whole shard is invalid and must be
  rerun.
- No condition or question is removed because of poor performance.

There is no early stopping based on outcomes or significance.

## 21. TechQA semantic evaluator

### 21.1 Judge

Pinned Qwen2.5-7B-Instruct, float16, native chat template, maximum 256 new tokens, and
batch size 1. Its decoding configuration is exactly Section 14.1's deterministic
configuration except for that output limit. The complete judge input is limited to
8,192 native Qwen tokens. All frozen
questions, references, contexts, and candidates must fit without truncation; otherwise
E4 stops and the judge-input budget must be changed only by protocol amendment.

### 21.2 Judge prompt: techqa-judge-v1

System message:

You are evaluating a technical question-answering system. Judge only the candidate
answer using the question, reference answer, and consumed context. Do not reward wording
similarity by itself. Return valid JSON only.

User message:

Question:
{question}

Reference answer:
{reference}

Consumed context:
{context}

Candidate answer:
{candidate}

Assign integer scores:

- correctness: 0 incorrect, 1 partly correct, 2 fully correct;
- completeness: 0 misses the resolution, 1 contains part of the needed resolution,
  2 contains the information needed to resolve the question;
- groundedness: 0 contains a major unsupported claim, 1 has a minor unsupported or
  unverifiable detail, 2 is fully supported by the consumed context.

Return exactly:
{"correctness": 0, "completeness": 0, "groundedness": 0, "reason": "brief reason"}

The system identity and chunker name are never shown to the judge. The JSON is schema
validated. One identical retry is allowed for invalid JSON; a second failure receives
zero scores and is recorded. If more than 1% of records in any condition fail both
parses, evaluator validation automatically fails and every judge-based TechQA answer
analysis becomes exploratory.

## 22. Human annotation protocol

### 22.1 Sample

Select the first 60 frozen TechQA questions under:

SHA256("chunkrag-human-v1" + NUL + example_id).

For each question annotate six Mistral outputs under matched 4,096-token packing:

- fixed;
- recursive;
- sentence;
- semantic;
- semantic jitter seed 1103;
- gold evidence.

Total: 360 output records.

### 22.2 Blinding

- Remove model, policy, seed, score, and condition names.
- Randomize the six candidates per question using the hash of
  "chunkrag-human-order-v1" + NUL + question ID + NUL + candidate artifact hash,
  sorting ascending by digest.
- For correctness and completeness, annotators see the question, reference answer, and
  candidate answer, but not the consumed context.
- Groundedness is annotated on the first 10 of the 60 human questions under the same
  selection hash. For these 60 records only, annotators additionally see the consumed
  context. This confines the context-intensive task while preserving all six conditions.
- Annotators do not see automatic F1 or judge scores.

### 22.3 Annotators

Two English-fluent annotators independently label every record. At least one annotator
must not have implemented the system. Both complete the same 20-record training set,
which is excluded from analysis, using a written rubric and adjudicated examples. The
training records are the first 20 candidate records, in artifact-hash order, from frozen
TechQA questions ranked 61 onward by the human-selection hash.

Each annotator is budgeted 17 hours and is compensated at USD 25 per hour or the higher
applicable institutional minimum, reported exactly in the paper. Annotators receive a
plain-language consent sheet explaining the public technical-support content, stored
labels, compensation, and right to stop. Names and contact details are never stored in
the released artifact.

### 22.4 Labels

Each dimension is ordinal 0, 1, or 2 using the exact judge definitions:

- correctness;
- completeness;
- groundedness, on the frozen 60-record groundedness subset only.

Annotators may flag cannot_assess with a required reason.

### 22.5 Agreement and adjudication

Report:

- quadratic weighted kappa per dimension;
- ordinal Krippendorff alpha per dimension;
- exact agreement.

Every disagreement and every cannot_assess record is adjudicated by the two annotators
together without seeing system identity. Adjudication must produce an integer consensus
score in {0,1,2}; if consensus is impossible, the record remains cannot_assess and is
handled by the frozen threshold below. No numerical averaging replaces adjudication.

### 22.6 Judge acceptance criteria

Against adjudicated human labels, the judge must achieve:

- on all 360 records, Spearman correlation at least 0.60 for correctness and at least
  0.60 for completeness;
- on all 360 records, quadratic weighted kappa at least 0.50 for correctness and at
  least 0.50 for completeness;
- on the frozen 60-record groundedness subset, Spearman correlation at least 0.50 and
  quadratic weighted kappa at least 0.50 for groundedness;
- no absolute mean bias greater than 0.25 on the 0–2 scale for correctness or
  completeness over 360 records or for groundedness over its 60 records.

Spearman correlation uses average ranks for ties. Any undefined correlation or kappa,
including a constant-score case, fails validation.

If any criterion fails, TechQA judge results are exploratory. Confirmatory TechQA tests
are canceled by protocol, not replaced. Human-subset results remain reportable.

If both annotators mark more than 10% of the 360 answer records cannot_assess, or more
than 10% of the 60 groundedness records cannot_assess, TechQA answer evaluation is
removed from the main paper; retrieval and evidence results remain.

## 23. Artifact schema

All records include schema_version="chunkrag-main-v1".

Canonical artifact root:

`artifacts/chunkrag-main-v1/{manifests,chunks,retrieval,generation,evaluation,analysis,audit}`

Condition IDs use exactly `fixed192`, `recursive192`, `sentence192`, `semantic192`, and
`{policy}-jitter-{seed}`. Packing IDs use exactly `operational-1024`,
`operational-4096`, `matched-1024`, `matched-4096`, `gold-1024`, and `gold-4096`.

Primary record identifiers are:

- chunk ID: SHA256(dataset + NUL + condition ID + NUL + document ID + NUL + canonical
  start + NUL + canonical end + NUL + text hash);
- retrieval ID: SHA256(protocol hash + NUL + question ID + NUL + condition ID + NUL +
  retrieval-config hash);
- generation ID: SHA256(retrieval-or-gold hash + NUL + question ID + NUL + model
  snapshot hash + NUL + packing ID + NUL + prompt-version hash);
- evaluation ID: SHA256(generation ID + NUL + evaluator-config hash).

Integers in identifier formulas use unsigned base-10 ASCII. All other fields use their
canonical UTF-8 representation. A hash collision stops the stage.

### 23.1 Run manifest

- protocol identifier and hash;
- clean Git commit;
- dirty-worktree flag, required false;
- source-tree hash;
- configuration hash;
- environment-lock hash;
- dataset, question, corpus, chunk, retrieval, generation, and evaluation hashes;
- model repositories, revisions, and local snapshot hashes;
- hardware and CUDA details;
- start/end UTC;
- planned and completed record counts;
- shard IDs;
- status and failure summary.

### 23.2 Chunk record

Contains all metadata in Section 11.5 plus control seed, parent-policy chunk count, and
boundary-generation hash.

### 23.3 Retrieval trace

One record per question, policy, and retrieval stack:

- question and corpus hashes;
- all dense top-50 chunk IDs and scores;
- all BM25 top-50 chunk IDs and scores;
- fused candidate IDs, component ranks, and fused scores;
- reranked top-50 IDs and scores;
- frozen top-16 IDs;
- latency and memory;
- retriever/model/config hashes.

### 23.4 Generation trace

- question ID;
- retrieval-trace hash or gold-evidence hash;
- policy and control seed;
- packing condition and budget;
- ranked source spans;
- full rendered context;
- consumed context;
- per-chunk consumed token counts;
- prompt version and exact messages;
- full and used input tokens;
- context target M(q,b);
- truncation location;
- model/revision/dtype/hardware;
- raw output;
- normalized output;
- generated tokens;
- stopping reason;
- latency;
- attempt history;
- record hash.

### 23.5 Evaluation trace

- generation-record hash;
- references and gold evidence IDs;
- every metric component;
- judge prompt/version/model/revision/raw JSON;
- parsed semantic scores;
- human-label linkage where applicable;
- evaluator code hash.

### 23.6 Human annotation records

The blinded package contains annotation-record ID, question, reference, candidate,
groundedness-subset flag, and consumed context only when that flag is true. The private
label file contains annotation-record ID, annotator code A or B, each applicable ordinal
label, cannot_assess reason, UTC completion time, and rubric version. The adjudication
file contains both source-label hashes, consensus values, adjudication status, and no
personal identifier. Only these deidentified records enter the released artifact.

## 24. Hashing strategy

All hashes use SHA-256.

An external snapshot hash is computed over dereferenced file content in lexicographic
relative-path order as repeated bytes `path + NUL + SHA256(file_bytes) + LF`. This rule
is used for every model snapshot and dataset cache snapshot, so Hugging Face symlink
locations and machine-specific cache roots cannot alter the hash.

Canonical JSON:

- UTF-8;
- Unicode NFC;
- keys sorted lexicographically;
- separators comma and colon without added spaces;
- no NaN or Infinity;
- newline terminated.

Canonical JSONL:

- records sorted by their primary ID;
- each line is one canonical JSON record;
- file hash is over the exact UTF-8 bytes.

Hash chain:

protocol -> source/config/environment -> dataset -> questions/corpus -> chunks ->
retrieval -> generation -> evaluation -> tables/figures.

Every downstream record contains the immediate upstream artifact hash. Merge operations
reject missing, duplicated, or conflicting IDs.

The source hash is computed over the clean Git commit's tracked experiment source, in
lexicographic path order, as repeated bytes
`path + NUL + SHA256(file_bytes) + LF`. The included paths are `src/**`, `scripts/**`,
`configs/**`, `tests/**`, `pyproject.toml`, and the resolved environment lock. The run
manifest records both the Git commit and this SHA-256 source hash. Untracked or ignored
files can never enter a canonical artifact except the declared model/dataset caches and
the hashed output root.

## 25. Reproducibility guarantees

Primary runs require:

- clean committed worktree;
- pinned dataset and model revisions;
- exact environment lock;
- deterministic seeds;
- no network-dependent inference after artifacts are cached;
- complete raw traces;
- shard-order-independent merge;
- one command to validate the full hash chain;
- one command to regenerate every analysis table and figure without model inference.

The global implementation seed is 8677 for Python, NumPy, Torch CPU, and every CUDA
device except where a frozen control/bootstrap seed is explicitly specified. Set
`torch.use_deterministic_algorithms(True)`,
`torch.backends.cuda.matmul.allow_tf32=False`,
`torch.backends.cudnn.allow_tf32=False`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and run all models in evaluation mode.
Dense and semantic embedding use float32 with batch size 64; cross-encoder inference
uses float32 with batch size 32; every generator and judge uses float16 with batch size
1. A backend that cannot honor these settings cannot produce canonical outputs.

A reproducibility audit reruns the first 25 questions by canonical hash in each dataset
after the canonical run:

- top-16 retrieved chunk IDs must match exactly;
- reranker scores must agree within 1e-5;
- prompt token IDs must match exactly;
- normalized greedy outputs must match exactly on the same A100 class and environment;
- all recomputed aggregate metrics must match stored values within 1e-12.

Any failure blocks paper-table generation until explained and resolved.

## 26. Experiments

### E0: Frozen data and corpus materialization

Objective:

Create immutable questions, corpora, clusters, gold evidence, and hashes.

Exact inputs:

- pinned datasets and revisions;
- eligibility, selection, corpus, normalization, and clustering rules in Sections 8–10.

Exact outputs:

- dataset, question, corpus, cluster, and gold-evidence manifests;
- outcome-free sample-size sensitivity report;
- validation report;
- hash manifest.

Stopping criteria:

- all 500/500/300 questions are selected;
- every selected question maps to at least one gold document;
- cluster requirements pass;
- no TechQA filename conflict exists.

Validation:

- independent second materialization is byte-identical;
- every document and question hash recomputes;
- no selected ID is duplicated.

### E1: Chunking, controls, and primary retrieval

Objective:

Materialize exact and randomized policies, measure boundary-level retrieval behavior,
and freeze score-complete top-16 retrieval traces.

Exact inputs:

- E0 manifests;
- four policies;
- five jitter seeds per policy;
- pinned BGE, BM25, hybrid, and reranker.

Exact outputs:

- chunk manifests for 24 policy/control systems per dataset;
- encoder exposure audit;
- retrieval traces;
- retrieval metrics at k=4 and k=8;
- cost traces.

Stopping criteria:

- every planned system/dataset trace is complete;
- all randomized-control validations pass;
- no chunk exceeds 254 canonical tokens;
- no reranker or embedding shard remains failed.

Validation:

- source coverage exact;
- index sizes equal chunk-manifest counts;
- score sorting and tie rules recompute;
- top-16 IDs reproduce on the audit subset;
- no answers were read during chunking, control generation, or retrieval.

### E2: Primary Mistral treatment experiment

Objective:

Estimate H1, H2, and H3 using frozen retrieval traces.

Exact inputs:

- E1 primary retrieval traces;
- Mistral primary generator;
- frozen prompts;
- budgets and packing rules.

Generation conditions per dataset:

1. Boundary set: recursive, sentence, and semantic plus each policy's five jitter
   controls under matched 4,096-token packing: 18 conditions.
2. Exposure set: fixed, recursive, sentence, and semantic under operational 1,024,
   matched 1,024, and operational 4,096 packing: 12 conditions.
3. Fixed under matched 4,096 packing: one condition.

The three structured matched-4,096 conditions in the boundary set are reused, not
regenerated. Total unique retrieved-evidence conditions: 31 per dataset.

Exact outputs:

- 31 generation traces per frozen question;
- 15,500 SQuAD traces;
- 15,500 HotpotQA traces;
- 9,300 TechQA traces;
- prompt/context audit;
- answer and evidence metrics.

Stopping criteria:

- every planned record is valid or receives the frozen failure score;
- no condition exceeds 1% unresolved infrastructure failure;
- all hash chains validate.

Validation:

- retrieval IDs equal E1;
- matched token counts are within two tokens of M(q,b);
- operational packing reproduces from stored context;
- raw output normalizes deterministically;
- result counts equal the specification.

### E3: Gold-evidence upper bounds

Objective:

Estimate the remaining generator/evaluation gap when annotated or answer-informed gold
evidence is supplied.

Exact inputs:

- E0 gold-evidence manifests;
- Mistral;
- dataset prompt;
- 1,024 and 4,096 budgets.

Exact outputs:

- two gold-evidence generation traces per question;
- 1,000 SQuAD, 1,000 HotpotQA, and 600 TechQA traces;
- gold-gap metrics.

Stopping and validation:

- same generation completion rules as E2;
- evidence spans must be traceable to gold documents;
- no retrieved-system score is used in gold-evidence selection.

### E4: TechQA semantic and human evaluation

Objective:

Validate semantic evaluation and measure TechQA utility and groundedness.

Exact inputs:

- all 9,900 Mistral TechQA traces from E2 and E3;
- pinned Qwen judge;
- frozen judge prompt;
- 360-record human subset.

Exact outputs:

- judge traces for every TechQA answer;
- two raw human annotation files;
- adjudicated labels;
- agreement and judge-validation report;
- confirmatory-status flag.

Stopping criteria:

- every judge record is parsed or receives the frozen zero score;
- all 360 records receive two correctness and completeness labels or an explicit
  cannot_assess;
- all 60 frozen groundedness records receive two groundedness labels or an explicit
  cannot_assess;
- acceptance or fallback criteria are applied mechanically.

Validation:

- system names remain blinded;
- output order matches deterministic hashes;
- agreement and bias calculations reproduce from raw labels.

### E5: Secondary retrieval-stack robustness

Objective:

Test whether deterministic-policy retrieval patterns persist outside the primary stack.

Exact inputs:

- E0 manifests;
- fixed, recursive, sentence, semantic only;
- embedders BGE and MiniLM;
- retrievers dense, hybrid, and hybrid-plus-rerank;
- top-16 traces.

Exact outputs:

- retrieval metrics at k=4 and k=8;
- complete paired effect tables;
- no selected-winner table.

Stopping and validation:

- all 72 dataset/embedder/retriever/policy cells complete;
- BM25 components are identical wherever the corpus/chunker is identical;
- model revisions and scores are preserved.

No confirmatory p-values are assigned to E5.

### E6: Secondary Qwen generator replication

Objective:

Check whether deterministic matched-budget policy patterns depend on the primary
generator checkpoint.

Exact inputs:

- SQuAD and HotpotQA only;
- Qwen2.5-7B;
- fixed, recursive, sentence, semantic under matched 4,096;
- gold evidence at 4,096;
- same retrieval traces and prompts as E2/E3.

For Qwen, define M_Q(q,4096) independently with Qwen's native tokenizer: take the
minimum of 3,072, the question-specific prompt capacity after the 16-token safety
margin, and the available rendered-context tokens across the four deterministic
retrieved systems. Each retrieved system consumes the longest prefix within two tokens
of M_Q. Gold evidence uses the same M_Q when sufficient gold text exists and otherwise
records its shortfall. Mistral token counts are never reused for Qwen packing.

Exact outputs:

- 2,500 SQuAD and 2,500 HotpotQA generation traces;
- paired descriptive effects and intervals.

Stopping and validation:

- same rules as E2;
- Qwen does not judge these outputs;
- no new confirmatory family is opened.

### E7: Cost and reproducibility audit

Objective:

Measure operational cost and verify reproducibility before paper generation.

Exact inputs:

- completed E0–E6 artifacts;
- canonical SCC A100 environment;
- frozen 25-question audit subsets.

Exact outputs:

- latency, storage, token, and memory table;
- reproducibility report;
- full artifact validation report.

Stopping criteria:

- every reproducibility criterion in Section 25 passes;
- every reported table/figure value can be traced to immutable records.

Failure blocks paper generation.

Validation:

- audit records reference the original immutable artifact hashes;
- recomputed outputs are written to a separate audit namespace and byte-compared;
- timing rows identify the exact node, GPU class, CUDA build, batch size, warm-up
  status, and number of measured questions;
- all eight experiment completion flags (E0 through E7) are true.

## 27. Compute plan

### Canonical platform

Primary retrieval timing and canonical generation use NVIDIA A100 GPUs under the frozen
Python 3.11 environment.

Exact direct dependency versions:

- accelerate 0.34.2;
- datasets 2.21.0;
- faiss-cpu 1.14.3;
- huggingface-hub 0.36.2;
- langchain-text-splitters 0.3.11;
- matplotlib 3.11.0;
- numpy 1.26.4;
- pandas 2.3.3;
- rank-bm25 0.2.2;
- sentence-transformers 3.4.1;
- sentencepiece 0.2.1;
- spacy 3.8.14;
- torch 2.13.0;
- transformers 4.57.6;
- tqdm 4.68.4.

The complete transitive lock and CUDA/Python build strings are hashed in the run
manifest.

Pre-run capacity reservation is 160 A100 GPU-hours and 500 GB of artifact storage:

- E1 retrieval/reranking: 24 GPU-hours;
- E2 primary generation: 68 GPU-hours;
- E3 gold generation: 8 GPU-hours;
- E4 judging: 22 GPU-hours;
- E5 retrieval robustness: 18 GPU-hours;
- E6 Qwen generation: 16 GPU-hours;
- E7 reproduction reserve: 4 GPU-hours.

These are scheduling estimates, not stopping thresholds. Overrun is recorded and
additional compute is requested; no planned condition is dropped to meet an estimate.

### Resource assignment

- GPU 0: SQuAD Mistral shards.
- GPU 1: HotpotQA Mistral shards.
- GPU 2: TechQA Mistral shards.
- GPU 3: Qwen secondary generation, TechQA judge, and failed-shard recovery.

Retrieval and reranking are materialized before generation and may use any GPU, but
timing is measured on one designated A100.

## 28. Colab sharding plan

Google Colab is an overflow execution surface, not a source of hidden state.

Canonical Colab output is accepted only when:

- the runtime GPU reports NVIDIA A100;
- the environment-lock hash matches;
- the clean Git commit and protocol hash match;
- the model snapshot hash matches;
- the validation notebook reports canonical=true.

L4, T4, or other GPUs may be used for smoke tests but their outputs cannot enter primary
tables.

Drive root:

MyDrive/chunkrag-main-v1/

One notebook invocation accepts exactly:

- protocol hash;
- Git commit;
- model;
- dataset;
- condition ID;
- shard index;
- question-manifest hash;
- retrieval or gold-evidence hash.

Questions are sorted by example ID and partitioned into consecutive shards of 50.

Checkpoint rules:

- write one append-only record after each question;
- fsync/copy to Drive every 10 questions;
- write to part-NNN.jsonl.tmp;
- atomically rename to part-NNN.jsonl only after all 50 records validate;
- maintain part-NNN.state.json with completed IDs and record hashes;
- resume only from individually valid hashes;
- never overwrite a complete shard;
- merge rejects duplicate IDs, missing IDs, mixed hashes, and mixed environments.

No password, access token, notebook output, or Drive path is stored in the anonymous
artifact.

## 29. Execution order

Execution is strictly sequential at the stage level:

1. Commit protocol, implementation, tests, environment lock, and configs in a clean
   worktree.
2. Run E0 and commit the frozen manifests and hashes.
3. Run all synthetic statistical tests without primary outcomes.
4. Run E1 chunk/control validation.
5. Run E1 retrieval and freeze top-16 traces.
6. Run E1 validation; do not proceed on failure.
7. Run E2 Mistral generation by dataset/shard.
8. Run E3 gold-evidence generation.
9. Merge and validate all Mistral traces.
10. Freeze the 360-record human annotation package without judge scores.
11. Collect human labels.
12. Run E4 Qwen judge and evaluator validation.
13. Run E5 secondary retrieval robustness.
14. Run E6 secondary Qwen generation.
15. Run E7 cost and reproducibility audit.
16. Lock all result artifacts read-only.
17. Execute the confirmatory analysis once.
18. Generate every table and figure.
19. Rewrite the paper from the locked outputs.
20. Build and validate the anonymous artifact.

No confirmatory script is modified after Step 17 without an amendment and a complete
rerun from locked raw traces.

## 30. Paper table and figure assignments

### Main figures

Figure 1: Treatment and evidence-exposure diagram.

- Inputs: protocol Sections 5, 13, 16, and Experiments E1–E3.
- Contains no outcome-dependent design choice.

Figure 2: Boundary identification forest plot.

- Inputs: E2 H1 contrasts.
- Panels: SQuAD and HotpotQA.
- Rows: recursive, sentence, semantic.
- Shows exact-minus-random mean F1, 95% cluster interval, and equivalence region.

Figure 3: Exposure distortion and mechanism.

- Inputs: E2 H2/H3 effects and consumed-evidence metrics.
- Columns are SQuAD, HotpotQA, and TechQA. Each column has two aligned panels.
- The upper panel shows structured-minus-fixed answer effects for operational 1,024,
  matched 1,024, and operational 4,096 with 95% cluster intervals.
- The lower panel shows the same contrasts for fully consumed gold-evidence fraction.
- That fraction is the binary fully visible answer span for SQuAD, the fraction of
  annotated supporting sentences fully consumed for HotpotQA, and the fraction of gold
  documents represented in consumed context for TechQA.
- TechQA answer effects are explicitly labeled exploratory if judge validation fails;
  if the human cannot-assess rule removes TechQA answer evaluation, only its lower
  evidence panel remains.

### Main tables

Table 1: Frozen datasets, corpora, clusters, questions, and manifest hashes.

- Input: E0.

Table 2: Primary H1/H2 contrasts.

- Input: E2.
- Includes mean F1 difference, interval, rank-biserial effect, raw p, Holm p, n, and
  cluster count.

Table 3: Gold-evidence gaps and TechQA evaluation.

- Inputs: E3 and E4.
- Includes semantic utility, groundedness, human agreement, and judge-validation status.

### Appendix figures

- A1: Archived pilot token-budget audit; archived audit.
- A2: Full boundary-control retrieval distributions; E1.
- A3: Retrieval metrics at k=4 and k=8; E1/E5.
- A4: TechQA semantic utility by policy and condition; E4.
- A5: Cost-quality Pareto front; E7.
- A6: Human-versus-judge calibration and bias; E4.
- A7: Chunk-length and randomized-boundary diagnostics; E1.
- A8: Secondary Qwen paired effects; E6.

### Appendix tables

- B1: Complete chunk configuration and realized lengths; E1.
- B2: Complete primary retrieval metrics; E1.
- B3: Complete Mistral answer metrics; E2/E3.
- B4: Budget and truncation audit; E2.
- B5: TechQA human labels and agreement summary; E4.
- B6: Secondary embedder/retriever matrix; E5.
- B7: Secondary Qwen results; E6.
- B8: Runtime, storage, token, and memory measurements; E7.
- B9: Reproducibility audit; E7.
- B10: Archived pilot results and corrected failure analysis; archived appendix case.
- B11: Exact prompts, evaluator rubric, and annotation instructions; protocol/E4.
- B12: Outcome-free sample-size sensitivity analysis; E0.

No main table may display only selected observed winners.

## 31. Outcome-independent stopping policy

The study ends when:

- every planned E0–E7 record is complete or handled by its frozen failure rule;
- all validation and reproducibility requirements pass;
- the human/judge fallback is applied mechanically;
- confirmatory analysis and multiplicity correction run once;
- every table and figure traces to locked raw artifacts.

The study does not stop because:

- a hypothesis is significant or non-significant;
- structured policies lose to controls;
- effect sizes are small;
- a preferred narrative is unsupported;
- the secondary generator disagrees with Mistral.

## 32. Publishable-outcome policy

All of the following outcomes remain publishable:

1. Structured boundaries outperform randomized controls after exposure matching.
2. Structured boundaries do not outperform randomized controls and meet equivalence
   criteria.
3. Apparent operational effects shrink or reverse under exposure matching.
4. Operational and matched effects agree, showing that exposure control does not explain
   the observed differences.
5. Reranking dominates chunker differences.
6. Effects differ by corpus, provided the interaction and corpus construction are
   reported without a universal claim.

The project is scientifically invalid only if the trace, corpus, randomization,
evaluation, or reproducibility requirements fail and cannot be repaired without viewing
outcomes.

## 33. Phase 2 closure

This specification is complete when its SHA-256 is recorded in
`reports/phase2_immutable_specification.sha256`. That file and this specification must
be committed unchanged at the start of Phase 3 before implementation changes.

Phase 3 may implement this protocol but may not:

- change a scientific parameter;
- add or remove a primary condition;
- inspect primary outcomes during implementation;
- substitute a model or dataset;
- relax a validation threshold;
- redefine a metric or contrast.

Any necessary change follows the amendment process in Section 1.
