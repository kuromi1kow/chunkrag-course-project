# Current results summary

This summary replaces the earlier FLAN-era report. It describes only experiments with
committed result artifacts and keeps the archived Mistral pilot separate from the
reviewer-driven robustness extension.

## 1. Archived Mistral pilot

The archived run evaluates 60 answerable SQuAD 2.0 questions and 30 HotpotQA distractor
questions with MiniLM dense retrieval at `k=4` and Mistral-7B-Instruct-v0.3 generation.
All values below are percentage points.

| Dataset | Highest observed chunker F1 | F1 (95% question-bootstrap CI) | Retrieval observation |
|---|---|---:|---|
| SQuAD | recursive-256 | 60.6 [50.5, 70.5] | fixed-128 and semantic-256 both have 95.0 AllHit@4 but miss different questions |
| HotpotQA | recursive-256 | 42.6 [26.8, 58.7] | recursive, fixed-512, sentence, and semantic all have 53.3 AllHit@4 |

These are observed leaders, not statistically established winners. None of the five
recursive-versus-competitor F1 contrasts survives Holm correction within either dataset.
For example, recursive-256 minus semantic-256 is `+3.7` on SQuAD (paired 95% CI
`[-3.2, 11.0]`) and `+0.2` on HotpotQA (`[-6.5, 8.5]`).

### Audit findings that change the interpretation

- MiniLM accepts 256 encoded positions including two special tokens. On SQuAD it
  truncates 88.2% of fixed-256 corpus chunks and 93.9% of fixed-512 chunks during
  embedding, compared with 1.8% of recursive-256 chunks.
- The 1,024-token Mistral chat budget truncates all 60 SQuAD fixed-512 prompts. Only
  1.07 of four retrieved chunks are completely consumed on average.
- Among 22 HotpotQA recursive-256 exact-match failures, 13 are evidence-limited, 6 are
  response-form candidates, and 3 are content mismatches. The former “72.7% fixable”
  claim is withdrawn.
- Fixed-128 and semantic-256 each retrieve the SQuAD gold article for 57 of 60
  questions, but the overlap is 55 both-hit, 2 fixed-only, 2 semantic-only, and 1
  shared miss.

Sources: `outputs/midway_mistral_endpoint_v2/` and `outputs/revision_audit/`.

## 2. Multi-seed retrieval robustness

The extension uses three data seeds (13, 21, and 34), two embedders (MiniLM and
BGE-small-en-v1.5), three retrievers (dense, BM25, and hybrid), and four chunkers
(fixed-128, fixed-254, recursive-254, and sentence-254). Per seed it evaluates 150
SQuAD, 75 HotpotQA, and 200 answerable TechQA questions. TechQA always searches the same
496-document corpus built from all answerable rows before question sampling. The primary
endpoint is document AllHit@4; AnsVis@4 requires an exact normalized answer-token sequence
and excludes HotpotQA yes/no labels.

Selected observed leaders are shown as mean ± sample standard deviation across the
three seeds:

| Dataset | Embedder | Retriever | Observed leader | AllHit@4 |
|---|---|---|---|---:|
| SQuAD | MiniLM | dense | fixed-128 | 97.1 ± 1.4 |
| SQuAD | MiniLM | hybrid | fixed-128 | 98.9 ± 0.4 |
| SQuAD | BGE | dense | fixed-128 | 97.8 ± 0.4 |
| SQuAD | BGE | hybrid | fixed-128 | 99.1 ± 1.0 |
| HotpotQA | MiniLM | dense | sentence-254 | 50.7 ± 4.8 |
| HotpotQA | MiniLM | hybrid | fixed-254 | 48.0 ± 5.8 |
| HotpotQA | BGE | dense | recursive-254 / sentence-254 (tie) | 70.2 ± 8.9 |
| HotpotQA | BGE | hybrid | recursive-254 | 58.7 ± 3.5 |
| TechQA | MiniLM | dense | fixed-128 | 88.0 ± 3.0 |
| TechQA | BGE | dense | fixed-254 | 88.8 ± 0.8 |

The complete table is generated from the two retrieval output roots. Sentence-254 has
the highest observed AnsVis@4 in every dense and hybrid cell even when another chunker
leads AllHit@4; on TechQA, the best observed AnsVis remains below 52%. A hit on a long
gold document therefore often does not expose the answer-bearing region. BM25 does not
use the embedding model, so its duplicate MiniLM/BGE rows are identical by construction.
The important pattern is the change in observed leader across dataset, endpoint, and
retrieval setup, not any single maximum.

Sources:

- `outputs/reviewer_robustness_retrieval_minilm/`
- `outputs/reviewer_robustness_retrieval_bge/`
- `configs/reviewer_robustness_retrieval_minilm.json`
- `configs/reviewer_robustness_retrieval_bge.json`

## 3. Controlled local generation robustness

The local Qwen and Mistral runs use BGE hybrid retrieval, seed 42, identical questions
and retrieved chunk IDs, the same instructions and output limits, and a 1,536-token
complete-chat budget under each model's native tokenizer. Each evaluates 60 SQuAD, 30
HotpotQA, and 50 TechQA questions and records raw/final generation and token traces.
SQuAD and HotpotQA use an extractive answer style; TechQA uses a complete-answer style.

### Qwen2.5-1.5B

| Dataset | No-context F1 | fixed-128 | fixed-254 | recursive-254 | sentence-254 |
|---|---:|---:|---:|---:|---:|
| SQuAD | 17.26 | 66.58 | 61.52 | 65.58 | 65.35 |
| HotpotQA | 21.91 | 50.32 | 50.32 | 46.99 | 53.37 |
| TechQA | 13.97 | 21.29 | 23.72 | 22.22 | 20.42 |

The observed Qwen leader differs by dataset. Every RAG mean exceeds its descriptive
no-context comparator, but no recursive-versus-comparator contrast is significant; all
nine Qwen Holm-adjusted p-values are 1.0. No Qwen output reaches its generation cap, and
only four TechQA RAG prompts are truncated across the 200 RAG question-cells.

### Mistral-7B

| Dataset | No-context F1 | fixed-128 | fixed-254 | recursive-254 | sentence-254 |
|---|---:|---:|---:|---:|---:|
| SQuAD | 27.78 | 61.59 | 55.81 | 54.36 | 53.71 |
| HotpotQA | 23.36 | 45.40 | 47.21 | 45.73 | 49.64 |
| TechQA | 16.04 | 24.15 | 26.50 | 23.80 | 25.03 |

The Mistral leader is likewise dataset-specific: fixed-128 on SQuAD, sentence-254 on
HotpotQA, and fixed-254 on TechQA. The smallest raw paired value is recursive-254 minus
fixed-128 on SQuAD (`-7.23` F1, paired 95% CI `[-13.62, -1.77]`, raw `p=.0156`), but
its global Holm-adjusted value is `.2817`. None of the 18 planned contrasts across both
generators survives the primary correction. No Mistral output reaches its generation
cap; five TechQA RAG prompts are truncated, versus none on SQuAD or HotpotQA.

Sources:

- `outputs/reviewer_robustness_qwen/`
- `outputs/reviewer_robustness_mistral/`
- `outputs/reviewer_robustness_analysis.json`
- `outputs/reviewer_robustness_analysis.md`
- `configs/reviewer_robustness_qwen.json`
- `configs/reviewer_robustness_mistral.json`
- `scripts/analyze_reviewer_robustness.py`

## Bottom line

The revised evidence supports a conditional conclusion: chunking behavior depends on
the corpus, retriever, embedder, token budget, and generator. The original sample is too
small and confounded by token truncation for general ranking claims. The robustness
extension broadens the empirical base and demonstrates configuration sensitivity, but
it remains a sampled study rather than a full-benchmark, systematic model-scale, or
multi-temperature evaluation. The local Qwen and Mistral follow-ups form a controlled
same-stack checkpoint comparison; the separate archived endpoint run uses different
retrieval and context settings and is not pooled with them.
