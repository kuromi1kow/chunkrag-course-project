# Reviewer-driven revision matrix

This document maps the three official reviews of submission 8677 to the revised
manuscript, code, and result artifacts. “Partially addressed” means that a concern now
has empirical coverage but remains broader than the completed experiments.

## Central changes

1. The original Mistral experiment is framed as a controlled pilot rather than a
   universal ranking of chunking strategies.
2. A reviewer-driven retrieval extension evaluates 150 SQuAD, 75 HotpotQA, and 200
   answerable TechQA questions per seed across seeds 13, 21, and 34; MiniLM and BGE
   embeddings; dense, BM25, and hybrid retrieval; and four encoder-compatible chunkers.
3. Controlled local generation experiments use pinned Qwen2.5-1.5B-Instruct and
   Mistral-7B-Instruct-v0.3 checkpoints with identical BGE hybrid retrieval, questions,
   retrieved chunks, instructions, and token limits on 60 SQuAD, 30 HotpotQA, and 50
   TechQA questions. Both retain raw generations and prompt traces.
4. The original answer tables now report question-bootstrap intervals. Paired analyses
   use aligned-question bootstrap intervals, paired sign-flip tests, and Holm correction.
5. The complete prompt, decoding limits, prefix-truncation algorithm, optional refinement
   call, and deterministic post-processing are documented.
6. The archived top-four chunks are reconstructed and replayed through both tokenizer
   limits. This reveals extensive SQuAD generator truncation and MiniLM embedding
   truncation. The prospective chunkers use a strict post-decode token check; an
   executable audit finds zero target or encoder-limit violations.
7. The original 65.7%/72.7% “fixable failure” headline is withdrawn. The corrected
   evidence-aware taxonomy treats partial HotpotQA support retrieval as incomplete and
   does not label any category empirically fixable.
8. The related-work section is expanded to cover retrieval granularity, RAG evaluation,
   QA robustness, statistical testing, and context-position effects.

The extension strengthens the empirical basis but does not test full dataset splits,
temperature variation, a systematic generator-size sweep, or a held-out prompt-repair
intervention. The manuscript preserves those as limitations.

## Review soAh

| Concern | Revision | Evidence/location | Status |
|---|---|---|---|
| Evaluation is too small and narrow to support general RAG claims. | Universal, causal, “optimal,” and “best” claims were removed. The retrieval extension adds three seeds, larger samples, and TechQA as a technical-support domain. | `reports/final_report_acl.tex`; `configs/reviewer_robustness_retrieval_*.json`; `outputs/reviewer_robustness_retrieval_*/` | **Partially addressed empirically.** The study is broader, but it still samples rather than evaluating complete benchmark splits. |
| Results may not generalize across tasks, model sizes/families, temperatures, embeddings, or retrievers. | The extension adds TechQA, BGE alongside MiniLM, dense/BM25/hybrid retrieval, and a controlled same-stack comparison of local Qwen2.5-1.5B and Mistral-7B checkpoints. The observed leader changes across settings. | Robustness section and tables; `configs/reviewer_robustness_*.json`; `scripts/analyze_reviewer_robustness.py` | **Partially addressed empirically.** Dataset, embedder, retriever, and generator coverage improved; family and size remain confounded, and a systematic scale/temperature sweep remains untested. |
| The 60/30 sample is unstable; one or two questions can move results substantially. | The original paper quantifies one-question changes and reports 95% intervals. The retrieval follow-up increases per-seed samples to 150/75 and adds 200 TechQA questions across three seeds. | Original result tables; robustness retrieval tables; `outputs/revision_audit/statistics.json` | **Partially addressed.** Uncertainty and data-seed variation are visible, but generation still uses small single-seed samples. |
| “Most failures are fixable” was not tested. | The claim is withdrawn. “Response-form candidate” is explicitly a diagnostic label, and the paper states that only a prospective held-out intervention can establish repairability. | Failure-audit section and table; `scripts/failure_reanalysis.py` | **Addressed by correction, not by claiming an unrun intervention.** Prompt repair remains future work. |
| Related work is sparse and omits broader RAG evaluation and QA robustness. | Added primary work on Dense X Retrieval, TextTiling, Late Chunking, KILT, RAGAs, ARES, RAGTruth, adversarial QA, significance testing, and long-context position effects. | Related Work; `reports/references.bib` | **Addressed.** |
| How can top-4 fixed-512 fit a 1,024-token input? | It does not. The original implementation retains the longest ranked-context prefix. All 60 SQuAD fixed-512 chats are truncated; 56 retain one complete chunk and part of the second, 4 retain two complete chunks and part of the third, and none retains all four. | Context-budget section and appendix; `outputs/revision_audit/context_budget_audit.json` | **Addressed and elevated to a central confound.** |
| Error categories are referred to by numbers that are not displayed. | Categories are now named explicitly and the executable audit uses a fixed, mutually exclusive priority order. | Failure-audit methods; `scripts/failure_reanalysis.py` | **Addressed.** |

## Review 6G9z

| Concern | Revision | Evidence/location | Status |
|---|---|---|---|
| The study is exploratory and may be more appropriate as a short paper. | The title, abstract, discussion, limitations, and conclusion describe the archived experiment as a pilot and make all rankings conditional on the recorded setup. The robustness extension is presented as follow-up evidence rather than retroactive proof of a general ranking. | `reports/final_report_acl.tex` | **Addressed by scope and framing.** Venue/length remains an editorial decision. |
| Only 60 SQuAD and 30 HotpotQA questions are evaluated. | The retrieval extension evaluates 150 SQuAD, 75 HotpotQA, and 200 answerable TechQA questions per seed across three seeds. TechQA retrieval searches one fixed 496-document corpus built before question sampling. Both generator follow-ups include 50 TechQA questions in addition to the original-sized SQuAD/HotpotQA samples. | `configs/reviewer_robustness_*.json`; committed output roots | **Partially addressed empirically.** The extension is materially larger and adds a domain, but it is not a full-split evaluation. |
| Single runs are reported without confidence intervals or significance tests. | The original run now has 20,000-draw marginal and paired bootstrap intervals, 100,000-draw paired sign-flip tests, and Holm correction. The retrieval extension reports mean and seed standard deviation over three data seeds. Both local generator runs receive the same paired treatment, with one 18-test Holm family across models. | Generated statistical tables; `scripts/statistical_analysis.py`; `scripts/analyze_reviewer_robustness.py` | **Addressed for question uncertainty; partially addressed for run variation.** Each generator still has one deterministic run. |
| Multiple retrievers and generators should be evaluated. | The extension runs dense, BM25, and hybrid retrieval with MiniLM and BGE, then evaluates local Qwen2.5-1.5B and Mistral-7B under the same data, retrieval, chunking, prompts, and decoding budgets. | Robustness configs, outputs, and generated tables | **Partially addressed empirically.** Two checkpoints are included, but family and size are confounded and temperature is fixed. |

## Review JYbm

| Concern | Revision | Evidence/location | Status |
|---|---|---|---|
| The full prompt and generator sampling parameters are missing. | Added exact contextual and no-context messages, temperature 0, 48-token first completion, 24-token optional refinement, 1,024-token complete-chat limit, prefix truncation, and post-processing. Unrecorded vLLM defaults are identified as unrecorded rather than guessed. Both prospective configs pin model revisions, a 1,536-token input budget, and dataset-specific 96/96/512 output limits. | Generation methods and prompt appendix; `configs/reviewer_robustness_{qwen,mistral}.json` | **Addressed.** |
| The prompt-fix narrative cannot be judged because the current prompt and post-processing rules are invisible. | The original prompt and executable rules are now described, including the fact that it already requests the shortest span and that archived endpoint predictions may include a second model call. Both local extensions log raw predictions, final predictions, prompt lengths, truncation, refinement state, answer style, generated-token counts, and length caps per question. | Prompt appendix; `src/chunkrag/generation.py`; completed local generator output roots | **Addressed for transparency.** The archived raw drafts cannot be recovered, and repair efficacy remains untested. |
| Fixed-128 and semantic-256 share 95% Recall@4, but it is unknown whether they miss the same questions. | Added a 2×2 per-question overlap table: 55 both hit, 2 fixed-only hits, 2 semantic-only hits, and 1 shared miss. The exact missed questions are listed in the appendix. | Retrieval-overlap table and appendix; `outputs/revision_audit/statistics.json` | **Addressed.** |

## Additional corrections discovered during the audit

- MiniLM embeds at most 256 positions including special tokens, leaving 254 content
  positions. This affects 284 of 322 SQuAD fixed-256 chunks, 154 of 164 fixed-512
  chunks, 7 of 389 recursive-256 chunks, and 23 of 302 sentence-256 chunks.
- Retrieval metrics in the archived experiment were computed over all four retrieved
  chunks, including later chunks that could be removed from the generator prompt.
- The old HotpotQA taxonomy treated one of two supporting documents as adequate. In the
  corrected recursive-256 audit, 13 of 22 EM failures are evidence-limited, 6 are
  response-form candidates, and 3 are content mismatches.
- The original “Precision@4” mixes document identity and answer-string containment. It
  is omitted from the main interpretation in favor of supporting-document coverage and
  all-support hit rate.
- Equal SQuAD aggregate document coverage hides different per-question misses.
- The extension targets 254 content tokens for encoder compatibility and now subdivides
  a single overlong sentence instead of passing it through unsplit. A post-decode
  round-trip check and executable audit show zero target or encoder-limit violations.
- AnsVis originally used normalized character substrings, which could match short answers
  inside larger words. It now requires an exact normalized token sequence and excludes
  HotpotQA yes/no questions from this evidence diagnostic.
- Claims that “DPR serves as our retriever,” “titles should be surfaced for HotpotQA,”
  and “the prompt lacks length constraints” were factually incorrect and were removed.

## Statistical interpretation of the extension

The multi-seed retrieval results show that the observed leading chunker changes with the
dataset, embedder, and retriever. They are descriptive robustness evidence, not a new
global ranking. In the Qwen end-to-end follow-up, no recursive-versus-comparator
contrast is significant after correction across the planned two-model family; all nine
Qwen adjusted values are 1.0. Mistral's smallest raw value is for recursive-254 minus
fixed-128 on SQuAD (`-7.23` F1, paired 95% CI `[-13.62, -1.77]`, raw `p=.0156`), but
its adjusted value is `.2817`. None of the 18 planned contrasts across both generators
survives the primary correction.

## Remaining work for a stronger resubmission

A stronger confirmatory study should evaluate full or substantially larger benchmark
splits, pre-register a small number of primary contrasts, vary generator sizes and
temperatures, repeat generation runs, equalize consumed-context budgets, and conduct a
held-out prompt-repair intervention. Human review or inter-annotator agreement would
also strengthen the automatic failure taxonomy. None of these remaining items is
presented as completed evidence.
