# Phase 3H Final Audit

Protocol: `chunkrag-main-v1`  
Protocol SHA-256: `567b652fc403e7ff7e00e349de86357f9a293cac77e7e7f4d3612284eb2c89bf`  
Phase 3 scope: implementation and synthetic validation only.  
Experimental datasets loaded: none.  
Models loaded or executed: none.  
Experimental outputs or statistics inspected: none.

## Final status

Phase 3 implementation is complete. The repository has an isolated, protocol-frozen
main-study package covering all 33 specification sections and E0--E7. The unified DAG
contains 1,307 immutable work items. Every stage has a concrete handler, artifact schema,
hash/provenance path, checkpoint strategy, validation gate, and synthetic test coverage.

The archived pipeline remains unchanged and cannot be selected by the main-study runner.

## Implementation step 1: protocol authority

- Rationale: prevent silent drift from the frozen Phase 2 decisions.
- Protocol sections: 1, 7, 23, 33.
- Files changed: `constants.py`, `protocol.py`, `configs/main_study.json`, frozen checksum
  integration.
- Modification: exact IDs, revisions, conditions, budgets, counts, and checksum gates.
- Validation: checksum verification and frozen-config invariant tests pass.

## Implementation step 2: canonical artifacts and schemas

- Rationale: replace legacy mutable aggregate JSON with traceable records.
- Protocol sections: 10, 11.5, 23--25.
- Files changed: `canonical.py`, `schemas.py`, `artifacts.py`.
- Modification: NFC canonical JSON/JSONL, SHA-256 identifiers, strict schemas, immutable
  atomic writes, external/source tree hashes, record-link validation, read-only locking.
- Validation: golden serialization, duplicate, corruption, unknown-field, overwrite, and
  broken-link tests pass.

## Implementation step 3: deterministic runtime and provenance

- Rationale: canonical execution must reject hidden state.
- Protocol sections: 20, 23--25, 27.
- Files changed: `determinism.py`, `environment.py`, `logging.py`,
  `requirements-main-study.lock`.
- Modification: derived PCG64 seeds, Torch/CUDA determinism, clean-Git gate, exact direct
  dependency verifier, transitive environment freezer, hardware manifest, structured
  logs, source hash, run manifests.
- Validation: deterministic seed and repository-verification tests pass. Canonical A100
  and exact installed-package checks are execution gates and were not bypassed locally.

## Implementation step 4: E0 materialization

- Rationale: question/corpus construction must be independent of outcomes and legacy
  sampling.
- Protocol sections: 8--10, 17, E0.
- Files changed: `data.py`, `gold.py`, `power.py`, E0 handler.
- Modification: pinned loaders, normalization, SHA ordering/caps, SQuAD span remapping,
  Hotpot sentence provenance, TechQA conflict detection, connected components, dataset
  cache manifests, gold-source manifests, duplicate materialization check, outcome-free
  power sensitivity.
- Validation: synthetic normalization, selection, cluster, provenance, and gold tests
  pass. No external dataset was loaded.

## Implementation step 5: E1 chunking and controls

- Rationale: boundary treatment must preserve source coverage and isolate cut location.
- Protocol sections: 11--12, E1.
- Files changed: `chunking.py`, `controls.py`.
- Modification: exact fixed/recursive/sentence/semantic policies, offset round trips,
  strict length rules, five SHA-derived jitter controls, feasibility accounting, 80%
  changed-boundary gate.
- Validation: deterministic fixed/recursive/sentence/semantic and jitter tests pass,
  including exact source reconstruction and length bounds.

## Implementation step 6: retrieval and context exposure

- Rationale: retain every retrieval score and separate retrieval from generation.
- Protocol sections: 13, 16, 18.3--18.5, E1, E5.
- Files changed: `retrieval.py`, `packing.py`, `prompts.py`.
- Modification: exact float32 dense/IndexFlatIP adapter, frozen BM25 tokens, weighted RRF,
  cross-encoder `only_second` truncation with audits, top-50/top-16 traces, top-4
  operational packing, 19-system matched targets, complete prompt IDs and truncation
  locations.
- Validation: RRF/tie, lexical, operational-budget, matched-budget, and two-token
  tolerance tests pass.

## Implementation step 7: generation and evaluation

- Rationale: local pinned inference must produce score-complete, retryable traces.
- Protocol sections: 14--19, 23.4--23.5, E2, E3, E6.
- Files changed: `generation.py`, `evaluation.py`, `execution.py`.
- Modification: offline pinned Mistral/Qwen adapters, resolved greedy settings, snapshot
  hashes, exact prompts, three-attempt infrastructure policy, invalid-shard preservation,
  output normalization, answer/evidence intervals, gold ordering/centering, automatic
  evaluation records.
- Validation: prompt, normalization, resolved generation config, F1, evidence coverage,
  ranking, and retry/checkpoint paths have synthetic tests. No generation was executed.

## Implementation step 8: TechQA judge and human evaluation

- Rationale: semantic utility is confirmatory only after human validation.
- Protocol sections: 21--22, E4.
- Files changed: `human.py`, judge functions in `evaluation.py`, E4 handlers,
  `docs/main_study_human_annotation.md`.
- Modification: exact judge prompt/JSON, invalid parse handling, 360-record blinded
  package, 60-record groundedness subset, 20-record training package, private linkage,
  two-annotator schemas, agreement/adjudication, parse/bias/correlation/kappa thresholds,
  cannot-assess fallback.
- Validation: strict judge parsing, package size/order/blinding, training size, Spearman,
  weighted kappa, and ordinal alpha tests pass.

## Implementation step 9: statistical engine

- Rationale: inference must be cluster-aware and familywise-corrected before outcomes.
- Protocol sections: 8.5, 20, E0, E4.
- Files changed: `statistics.py`, `analysis.py`.
- Modification: 20,000-draw cluster bootstrap, exact/99,999 sign flips, CR1 regression,
  TOST, Holm families, H1--H4, TechQA validation gate, rank-biserial, Cliff's delta,
  power sensitivity, locked analysis gate.
- Validation: synthetic effect, seed, bootstrap, sign-flip, Holm, correlation, and
  agreement tests pass. No primary statistics were computed.

## Implementation step 10: orchestration, checkpoints, and Colab

- Rationale: local/SCC/Colab shards must use one execution path.
- Protocol sections: 26--29.
- Files changed: `experiments.py`, `stages.py`, `completion.py`, `checkpoint.py`,
  `runner.py`, `colab.py`, `scripts/run_main_study.py`, Colab notebook and execution guide.
- Modification: E0--E7 DAG, 50-question shards, append/resume/invalidate/finalize, atomic
  merge, hashed work/stage markers, mixed-provenance rejection, dry-run/run/validation/
  merge modes, A100 and Drive gates.
- Validation: all 1,307 work items dry-run successfully; resume, conflicting record,
  missing merge ID, stage completion, and noncanonical state tests pass.

## Implementation step 11: E7 and paper regeneration

- Rationale: publication values must trace to locked records and reproduce.
- Protocol sections: 25, 30, E7.
- Files changed: `reproducibility.py`, E7 handler, `paper.py`,
  `scripts/regenerate_main_analysis.py`.
- Modification: first-25 recomputation namespace, exact top-16/prompt/output checks,
  score/metric tolerances, complete record-link audit, cost/storage/token/memory report,
  one-command H1--H4/TechQA analysis, three main tables, and three main figures.
- Validation: comparison tests and a synthetic render of all three tables and figures
  pass.

## Implementation step 12: coverage and commands

- Rationale: every frozen requirement must be mechanically discoverable.
- Protocol sections: all 1--33.
- Files changed: `coverage.py`, `validation.py`, `scripts/verify_main_study.py`,
  `scripts/prefetch_main_study.py`, Makefile, pyproject, 12 main-study test modules.
- Modification: protocol-to-code/test registry, one-command verification, offline cache
  prefetch, exact optional dependencies, additive Make targets.
- Validation: registry reports all 33 sections covered with no missing implementation or
  test file.

## Validation evidence

- Frozen protocol checksum: pass.
- JSON config and Colab notebook syntax: pass.
- Python compilation: pass.
- E0--E7 dry-run: pass.
- Work-item count: 1,307.
- Main-study synthetic tests: 40/40 pass.
- Entire repository regression suite: 71/71 pass using the existing Python 3.11 ML
  environment.
- Synthetic paper outputs: 3/3 tables and 3/3 figures rendered successfully.
- Protocol coverage: 33/33 sections.
- Model calls: zero.
- Dataset loads: zero external loads; legacy loader tests used only in-memory synthetic
  fixtures.

## Completion assessment

No unresolved architectural item remains. Phase 4 can begin from a clean committed
checkout by prefetching pinned artifacts, satisfying the exact environment/A100 gates,
and executing E0 in the frozen order. Any runtime contradiction must use the formal
amendment process; Phase 3 code does not contain a scientific fallback.
