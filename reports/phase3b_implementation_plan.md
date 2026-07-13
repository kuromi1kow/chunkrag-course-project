# Phase 3B Implementation Plan

Protocol: `chunkrag-main-v1`  
Scope: repository implementation only; no dataset materialization or model execution.  
Governing document: `reports/phase2_immutable_specification.md`.

## Architecture decision

Implement a new package at `src/chunkrag/mainstudy/`. The package is the sole execution
path for E0--E7. Legacy code remains unchanged and supports only archived appendix work.
The new runner uses lazy external-library imports so planning, schema validation, merge,
and hash verification require only the Python standard library plus NumPy where
statistics are explicitly invoked.

Artifact root is exactly Section 23's
`artifacts/chunkrag-main-v1/{manifests,chunks,retrieval,generation,evaluation,analysis,audit}`.

## Task 1: protocol authority and frozen configuration

- Protocol sections: 1, 7, 11--17, 20, 23--29.
- Files affected: none in legacy package.
- New files: `protocol.py`, `constants.py`, `configs/main_study.json`.
- Functions: protocol checksum verification, typed configuration loading, immutable
  condition/stage registry, dependency/version checks.
- Deleted files: none.
- Migration: main-study commands reject legacy configs.
- Validation: checksum, exact model/dataset revisions, exact condition counts, E0--E7
  DAG, and cross-field invariants.
- Estimated effort: medium.

## Task 2: canonical serialization and hash chain

- Protocol sections: 23--25.
- New files: `canonical.py`, `artifacts.py`.
- Functions: NFC canonical JSON, canonical JSONL ordering, file/tree/snapshot/source
  hashes, atomic immutable writes, record IDs, upstream-chain verification.
- Deleted files: none.
- Migration: legacy artifacts are never silently upgraded; only schema version
  `chunkrag-main-v1` is accepted.
- Validation: golden hashes, order invariance, corruption rejection, duplicate rejection,
  immutable overwrite rejection, end-to-end synthetic chain.
- Estimated effort: high.

## Task 3: schemas

- Protocol sections: 10, 11.5, 23.
- New files: `schemas.py`.
- Functions: strict validators for run, dataset, question, corpus, cluster, gold, chunk,
  retrieval, generation, evaluation, human-label, checkpoint, and analysis records.
- Deleted files: none.
- Migration: legacy dataclasses remain; main study accepts mappings validated by schema
  name and version.
- Validation: required/unknown fields, numeric domains, enum values, hash formats,
  record-ID recomputation, nested candidate lengths.
- Estimated effort: high.

## Task 4: deterministic runtime and logging

- Protocol sections: 25, 27.
- New files: `determinism.py`, `logging.py`, `environment.py`.
- Functions: seed derivation, canonical Torch settings, clean-Git gate, environment-lock
  hash, hardware/runtime capture, structured JSONL logs.
- Deleted files: none.
- Migration: no reuse of `pipeline.runtime_manifest`.
- Validation: stable seed vectors, dirty-tree rejection, protocol-authorized Phase 3
  development override restricted to dry-run/tests, log hash validation.
- Estimated effort: medium.

## Task 5: E0 materialization

- Protocol sections: 8--10, 17, E0.
- New files: `data.py`, `gold.py`.
- Functions: pinned snapshot loading, NFC/CRLF normalization, SQuAD span remapping,
  Hotpot sentence provenance, TechQA conflict checks, SHA selection/caps, connected
  components, question/corpus/cluster/gold manifests, outcome-free sensitivity plan.
- Deleted files: none.
- Migration: new IDs intentionally differ from legacy IDs.
- Validation: byte-identical second materialization hook, sample counts/caps, exact span
  round trips, gold mapping, cluster constraints, snapshot hashes.
- Estimated effort: high.

## Task 6: exact chunking and randomized controls

- Protocol sections: 11--12, E1.
- New files: `chunking.py`, `controls.py`.
- Functions: offset-preserving fixed/recursive/sentence/semantic 192 policies, separator
  rules, pseudo-sentence handling, chunk IDs, jitter intervals/clamping, coverage and
  changed-boundary validation.
- Deleted files: none.
- Migration: legacy overlap chunkers remain isolated.
- Validation: synthetic tokenizer fixtures, Unicode whitespace, long sentences, short
  finals, exact source reconstruction, deterministic controls, infeasible-boundary
  accounting.
- Estimated effort: high.

## Task 7: retrieval and packing

- Protocol sections: 13, 16, 18.3--18.5, E1, E5.
- New files: `retrieval.py`, `packing.py`.
- Functions: lazy dense/BM25/RRF/cross-encoder execution, score-complete traces, fixed
  tie rules, document deduplication, metric derivation, operational/matched/Qwen/gold
  pack plans and token audits.
- Deleted files: none.
- Migration: retrievers consume immutable chunk manifests and write immutable traces;
  they never call generation.
- Validation: deterministic fake encoders/rerankers, RRF golden example, top-50/top-16,
  no encoder truncation, top-4 and matched-prefix behavior, two-token tolerance.
- Estimated effort: high.

## Task 8: generation, judge, and evaluation traces

- Protocol sections: 14--15, 18--22, E2--E4, E6.
- New files: `prompts.py`, `generation.py`, `evaluation.py`, `human.py`.
- Functions: exact prompt rendering, pinned local model adapter, resolved generation
  config validation, per-record attempts, normalization, SQuAD/Hotpot metrics, evidence
  intervals, TechQA judge JSON parsing/retry/fallback, blinded annotation package.
- Deleted files: none.
- Migration: endpoint inference is forbidden by the new adapter.
- Validation: prompt goldens, mock tokenizer/model traces, retry/failure rules, judge
  schema, blindness scan, deterministic candidate order.
- Estimated effort: high.

## Task 9: statistical and analysis engine

- Protocol sections: 8.5, 20, 30--32.
- New files: `statistics.py`, `analysis.py`, `paper.py`.
- Functions: derived PCG64 seeds, cluster bootstrap, sign flips, Holm, CR1 regression,
  TOST, rank-biserial, Cliff's delta, judge acceptance, fixed table/figure data
  builders, outcome inspection gate.
- Deleted files: none.
- Migration: legacy post-hoc scripts are excluded.
- Validation: synthetic known-effect fixtures, exact enumeration, Monte Carlo
  reproducibility, multiplicity golden cases, table/figure assignment coverage.
- Estimated effort: high.

## Task 10: experiment DAG and unified runner

- Protocol sections: E0--E7 and 29.
- New files: `experiments.py`, `runner.py`, `scripts/run_main_study.py`.
- Functions: one CLI supporting run, dry-run, resume, shard, local, Colab,
  validation-only, and merge-only; stage dependency gates; expected-count planners;
  outcome lock.
- Deleted files: none.
- Migration: the main runner has no import or dispatch path to legacy pipeline.
- Validation: synthetic stage handlers, invalid-order rejection, dry-run with no heavy
  imports, exact expected counts, mode matrix.
- Estimated effort: high.

## Task 11: checkpointing, merge, and Colab

- Protocol sections: 28--29.
- New files: `checkpoint.py`, `colab.py`, `notebooks/chunkrag_main_study_colab.ipynb`,
  `docs/main_study_execution.md`.
- Functions: 50-question shards, append-only per-question records, ten-record sync
  marker, atomic `.tmp` finalization, state hashes, canonical A100/environment checks,
  Drive path planner, duplicate/missing/mixed merge rejection.
- Deleted files: none.
- Migration: no credentials or Drive user path enter artifacts.
- Validation: interruption/resume simulation, tampering tests, partial-final-shard tests,
  merge order invariance, non-A100 rejection.
- Estimated effort: high.

## Task 12: validation and reproducibility commands

- Protocol sections: 24--25, E7.
- New files: `validation.py`, `coverage.py`, `scripts/verify_main_study.py`,
  `scripts/regenerate_main_analysis.py`, `tests/mainstudy/**`.
- Modified files: `Makefile`, `pyproject.toml`, `.gitignore` only where required.
- Functions: one-command full verification, one-command analysis regeneration,
  protocol-to-code coverage matrix, synthetic audit subset, artifact read-only lock.
- Deleted files: none.
- Migration: default legacy tests remain; new test target is additive.
- Validation: compile, unittest discovery, CLI smoke tests, checksum verification,
  no-experiment guard.
- Estimated effort: high.

## Task 13: exact environment lock

- Protocol section: 27.
- New files: `requirements-main-study.lock` and environment metadata template.
- Modified files: `pyproject.toml` main-study optional dependency group.
- Functions: lock verifier; no environment installation during Phase 3.
- Deleted files: none.
- Migration: reviewer robustness lock remains archived.
- Validation: all 17 direct versions match protocol and lock hash is included in dry-run
  plan.
- Estimated effort: low.

## Task 14: final protocol coverage audit

- Protocol sections: all, especially 30 and E0--E7.
- New file: `reports/phase3h_final_audit.md`.
- Functions: machine-readable coverage registry maps every protocol requirement to code
  and tests.
- Deleted files: none.
- Migration: unresolved items block Phase 3 completion.
- Validation: zero missing coverage entries; all non-experimental tests pass; no model
  output, dataset manifest, result statistic, table, or figure is generated.
- Estimated effort: medium.

## Implementation order

1. Authority/config/hash/schemas.
2. Artifact store/checkpoints/environment.
3. E0 data and gold interfaces.
4. E1 chunk/control/retrieval interfaces.
5. Packing/generation/evaluation/human interfaces.
6. Statistics/analysis/paper builders.
7. E0--E7 DAG and runner.
8. Colab/merge.
9. Tests and one-command commands.
10. Coverage audit, intentional commit, and push.

## Definition of done

Phase 3 is complete only if a clean checkout can:

- verify the frozen protocol and environment lock;
- dry-run E0--E7 without importing or executing models;
- execute any planned shard through one runner when Phase 4 begins;
- resume and merge synthetic shards deterministically;
- validate every artifact and upstream hash;
- regenerate analysis/table/figure artifacts from locked synthetic result fixtures;
- pass every protocol-compliance test;
- prove that no Phase 3 command executed a real experiment.
