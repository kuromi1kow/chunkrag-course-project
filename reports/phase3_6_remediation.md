# Phase 3.6 P0 remediation record

Protocol authority: `reports/phase2_immutable_specification.md`, SHA-256
`567b652fc403e7ff7e00e349de86357f9a293cac77e7e7f4d3612284eb2c89bf`.

No dataset, model, smoke, or canonical experiment was executed in this phase.

| P0 class | Root cause | Canonical-path remediation | Regression coverage |
|---|---|---|---|
| Immutable configuration | Partial field checks allowed protocol drift | Require the canonical hash of the complete parsed configuration | `test_phase36_unit.py` mutation subtests |
| Determinism and hardware | Determinism helper and A100 gate were not on the runner path | Configure Python/NumPy/Torch before environment/model loading; require Python 3.11 and A100/CUDA for E1--E7 | Unit runtime rejection and mocked runner integration |
| Environment lock | Direct requirements were mistaken for a complete lock | Track and verify the exact transitive distribution set; record Python, CUDA, cuDNN, node, driver, and GPUs | Adversarial package-drift test and repository verifier |
| Source hash | Recursive filesystem hashing admitted ignored bytecode | Hash only `git ls-files` under the frozen source paths and tracked resolved lock | Temporary-Git adversarial test with ignored `pyc` |
| Record identities | Retrieval used protocol name; prompts were not hashed; generation lacked a self hash | Use protocol SHA, hash exact prompt templates, recompute all identifiers in E7, and store/validate generation payload hashes | Unit identity tests plus E7 formula validation |
| Context packing | Character binary search assumed monotonic BPE counts | Descend native tokenizer prefix boundaries and validate actual complete-chat counts; count passage text only | Original non-monotonic-tokenizer regression |
| Checkpointing and merge | Data/state update gap was unrecoverable; final shards and merges trusted stale provenance | Reconcile individually valid fsynced rows, trim partial tails, validate final state, reject mixed/stale merges, and merge E2/E3 before stage completion | Crash, truncated-tail, final-state, missing-ID, and mixed-environment tests |
| Completion and locking | Status booleans and bare hashes could be forged | Store path/hash/size references, recompute every work/stage marker, inventory artifacts, lock raw results, and validate the canonical completion path | Completion unit tests and fabricated/writable-manifest adversarial tests |
| Analysis locking | User-authored manifests and alternate outputs could open analysis repeatedly | Require validated E0--E7 registry, clean matching Git/source, locked inventory, canonical output path, and one immutable analysis lock | Adversarial gate tests |
| E4 blinding order | Judge shards preceded the human package and labels | Order package first and require its work marker plus both labels and adjudication before judge inference | Plan unit test and two handler integration tests |
| Required outputs | Raw traces alone could complete E1/E3/E5/E6 | Materialize retrieval/evidence metrics, encoder and cost audits, gold gaps, human summaries, E5 paired effects, E6 intervals, visibility, equivalence, and symmetry diagnostics | Synthetic output integration test and E7 required-path audit |
| E7 audit | Repository-shape checks were labeled full validation | Validate schemas, counts, IDs, model snapshots, hash links, merged traces, operational audits, 25-question reruns, aggregate equality, and E0--E7 completion before locking | Artifact-audit fail-closed checks and full protocol suite |

## Test classes

- Unit: `tests/mainstudy/test_phase36_unit.py`
- Integration: `tests/mainstudy/test_phase36_integration.py`
- Regression: `tests/mainstudy/test_phase36_regressions.py`
- Adversarial: `tests/mainstudy/test_phase36_adversarial.py`

The older main-study tests remain active and are run together with these tests.
