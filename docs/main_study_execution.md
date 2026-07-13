# Main-study execution guide

This guide implements Immutable Specification Sections 23--29. It is for the new
`chunkrag-main-v1` study only; legacy experiment commands are not interchangeable.

## Non-experimental verification

```bash
make main-study-verify
make main-study-test
make main-study-plan
```

These commands verify or dry-run only. They never download a dataset or load a model.

## Canonical execution prerequisites

1. Use Python 3.11 and install `requirements-main-study.lock`.
2. Cache every pinned model and dataset revision.
3. Use an NVIDIA A100 for canonical GPU stages.
4. Commit all implementation and ensure `git status --porcelain` is empty.
5. Verify the protocol SHA-256 is
   `567b652fc403e7ff7e00e349de86357f9a293cac77e7e7f4d3612284eb2c89bf`.

The runner rejects dirty Git state, mismatched dependencies, mixed protocol/config
hashes, noncanonical Colab GPUs, broken prerequisites, and missing work-item filters.

## Unified runner

```bash
PYTHONPATH=src python scripts/run_main_study.py \
  --experiment E0 \
  --mode run \
  --platform local \
  --dataset squad_v2
```

Modes are `run`, `dry-run`, `validation-only`, and `merge-only`. Every stage supports
dataset/condition/shard filtering. E1--E7 require their frozen prerequisite completion
IDs through `--completed`.

Example canonical generation shard:

```bash
PYTHONPATH=src python scripts/run_main_study.py \
  --experiment E2 --mode run --platform local \
  --completed E1 --dataset squad_v2 \
  --condition-id recursive192__matched-4096 --shard-index 0
```

## Checkpoints and merge

Each shard appends one canonical record per question to `part-NNN.jsonl.tmp`, updates a
hash-validated state file, and atomically renames only after completeness validation.
Resume repeats the identical command. Complete shards are immutable.

```bash
PYTHONPATH=src python scripts/run_main_study.py \
  --experiment E2 --mode merge-only --completed E1 \
  --dataset squad_v2 --condition-id recursive192__matched-4096 \
  --shard-dir artifacts/chunkrag-main-v1/generation/mistral/squad_v2/recursive192__matched-4096 \
  --merge-output artifacts/chunkrag-main-v1/generation/mistral/squad_v2/recursive192__matched-4096.jsonl
```

## Colab Pro

Use `notebooks/chunkrag_main_study_colab.ipynb`. It mounts Drive, checks the invocation
fields, verifies A100 hardware, checks Git/protocol/environment hashes, and delegates to
the same runner. T4/L4 sessions may dry-run or smoke-test but cannot emit canonical
artifacts.

Drive layout:

```text
MyDrive/chunkrag-main-v1/
  <git-commit>/<model>/<dataset>/<condition>/
    part-NNN.jsonl.tmp
    part-NNN.state.json
    part-NNN.jsonl
```

Credentials and user-specific Drive paths are never copied into released artifacts.

## Locked analysis regeneration

After E0--E7 complete, artifacts are read-only, and the completion manifest is frozen:

```bash
make main-study-analysis COMPLETION_MANIFEST=artifacts/chunkrag-main-v1/audit/completion.json
```

This is the only command authorized to execute the confirmatory analysis and regenerate
the frozen paper outputs.
