# EAAI Phase 2 execution guide

## Scientific freeze

- Baseline manifest: `reports/eaai_phase2_baseline_manifest.json`
- Baseline tree SHA-256: `7572ec911f852c9d420c6728f74fedbfbc88f2652e2df4b9b4963a363fda9ac6`
- Prospective protocol commit: `5df8cc7bf5f26a1b3cde37df2023cd6590352ad2`
- Scientific implementation commit: `ed4aabf2427015773442f01b084647697ce2a222`
- Frozen configuration SHA-256: `be12b0ca2a37713da0702a5d3991e1e38292bc3bd424ba3dc79ad22daf1fbe72`

Existing scientific artifacts remain immutable. New rows are written only below
`results/eaai_phase2/techqa_adaptive_v1/` and
`artifacts/eaai_phase2/techqa_adaptive_v1/`.

## Colab execution

Open `notebooks/eaai_phase2_colab.ipynb` in a GPU Colab runtime and run cells in
order. The notebook checks out the exact implementation commit, verifies the
private frozen-baseline archive, and checkpoints every new row to Google Drive.

The following private files must remain at
`MyDrive/chunkrag_outputs/eaai_phase2/`:

- `eaai_phase2_baseline_private.tar.gz`
- `eaai_phase2_baseline_private.tar.gz.sha256`

The archive SHA-256 is
`d1491fc016e8cf83771026aaf120e833af3e8278e2fc64abb8a89839d2c1f697`.
It contains controlled benchmark-derived inputs and must not be committed or
published.

The required order is:

1. Restore and verify the frozen baseline.
2. Materialize the deterministic development/test/reserve partition.
3. Run development retrieval and Qwen generation.
4. Fit and freeze the adaptive gate.
5. Run held-out retrieval and Qwen generation.
6. Run the single prespecified primary analysis.
7. Optionally run the Mistral replication without refitting the gate.

Do not inspect held-out outcomes before step 4. Interrupted retrieval and
generation stages are resumable because complete rows are content-hashed and
conflicting rows are rejected.

## Local validation

The following commands are inference-free:

```bash
make eaai-phase2-preflight
make eaai-phase2-test
make eaai-phase2-dry-run
```

The GPU/network commands are intentionally separate:

```bash
make eaai-phase2-qwen
make eaai-phase2-analysis
make eaai-phase2-mistral
```

The manuscript must not be updated with the adaptive method until the held-out
analysis is complete and reported without suppressing neutral or negative
results.
