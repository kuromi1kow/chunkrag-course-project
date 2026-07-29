# Anonymous paper and reproducibility bundle

This directory contains the anonymous research paper, build inputs, reviewer-response
matrix, summary-level run artifacts, audits, exact experiment configurations,
and the code/tests needed to reproduce the reported pipeline. Run
`tectonic paper.tex` in this directory to rebuild the paper.

## IP&MC 2026 FIRM extension

The active venue target is the FIRM track at IP&MC 2026. The anonymous manuscript now
uses Elsevier's CAS single-column format and is titled *Ranking Before Generation:
Auditing Chunking, Retrieval Fusion, and Evidence Exposure in RAG*.

Run the missing paired hybrid-versus-reranking experiment in Colab:

[Open the FIRM reranking notebook in Colab](https://colab.research.google.com/github/kuromi1kow/chunkrag-course-project/blob/main/notebooks/colab_ipmc_firm_rerank.ipynb)

The notebook evaluates BGE hybrid retrieval with and without a revision-pinned
cross-encoder over 3 datasets, 3 seeds, and 4 chunkers. It writes 72 paired cells to
`MyDrive/chunkrag_outputs/ipmc_firm_rerank_bge`. After copying that directory to
`outputs/ipmc_firm_rerank_bge`, run:

```bash
make firm-rerank-analysis
make paper
```

The venue-specific readiness checklist is in
`reports/ipmc_firm_submission_checklist.md`.

## Included and intentionally omitted

- `paper.pdf`, `paper.tex`, `references.bib`, ACL styles, `generated/`, and
  `figures/` are the self-contained paper build.
- `artifacts/` contains final aggregate/summary results and reviewer-driven audits.
  Run manifests omit Git commit and worktree fields to preserve anonymity while
  retaining configuration hashes, source-tree hashes, package versions, and devices.
- `configs/`, `src/`, `scripts/`, `tests/`, `pyproject.toml`, and the exact pinned
  requirements support reproduction.
- Raw `*_predictions.json` files are intentionally excluded because they reproduce
  retrieved benchmark passages. Model weights, downloaded datasets, caches, smoke
  runs, cluster deployment files, historical midway reports, and build auxiliaries
  are also excluded. Audit JSON may retain question text and answer strings needed
  to substantiate the reported diagnostic analysis.

## Dataset provenance and licenses

No dataset is claimed as an original contribution of this project.

- SQuAD 2.0 (`3ffb306f725f7d2ce8394bc1873b24868140c412`): Stanford Question Answering Dataset;
  its dataset metadata lists CC BY-SA 4.0.
  <https://huggingface.co/datasets/rajpurkar/squad_v2>
- HotpotQA distractor (`1908d6afbbead072334abe2965f91bd2709910ab`): multi-hop Wikipedia QA;
  its dataset metadata lists CC BY-SA 4.0.
  <https://huggingface.co/datasets/hotpotqa/hotpot_qa>
- NVIDIA TechQA-RAG-Eval (`0b5bbc84b7f07d6d09d063130e90b716d8d4a32a`): a RAG-oriented distribution
  derived from IBM TechQA; its dataset card lists Apache-2.0.
  <https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval>

The bundle does not redistribute model weights. Model identifiers and immutable
revisions are recorded in `configs/`; downstream users must follow each model card's
license and access terms. The source repository did not declare a project-code license
at bundle-build time, so inclusion for anonymous review does not itself grant broader
reuse rights.

## Integrity and anonymity

`MANIFEST.sha256` records every other bundled file. The builder rejects incomplete
runs, inconsistent source/configuration hashes, non-anonymous TeX, identity strings,
absolute local paths, high-confidence secret patterns, raw predictions, model weights,
cache/build metadata, and unsanitized run manifests. ZIP entries use fixed timestamps,
permissions, ordering, and compression settings for deterministic output.
