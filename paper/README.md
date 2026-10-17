# Manuscript sources

The current manuscript and supplement are [../paper.pdf](../paper.pdf) and [../supplement.pdf](../supplement.pdf). The [combined preprint](arxiv/main.pdf) includes both the paper and appendices.

`source/` contains the final LaTeX bodies, bibliography, number macros, tables and figures. The repository-root entry points `paper.tex` and `supplement.tex` load these sources. The PDFs and scientific source content are preserved from the supplied revision dated 16 October 2026.

From the repository root, with Tectonic and the required LaTeX packages available:

```bash
make paper
make supplement
```

For the named documents, combined preprint, arXiv source archive and anonymous journal candidates:

```bash
make paper-all
```

The latter command runs `paper/build_documents.py` and writes all generated copies, intermediate files and the compilation report under `build/paper/`, which Git ignores. Anonymous candidates and source ZIPs can be regenerated there as needed. The [arXiv metadata](arxiv/ARXIV_METADATA.json) records the prepared preprint information; it does not report a submission.

The selected code and aggregate-results companion is [chunkrag-reranking-study](https://github.com/zukhriddingit/chunkrag-reranking-study). See the paper and supplement for release scope and replay limitations.
