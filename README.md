# ChunkRAG: literal-match reranker supervision

**Does literal-match reranker supervision improve support-answer generation? A controlled study with external transfer**

Assylkhan Geniyat, Zukhriddin Fakhriddinov and Atharv Gulati. Revision dated 16 October 2026. Preprint; not peer reviewed.

## Paper and sources

| Material | File |
| --- | --- |
| Current main manuscript | [paper.pdf](paper.pdf) |
| Main LaTeX entry point | [paper.tex](paper.tex) |
| Full supplement | [supplement.pdf](supplement.pdf) |
| Supplement LaTeX entry point | [supplement.tex](supplement.tex) |
| Combined preprint with appendices | [arxiv/main.pdf](author_review_package/arxiv/main.pdf) |
| Preprint LaTeX source archive | [arxiv_source.zip](author_review_package/arxiv/arxiv_source.zip) |
| Editable bodies, preamble, tables and figures | [author_review_package/source/](author_review_package/source/) |
| Anonymous document candidates and build details | [author_review_package/README.md](author_review_package/README.md) |

The root manuscript and supplement PDFs are exact copies of the final supplied revision. The root LaTeX entry points load the final sources in `author_review_package/source/`. The bibliography is also available as [references.bib](references.bib).

## Build and integrity checks

With Tectonic and the required LaTeX packages available, run from the repository root:

```bash
make paper
make supplement
```

To build all five document variants inside `author_review_package/`, including the combined preprint and anonymous candidates:

```bash
python3 author_review_package/build_documents.py
```

To verify the checked-in files:

```bash
shasum -a 256 -c MANIFEST.sha256
```

The supplied final documents also retain their original [checksums](author_review_package/SHA256SUMS_FINAL.txt), which can be checked from `author_review_package/`.

## Research code and reproducibility

The selected code and aggregate-results companion cited by the current paper is [chunkrag-reranking-study](https://github.com/zukhriddingit/chunkrag-reranking-study). The paper and supplement describe its release scope and remaining replay limitations.

This repository also retains the code, configurations, notebooks, aggregate results and reports from the earlier chunking and retrieval study. Their original documentation is preserved in [docs/README_earlier_study.md](docs/README_earlier_study.md); it describes that earlier study and its build and experiment workflows. The earlier root paper remains available in Git history.
