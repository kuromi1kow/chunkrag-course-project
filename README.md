# ChunkRAG: literal-match reranker supervision

**Does literal-match reranker supervision improve support-answer generation? A controlled study with external transfer**

Assylkhan Geniyat, Zukhriddin Fakhriddinov and Atharv Gulati. Revision dated 16 October 2026. Preprint; not peer reviewed.

## Paper

| Material | Location |
| --- | --- |
| Main manuscript | [paper.pdf](paper.pdf) |
| Full supplement | [supplement.pdf](supplement.pdf) |
| Combined preprint with appendices | [paper/arxiv/main.pdf](paper/arxiv/main.pdf) |
| LaTeX entry points | [paper.tex](paper.tex), [supplement.tex](supplement.tex) |
| Editable sources, bibliography, tables and figures | [paper/source/](paper/source/) |
| Document build instructions | [paper/README.md](paper/README.md) |

With Tectonic and the required LaTeX packages available:

```bash
make paper
make supplement
```

To generate all document variants and the arXiv source archive in `build/paper/`:

```bash
make paper-all
```

Verify the checked-in files with `shasum -a 256 -c MANIFEST.sha256`.

## Code and results

The selected code and aggregate-results companion cited by the current paper is [chunkrag-reranking-study](https://github.com/zukhriddingit/chunkrag-reranking-study). The paper and supplement describe its release scope and replay limitations.

This repository also retains the earlier chunking and retrieval study:

- `src/`, `configs/`, `scripts/`, `tests/` and `notebooks/`: pipeline, experiment definitions and validation.
- `artifacts/` and `outputs/`: retained results and audit records.
- [reports/](reports/README.md): earlier reports and review documents.
- [examples/demo_results/](examples/README.md): the saved sample used by the dashboard.
- [docs/README_earlier_study.md](docs/README_earlier_study.md): archived project documentation.

System files, disposable smoke runs, duplicate documents and generated build files are excluded from the current tree. Earlier versions remain available in Git history.
