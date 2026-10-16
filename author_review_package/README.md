# Literal-match reranker supervision: October 2026 manuscript

**Does literal-match reranker supervision improve support-answer generation? A controlled study with external transfer**

Assylkhan Geniyat, Zukhriddin Fakhriddinov and Atharv Gulati

Revision dated 16 October 2026. Preprint; not peer reviewed.

| Material | File |
| --- | --- |
| Main manuscript | [manuscript.pdf](manuscript.pdf) |
| Full supplement | [supplement.pdf](supplement.pdf) |
| Combined preprint with appendices | [arxiv/main.pdf](arxiv/main.pdf) |
| Preprint LaTeX source archive | [arxiv/arxiv_source.zip](arxiv/arxiv_source.zip) |
| Editable LaTeX sources, tables and figures | [source/](source/) |
| Anonymous manuscript candidate | [journal_revision/manuscript_anonymous.pdf](journal_revision/manuscript_anonymous.pdf) |
| Anonymous supplement candidate | [journal_revision/supplement_anonymous.pdf](journal_revision/supplement_anonymous.pdf) |
| PDF compilation report | [build/compilation_report.json](build/compilation_report.json) |
| Checksums of the supplied final files | [SHA256SUMS_FINAL.txt](SHA256SUMS_FINAL.txt) |

The PDFs, LaTeX sources and build script are copied unchanged from the final local revision. The selected code and aggregate-results companion cited by the manuscript is [chunkrag-reranking-study](https://github.com/zukhriddingit/chunkrag-reranking-study); see the paper and supplement for its scope and replay limitations.

To verify the supplied final files, run from this directory:

```bash
shasum -a 256 -c SHA256SUMS_FINAL.txt
```

To rebuild the documents with Tectonic and the required LaTeX packages available:

```bash
python3 build_documents.py
```

The script builds the named manuscript and supplement, the combined preprint, and the anonymous journal candidates. The metadata and anonymous PDFs are prepared candidates; inclusion here does not report an arXiv or journal submission.
