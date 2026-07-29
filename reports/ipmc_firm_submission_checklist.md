# IP&MC 2026 FIRM submission checklist

Status checked: 30 July 2026.

## Portal and venue

- [x] Official conference homepage says the call is open through 31 July 2026.
- [x] Official FIRM track page says paper and poster manuscripts are invited through
  31 July 2026.
- [x] Oxford Abstracts stage `80479` is live and redirects to authentication.
- [ ] Sign in and confirm that `Create submission` is visible:
  <https://app.oxfordabstracts.com/auth?redirect=/stages/80479/submitter>
- [ ] Select submission type `Full Paper`.
- [ ] Select category `FIRM: Foundation-model Integrity and Ranking Methods in the LLM Era`.
- [ ] Confirm that every author can attend and present in person in Wuhan if accepted.
- [ ] Confirm the time zone shown by Oxford Abstracts for the 31 July deadline.

The conference pages previously disagreed between 30 June and 31 July. The current
conference and track pages show 31 July, and a July call points to the same live Oxford
Abstracts stage. Authentication is still required to verify the final create-submission
screen.

## Scientific readiness

- [x] Dense retrieval baseline: MiniLM and BGE.
- [x] Lexical baseline: BM25.
- [x] Hybrid baseline: weighted reciprocal-rank fusion of dense and BM25 ranks.
- [x] Cross-encoder reranker implementation with a pinned model revision.
- [x] Run the 72-cell paired BGE hybrid versus hybrid-rerank Colab experiment.
- [x] Copy the completed reranking artifacts from Drive into the repository.
- [x] Regenerate the retrieval table with hybrid versus hybrid-rerank quality and latency.
- [x] Report AllHit@4, AnsVis@4, and the mean retrieval-latency ratio; DocCov@4 and
  indexed chunk counts remain in the complete retrieval appendix/artifacts.
- [x] Add paired per-question hybrid-versus-rerank analysis on each dataset.
- [x] Add reranking failure cases: document hit without answer visibility, incomplete
  multi-hop support, and answer-bearing evidence dropped by reranking.
- [x] Keep claims conditional: do not call an observed winner statistically superior
  unless the planned paired test supports it.

## Manuscript positioning

- [x] Working title: *Ranking Before Generation: Auditing Chunking, Retrieval Fusion,
  and Evidence Exposure in RAG*.
- [x] Abstract reframed around multi-stage ranking, integrity, evidence exposure,
  quality, and efficiency.
- [x] Replace the ACL front matter with Elsevier's CAS single-column format.
- [x] Keep the review manuscript anonymized.
- [ ] Prepare a separate title page with author names, affiliations, acknowledgements,
  declarations of interest, corresponding-author postal address, email, and phone.
- [x] Keep the abstract at no more than 250 words.
- [x] Provide 1--7 English keywords.
- [ ] Add a glossary for field-specific terms if required by the final form.
- [ ] Add CRediT author contributions.
- [ ] Add funding and competing-interest declarations.
- [x] Add the required generative-AI-use declaration before the references.
- [ ] Remove or anonymize repository URLs and identity-bearing artifact paths for review.

## Evidence and reproducibility

- [x] Verify every number in the abstract against generated artifacts.
- [x] Verify the new BM25, RRF, and cross-encoder citations against DOI or arXiv;
  complete one final citation audit for the pre-existing bibliography.
- [x] Ensure all reranking model and dataset revisions are immutable and reported in
  the retained config and run manifest.
- [x] Preserve run manifests, exact configs, package versions, hardware, seeds, raw
  predictions, and latency traces.
- [ ] Verify every remaining cited work against DOI, ACL Anthology, arXiv, or the official model
  and dataset page.
- [ ] Build the anonymous supplementary ZIP and run its integrity/anonymity audit.
- [ ] Confirm the public repository has an explicit software license before promising
  reuse rights.
- [ ] Confirm no manuscript is under review elsewhere.

## Submission package

- [ ] Anonymous manuscript PDF.
- [ ] Editable LaTeX source, bibliography, class/style files, tables, and figures.
- [ ] Separate title page.
- [ ] Portal abstract of no more than 250 words.
- [ ] Cover letter naming IP&MC 2026 and the FIRM track.
- [ ] Supplementary reproducibility archive.
- [ ] Figure files supplied separately with readable labels and embedded fonts.
- [ ] Final spell, grammar, citation, anonymity, and PDF rendering checks.
- [ ] All authors approve the submitted version and author order.

## Completed reranking run

The T4 Colab run completed in 22 minutes on commit `d2bbd3d` and wrote:

`MyDrive/chunkrag_outputs/ipmc_firm_rerank_bge`

The retained manifest reports `status=complete`, CUDA, a clean source tree, and exactly
72 paired summary cells. Local analysis is regenerated with:

```bash
make firm-rerank-analysis
make paper
```
