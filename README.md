# Comparing Chunking Strategies for Retrieval-Augmented Question Answering

This repository contains a reproducible NLP course project built from the proposal in `/Users/assylkhan/Downloads/proposal.pdf`.

The main study compares four chunking strategies in a retrieval-augmented QA pipeline:

- Fixed-size token chunking
- Recursive chunking
- Sentence-based chunking
- Semantic chunking

An auxiliary comparison also evaluates Chonkie's recursive and semantic chunkers against the in-repo implementations.

The upgraded experiment runner now supports:

- Multi-seed sweeps with aggregated summaries
- Dense, BM25, hybrid, and reranked retrieval
- Bootstrap confidence intervals for headline metrics
- Hotpot supporting-document coverage metrics
- Figure generation from result JSONs

The default experimental setup uses:

- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- `google/flan-t5-base` for generation
- `squad_v2` and `hotpot_qa` from Hugging Face
- FAISS for dense retrieval

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python scripts/run_experiments.py --config configs/quickstart.json
```

Outputs are written under `outputs/`.

## Optional Chonkie comparison

```bash
source .venv/bin/activate
pip install -e .[chonkie]
python scripts/run_experiments.py --config configs/chonkie_comparison.json
```

## Rigorous local sweep

```bash
source .venv/bin/activate
pip install -e .
python scripts/run_experiments.py --config configs/rigorous_local.json --output-dir outputs/rigorous_local
python scripts/plot_results.py --results outputs/rigorous_local/aggregate_results.json --output-dir outputs/rigorous_local/figures
```

## SCC sweep

The SCC path uses BU's `qsub` scheduler and the `academic-ml/spring-2026` environment.

```bash
bash scripts/submit_scc_rigorous.sh configs/scc_rigorous_qwen.json /projectnb/cs505am/projects/kuromiqo_chunkrag_project/outputs/scc_rigorous_qwen
```

Useful overrides:

- `SCC_QUEUE=academic-gpu`
- `SCC_GPU_TYPE=A100`
- `SCC_GPU_MEMORY=80G`
- `SCC_WALLTIME=24:00:00`
- `SCC_THREADS=8`

## Project deliverables

- Final report: [`reports/final_report.md`](/Users/assylkhan/Documents/NLP/reports/final_report.md)
- Midway report: [`reports/midway_report.md`](/Users/assylkhan/Documents/NLP/reports/midway_report.md)
- Reference list: [`reports/references.bib`](/Users/assylkhan/Documents/NLP/reports/references.bib)
- Final experiment run used in the report: [`outputs/report_run/all_results.json`](/Users/assylkhan/Documents/NLP/outputs/report_run/all_results.json)
- Auxiliary Chonkie comparison: [`outputs/chonkie_report_run/all_results.json`](/Users/assylkhan/Documents/NLP/outputs/chonkie_report_run/all_results.json)
- Rigorous smoke verification run: [`outputs/rigorous_smoke/aggregate_results.json`](/Users/assylkhan/Documents/NLP/outputs/rigorous_smoke/aggregate_results.json)
