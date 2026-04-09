# Comparing Chunking Strategies for Retrieval-Augmented Question Answering

This repository contains a reproducible NLP course project built from the proposal in `/Users/assylkhan/Downloads/proposal.pdf`.

The main study compares four chunking strategies in a retrieval-augmented QA pipeline:

- Fixed-size token chunking
- Recursive chunking
- Sentence-based chunking
- Semantic chunking

An auxiliary comparison also evaluates Chonkie's recursive and semantic chunkers against the in-repo implementations.

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

## Project deliverables

- Final report: [`reports/final_report.md`](/Users/assylkhan/Documents/NLP/reports/final_report.md)
- Midway report: [`reports/midway_report.md`](/Users/assylkhan/Documents/NLP/reports/midway_report.md)
- Reference list: [`reports/references.bib`](/Users/assylkhan/Documents/NLP/reports/references.bib)
- Final experiment run used in the report: [`outputs/report_run/all_results.json`](/Users/assylkhan/Documents/NLP/outputs/report_run/all_results.json)
- Auxiliary Chonkie comparison: [`outputs/chonkie_report_run/all_results.json`](/Users/assylkhan/Documents/NLP/outputs/chonkie_report_run/all_results.json)
