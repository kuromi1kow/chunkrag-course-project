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
- OpenWebUI + SCC deployment scripts
- Streamlit demo dashboard for visual inspection

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

## OpenWebUI + SCC model serving

Recommended deployment:

- `SCC` hosts the model through `vLLM`
- `OpenWebUI` runs locally
- an SSH tunnel connects the UI to the SCC model server

### 1. Start the SCC model server

```bash
bash scripts/submit_scc_vllm.sh /projectnb/cs505am/projects/kuromiqo_chunkrag_project/outputs/openwebui_vllm
```

### 2. Open the tunnel locally

```bash
bash scripts/tunnel_scc_vllm.sh /projectnb/cs505am/projects/kuromiqo_chunkrag_project
```

### 3. Set up and run OpenWebUI locally

```bash
bash scripts/setup_openwebui_local.sh
bash scripts/run_openwebui_local.sh
```

Then in OpenWebUI add an OpenAI-compatible connection with:

- URL: `http://127.0.0.1:8000/v1`
- API key: `chunkrag-demo-key`
- Model: `Qwen/Qwen2.5-7B-Instruct`

## Visual dashboard

```bash
bash scripts/setup_demo_dashboard.sh
bash scripts/run_demo_dashboard.sh
```

This launches a Streamlit dashboard that shows:

- architecture choices
- aggregate metrics
- example-level predictions
- generated figures

## Project deliverables

- Final report: [`reports/final_report.md`](/Users/assylkhan/Documents/NLP/reports/final_report.md)
- Midway report: [`reports/midway_report.md`](/Users/assylkhan/Documents/NLP/reports/midway_report.md)
- Reference list: [`reports/references.bib`](/Users/assylkhan/Documents/NLP/reports/references.bib)
- OpenWebUI deployment guide: [`docs/openwebui_scc_deployment.md`](/Users/assylkhan/Documents/NLP/docs/openwebui_scc_deployment.md)
- RAG upgrade roadmap: [`docs/rag_roadmap.md`](/Users/assylkhan/Documents/NLP/docs/rag_roadmap.md)
- Final experiment run used in the report: [`outputs/report_run/all_results.json`](/Users/assylkhan/Documents/NLP/outputs/report_run/all_results.json)
- Auxiliary Chonkie comparison: [`outputs/chonkie_report_run/all_results.json`](/Users/assylkhan/Documents/NLP/outputs/chonkie_report_run/all_results.json)
- Rigorous smoke verification run: [`outputs/rigorous_smoke/aggregate_results.json`](/Users/assylkhan/Documents/NLP/outputs/rigorous_smoke/aggregate_results.json)
