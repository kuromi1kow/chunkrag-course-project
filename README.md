# Comparing Chunking Strategies for Retrieval-Augmented Question Answering

This repository contains our CS505 NLP course project on how chunking strategy affects retrieval-augmented question answering.

The project includes:

- chunking experiments on `SQuAD v2` and `HotpotQA`
- dense, BM25, hybrid, and reranked retrieval
- ACL-format midway and final reports
- a Streamlit dashboard for visual inspection
- a local OpenWebUI frontend
- an SCC-hosted `Mistral-7B` backend served through `vLLM`

## Current project status

The codebase supports two main usage modes:

1. Local experimentation and report building
2. Local UI + `SCC`-hosted `Mistral-7B-Instruct-v0.3` through an SSH tunnel

The current midpoint narrative uses:

- embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- retriever: FAISS dense retrieval
- generator: `mistralai/Mistral-7B-Instruct-v0.3` on `SCC`

## Requirements

- Python `3.11`
- macOS or Linux shell with `bash`
- `tectonic` if you want to rebuild the report PDFs
- Boston University `SCC` access if you want the remote `Mistral` backend

## Quick start

The easiest way to see available commands is:

```bash
make help
```

## Installation

### Core experiment environment

```bash
make setup
```

This creates `.venv` and installs the core project dependencies.

### Streamlit demo environment

```bash
make setup-demo
```

This creates `.venv_demo` and installs the dashboard dependencies.

### OpenWebUI environment

```bash
make setup-openwebui
```

This creates `.venv_openwebui` and installs `OpenWebUI`.

## Running the project locally

### 1. Streamlit dashboard

```bash
make dashboard
```

Default URL:

- [http://localhost:8501](http://localhost:8501)

The dashboard reads:

- results from `outputs/rigorous_smoke/aggregate_results.json`
- figures and artifacts from `outputs/rigorous_smoke`

You can override these when needed:

```bash
RESULTS_PATH=/absolute/path/to/aggregate_results.json OUTPUT_DIR=/absolute/path/to/output_dir make dashboard
```

### 2. OpenWebUI with no SCC backend

```bash
make openwebui
```

Default URL:

- [http://127.0.0.1:8080](http://127.0.0.1:8080)

This starts the local `OpenWebUI` instance only. If you want it connected to the remote `Mistral` model on `SCC`, use the SCC flow below instead.

## Running OpenWebUI with Mistral on SCC

Recommended architecture:

- `SCC` runs `vLLM`
- your laptop runs `OpenWebUI`
- an SSH tunnel connects `localhost:8000` to the live `SCC` model server

### Step 1. Submit the SCC vLLM job

Run this from an `SCC` login shell inside the project directory:

```bash
make scc-vllm
```

This submits a `Mistral-7B-Instruct-v0.3` `vLLM` server job using the default SCC settings.

Useful overrides:

```bash
SCC_QUEUE=l40s SCC_GPU_TYPE=L40S SCC_GPU_MEMORY=48G make scc-vllm
SCC_QUEUE=a100 SCC_GPU_TYPE=A100 SCC_GPU_MEMORY=80G make scc-vllm
SCC_QUEUE=h200 SCC_GPU_TYPE=H200 SCC_GPU_MEMORY=80G make scc-vllm
```

You can check job state on `SCC` with:

```bash
qstat -u <your_scc_username>
```

### Step 2. Open the SSH tunnel locally

Run this on your local machine after the SCC job is `running`:

```bash
make scc-tunnel
```

By default it reads runtime information from:

- `/projectnb/cs505am/projects/kuromiqo_chunkrag_project/outputs/openwebui_vllm/runtime.env`

You can override the remote root if needed:

```bash
REMOTE_ROOT=/projectnb/cs505am/projects/kuromiqo_chunkrag_project make scc-tunnel
```

When the tunnel is up, this should work locally:

```bash
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer chunkrag-demo-key"
```

### Step 3. Start OpenWebUI already pointed at the SCC model

```bash
make openwebui-scc
```

This starts `OpenWebUI` with:

- base URL: `http://127.0.0.1:8000/v1`
- API key: `chunkrag-demo-key`

Then open:

- [http://127.0.0.1:8080](http://127.0.0.1:8080)

If the model list looks stale, refresh the page once after the tunnel is live.

## Experiments

### Local quickstart experiment

```bash
make quickstart
```

### Local rigorous run

```bash
make rigorous-local
make plot-rigorous
```

### Optional Chonkie comparison

```bash
make chonkie
```

### SCC rigorous experiment sweep

Run from an `SCC` login shell:

```bash
make scc-rigorous
```

Default config:

- `configs/scc_rigorous_mistral.json`

Default output directory:

- `/projectnb/cs505am/projects/kuromiqo_chunkrag_project/outputs/scc_rigorous_mistral`

## Reports

### Regenerate tables and rebuild both PDFs

```bash
make reports
```

### Midway report only

```bash
make reports-midway
```

### Final report only

```bash
make reports-final
```

Generated report tables are written to:

- `reports/generated/midway_tables.tex`
- `reports/generated/chonkie_table.tex`
- `reports/generated/midway_tables.md`

## Validation

Run code compilation plus unit tests:

```bash
make test
```

## Main files and directories

- experiment code: `src/chunkrag`
- configs: `configs`
- scripts: `scripts`
- saved outputs: `outputs`
- ACL reports: `reports`
- Streamlit app: `apps/rag_demo_dashboard.py`

## Project deliverables

- midway report PDF: [`reports/midway_report.pdf`](/Users/assylkhan/Documents/NLP/reports/midway_report.pdf)
- final report PDF: [`reports/final_report.pdf`](/Users/assylkhan/Documents/NLP/reports/final_report.pdf)
- midway ACL source: [`reports/midway_report_acl.tex`](/Users/assylkhan/Documents/NLP/reports/midway_report_acl.tex)
- final ACL source: [`reports/final_report_acl.tex`](/Users/assylkhan/Documents/NLP/reports/final_report_acl.tex)
- bibliography: [`reports/references.bib`](/Users/assylkhan/Documents/NLP/reports/references.bib)
- SCC/OpenWebUI deployment notes: [`docs/openwebui_scc_deployment.md`](/Users/assylkhan/Documents/NLP/docs/openwebui_scc_deployment.md)
- roadmap: [`docs/rag_roadmap.md`](/Users/assylkhan/Documents/NLP/docs/rag_roadmap.md)
