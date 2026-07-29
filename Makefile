SHELL := /bin/bash

ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3.11

REMOTE_ROOT ?= /projectnb/cs505am/projects/kuromiqo_chunkrag_project
RESULTS_PATH ?= $(ROOT_DIR)/outputs/rigorous_smoke/aggregate_results.json
OUTPUT_DIR ?= $(ROOT_DIR)/outputs/rigorous_smoke
FIRM_RERANK_OUTPUT ?= $(ROOT_DIR)/outputs/ipmc_firm_rerank_bge
OPENAI_TUNNEL_URL ?= http://127.0.0.1:8000/v1
OPENAI_TUNNEL_KEY ?= chunkrag-demo-key

.PHONY: help setup setup-demo setup-openwebui dashboard openwebui openwebui-scc quickstart rigorous-local plot-rigorous chonkie reports reports-midway reports-final paper paper-figures firm-rerank firm-rerank-analysis test main-study-verify main-study-test main-study-plan main-study-analysis scc-vllm scc-rigorous scc-tunnel

help:
	@echo "Available targets:"
	@echo "  make setup             - create .venv and install core dependencies"
	@echo "  make setup-demo        - create .venv_demo for Streamlit dashboard"
	@echo "  make setup-openwebui   - create .venv_openwebui for OpenWebUI"
	@echo "  make dashboard         - run Streamlit dashboard on localhost:8501"
	@echo "  make openwebui         - run local OpenWebUI on 127.0.0.1:8080"
	@echo "  make openwebui-scc     - run OpenWebUI pointed at localhost:8000/v1"
	@echo "  make quickstart        - run quickstart local experiment"
	@echo "  make rigorous-local    - run rigorous local experiment"
	@echo "  make plot-rigorous     - generate plots for rigorous local experiment"
	@echo "  make chonkie           - run optional Chonkie comparison"
	@echo "  make reports           - rebuild midway and final PDFs (regenerates figures + tables)"
	@echo "  make reports-midway    - rebuild midway PDF only"
	@echo "  make reports-final     - rebuild final PDF only"
	@echo "  make paper             - rebuild the anonymous Elsevier CAS manuscript"
	@echo "  make paper-figures     - rebuild reports/figures/*.pdf used by the final report"
	@echo "  make firm-rerank       - run paired BGE hybrid and cross-encoder reranking"
	@echo "  make firm-rerank-analysis - validate reranking artifacts and build its table"
	@echo "  make test              - compile code and run unit tests"
	@echo "  make main-study-verify - verify frozen protocol, config, lock, DAG, and output map"
	@echo "  make main-study-test   - run synthetic-only main-study validation tests"
	@echo "  make main-study-plan   - dry-run the complete E0 work plan (no dataset/model load)"
	@echo "  make main-study-analysis COMPLETION_MANIFEST=... - regenerate locked analyses"
	@echo "  make scc-vllm          - submit SCC vLLM job (run from SCC shell)"
	@echo "  make scc-rigorous      - submit SCC rigorous run (run from SCC shell)"
	@echo "  make scc-tunnel        - open local SSH tunnel to SCC vLLM endpoint"

setup:
	$(PYTHON) -m venv .venv
	source .venv/bin/activate && pip install --upgrade pip && pip install -e .

setup-demo:
	bash scripts/setup_demo_dashboard.sh

setup-openwebui:
	bash scripts/setup_openwebui_local.sh

dashboard:
	RESULTS_PATH="$(RESULTS_PATH)" OUTPUT_DIR="$(OUTPUT_DIR)" bash scripts/run_demo_dashboard.sh

openwebui:
	bash scripts/run_openwebui_local.sh

openwebui-scc:
	source .venv_openwebui/bin/activate && \
	DATA_DIR="$(ROOT_DIR)/.open-webui" \
	OPENAI_API_BASE_URL="$(OPENAI_TUNNEL_URL)" \
	OPENAI_API_KEY="$(OPENAI_TUNNEL_KEY)" \
	open-webui serve --host 127.0.0.1 --port 8080

quickstart:
	source .venv/bin/activate && python scripts/run_experiments.py --config configs/quickstart.json

rigorous-local:
	source .venv/bin/activate && python scripts/run_experiments.py --config configs/rigorous_local.json --output-dir outputs/rigorous_local

plot-rigorous:
	source .venv/bin/activate && python scripts/plot_results.py --results outputs/rigorous_local/aggregate_results.json --output-dir outputs/rigorous_local/figures

chonkie:
	source .venv/bin/activate && pip install -e .[chonkie] && python scripts/run_experiments.py --config configs/chonkie_comparison.json

reports:
	bash scripts/build_reports.sh

reports-midway:
	bash scripts/build_reports.sh midway

reports-final:
	bash scripts/build_reports.sh final

paper:
	tectonic paper.tex

paper-figures:
	if [ -x .venv/bin/python ]; then .venv/bin/python scripts/build_paper_figures.py; \
	elif [ -x .venv_figs/bin/python ]; then .venv_figs/bin/python scripts/build_paper_figures.py; \
	else $(PYTHON) scripts/build_paper_figures.py; fi

firm-rerank:
	source .venv/bin/activate && python scripts/run_experiments.py \
		--config configs/ipmc_firm_rerank_bge.json \
		--output-dir "$(FIRM_RERANK_OUTPUT)"

firm-rerank-analysis:
	$(PYTHON) scripts/analyze_ipmc_firm_rerank.py \
		--run-root "$(FIRM_RERANK_OUTPUT)"

test:
	$(PYTHON) -m compileall src scripts tests
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v

main-study-verify:
	PYTHONPATH=src $(PYTHON) scripts/verify_main_study.py

main-study-test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests/mainstudy -v

main-study-plan:
	PYTHONPATH=src $(PYTHON) scripts/run_main_study.py --experiment E0 --mode dry-run

main-study-analysis:
	test -n "$(COMPLETION_MANIFEST)"
	PYTHONPATH=src $(PYTHON) scripts/regenerate_main_analysis.py --completion-manifest "$(COMPLETION_MANIFEST)"

scc-vllm:
	bash scripts/submit_scc_vllm.sh "$(REMOTE_ROOT)/outputs/openwebui_vllm"

scc-rigorous:
	bash scripts/submit_scc_rigorous.sh configs/scc_rigorous_mistral.json "$(REMOTE_ROOT)/outputs/scc_rigorous_mistral"

scc-tunnel:
	REMOTE_ROOT="$(REMOTE_ROOT)" bash scripts/tunnel_scc_vllm.sh "$(REMOTE_ROOT)"
