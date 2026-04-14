#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSISTENT_VENV="${SCC_VLLM_VENV:-$REPO_ROOT/.venv_scc_vllm}"

module load miniconda/25.3.1
module load academic-ml/spring-2026

source "$(conda info --base)/etc/profile.d/conda.sh"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

if [[ -f "$PERSISTENT_VENV/bin/activate" ]]; then
  source "$PERSISTENT_VENV/bin/activate"
else
  conda activate spring-2026-pyt
fi

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE"

if ! python -c "import vllm" >/dev/null 2>&1; then
  python -m pip install --upgrade pip
  python -m pip install "vllm>=0.8,<1" "openai>=1,<2"
  python -m pip install "transformers==4.51.3" "tokenizers<0.22"
fi
