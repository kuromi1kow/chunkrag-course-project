#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

module load miniconda/25.3.1
module load academic-ml/spring-2026

source "$(conda info --base)/etc/profile.d/conda.sh"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
conda activate spring-2026-pyt

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE"

python -m pip install --user --upgrade pip
python -m pip install --user -e "${REPO_ROOT}[chonkie]"
