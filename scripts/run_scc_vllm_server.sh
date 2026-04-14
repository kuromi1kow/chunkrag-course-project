#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$REPO_ROOT/outputs/openwebui_vllm}"
MODEL_NAME="${VLLM_MODEL_NAME:-mistralai/Mistral-7B-Instruct-v0.3}"
PORT="${VLLM_PORT:-8000}"
API_KEY="${VLLM_API_KEY:-chunkrag-demo-key}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"

mkdir -p "$OUTPUT_DIR"
source "$REPO_ROOT/scripts/setup_scc_vllm_env.sh"

HOSTNAME_VALUE="$(hostname)"
cat > "$OUTPUT_DIR/runtime.env" <<EOF
HOST=$HOSTNAME_VALUE
PORT=$PORT
MODEL_NAME=$MODEL_NAME
API_KEY=$API_KEY
BASE_URL=http://127.0.0.1:$PORT/v1
EOF

echo "Starting vLLM on $HOSTNAME_VALUE:$PORT for model $MODEL_NAME"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --tokenizer-mode mistral \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.90 \
  --api-key "$API_KEY"
