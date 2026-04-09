#!/bin/bash
set -euo pipefail

REMOTE_ROOT="${1:-/projectnb/cs505am/projects/kuromiqo_chunkrag_project}"
RUNTIME_PATH="$REMOTE_ROOT/outputs/openwebui_vllm/runtime.env"
LOCAL_PORT="${LOCAL_PORT:-8000}"
REMOTE_USER="${REMOTE_USER:-kuromiqo}"
LOGIN_HOST="${LOGIN_HOST:-scc1.bu.edu}"

HOST="$(ssh "$REMOTE_USER@$LOGIN_HOST" "source '$RUNTIME_PATH' && printf '%s' \"\$HOST\"")"
PORT="$(ssh "$REMOTE_USER@$LOGIN_HOST" "source '$RUNTIME_PATH' && printf '%s' \"\$PORT\"")"

echo "Forwarding http://127.0.0.1:$LOCAL_PORT -> $HOST:$PORT via $LOGIN_HOST"
ssh -N -L "$LOCAL_PORT:$HOST:$PORT" "$REMOTE_USER@$LOGIN_HOST"
