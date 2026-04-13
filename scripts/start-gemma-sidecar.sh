#!/usr/bin/env bash
set -euo pipefail

# alter.ai v0.1 — Gemma 4 inference sidecar launcher
#
# This script starts the Python FastAPI sidecar that loads Gemma 4 locally
# and exposes http://localhost:8090/generate for the Node.js API server.
#
# Prerequisites:
#   pip install -r sidecar/requirements.txt
#   HF weights downloaded to ALTER_AI_MODEL_PATH
#
# Integration: artifacts/api-server/src/lib/sidecarClient.ts calls this sidecar.
# The Node.js API auto-detects sidecar availability and falls back to demo mode.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

: "${ALTER_AI_MODEL_PATH:?ALTER_AI_MODEL_PATH must be set}"
: "${ALTER_AI_DEVICE:=auto}"
: "${HF_HOME:=/mnt/blockstorage/alter-ai/hf-home}"
: "${HF_HUB_CACHE:=/mnt/blockstorage/alter-ai/hf-cache}"
: "${SIDECAR_PORT:=8090}"
: "${ALTER_AI_DTYPE:=bfloat16}"
: "${ALTER_AI_LOGS_DIR:=/mnt/blockstorage/alter-ai/logs}"

echo "[alter-ai] Sidecar configuration:"
echo "  Model path:  $ALTER_AI_MODEL_PATH"
echo "  Device:      $ALTER_AI_DEVICE"
echo "  dtype:       $ALTER_AI_DTYPE"
echo "  HF home:     $HF_HOME"
echo "  HF cache:    $HF_HUB_CACHE"
echo "  Port:        $SIDECAR_PORT"
echo "  Logs dir:    $ALTER_AI_LOGS_DIR"
echo ""

if [ ! -d "$ALTER_AI_MODEL_PATH" ]; then
  echo "[alter-ai] ERROR: Model path does not exist: $ALTER_AI_MODEL_PATH"
  echo "[alter-ai] Download weights first:"
  echo "  huggingface-cli download google/gemma-4-E4B-it \\"
  echo "    --local-dir $ALTER_AI_MODEL_PATH \\"
  echo "    --local-dir-use-symlinks False"
  exit 1
fi

if [ ! -f "$ALTER_AI_MODEL_PATH/config.json" ]; then
  echo "[alter-ai] ERROR: config.json not found. Weights download may be incomplete."
  exit 1
fi

echo "[alter-ai] Starting Gemma 4 sidecar..."
cd "$REPO_ROOT/sidecar"
exec uvicorn main:app \
  --host 0.0.0.0 \
  --port "$SIDECAR_PORT" \
  --log-level info
