#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment variables if .env exists
if [ -f "$REPO_ROOT/.env" ]; then
  echo "[alter-ai] Loading environment from $REPO_ROOT/.env"
  set -a
  source "$REPO_ROOT/.env"
  set +a
else
  echo "[alter-ai] WARNING: No .env file found at $REPO_ROOT/.env"
  echo "[alter-ai] Proceeding with existing environment variables."
fi

# Required variables check
: "${ALTER_AI_PORT:?ALTER_AI_PORT must be set}"
: "${ALTER_AI_MODEL_ID:?ALTER_AI_MODEL_ID must be set}"

echo "[alter-ai] Building API server..."
pnpm --filter @workspace/api-server run build

echo "[alter-ai] Starting API server on port $ALTER_AI_PORT..."
PORT="$ALTER_AI_PORT" \
  node --enable-source-maps "$REPO_ROOT/artifacts/api-server/dist/index.mjs"
