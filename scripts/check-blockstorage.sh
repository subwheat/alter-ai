#!/usr/bin/env bash
set -euo pipefail

MOUNT_TARGET="${1:-/mnt/blockstorage/alter-ai}"

REQUIRED_DIRS=(
  "models/google/gemma-4-E4B-it"
  "hf-home"
  "hf-cache"
  "logs"
  "conversations"
  "exports/datasets"
  "exports/training"
  "traces"
  "profiles/versions"
  "snapshots/config"
)

echo "[check-blockstorage] Mount target: $MOUNT_TARGET"
echo ""

# Check mount
if mountpoint -q "$MOUNT_TARGET" 2>/dev/null; then
  echo "[OK]  $MOUNT_TARGET is a mounted filesystem"
elif [ -d "$MOUNT_TARGET" ]; then
  echo "[WARN] $MOUNT_TARGET exists but is NOT a mountpoint (may be local disk)"
else
  echo "[FAIL] $MOUNT_TARGET does not exist"
  echo "       Create it and mount your block storage volume first."
  exit 1
fi

echo ""

# Check disk space
AVAILABLE_KB=$(df -k "$MOUNT_TARGET" 2>/dev/null | awk 'NR==2 {print $4}')
AVAILABLE_GB=$(echo "scale=1; $AVAILABLE_KB / 1048576" | bc 2>/dev/null || echo "unknown")
echo "[INFO] Available space: ${AVAILABLE_GB} GB"
if [ "$AVAILABLE_KB" != "" ] && [ "$AVAILABLE_KB" -lt 20971520 ]; then
  echo "[WARN] Less than 20 GB available. Gemma 4 E4B weights require ~10-20 GB."
fi

echo ""

# Check required directories
ALL_OK=true
for dir in "${REQUIRED_DIRS[@]}"; do
  full_path="$MOUNT_TARGET/$dir"
  if [ -d "$full_path" ]; then
    echo "[OK]  $full_path"
  else
    echo "[MISS] $full_path — will be created by storageService on first run"
    ALL_OK=false
  fi
done

echo ""

# Check model weights
MODEL_PATH="$MOUNT_TARGET/models/google/gemma-4-E4B-it"
if [ -f "$MODEL_PATH/config.json" ]; then
  echo "[OK]  Model config.json found at $MODEL_PATH"
  WEIGHT_FILES=$(find "$MODEL_PATH" -name "*.safetensors" -o -name "*.bin" 2>/dev/null | wc -l)
  echo "[INFO] Weight shard files found: $WEIGHT_FILES"
  if [ "$WEIGHT_FILES" -eq 0 ]; then
    echo "[WARN] No weight shards (.safetensors or .bin) found."
    echo "       Run: huggingface-cli download google/gemma-4-E4B-it --local-dir $MODEL_PATH --local-dir-use-symlinks False"
  fi
else
  echo "[MISS] $MODEL_PATH/config.json not found"
  echo "       Weights not yet downloaded. Run:"
  echo "       huggingface-cli download google/gemma-4-E4B-it \\"
  echo "         --local-dir $MODEL_PATH \\"
  echo "         --local-dir-use-symlinks False"
fi

echo ""

# Check write access
TEST_FILE="$MOUNT_TARGET/logs/.write_test_$$"
if touch "$TEST_FILE" 2>/dev/null; then
  rm -f "$TEST_FILE"
  echo "[OK]  Block storage is writable"
else
  echo "[FAIL] Block storage is NOT writable at $MOUNT_TARGET/logs/"
  echo "       Check permissions: chown -R \$(whoami) $MOUNT_TARGET"
fi

echo ""
echo "[check-blockstorage] Done."
