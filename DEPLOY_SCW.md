# alter.ai v0.1 — Scaleway GPU Instance Deployment

## Prerequisites

- Scaleway GPU Instance (L4 or better recommended)
- Block storage volume attached and formatted
- Ubuntu 22.04 or Debian 12
- Node.js 20+ installed
- pnpm installed (`npm install -g pnpm`)
- Python 3.10+ with pip
- Git access to this repo

---

## 1. Mount block storage

```bash
# Identify your block device (usually /dev/sdb or /dev/vdb)
lsblk

# Format if new (skip if already formatted)
mkfs.ext4 /dev/sdb

# Mount
mkdir -p /mnt/blockstorage
mount /dev/sdb /mnt/blockstorage

# Persist across reboots — add to /etc/fstab
echo '/dev/sdb /mnt/blockstorage ext4 defaults,nofail 0 2' >> /etc/fstab
```

Verify:

```bash
bash scripts/check-blockstorage.sh
```

---

## 2. Create the directory layout

```bash
mkdir -p /mnt/blockstorage/alter-ai/models/google/gemma-4-E4B-it
mkdir -p /mnt/blockstorage/alter-ai/hf-home
mkdir -p /mnt/blockstorage/alter-ai/hf-cache
mkdir -p /mnt/blockstorage/alter-ai/logs
mkdir -p /mnt/blockstorage/alter-ai/conversations
mkdir -p /mnt/blockstorage/alter-ai/exports/datasets
mkdir -p /mnt/blockstorage/alter-ai/exports/training
mkdir -p /mnt/blockstorage/alter-ai/traces
mkdir -p /mnt/blockstorage/alter-ai/profiles/versions
mkdir -p /mnt/blockstorage/alter-ai/snapshots/config
```

---

## 3. Download Gemma 4 E4B weights

Install the Hugging Face CLI:

```bash
pip install huggingface_hub
```

Authenticate (requires HF account with access to google/gemma-4-E4B-it):

```bash
huggingface-cli login
```

Download weights to block storage:

```bash
export HF_HOME=/mnt/blockstorage/alter-ai/hf-home
export HF_HUB_CACHE=/mnt/blockstorage/alter-ai/hf-cache

huggingface-cli download google/gemma-4-E4B-it \
  --local-dir /mnt/blockstorage/alter-ai/models/google/gemma-4-E4B-it \
  --local-dir-use-symlinks False
```

This will take time and significant disk space (~10–20 GB). Verify:

```bash
ls /mnt/blockstorage/alter-ai/models/google/gemma-4-E4B-it/
```

You should see `config.json`, `tokenizer.json`, and model shard files.

---

## 4. Set environment variables

```bash
cp .env.scaleway.example .env
# Edit .env as needed (model path, device, etc.)
set -a && source .env && set +a
```

---

## 5. Install dependencies

```bash
# Node.js dependencies
pnpm install

# Python sidecar dependencies
pip install -r sidecar/requirements.txt
```

---

## 6. Start the Gemma 4 sidecar (GPU process)

```bash
bash scripts/start-gemma-sidecar.sh
```

Test it:

```bash
curl http://localhost:8090/health
# Expected: {"available":true,"model":"gemma-4-E4B-it","device":"cuda:0","error":null}
```

Leave this running in a separate terminal or manage with pm2/systemd.

---

## 7. Start the Node.js API server

```bash
bash scripts/start-api-server.sh
```

Verify:

```bash
curl http://localhost:8080/api/healthz
curl http://localhost:8080/api/drug/health
curl http://localhost:8080/api/drug/substances
```

---

## 8. Test the drug endpoint

```bash
# Test sober baseline
curl -s -X POST http://localhost:8080/api/drug \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is consciousness?","substance":"sober"}' | jq .

# Test LSD multi-run (replicate_count=3)
curl -s -X POST http://localhost:8080/api/drug \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is the nature of time?","substance":"lsd","replicate_count":3}' | jq .

# Test ketamine
curl -s -X POST http://localhost:8080/api/drug \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Who are you?","substance":"ketamine"}' | jq .
```

---

## 9. Build and serve the frontend

```bash
pnpm --filter @workspace/alter-ai run build
# Then serve with nginx pointing to artifacts/alter-ai/dist/public
```

---

## 10. Process management (production)

```bash
# pm2 — sidecar
pm2 start bash --name alter-ai-sidecar -- scripts/start-gemma-sidecar.sh
# pm2 — API server
pm2 start bash --name alter-ai-api -- scripts/start-api-server.sh

pm2 save
pm2 startup
```

---

## Substance → sampling mapping summary

| Substance  | Temp | top_p | top_k | rep_penalty | max_tokens |
|------------|------|-------|-------|-------------|------------|
| sober      | 0.70 | 0.90  | 50    | 1.10        | 512        |
| caffeine   | 0.85 | 0.92  | 60    | 1.15        | 600        |
| alcohol    | 1.10 | 0.95  | 80    | 1.00        | 550        |
| cannabis   | 1.05 | 0.97  | 100   | 1.05        | 600        |
| mdma       | 1.15 | 0.97  | 120   | 0.95        | 650        |
| lsd        | 1.35 | 0.99  | off   | 0.90        | 700        |
| psilocybin | 1.20 | 0.98  | 110   | 0.98        | 700        |
| cocaine    | 1.25 | 0.95  | 70    | 1.20        | 750        |
| ketamine   | 1.40 | 0.99  | off   | 1.30        | 400        |

`top_k=0` means disabled (nucleus sampling only via top_p).

---

## EGO → LLM Bridge v0.1 log fields

Logs are written to `ALTER_AI_LOGS_DIR/chat_runs.jsonl` (default `/mnt/blockstorage/alter-ai/logs/chat_runs.jsonl`).

Per-token traces are written to `/mnt/blockstorage/alter-ai/traces/{run_id}.tokens.jsonl`.

See `artifacts/api-server/src/lib/egoLogger.ts` for the full schema.

---

## File paths reference

| Purpose | Path |
|---|---|
| Model weights | `/mnt/blockstorage/alter-ai/models/google/gemma-4-E4B-it/` |
| HF home | `/mnt/blockstorage/alter-ai/hf-home/` |
| HF cache | `/mnt/blockstorage/alter-ai/hf-cache/` |
| Chat run logs | `/mnt/blockstorage/alter-ai/logs/chat_runs.jsonl` |
| Per-token traces | `/mnt/blockstorage/alter-ai/traces/{run_id}.tokens.jsonl` |
| Conversations | `/mnt/blockstorage/alter-ai/conversations/<id>.json` |
| Exports | `/mnt/blockstorage/alter-ai/exports/` |

---

## Current limitations (v0.1)

- `alpha_s_est` and `recoverability_est` are null placeholders — require multi-step rollout or ablation context (planned for v0.2)
- `first_token_ms` requires streaming hooks (planned for v0.2)
- `embedding_metrics.output_embedding_norm` requires dedicated embedding layer forward pass (planned for v0.2)
- `lmi_empirical` uses Jaccard token-set distance (fast proxy) — replace with embedding cosine distance in v0.2
- `r_eff_empirical` uses length variance proxy — replace with PCA/spectral entropy on embeddings in v0.2
- The frontend currently uses demo mode responses when no Gemma sidecar is running. Deploy the sidecar to activate real inference.

## Next recommended step

Wire real token-level logprobs from the sidecar into `token_metrics` fields, then implement `lmi_empirical` via embedding cosine distance (use a lightweight sentence-transformers model already on the block storage).
