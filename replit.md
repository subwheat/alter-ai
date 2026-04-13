# alter.ai — LLM Cognition Simulator / Drug Lab

## Overview

alter.ai is an MVP "LLM Cognition Simulator" that lets users select a psychoactive substance and observe how a language model's output style shifts accordingly. It includes a real EGO → LLM bridge for structured logging and metrics.

## Architecture

pnpm workspace monorepo:
- `artifacts/alter-ai` — React + Vite frontend (IBM Plex Mono / Cormorant Garant, dark UI)
- `artifacts/api-server` — Express 5 + TypeScript API with drug endpoint and EGO bridge
- `sidecar/` — Python FastAPI sidecar for Gemma 4 local inference (GPU)
- `lib/api-spec/` — OpenAPI 3.1 spec (source of truth for API contract)
- `lib/api-client-react/` — Generated React Query hooks
- `lib/api-zod/` — Generated Zod validation schemas

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Validation**: Zod (`zod/v4`)
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)
- **Frontend**: React + Vite + Tailwind CSS
- **Python sidecar**: FastAPI + Uvicorn + HuggingFace Transformers + PyTorch

## Key Endpoints

- `POST /api/drug` — substance-modulated LLM generation (demo or sidecar mode)
- `GET /api/drug/substances` — list all 9 substances with sampling configs
- `GET /api/drug/health` — Gemma 4 sidecar availability
- `GET /api/healthz` — API health

## Substances

sober | caffeine | alcohol | cannabis | mdma | lsd | psilocybin | cocaine | ketamine

## EGO → LLM Bridge v0.1

Files:
- `artifacts/api-server/src/lib/egoLogger.ts` — append-only JSONL logging
- `artifacts/api-server/src/lib/egoMetrics.ts` — cost_dyn, lmi_empirical, r_eff_empirical, clei_llm
- `artifacts/api-server/src/lib/substances.ts` — substance → sampling config mapping
- `artifacts/api-server/src/lib/sidecarClient.ts` — HTTP client for Python sidecar
- `artifacts/api-server/src/lib/demoModel.ts` — fallback demo responses

## Python Sidecar

- `sidecar/main.py` — FastAPI sidecar, loads Gemma 4 from block storage
- `sidecar/requirements.txt` — Python dependencies
- Listens on port 8090, called by Node.js API when available

## Scripts

- `scripts/check-blockstorage.sh` — validate Scaleway block storage mount
- `scripts/start-api-server.sh` — launch Node.js API
- `scripts/start-gemma-sidecar.sh` — launch Python sidecar

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas
- `pnpm --filter @workspace/api-server run dev` — run API server locally
- `bash scripts/start-gemma-sidecar.sh` — start Python Gemma 4 sidecar (requires model weights)
- `bash scripts/check-blockstorage.sh` — check block storage mount (Scaleway)

## Deployment (Scaleway GPU)

See `DEPLOY_SCW.md` for full deployment guide.
Environment template: `.env.scaleway.example`

## Log locations

- Block storage (mounted): `/mnt/blockstorage/alter-ai/logs/chat_runs.jsonl`
- Fallback (dev): `./logs/chat_runs.jsonl`
- Per-token traces: `/mnt/blockstorage/alter-ai/traces/{run_id}.tokens.jsonl`

## Placeholder fields (v0.2)

- `alpha_s_est` — requires ablation context
- `recoverability_est` — requires multi-step rollout
- `first_token_ms` — requires streaming hooks
- `embedding_metrics.output_embedding_norm` — requires dedicated embedding forward pass
- `lmi_empirical` and `r_eff_empirical` — currently Jaccard/length proxies; replace with cosine on embeddings in v0.2
