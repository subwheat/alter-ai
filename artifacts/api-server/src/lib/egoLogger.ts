/**
 * EGO → LLM Run Logger v0.1
 *
 * Appends structured JSONL log entries to the block storage logs directory.
 * Falls back gracefully when the block storage is not mounted.
 *
 * Log targets:
 *   Primary:  ALTER_AI_LOGS_DIR (default /mnt/blockstorage/alter-ai/logs)
 *   Fallback: ./logs/ in the repo root (dev mode)
 *
 * Files written:
 *   chat_runs.jsonl — one entry per LLM generation call
 */

import fs from "fs";
import path from "path";
import crypto from "crypto";

const BLOCK_LOGS =
  process.env["ALTER_AI_LOGS_DIR"] ?? "/mnt/blockstorage/alter-ai/logs";
const FALLBACK_LOGS = path.resolve(process.cwd(), "logs");

function resolveLogsDir(): string | null {
  if (fs.existsSync(BLOCK_LOGS)) {
    try {
      fs.accessSync(BLOCK_LOGS, fs.constants.W_OK);
      return BLOCK_LOGS;
    } catch {
      // not writable
    }
  }
  try {
    fs.mkdirSync(FALLBACK_LOGS, { recursive: true });
    return FALLBACK_LOGS;
  } catch {
    return null;
  }
}

export interface EgoRunLog {
  schema_version: "ego-llm-v0.1";
  run_id: string;
  timestamp_utc: string;
  experiment_id: string | null;
  batch_id: string | null;
  replicate_index: number;

  model_name: string;
  model_path: string | null;
  dtype: string | null;
  device: string | null;

  prompt_text: string;
  prompt_hash: string;
  prompt_tokens: number | null;
  context_total_tokens: number | null;

  perturbation: {
    family: string;
    label: string;
    parameters: Record<string, unknown>;
  };

  completion_text: string;
  completion_hash: string;
  completion_tokens: number | null;
  finish_reason: string | null;

  latency_ms: number;
  prefill_ms: number | null;
  decode_ms: number | null;
  first_token_ms: number | null;
  tokens_per_second: number | null;
  peak_vram_gb: number | null;

  token_metrics: {
    mean_logprob: number | null;
    mean_entropy: number | null;
    mean_margin_top1_top2: number | null;
    entropy_std: number | null;
    margin_std: number | null;
    num_low_margin_steps: number | null;
    num_high_entropy_steps: number | null;
  };

  embedding_metrics: {
    output_embedding_norm: number | null;
  };

  ego_metrics: {
    cost_dyn: number | null;
    cost_dyn_replicate: number | null;
    cost_dyn_mean: number | null;
    cost_dyn_std: number | null;
    coherence_gate_pass: boolean | null;
    coherence_gate_pass_rate: number | null;
    lmi_empirical: number | null;
    r_eff_empirical: number | null;
    clei_llm: number | null;
    alpha_s_est: number | null;
    recoverability_est: number | null;
  };

  mode: "sidecar" | "demo";
}

export function sha256(text: string): string {
  return crypto.createHash("sha256").update(text).digest("hex").slice(0, 16);
}

export function appendRunLog(entry: EgoRunLog): void {
  const logsDir = resolveLogsDir();
  if (!logsDir) return;
  const logFile = path.join(logsDir, "chat_runs.jsonl");
  try {
    fs.appendFileSync(logFile, JSON.stringify(entry) + "\n", "utf8");
  } catch {
    // silent — logging must never crash the API
  }
}

export function buildRunLog(params: {
  run_id: string;
  replicate_index: number;
  model_name: string;
  model_path: string | null;
  dtype: string | null;
  device: string | null;
  prompt_text: string;
  prompt_tokens: number | null;
  context_total_tokens: number | null;
  substance_id: string;
  substance_label: string;
  substance_family: string;
  sampling_config: Record<string, unknown>;
  completion_text: string;
  completion_tokens: number | null;
  finish_reason: string | null;
  latency_ms: number;
  prefill_ms: number | null;
  decode_ms: number | null;
  first_token_ms: number | null;
  tokens_per_second: number | null;
  peak_vram_gb: number | null;
  token_metrics?: {
    mean_logprob: number | null;
    mean_entropy: number | null;
    mean_margin_top1_top2: number | null;
    entropy_std: number | null;
    margin_std: number | null;
    num_low_margin_steps: number | null;
    num_high_entropy_steps: number | null;
  };
  ego_metrics: {
    cost_dyn: number | null;
    lmi_empirical: number | null;
    r_eff_empirical: number | null;
    clei_llm: number | null;
    alpha_s_est: number | null;
    recoverability_est: number | null;
  };
  mode: "sidecar" | "demo";
}): EgoRunLog {
  return {
    schema_version: "ego-llm-v0.1",
    run_id: params.run_id,
    timestamp_utc: new Date().toISOString(),
    experiment_id: null,
    batch_id: null,
    replicate_index: params.replicate_index,

    model_name: params.model_name,
    model_path: params.model_path,
    dtype: params.dtype,
    device: params.device,

    prompt_text: params.prompt_text,
    prompt_hash: sha256(params.prompt_text),
    prompt_tokens: params.prompt_tokens,
    context_total_tokens: params.context_total_tokens,

    perturbation: {
      family: params.substance_family,
      label: params.substance_label,
      parameters: params.sampling_config,
    },

    completion_text: params.completion_text,
    completion_hash: sha256(params.completion_text),
    completion_tokens: params.completion_tokens,
    finish_reason: params.finish_reason,

    latency_ms: params.latency_ms,
    prefill_ms: params.prefill_ms,
    decode_ms: params.decode_ms,
    first_token_ms: params.first_token_ms,
    tokens_per_second: params.tokens_per_second,
    peak_vram_gb: params.peak_vram_gb,

    token_metrics: {
      mean_logprob: params.token_metrics?.mean_logprob ?? null,
      mean_entropy: params.token_metrics?.mean_entropy ?? null,
      mean_margin_top1_top2: params.token_metrics?.mean_margin_top1_top2 ?? null,
      entropy_std: params.token_metrics?.entropy_std ?? null,
      margin_std: params.token_metrics?.margin_std ?? null,
      num_low_margin_steps: params.token_metrics?.num_low_margin_steps ?? null,
      num_high_entropy_steps: params.token_metrics?.num_high_entropy_steps ?? null,
    },

    embedding_metrics: {
      output_embedding_norm: null,
    },

    ego_metrics: params.ego_metrics,
    mode: params.mode,
  };
}
