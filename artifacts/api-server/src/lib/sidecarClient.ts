/**
 * HTTP client for the Python Gemma 4 inference sidecar.
 *
 * The sidecar runs at GEMMA_SIDECAR_URL (default http://localhost:8090).
 * All calls time out after GEMMA_SIDECAR_TIMEOUT_MS (default 120 000 ms).
 *
 * When the sidecar is unreachable or times out, functions throw so the
 * caller can fall back to demo mode.
 */

const SIDECAR_URL =
  process.env["GEMMA_SIDECAR_URL"] ?? "http://localhost:8090";
const TIMEOUT_MS = Number(
  process.env["GEMMA_SIDECAR_TIMEOUT_MS"] ?? "120000"
);

async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutMs: number
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(timer);
  }
}

export interface SidecarGenerateRequest {
  prompt: string;
  substance: string;
  sampling_config: {
    temperature: number;
    top_p: number;
    top_k: number;
    repetition_penalty: number;
    max_new_tokens: number;
    do_sample: boolean;
  };
  system_prompt: string;
  messages?: { role: string; content: string }[];
  replicate_count?: number;
}

export interface TokenMetricsSummary {
  mean_logprob: number | null;
  mean_entropy: number | null;
  mean_margin_top1_top2: number | null;
  entropy_std: number | null;
  margin_std: number | null;
  num_low_margin_steps: number | null;
  num_high_entropy_steps: number | null;
}

export interface SidecarGenerateResponse {
  texts: string[];
  prompt_tokens: number | null;
  completion_tokens: number[];
  latency_ms: number;
  prefill_ms: number | null;
  decode_ms: number | null;
  first_token_ms: number | null;
  tokens_per_second: number | null;
  peak_vram_gb: number | null;
  finish_reasons: string[];
  model_name: string;
  model_path: string;
  dtype: string;
  device: string;
  token_metrics_per_replicate: TokenMetricsSummary[];
}

export interface SidecarHealthResponse {
  available: boolean;
  model: string | null;
  device: string | null;
  error: string | null;
}

export async function sidecarGenerate(
  req: SidecarGenerateRequest
): Promise<SidecarGenerateResponse> {
  const res = await fetchWithTimeout(
    `${SIDECAR_URL}/generate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    },
    TIMEOUT_MS
  );
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Sidecar error ${res.status}: ${body}`);
  }
  return res.json() as Promise<SidecarGenerateResponse>;
}

export async function sidecarHealth(): Promise<SidecarHealthResponse> {
  const res = await fetchWithTimeout(
    `${SIDECAR_URL}/health`,
    { method: "GET" },
    5000
  );
  if (!res.ok) {
    return { available: false, model: null, device: null, error: `HTTP ${res.status}` };
  }
  return res.json() as Promise<SidecarHealthResponse>;
}

export async function isSidecarAvailable(): Promise<boolean> {
  try {
    const h = await sidecarHealth();
    return h.available;
  } catch {
    return false;
  }
}
