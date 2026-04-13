/**
 * EGO → LLM Bridge v0.1
 *
 * Computes and structures the EGO metrics for each LLM run.
 * Most real metrics require actual token logprob data from the model.
 * In demo mode, placeholders are used where the real sidecar is not available.
 *
 * Metric definitions:
 *
 * cost_dyn:
 *   A scalar representing the dynamic cost of the generation:
 *   cost_dyn = 1.0*prefill_ms + 1.0*decode_ms + 0.01*prompt_tokens
 *              + 0.02*completion_tokens + 10.0*peak_vram_gb
 *
 * lmi_empirical (Language Model Instability):
 *   Estimated from output diversity across N replicates.
 *   Uses simple Jaccard distance proxy on token sets when no embeddings available.
 *   Range: [0,1], higher = more instability.
 *
 * r_eff_empirical (Effective Rank):
 *   Proxy for covariance-based effective rank of output embeddings.
 *   Computed from spectral entropy of pairwise similarity matrix.
 *   Placeholder in demo mode.
 *
 * clei_llm (Cognitive Load / Ego Integrity index):
 *   Non-circular proxy combining:
 *   - output diversity (inter-run Jaccard)
 *   - mean response length stability
 *   - substance intensity as prior
 *   Range: [0,1], higher = lower coherence/stability.
 *
 * alpha_s_est: Placeholder — requires ablation context (not yet implemented)
 * recoverability_est: Placeholder — requires multi-step rollout (not yet implemented)
 */

export interface EgoRunInput {
  prefill_ms?: number;
  decode_ms?: number;
  latency_ms: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  peak_vram_gb?: number;
  texts: string[];
  substance_intensity: number;
}

export interface EgoMetricsResult {
  cost_dyn: number | null;
  lmi_empirical: number | null;
  r_eff_empirical: number | null;
  clei_llm: number | null;
  alpha_s_est: number | null;
  recoverability_est: number | null;
  latency_ms: number;
  completion_tokens: number | null;
  prompt_tokens: number | null;
  peak_vram_gb: number | null;
}

function tokenSet(text: string): Set<string> {
  return new Set(text.toLowerCase().split(/\W+/).filter((t) => t.length > 2));
}

function jaccardDistance(a: Set<string>, b: Set<string>): number {
  const intersection = [...a].filter((t) => b.has(t)).length;
  const union = new Set([...a, ...b]).size;
  if (union === 0) return 0;
  return 1 - intersection / union;
}

function computeLMI(texts: string[]): number | null {
  if (texts.length < 2) return null;
  const sets = texts.map(tokenSet);
  let totalDist = 0;
  let count = 0;
  for (let i = 0; i < sets.length; i++) {
    for (let j = i + 1; j < sets.length; j++) {
      totalDist += jaccardDistance(sets[i]!, sets[j]!);
      count++;
    }
  }
  return count > 0 ? totalDist / count : 0;
}

function computeREffProxy(texts: string[]): number | null {
  if (texts.length < 2) return null;
  const lengths = texts.map((t) => t.length);
  const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  const variance = lengths.reduce((a, b) => a + (b - mean) ** 2, 0) / lengths.length;
  const cv = mean > 0 ? Math.sqrt(variance) / mean : 0;
  return Math.min(cv * texts.length, texts.length);
}

function computeCLEI(
  texts: string[],
  lmi: number | null,
  intensityNorm: number
): number | null {
  if (texts.length === 0) return null;
  const lengths = texts.map((t) => t.length);
  const meanLen = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  const lenStability =
    meanLen > 0
      ? 1 -
        Math.min(
          1,
          lengths.reduce((a, b) => a + Math.abs(b - meanLen), 0) /
            (meanLen * lengths.length)
        )
      : 1;
  const lmiWeight = lmi ?? 0;
  const clei = 0.4 * lmiWeight + 0.3 * (1 - lenStability) + 0.3 * intensityNorm;
  return Math.min(1, Math.max(0, clei));
}

export function computeEgoMetrics(input: EgoRunInput): EgoMetricsResult {
  const {
    prefill_ms,
    decode_ms,
    latency_ms,
    prompt_tokens,
    completion_tokens,
    peak_vram_gb,
    texts,
    substance_intensity,
  } = input;

  const prefill = prefill_ms ?? latency_ms * 0.2;
  const decode = decode_ms ?? latency_ms * 0.8;

  const cost_dyn =
    prompt_tokens !== undefined &&
    completion_tokens !== undefined &&
    peak_vram_gb !== undefined
      ? 1.0 * prefill +
        1.0 * decode +
        0.01 * prompt_tokens +
        0.02 * completion_tokens +
        10.0 * peak_vram_gb
      : 1.0 * prefill + 1.0 * decode;

  const lmi_empirical = computeLMI(texts);
  const r_eff_empirical = computeREffProxy(texts);
  const intensityNorm = substance_intensity / 4;
  const clei_llm = computeCLEI(texts, lmi_empirical, intensityNorm);

  return {
    cost_dyn: Math.round(cost_dyn * 100) / 100,
    lmi_empirical: lmi_empirical !== null ? Math.round(lmi_empirical * 1000) / 1000 : null,
    r_eff_empirical:
      r_eff_empirical !== null ? Math.round(r_eff_empirical * 1000) / 1000 : null,
    clei_llm: clei_llm !== null ? Math.round(clei_llm * 1000) / 1000 : null,
    alpha_s_est: null,
    recoverability_est: null,
    latency_ms,
    completion_tokens: completion_tokens ?? null,
    prompt_tokens: prompt_tokens ?? null,
    peak_vram_gb: peak_vram_gb ?? null,
  };
}
