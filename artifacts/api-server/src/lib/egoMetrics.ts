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
 *   Canonical per-completion dynamic cost proxy.
 *   For multi-replicate calls, wall-clock latency is allocated across replicates,
 *   and completion tokens are averaged per replicate:
 *   cost_dyn = ((prefill_ms + decode_ms) / n_replicates)
 *              + 0.1*prompt_tokens
 *              + 0.05*avg_completion_tokens
 *              + 2.0*peak_vram_gb
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

export interface TokenMetricsAggregateInput {
  mean_logprob?: number | null;
  mean_entropy?: number | null;
  mean_margin_top1_top2?: number | null;
  num_low_margin_steps?: number | null;
  num_high_entropy_steps?: number | null;
}

export interface EgoRunInput {
  prefill_ms?: number;
  decode_ms?: number;
  latency_ms: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  completion_tokens_per_replicate?: (number | null | undefined)[];
  peak_vram_gb?: number;
  texts: string[];
  substance_intensity: number;
  token_metrics_per_replicate?: (TokenMetricsAggregateInput | null)[];
}

export interface EgoMetricsResult {
  cost_dyn: number | null;
  cost_dyn_mean: number | null;
  cost_dyn_std: number | null;
  coherence_gate_pass_rate: number | null;
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

export interface ReplicateDiagnosticsResult {
  cost_dyn_replicates: (number | null)[];
  coherence_gate_passes: (boolean | null)[];
  cost_dyn_mean: number | null;
  cost_dyn_std: number | null;
  coherence_gate_pass_rate: number | null;
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

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function meanOrNull(values: Array<number | null | undefined>): number | null {
  const xs = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  if (xs.length === 0) return null;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function roundOrNull(value: number | null, decimals: number): number | null {
  if (value === null || !Number.isFinite(value)) return null;
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function stdOrNull(values: number[]): number | null {
  if (values.length === 0) return null;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function computePerCompletionCost(
  latencyPerCompletion: number,
  promptTokens: number | undefined,
  completionTokens: number | null | undefined,
  peakVramGb: number | undefined
): number {
  if (
    promptTokens !== undefined &&
    completionTokens !== undefined &&
    completionTokens !== null &&
    peakVramGb !== undefined
  ) {
    return (
      latencyPerCompletion +
      0.1 * promptTokens +
      0.05 * completionTokens +
      2.0 * peakVramGb
    );
  }
  return latencyPerCompletion;
}

function computeCoherenceGatePass(
  text: string,
  tokenMetrics: TokenMetricsAggregateInput | null | undefined
): boolean {
  const trimmed = text.trim();
  const wordCount = trimmed.split(/\s+/).filter((t) => t.length > 0).length;

  if (trimmed.length < 80) return false;
  if (wordCount < 12) return false;
  if (!/[.!?]/.test(trimmed)) return false;

  const lowMarginSteps = tokenMetrics?.num_low_margin_steps;
  const meanEntropy = tokenMetrics?.mean_entropy;

  if (typeof lowMarginSteps === "number" && lowMarginSteps > 25) return false;
  if (typeof meanEntropy === "number" && meanEntropy > 2.5) return false;

  return true;
}

function aggregateTokenMetrics(
  summaries: (TokenMetricsAggregateInput | null)[] | undefined
): TokenMetricsAggregateInput | null {
  if (!summaries || summaries.length === 0) return null;

  const mean_logprob = meanOrNull(summaries.map((s) => s?.mean_logprob));
  const mean_entropy = meanOrNull(summaries.map((s) => s?.mean_entropy));
  const mean_margin_top1_top2 = meanOrNull(
    summaries.map((s) => s?.mean_margin_top1_top2)
  );
  const num_low_margin_steps = meanOrNull(
    summaries.map((s) => s?.num_low_margin_steps)
  );
  const num_high_entropy_steps = meanOrNull(
    summaries.map((s) => s?.num_high_entropy_steps)
  );

  if (
    mean_logprob === null &&
    mean_entropy === null &&
    mean_margin_top1_top2 === null &&
    num_low_margin_steps === null &&
    num_high_entropy_steps === null
  ) {
    return null;
  }

  return {
    mean_logprob,
    mean_entropy,
    mean_margin_top1_top2,
    num_low_margin_steps,
    num_high_entropy_steps,
  };
}

export function computeReplicateDiagnostics(input: EgoRunInput): ReplicateDiagnosticsResult {
  const {
    prefill_ms,
    decode_ms,
    latency_ms,
    prompt_tokens,
    completion_tokens,
    completion_tokens_per_replicate,
    peak_vram_gb,
    texts,
    token_metrics_per_replicate,
  } = input;

  const nReplicates = Math.max(
    1,
    completion_tokens_per_replicate?.length ?? texts.length ?? 1
  );

  const prefillTotal = prefill_ms ?? latency_ms * 0.2;
  const decodeTotal = decode_ms ?? latency_ms * 0.8;
  const latencyPerCompletion = (prefillTotal + decodeTotal) / nReplicates;

  const localCompletionTokens =
    completion_tokens_per_replicate && completion_tokens_per_replicate.length > 0
      ? completion_tokens_per_replicate
      : Array.from({ length: texts.length }, () =>
          completion_tokens !== undefined ? completion_tokens / nReplicates : null
        );

  const cost_dyn_replicates = texts.map((_, i) =>
    roundOrNull(
      computePerCompletionCost(
        latencyPerCompletion,
        prompt_tokens,
        localCompletionTokens[i],
        peak_vram_gb
      ),
      2
    )
  );

  const validCosts = cost_dyn_replicates.filter(
    (v): v is number => typeof v === "number" && Number.isFinite(v)
  );

  const coherence_gate_passes = texts.map((text, i) =>
    computeCoherenceGatePass(text, token_metrics_per_replicate?.[i] ?? null)
  );

  const coherencePassValues = coherence_gate_passes.map((v) => (v ? 1 : 0));

  return {
    cost_dyn_replicates,
    coherence_gate_passes,
    cost_dyn_mean: roundOrNull(
      validCosts.length > 0
        ? validCosts.reduce((a, b) => a + b, 0) / validCosts.length
        : null,
      2
    ),
    cost_dyn_std: roundOrNull(stdOrNull(validCosts), 2),
    coherence_gate_pass_rate: roundOrNull(
      coherencePassValues.length > 0
        ? coherencePassValues.reduce((a, b) => a + b, 0) / coherencePassValues.length
        : null,
      3
    ),
  };
}

function computeCLEI(
  texts: string[],
  lmi: number | null,
  intensityNorm: number,
  aggregatedTokenMetrics: TokenMetricsAggregateInput | null
): number | null {
  if (texts.length === 0) return null;

  if (aggregatedTokenMetrics) {
    const entropyStress =
      aggregatedTokenMetrics.mean_entropy !== null &&
      aggregatedTokenMetrics.mean_entropy !== undefined
        ? clamp01(aggregatedTokenMetrics.mean_entropy / 2.5)
        : null;

    const lowConfidence =
      aggregatedTokenMetrics.mean_logprob !== null &&
      aggregatedTokenMetrics.mean_logprob !== undefined
        ? clamp01((-aggregatedTokenMetrics.mean_logprob) / 5)
        : null;

    const lowMargin =
      aggregatedTokenMetrics.mean_margin_top1_top2 !== null &&
      aggregatedTokenMetrics.mean_margin_top1_top2 !== undefined
        ? 1 - clamp01(aggregatedTokenMetrics.mean_margin_top1_top2)
        : null;

    const hasRealSignal =
      entropyStress !== null || lowConfidence !== null || lowMargin !== null;

    if (hasRealSignal) {
      const clei =
        0.45 * (entropyStress ?? 0) +
        0.2 * (lowConfidence ?? 0) +
        0.35 * (lowMargin ?? 0);
      return clamp01(clei);
    }
  }

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
  return clamp01(clei);
}

export function computeEgoMetrics(input: EgoRunInput): EgoMetricsResult {
  const {
    prefill_ms,
    decode_ms,
    latency_ms,
    prompt_tokens,
    completion_tokens,
    completion_tokens_per_replicate,
    peak_vram_gb,
    texts,
    substance_intensity,
    token_metrics_per_replicate,
  } = input;

  const replicateDiagnostics = computeReplicateDiagnostics({
    ...input,
    token_metrics_per_replicate,
    completion_tokens_per_replicate,
  });

  const cost_dyn = replicateDiagnostics.cost_dyn_mean;

  const lmi_empirical = computeLMI(texts);
  const r_eff_empirical = computeREffProxy(texts);
  const intensityNorm = substance_intensity / 4;
  const aggregatedTokenMetrics = aggregateTokenMetrics(token_metrics_per_replicate);
  const clei_llm = computeCLEI(
    texts,
    lmi_empirical,
    intensityNorm,
    aggregatedTokenMetrics
  );

  return {
    cost_dyn,
    cost_dyn_mean: replicateDiagnostics.cost_dyn_mean,
    cost_dyn_std: replicateDiagnostics.cost_dyn_std,
    coherence_gate_pass_rate: replicateDiagnostics.coherence_gate_pass_rate,
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
