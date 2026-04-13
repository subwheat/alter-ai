import { Router, type IRouter } from "express";
import crypto from "crypto";
import { getSubstance, listSubstances } from "../lib/substances.js";
import { buildDemoResponse } from "../lib/demoModel.js";
import { sidecarGenerate, sidecarHealth, isSidecarAvailable, type TokenMetricsSummary } from "../lib/sidecarClient.js";
import { computeEgoMetrics } from "../lib/egoMetrics.js";
import { appendRunLog, buildRunLog } from "../lib/egoLogger.js";

const router: IRouter = Router();

router.post("/drug", async (req, res) => {
  const { prompt, substance: substanceId, replicate_count = 1, messages = [] } = req.body ?? {};

  if (!prompt || typeof prompt !== "string") {
    res.status(400).json({ error: "prompt is required and must be a string" });
    return;
  }

  const substanceKey = typeof substanceId === "string" ? substanceId : "sober";
  const substance = getSubstance(substanceKey);

  const runId = crypto.randomUUID();
  const replicates = Math.min(Math.max(1, Number(replicate_count) || 1), 8);

  let texts: string[];
  let promptTokens: number | null = null;
  let completionTokens: number[] = [];
  let latencyMs: number;
  let prefillMs: number | null = null;
  let decodeMs: number | null = null;
  let firstTokenMs: number | null = null;
  let tokensPerSecond: number | null = null;
  let peakVramGb: number | null = null;
  let mode: "sidecar" | "demo" = "demo";
  let modelName = "demo";
  let modelPath: string | null = null;
  let dtype: string | null = null;
  let device: string | null = null;
  let finishReasons: string[] = [];
  let tokenMetricsPerReplicate: (TokenMetricsSummary | null)[] = [];

  const available = await isSidecarAvailable();

  if (available) {
    try {
      const sidecarRes = await sidecarGenerate({
        prompt,
        substance: substance.id,
        sampling_config: substance.sampling_config,
        system_prompt: substance.system_prompt,
        messages,
        replicate_count: replicates,
      });

      texts = sidecarRes.texts;
      promptTokens = sidecarRes.prompt_tokens;
      completionTokens = sidecarRes.completion_tokens;
      latencyMs = sidecarRes.latency_ms;
      prefillMs = sidecarRes.prefill_ms;
      decodeMs = sidecarRes.decode_ms;
      firstTokenMs = sidecarRes.first_token_ms;
      tokensPerSecond = sidecarRes.tokens_per_second;
      peakVramGb = sidecarRes.peak_vram_gb;
      modelName = sidecarRes.model_name;
      modelPath = sidecarRes.model_path;
      dtype = sidecarRes.dtype;
      device = sidecarRes.device;
      finishReasons = sidecarRes.finish_reasons;
      tokenMetricsPerReplicate = sidecarRes.token_metrics_per_replicate ?? [];
      mode = "sidecar";
    } catch (err) {
      req.log.warn({ err }, "Sidecar call failed, falling back to demo");
      const demo = buildDemoResponse(substance, replicates);
      texts = demo.texts;
      promptTokens = demo.prompt_tokens;
      completionTokens = demo.completion_tokens;
      latencyMs = demo.latency_ms;
      finishReasons = texts.map(() => "stop");
    }
  } else {
    const demo = buildDemoResponse(substance, replicates);
    texts = demo.texts;
    promptTokens = demo.prompt_tokens;
    completionTokens = demo.completion_tokens;
    latencyMs = demo.latency_ms;
    finishReasons = texts.map(() => "stop");
  }

  const totalCompletionTokens = completionTokens.reduce((a, b) => a + b, 0);

  const egoMetrics = computeEgoMetrics({
    prefill_ms: prefillMs ?? undefined,
    decode_ms: decodeMs ?? undefined,
    latency_ms: latencyMs,
    prompt_tokens: promptTokens ?? undefined,
    completion_tokens: totalCompletionTokens,
    peak_vram_gb: peakVramGb ?? undefined,
    texts,
    substance_intensity: substance.intensity,
  });

  for (let i = 0; i < texts.length; i++) {
    const logEntry = buildRunLog({
      run_id: runId,
      replicate_index: i,
      model_name: modelName,
      model_path: modelPath,
      dtype,
      device,
      prompt_text: prompt,
      prompt_tokens: promptTokens,
      context_total_tokens: promptTokens,
      substance_id: substance.id,
      substance_label: substance.label,
      substance_family: substance.family,
      sampling_config: substance.sampling_config as unknown as Record<string, unknown>,
      completion_text: texts[i] ?? "",
      completion_tokens: completionTokens[i] ?? null,
      finish_reason: finishReasons[i] ?? null,
      latency_ms: latencyMs,
      prefill_ms: prefillMs,
      decode_ms: decodeMs,
      first_token_ms: firstTokenMs,
      tokens_per_second: tokensPerSecond,
      peak_vram_gb: peakVramGb,
      token_metrics: tokenMetricsPerReplicate[i] ?? undefined,
      ego_metrics: egoMetrics,
      mode,
    });
    appendRunLog(logEntry);
  }

  // Build response-level token_metrics: aggregate across all replicates that have data.
  // When sidecar is available, each replicate has its own summary; we surface the array.
  // In demo mode the array is empty (all nulls in the log, not exposed in response).
  const responseTokenMetrics =
    tokenMetricsPerReplicate.length > 0 ? tokenMetricsPerReplicate : null;

  res.json({
    run_id: runId,
    mode,
    substance: substance.id,
    texts,
    sampling_config: substance.sampling_config,
    ego_metrics: egoMetrics,
    token_metrics: responseTokenMetrics,
  });
});

router.get("/drug/substances", (_req, res) => {
  res.json({
    substances: listSubstances().map((s) => ({
      id: s.id,
      label: s.label,
      family: s.family,
      intensity: s.intensity,
      sampling_config: s.sampling_config,
    })),
  });
});

router.get("/drug/health", async (_req, res) => {
  try {
    const h = await sidecarHealth();
    res.json(h);
  } catch (err) {
    res.json({
      available: false,
      model: null,
      device: null,
      error: err instanceof Error ? err.message : "sidecar unreachable",
    });
  }
});

export default router;
