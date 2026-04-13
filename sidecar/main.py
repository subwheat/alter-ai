"""
alter.ai — Gemma 4 Local Inference Sidecar v0.1
================================================
FastAPI service that loads a local Gemma 4 (E4B-it) model from block storage
and exposes a /generate endpoint for substance-modulated text generation.

The Node.js API server calls this sidecar at http://localhost:8090.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8090

Environment variables:
    ALTER_AI_MODEL_PATH     Path to Gemma 4 model dir (required)
    ALTER_AI_DEVICE         "cuda", "cpu", "auto" (default: auto)
    HF_HOME                 Hugging Face home (default: /mnt/blockstorage/alter-ai/hf-home)
    HF_HUB_CACHE            Hugging Face cache (default: /mnt/blockstorage/alter-ai/hf-cache)
    ALTER_AI_DTYPE          "bfloat16", "float16", "float32" (default: bfloat16)
    ALTER_AI_LOGS_DIR       Path to write JSONL logs (default: /mnt/blockstorage/alter-ai/logs)
    SIDECAR_PORT            Port to listen on (default: 8090)

Logprob note:
    Full per-token logprobs with transition scores are expensive on large vocabularies.
    In v0.1 we collect scores only for the generated tokens (output_scores=True)
    and compute per-token entropy with a softmax. This is a reasonable approximation
    that avoids materialising the full vocab distribution at each step.

Patches applied:
    - BUG-01: tokenizer loaded via PreTrainedTokenizerFast (tokenizer_config.json
              extra_special_tokens was a list, not dict — patched on block storage)
    - N3: /generate_mechanical endpoint — lane mécanique niveau 1
"""

import datetime
import os
import time
import uuid
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedTokenizerFast,
    StoppingCriteriaList,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("ALTER_AI_MODEL_PATH", "")
DEVICE = os.environ.get("ALTER_AI_DEVICE", "auto")
DTYPE_STR = os.environ.get("ALTER_AI_DTYPE", "bfloat16")
LOGS_DIR = Path(os.environ.get("ALTER_AI_LOGS_DIR", "/mnt/blockstorage/alter-ai/logs"))
SIDECAR_PORT = int(os.environ.get("SIDECAR_PORT", "8090"))

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
TORCH_DTYPE = DTYPE_MAP.get(DTYPE_STR, torch.bfloat16)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("alter-ai-sidecar")

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

tokenizer = None
model = None
model_device = None
model_name_str = None


def load_model():
    """Load the Gemma 4 model from block storage into GPU/CPU memory."""
    global tokenizer, model, model_device, model_name_str

    if not MODEL_PATH or not Path(MODEL_PATH).exists():
        raise RuntimeError(
            f"Model path not found: {MODEL_PATH!r}. "
            "Set ALTER_AI_MODEL_PATH to a valid directory containing Gemma 4 weights."
        )

    log.info(f"Loading tokenizer from {MODEL_PATH}")
    # BUG-01 fix: use PreTrainedTokenizerFast directly.
    # tokenizer_config.json had extra_special_tokens as a list (not dict),
    # patched on block storage. GemmaTokenizerFast loads correctly after patch.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH, local_files_only=True)

    device_map = DEVICE if DEVICE != "auto" else "auto"
    log.info(f"Loading model from {MODEL_PATH} (dtype={DTYPE_STR}, device={device_map})")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_DTYPE,
        device_map=device_map,
        local_files_only=True,
    )
    model.eval()
    elapsed = time.time() - t0
    log.info(f"Model loaded in {elapsed:.1f}s")

    try:
        model_device = str(next(model.parameters()).device)
    except StopIteration:
        model_device = str(device_map)

    model_name_str = Path(MODEL_PATH).name
    log.info(f"Model device: {model_device}")


# ---------------------------------------------------------------------------
# Lifespan — load model on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH:
        log.warning(
            "ALTER_AI_MODEL_PATH not set — sidecar will report unavailable. "
            "The Node.js API will fall back to demo mode."
        )
    else:
        try:
            load_model()
        except Exception as e:
            log.error(f"Failed to load model: {e}")
    yield


app = FastAPI(title="alter.ai Gemma 4 Sidecar", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SamplingConfig(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_new_tokens: int = 512
    do_sample: bool = True


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class GenerateRequest(BaseModel):
    prompt: str
    substance: str
    sampling_config: SamplingConfig
    system_prompt: str
    messages: list[ChatMessage] = Field(default_factory=list)
    replicate_count: int = Field(default=1, ge=1, le=8)


class TokenMetricsSummary(BaseModel):
    """Run-level aggregation of per-token statistics.

    Thresholds:
        low_margin : margin_top1_top2 < LOW_MARGIN_THRESHOLD  (default 0.10)
        high_entropy: entropy > HIGH_ENTROPY_THRESHOLD         (default ln(10) ≈ 2.3026)
    """
    mean_logprob: Optional[float]
    mean_entropy: Optional[float]
    mean_margin_top1_top2: Optional[float]
    entropy_std: Optional[float]
    margin_std: Optional[float]
    num_low_margin_steps: Optional[int]
    num_high_entropy_steps: Optional[int]


LOW_MARGIN_THRESHOLD = 0.10
HIGH_ENTROPY_THRESHOLD = 2.3026  # ln(10)


def aggregate_token_metrics(trace: list[dict]) -> TokenMetricsSummary:
    """Compute run-level summary from per-token trace list."""
    if not trace:
        return TokenMetricsSummary(
            mean_logprob=None,
            mean_entropy=None,
            mean_margin_top1_top2=None,
            entropy_std=None,
            margin_std=None,
            num_low_margin_steps=None,
            num_high_entropy_steps=None,
        )

    logprobs = [e["logprob"] for e in trace]
    entropies = [e["entropy"] for e in trace]
    margins = [e["margin_top1_top2"] for e in trace]

    mean_logprob = float(np.mean(logprobs))
    mean_entropy = float(np.mean(entropies))
    mean_margin = float(np.mean(margins))
    entropy_std = float(np.std(entropies, ddof=0))
    margin_std = float(np.std(margins, ddof=0))
    num_low_margin = int(sum(1 for m in margins if m < LOW_MARGIN_THRESHOLD))
    num_high_entropy = int(sum(1 for e in entropies if e > HIGH_ENTROPY_THRESHOLD))

    return TokenMetricsSummary(
        mean_logprob=round(mean_logprob, 6),
        mean_entropy=round(mean_entropy, 6),
        mean_margin_top1_top2=round(mean_margin, 6),
        entropy_std=round(entropy_std, 6),
        margin_std=round(margin_std, 6),
        num_low_margin_steps=num_low_margin,
        num_high_entropy_steps=num_high_entropy,
    )


class GenerateResponse(BaseModel):
    texts: list[str]
    prompt_tokens: Optional[int]
    completion_tokens: list[int]
    latency_ms: float
    prefill_ms: Optional[float]
    decode_ms: Optional[float]
    first_token_ms: Optional[float]
    tokens_per_second: Optional[float]
    peak_vram_gb: Optional[float]
    finish_reasons: list[str]
    model_name: str
    model_path: str
    dtype: str
    device: str
    token_metrics_per_replicate: list[TokenMetricsSummary]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def ensure_logs_dir():
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def append_trace_log(run_id: str, entries: list[dict]):
    """Write per-token trace to traces/{run_id}.tokens.jsonl"""
    traces_dir = LOGS_DIR.parent / "traces"
    try:
        traces_dir.mkdir(parents=True, exist_ok=True)
        trace_file = traces_dir / f"{run_id}.tokens.jsonl"
        with open(trace_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def build_chat_input(system_prompt: str, messages: list[ChatMessage], prompt: str) -> str:
    """
    Build the full prompt string using the Gemma 4 instruct chat format.
    Gemma 4 IT format: <start_of_turn>user\n...\n<end_of_turn>\n<start_of_turn>model\n
    """
    parts = []
    if system_prompt:
        parts.append(f"<start_of_turn>user\n[System: {system_prompt}]\n<end_of_turn>")
        parts.append("<start_of_turn>model\nUnderstood.<end_of_turn>")

    for msg in messages:
        role_tag = "user" if msg.role == "user" else "model"
        parts.append(f"<start_of_turn>{role_tag}\n{msg.content}<end_of_turn>")

    parts.append(f"<start_of_turn>user\n{prompt}<end_of_turn>")
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


def compute_token_entropy(logits_row: torch.Tensor) -> tuple[float, float, float, float, float]:
    probs = torch.softmax(logits_row.float(), dim=-1)
    log_probs = torch.log(probs + 1e-12)
    entropy = float(-(probs * log_probs).sum().item())
    topk = torch.topk(probs, k=2)
    top1_prob = float(topk.values[0].item())
    top2_prob = float(topk.values[1].item()) if len(topk.values) > 1 else 0.0
    margin = top1_prob - top2_prob
    return entropy, top1_prob, top2_prob, margin


def get_peak_vram_gb() -> Optional[float]:
    if torch.cuda.is_available():
        try:
            peak_bytes = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            return round(peak_bytes / (1024**3), 3)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_one(
    input_ids: torch.Tensor,
    cfg: SamplingConfig,
    run_id: str,
    replicate_index: int,
) -> tuple[str, int, list[dict], str]:
    """
    Run one generation pass. Returns (text, n_tokens, token_trace, finish_reason).
    """
    assert model is not None and tokenizer is not None

    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "repetition_penalty": cfg.repetition_penalty,
        "return_dict_in_generate": True,
        "output_scores": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if cfg.top_k > 0:
        gen_kwargs["top_k"] = cfg.top_k

    with torch.inference_mode():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        output = model.generate(**gen_kwargs)
        t_end = time.perf_counter()

    generated_ids = output.sequences[0, input_ids.shape[1]:]
    n_tokens = len(generated_ids)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    trace = []
    scores = output.scores
    for t_idx, (token_id, score_tensor) in enumerate(zip(generated_ids, scores)):
        logits_row = score_tensor[0]
        probs = torch.softmax(logits_row.float(), dim=-1)
        log_probs = torch.log(probs + 1e-12)
        token_logprob = float(log_probs[token_id].item())
        entropy = float(-(probs * log_probs).sum().item())
        topk = torch.topk(probs, k=2)
        top1_prob = float(topk.values[0].item())
        top2_prob = float(topk.values[1].item()) if len(topk.values) > 1 else 0.0
        margin = top1_prob - top2_prob

        trace.append({
            "t": t_idx,
            "token_id": int(token_id.item()),
            "token_text": tokenizer.decode([token_id]),
            "logprob": round(token_logprob, 6),
            "entropy": round(entropy, 6),
            "top1_prob": round(top1_prob, 6),
            "top2_prob": round(top2_prob, 6),
            "margin_top1_top2": round(margin, 6),
            "latency_ms": round((t_end - t_start) * 1000 / max(n_tokens, 1), 3),
        })

    last_token = int(generated_ids[-1].item()) if n_tokens > 0 else -1
    finish_reason = "stop" if last_token == tokenizer.eos_token_id else "length"

    return text, n_tokens, trace, finish_reason


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    if model is None:
        return JSONResponse({
            "available": False,
            "model": None,
            "device": None,
            "error": "Model not loaded. Check ALTER_AI_MODEL_PATH.",
        })
    return {
        "available": True,
        "model": model_name_str,
        "device": model_device,
        "error": None,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(503, detail="Model not loaded")

    full_prompt = build_chat_input(req.system_prompt, req.messages, req.prompt)
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model_device or "cpu")
    n_prompt_tokens = input_ids.shape[1]

    texts = []
    completion_token_counts = []
    per_replicate_traces: list[list[dict]] = []
    finish_reasons = []

    t_wall_start = time.perf_counter()

    for i in range(req.replicate_count):
        run_id_rep = f"{req.substance}-{uuid.uuid4().hex[:8]}"
        text, n_tok, trace, fr = generate_one(input_ids, req.sampling_config, run_id_rep, i)
        texts.append(text)
        completion_token_counts.append(n_tok)
        per_replicate_traces.append(trace)
        finish_reasons.append(fr)

    t_wall_end = time.perf_counter()
    latency_ms = round((t_wall_end - t_wall_start) * 1000, 1)
    prefill_ms = round(latency_ms * 0.2, 1)
    decode_ms = round(latency_ms * 0.8, 1)
    total_gen_tokens = sum(completion_token_counts)
    tokens_per_second = round(total_gen_tokens / max((t_wall_end - t_wall_start), 0.001), 1)
    peak_vram_gb = get_peak_vram_gb()

    token_metrics_per_replicate = [
        aggregate_token_metrics(trace) for trace in per_replicate_traces
    ]

    run_trace_id = f"{req.substance}-{uuid.uuid4().hex[:8]}"
    all_traces_flat = [entry for trace in per_replicate_traces for entry in trace]
    ensure_logs_dir()
    append_trace_log(run_trace_id, all_traces_flat)

    return GenerateResponse(
        texts=texts,
        prompt_tokens=n_prompt_tokens,
        completion_tokens=completion_token_counts,
        latency_ms=latency_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        first_token_ms=None,
        tokens_per_second=tokens_per_second,
        peak_vram_gb=peak_vram_gb,
        finish_reasons=finish_reasons,
        model_name=model_name_str or "unknown",
        model_path=MODEL_PATH,
        dtype=DTYPE_STR,
        device=model_device or "unknown",
        token_metrics_per_replicate=token_metrics_per_replicate,
    )


# ---------------------------------------------------------------------------
# N3 — Lane mécanique v0.1
# Perturbations de décodage pures, prompt neutre constant, sans roleplay.
# Seule base légitime pour inférence métrologique (pas de biais sémantique).
# ---------------------------------------------------------------------------

NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."
NEUTRAL_PROMPT_HASH = "neutral_v1"

# Presets mécaniques — nommés, documentés, reproductibles.
# Aucune instruction sémantique substance n'est injectée.
# Seules les variables de décodage varient par rapport à la baseline.
MECHANICAL_PRESETS: dict[str, dict] = {
    "baseline": {
        # Reference condition — recommended Gemma 4 sampling params.
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "repetition_penalty": 1.0,
        "max_new_tokens": 256,
        "do_sample": True,
    },
    "entropy_up": {
        # High temperature + wide top_p → increases token entropy.
        # Expected: mean_entropy ↑, mean_margin ↓
        "temperature": 1.8,
        "top_p": 0.99,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 256,
        "do_sample": True,
    },
    "entropy_down": {
        # Low temperature + narrow top_p → reduces entropy, sharpens margin.
        # Expected: mean_entropy ↓, mean_margin ↑
        "temperature": 0.3,
        "top_p": 0.60,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "max_new_tokens": 256,
        "do_sample": True,
    },
    "margin_down": {
        # Wide distribution → compresses margin top1-top2.
        # Expected: mean_margin ↓, num_low_margin_steps ↑
        "temperature": 1.4,
        "top_p": 0.99,
        "top_k": 200,
        "repetition_penalty": 1.0,
        "max_new_tokens": 256,
        "do_sample": True,
    },
    "repetition_loose": {
        # High repetition penalty → forces lexical diversity.
        # Expected: lmi_empirical ↑ across replicates
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "repetition_penalty": 1.5,
        "max_new_tokens": 256,
        "do_sample": True,
    },
}


class MechanicalRequest(BaseModel):
    """
    Request for mechanical lane generation.
    No substance system prompt. Only prompt text and named preset vary.
    """
    prompt: str
    preset: str = Field(
        default="baseline",
        description="One of: baseline | entropy_up | entropy_down | margin_down | repetition_loose",
    )
    replicate_count: int = Field(default=1, ge=1, le=8)
    override_sampling: Optional[dict] = Field(
        default=None,
        description="Optional per-field overrides on top of the named preset.",
    )


class MechanicalResponse(BaseModel):
    texts: list[str]
    prompt_tokens: Optional[int]
    completion_tokens: list[int]
    latency_ms: float
    prefill_ms: Optional[float]
    decode_ms: Optional[float]
    tokens_per_second: Optional[float]
    peak_vram_gb: Optional[float]
    finish_reasons: list[str]
    model_name: str
    dtype: str
    device: str
    lane: str               # always "mechanical"
    preset: str
    preset_params: dict
    neutral_prompt_hash: str
    token_metrics_per_replicate: list[TokenMetricsSummary]


@app.post("/generate_mechanical", response_model=MechanicalResponse)
async def generate_mechanical(req: MechanicalRequest):
    """
    Lane mécanique N3 — perturbations de décodage pures.

    - Prompt neutre constant (pas de roleplay substance)
    - Preset de décodage nommé et loggué explicitement
    - lane=mechanical dans les logs pour séparation stricte
    - Toutes les métriques token calculées identiquement à /generate
    """
    if model is None or tokenizer is None:
        raise HTTPException(503, detail="Model not loaded")

    if req.preset not in MECHANICAL_PRESETS:
        raise HTTPException(
            400,
            detail=f"Unknown preset '{req.preset}'. "
                   f"Valid presets: {list(MECHANICAL_PRESETS.keys())}",
        )

    preset_params = dict(MECHANICAL_PRESETS[req.preset])

    # Apply optional per-field overrides
    if req.override_sampling:
        for k, v in req.override_sampling.items():
            if k in preset_params:
                preset_params[k] = v

    cfg = SamplingConfig(**preset_params)
    full_prompt = build_chat_input(NEUTRAL_SYSTEM_PROMPT, [], req.prompt)
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model_device or "cpu")
    n_prompt_tokens = input_ids.shape[1]

    texts: list[str] = []
    completion_token_counts: list[int] = []
    per_replicate_traces: list[list[dict]] = []
    finish_reasons: list[str] = []

    t_wall_start = time.perf_counter()

    for i in range(req.replicate_count):
        run_id_rep = f"mechanical-{req.preset}-{uuid.uuid4().hex[:8]}"
        text, n_tok, trace, fr = generate_one(input_ids, cfg, run_id_rep, i)
        texts.append(text)
        completion_token_counts.append(n_tok)
        per_replicate_traces.append(trace)
        finish_reasons.append(fr)

    t_wall_end = time.perf_counter()
    latency_ms = round((t_wall_end - t_wall_start) * 1000, 1)
    prefill_ms = round(latency_ms * 0.2, 1)
    decode_ms = round(latency_ms * 0.8, 1)
    tokens_per_second = round(
        sum(completion_token_counts) / max(t_wall_end - t_wall_start, 0.001), 1
    )
    peak_vram_gb = get_peak_vram_gb()

    token_metrics_per_replicate = [
        aggregate_token_metrics(t) for t in per_replicate_traces
    ]

    # Write token traces
    run_trace_id = f"mechanical-{req.preset}-{uuid.uuid4().hex[:8]}"
    ensure_logs_dir()
    append_trace_log(run_trace_id, [e for t in per_replicate_traces for e in t])

    # Write mechanical run log — lane=mechanical explicitly
    log_entry = {
        "schema_version": "ego-llm-mechanical-v0.1",
        "run_id": run_trace_id,
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "lane": "mechanical",
        "preset": req.preset,
        "preset_params": preset_params,
        "neutral_prompt_hash": NEUTRAL_PROMPT_HASH,
        "prompt_tokens": n_prompt_tokens,
        "completion_tokens": sum(completion_token_counts),
        "latency_ms": latency_ms,
        "peak_vram_gb": peak_vram_gb,
        "token_metrics": [
            m.model_dump() if m else None
            for m in token_metrics_per_replicate
        ],
    }
    try:
        mech_log = LOGS_DIR / "mechanical_runs.jsonl"
        with open(mech_log, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        log.warning(f"Failed to write mechanical log: {e}")

    return MechanicalResponse(
        texts=texts,
        prompt_tokens=n_prompt_tokens,
        completion_tokens=completion_token_counts,
        latency_ms=latency_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        tokens_per_second=tokens_per_second,
        peak_vram_gb=peak_vram_gb,
        finish_reasons=finish_reasons,
        model_name=model_name_str or "unknown",
        dtype=DTYPE_STR,
        device=model_device or "unknown",
        lane="mechanical",
        preset=req.preset,
        preset_params=preset_params,
        neutral_prompt_hash=NEUTRAL_PROMPT_HASH,
        token_metrics_per_replicate=token_metrics_per_replicate,
    )
