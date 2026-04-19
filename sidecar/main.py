"""
alter.ai — Gemma 4 Local Inference Sidecar v0.2
================================================
FastAPI service that loads a local Gemma 4 (E4B-it) model from block storage
and exposes generation endpoints for substance-modulated text generation.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8090

Environment variables:
    ALTER_AI_MODEL_PATH     Path to Gemma 4 model dir (required)
    ALTER_AI_DEVICE         "cuda", "cpu", "auto" (default: auto)
    HF_HOME                 Hugging Face home
    HF_HUB_CACHE            Hugging Face cache
    ALTER_AI_DTYPE          "bfloat16", "float16", "float32" (default: bfloat16)
    ALTER_AI_LOGS_DIR       Path to write JSONL logs
    SIDECAR_PORT            Port to listen on (default: 8090)

Changelog:
    v0.1  initial — /generate, /generate_mechanical N3
    v0.1  N4 LogitsProcessor hooks (additive_noise, flatten, sharpen, margin_compress)
    v0.2  seed support in MechanicalRequest (torch.manual_seed before generation)
          transformers_version exposed in /health and MechanicalResponse
          BUG-01 fix: PreTrainedTokenizerFast (tokenizer_config patched on block storage)
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
import transformers
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
    AutoTokenizer,
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
TRANSFORMERS_VERSION = transformers.__version__

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
    global tokenizer, model, model_device, model_name_str

    if not MODEL_PATH or not Path(MODEL_PATH).exists():
        raise RuntimeError(
            f"Model path not found: {MODEL_PATH!r}. "
            "Set ALTER_AI_MODEL_PATH to a valid directory."
        )

    log.info(f"Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    device_map = DEVICE if DEVICE != "auto" else "auto"
    log.info(f"Loading model (dtype={DTYPE_STR}, device={device_map})")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_DTYPE,
        device_map=device_map,
        local_files_only=True,
    )
    model.eval()
    log.info(f"Model loaded in {time.time() - t0:.1f}s")

    try:
        model_device = str(next(model.parameters()).device)
    except StopIteration:
        model_device = str(device_map)

    model_name_str = Path(MODEL_PATH).name
    log.info(f"Model device: {model_device}")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH:
        log.warning("ALTER_AI_MODEL_PATH not set — sidecar will report unavailable.")
    else:
        try:
            load_model()
        except Exception as e:
            log.error(f"Failed to load model: {e}")
    yield


app = FastAPI(title="alter.ai Gemma 4 Sidecar", version="0.3.0", lifespan=lifespan)


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
    role: str
    content: str


class GenerateRequest(BaseModel):
    prompt: str
    substance: str
    sampling_config: SamplingConfig
    system_prompt: str
    messages: list[ChatMessage] = Field(default_factory=list)
    replicate_count: int = Field(default=1, ge=1, le=8)


class TokenMetricsSummary(BaseModel):
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
    if not trace:
        return TokenMetricsSummary(
            mean_logprob=None, mean_entropy=None, mean_margin_top1_top2=None,
            entropy_std=None, margin_std=None,
            num_low_margin_steps=None, num_high_entropy_steps=None,
        )
    logprobs = [e["logprob"] for e in trace]
    entropies = [e["entropy"] for e in trace]
    margins = [e["margin_top1_top2"] for e in trace]
    return TokenMetricsSummary(
        mean_logprob=round(float(np.mean(logprobs)), 6),
        mean_entropy=round(float(np.mean(entropies)), 6),
        mean_margin_top1_top2=round(float(np.mean(margins)), 6),
        entropy_std=round(float(np.std(entropies, ddof=0)), 6),
        margin_std=round(float(np.std(margins, ddof=0)), 6),
        num_low_margin_steps=int(sum(1 for m in margins if m < LOW_MARGIN_THRESHOLD)),
        num_high_entropy_steps=int(sum(1 for e in entropies if e > HIGH_ENTROPY_THRESHOLD)),
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
    traces_dir = LOGS_DIR.parent / "traces"
    try:
        traces_dir.mkdir(parents=True, exist_ok=True)
        with open(traces_dir / f"{run_id}.tokens.jsonl", "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def build_chat_input(system_prompt: str, messages: list[ChatMessage], prompt: str) -> str:
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



def is_qwen_model() -> bool:
    name = (model_name_str or Path(MODEL_PATH).name or "").lower()
    path = (MODEL_PATH or "").lower()
    return "qwen" in name or "qwen" in path


def build_model_inputs(system_prompt: str, messages: list[ChatMessage], prompt: str) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    device = model_device or "cpu"

    if is_qwen_model():
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            role = "user" if msg.role == "user" else "assistant"
            chat_messages.append({"role": role, "content": msg.content})
        chat_messages.append({"role": "user", "content": prompt})

        rendered = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    else:
        full_prompt = build_chat_input(system_prompt, messages, prompt)
        enc = tokenizer(full_prompt, return_tensors="pt")

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask

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
# N4 — LogitsProcessor hooks
# Post-forward, pre-sampling. Applied before temperature/top-k/top-p warping.
# ---------------------------------------------------------------------------

class AdditiveNoiseProcessor(LogitsProcessor):
    """Add Gaussian noise N(0, noise_std) to logits. Expected: entropy ↑, margin ↓."""
    def __init__(self, noise_std: float):
        self.noise_std = noise_std

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores + torch.randn_like(scores) * self.noise_std


class LogitFlattenProcessor(LogitsProcessor):
    """Scale logits by scale in (0,1] — flattens distribution. Expected: entropy ↑↑, margin ↓↓."""
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores * self.scale


class LogitSharpenProcessor(LogitsProcessor):
    """Scale logits by scale >= 1 — sharpens distribution. Expected: entropy ↓↓, margin ↑↑."""
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores * self.scale


class MarginCompressProcessor(LogitsProcessor):
    """Pull all logits toward max by compress_factor. Expected: margin ↓↓, num_low_margin ↑↑."""
    def __init__(self, compress_factor: float):
        self.compress_factor = compress_factor

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        max_logit = scores.max(dim=-1, keepdim=True).values
        return scores + self.compress_factor * (max_logit - scores)


def build_logits_processor(hook_type: str, hook_params: dict) -> Optional[LogitsProcessor]:
    if hook_type == "additive_noise":
        return AdditiveNoiseProcessor(noise_std=float(hook_params.get("noise_std", 1.0)))
    elif hook_type == "logit_flatten":
        return LogitFlattenProcessor(scale=float(hook_params.get("scale", 0.5)))
    elif hook_type == "logit_sharpen":
        return LogitSharpenProcessor(scale=float(hook_params.get("scale", 2.0)))
    elif hook_type == "margin_compress":
        return MarginCompressProcessor(compress_factor=float(hook_params.get("compress_factor", 0.7)))
    return None


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_one(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cfg: SamplingConfig,
    run_id: str,
    replicate_index: int,
    logits_processor_list: Optional[LogitsProcessorList] = None,
    seed: Optional[int] = None,
) -> tuple[str, int, list[dict], str]:
    """
    Run one generation pass.
    seed: if set, calls torch.manual_seed + torch.cuda.manual_seed before generation.
    Returns (text, n_tokens, token_trace, finish_reason).
    """
    assert model is not None and tokenizer is not None

    # Seed support — set before generation for reproducibility
    if seed is not None:
        torch.manual_seed(seed + replicate_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + replicate_index)

    gen_kwargs: dict = {
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
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask
    if cfg.top_k > 0:
        gen_kwargs["top_k"] = cfg.top_k
    if logits_processor_list is not None:
        gen_kwargs["logits_processor"] = logits_processor_list

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
    for t_idx, (token_id, score_tensor) in enumerate(zip(generated_ids, output.scores)):
        logits_row = score_tensor[0]
        probs = torch.softmax(logits_row.float(), dim=-1)
        log_probs = torch.log(probs + 1e-12)
        token_logprob = float(log_probs[token_id].item())
        entropy = float(-(probs * log_probs).sum().item())
        topk = torch.topk(probs, k=2)
        top1_prob = float(topk.values[0].item())
        top2_prob = float(topk.values[1].item()) if len(topk.values) > 1 else 0.0
        trace.append({
            "t": t_idx,
            "token_id": int(token_id.item()),
            "token_text": tokenizer.decode([token_id]),
            "logprob": round(token_logprob, 6),
            "entropy": round(entropy, 6),
            "top1_prob": round(top1_prob, 6),
            "top2_prob": round(top2_prob, 6),
            "margin_top1_top2": round(top1_prob - top2_prob, 6),
            "replicate_index": replicate_index,
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
            "available": False, "model": None, "device": None,
            "error": "Model not loaded.",
            "transformers_version": TRANSFORMERS_VERSION,
        })
    return {
        "available": True,
        "model": model_name_str,
        "device": model_device,
        "error": None,
        "transformers_version": TRANSFORMERS_VERSION,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(503, detail="Model not loaded")
    input_ids, attention_mask = build_model_inputs(req.system_prompt, req.messages, req.prompt)
    n_prompt_tokens = input_ids.shape[1]

    texts, completion_token_counts, per_replicate_traces, finish_reasons = [], [], [], []
    t_wall_start = time.perf_counter()

    for i in range(req.replicate_count):
        text, n_tok, trace, fr = generate_one(
            input_ids, attention_mask, req.sampling_config, f"{req.substance}-{uuid.uuid4().hex[:8]}", i
        )
        texts.append(text)
        completion_token_counts.append(n_tok)
        per_replicate_traces.append(trace)
        finish_reasons.append(fr)

    t_wall_end = time.perf_counter()
    latency_ms = round((t_wall_end - t_wall_start) * 1000, 1)
    tokens_per_second = round(sum(completion_token_counts) / max(t_wall_end - t_wall_start, 0.001), 1)
    peak_vram_gb = get_peak_vram_gb()

    run_trace_id = f"{req.substance}-{uuid.uuid4().hex[:8]}"
    ensure_logs_dir()
    append_trace_log(run_trace_id, [e for t in per_replicate_traces for e in t])

    return GenerateResponse(
        texts=texts, prompt_tokens=n_prompt_tokens, completion_tokens=completion_token_counts,
        latency_ms=latency_ms, prefill_ms=round(latency_ms * 0.2, 1),
        decode_ms=round(latency_ms * 0.8, 1), first_token_ms=None,
        tokens_per_second=tokens_per_second, peak_vram_gb=peak_vram_gb,
        finish_reasons=finish_reasons, model_name=model_name_str or "unknown",
        model_path=MODEL_PATH, dtype=DTYPE_STR, device=model_device or "unknown",
        token_metrics_per_replicate=[aggregate_token_metrics(t) for t in per_replicate_traces],
    )


# ---------------------------------------------------------------------------
# N3 + N4 — Lane mécanique
# ---------------------------------------------------------------------------

NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."
NEUTRAL_PROMPT_HASH = "neutral_v1"

MECHANICAL_PRESETS: dict[str, dict] = {
    "baseline":         {"temperature": 1.0, "top_p": 0.95, "top_k": 64,  "repetition_penalty": 1.0, "max_new_tokens": 256, "do_sample": True},
    "entropy_up":       {"temperature": 1.8, "top_p": 0.99, "top_k": 0,   "repetition_penalty": 1.0, "max_new_tokens": 256, "do_sample": True},
    "entropy_down":     {"temperature": 0.3, "top_p": 0.60, "top_k": 20,  "repetition_penalty": 1.0, "max_new_tokens": 256, "do_sample": True},
    "margin_down":      {"temperature": 1.4, "top_p": 0.99, "top_k": 200, "repetition_penalty": 1.0, "max_new_tokens": 256, "do_sample": True},
    "repetition_loose": {"temperature": 1.0, "top_p": 0.95, "top_k": 64,  "repetition_penalty": 1.5, "max_new_tokens": 256, "do_sample": True},
}

LOGITS_PRESETS: dict[str, dict] = {
    "noise_low":              {"hook": "additive_noise",  "noise_std": 0.5},
    "noise_high":             {"hook": "additive_noise",  "noise_std": 2.0},
    "flatten_mild":           {"hook": "logit_flatten",   "scale": 0.5},
    "flatten_strong":         {"hook": "logit_flatten",   "scale": 0.1},
    "sharpen_mild":           {"hook": "logit_sharpen",   "scale": 2.0},
    "sharpen_strong":         {"hook": "logit_sharpen",   "scale": 5.0},
    "margin_compress":        {"hook": "margin_compress", "compress_factor": 0.7},
    "margin_compress_strong": {"hook": "margin_compress", "compress_factor": 0.92},
}

NEUTRAL_SAMPLING_PARAMS = {
    "temperature": 1.0, "top_p": 1.0, "top_k": 0,
    "repetition_penalty": 1.0, "max_new_tokens": 256, "do_sample": True,
}


class MechanicalRequest(BaseModel):
    prompt: str
    preset: str = Field(default="baseline")
    replicate_count: int = Field(default=1, ge=1, le=8)
    override_sampling: Optional[dict] = None
    logits_hook: Optional[str] = None
    logits_hook_params: Optional[dict] = None
    # v0.2: seed for reproducibility
    seed: Optional[int] = Field(
        default=None,
        description="RNG seed applied via torch.manual_seed before each replicate (seed + replicate_index).",
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
    transformers_version: str
    lane: str
    hook_level: str
    preset: str
    preset_params: dict
    logits_hook: Optional[str]
    logits_hook_params: Optional[dict]
    seed: Optional[int]
    neutral_prompt_hash: str
    run_id: str
    token_metrics_per_replicate: list[TokenMetricsSummary]


@app.post("/generate_mechanical", response_model=MechanicalResponse)
async def generate_mechanical(req: MechanicalRequest):
    """
    Lane mécanique N3/N4.

    N3 only  : set preset, leave logits_hook=None
    N4 only  : set logits_hook (neutral sampling used automatically)
    N3 + N4  : set both
    seed     : if set, torch.manual_seed(seed + replicate_index) before each generation
    """
    if model is None or tokenizer is None:
        raise HTTPException(503, detail="Model not loaded")

    if req.preset not in MECHANICAL_PRESETS:
        raise HTTPException(400, detail=f"Unknown preset '{req.preset}'. Valid: {list(MECHANICAL_PRESETS.keys())}")

    if req.logits_hook is not None and req.preset == "baseline":
        preset_params = dict(NEUTRAL_SAMPLING_PARAMS)
    else:
        preset_params = dict(MECHANICAL_PRESETS[req.preset])

    if req.override_sampling:
        for k, v in req.override_sampling.items():
            if k in preset_params:
                preset_params[k] = v

    cfg = SamplingConfig(**preset_params)

    active_logits_hook: Optional[str] = None
    active_hook_params: Optional[dict] = None
    lp_list: Optional[LogitsProcessorList] = None

    if req.logits_hook is not None:
        if req.logits_hook not in LOGITS_PRESETS:
            raise HTTPException(400, detail=f"Unknown logits_hook '{req.logits_hook}'. Valid: {list(LOGITS_PRESETS.keys())}")
        hook_cfg = dict(LOGITS_PRESETS[req.logits_hook])
        hook_type = hook_cfg.pop("hook")
        if req.logits_hook_params:
            hook_cfg.update(req.logits_hook_params)
        processor = build_logits_processor(hook_type, hook_cfg)
        if processor is not None:
            lp_list = LogitsProcessorList([processor])
            active_logits_hook = req.logits_hook
            active_hook_params = {"hook": hook_type, **hook_cfg}

    hook_level = "logits" if active_logits_hook else "decode"
    input_ids, attention_mask = build_model_inputs(NEUTRAL_SYSTEM_PROMPT, [], req.prompt)
    n_prompt_tokens = input_ids.shape[1]

    texts: list[str] = []
    completion_token_counts: list[int] = []
    per_replicate_traces: list[list[dict]] = []
    finish_reasons: list[str] = []

    t_wall_start = time.perf_counter()

    for i in range(req.replicate_count):
        text, n_tok, trace, fr = generate_one(
            input_ids, attention_mask, cfg,
            f"mechanical-{req.preset}-{uuid.uuid4().hex[:8]}", i,
            logits_processor_list=lp_list,
            seed=req.seed,
        )
        texts.append(text)
        completion_token_counts.append(n_tok)
        per_replicate_traces.append(trace)
        finish_reasons.append(fr)

    t_wall_end = time.perf_counter()
    latency_ms = round((t_wall_end - t_wall_start) * 1000, 1)
    tokens_per_second = round(sum(completion_token_counts) / max(t_wall_end - t_wall_start, 0.001), 1)
    peak_vram_gb = get_peak_vram_gb()
    token_metrics_per_replicate = [aggregate_token_metrics(t) for t in per_replicate_traces]

    run_trace_id = f"mechanical-{req.preset}-{uuid.uuid4().hex[:8]}"
    ensure_logs_dir()
    append_trace_log(run_trace_id, [e for t in per_replicate_traces for e in t])

    log_entry = {
        "schema_version": "ego-llm-mechanical-v0.3",
        "run_id": run_trace_id,
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "lane": "mechanical",
        "hook_level": hook_level,
        "preset": req.preset,
        "preset_params": preset_params,
        "logits_hook": active_logits_hook,
        "logits_hook_params": active_hook_params,
        "seed": req.seed,
        "neutral_prompt_hash": NEUTRAL_PROMPT_HASH,
        "prompt_tokens": n_prompt_tokens,
        "completion_tokens": sum(completion_token_counts),
        "latency_ms": latency_ms,
        "peak_vram_gb": peak_vram_gb,
        "transformers_version": TRANSFORMERS_VERSION,
        "token_metrics": [m.model_dump() if m else None for m in token_metrics_per_replicate],
    }
    try:
        with open(LOGS_DIR / "mechanical_runs.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        log.warning(f"Failed to write mechanical log: {e}")

    return MechanicalResponse(
        texts=texts,
        prompt_tokens=n_prompt_tokens,
        completion_tokens=completion_token_counts,
        latency_ms=latency_ms,
        prefill_ms=round(latency_ms * 0.2, 1),
        decode_ms=round(latency_ms * 0.8, 1),
        tokens_per_second=tokens_per_second,
        peak_vram_gb=peak_vram_gb,
        finish_reasons=finish_reasons,
        model_name=model_name_str or "unknown",
        dtype=DTYPE_STR,
        device=model_device or "unknown",
        transformers_version=TRANSFORMERS_VERSION,
        lane="mechanical",
        hook_level=hook_level,
        preset=req.preset,
        preset_params=preset_params,
        logits_hook=active_logits_hook,
        logits_hook_params=active_hook_params,
        seed=req.seed,
        neutral_prompt_hash=NEUTRAL_PROMPT_HASH,
        run_id=run_trace_id,
        token_metrics_per_replicate=token_metrics_per_replicate,
    )
