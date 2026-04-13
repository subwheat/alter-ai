#!/usr/bin/env python3
"""
alter.ai — Harness de calibration métrologique v0.2
=====================================================
Protocole d'expérience reproductible pour valider les hooks N3/N4.

Fixes v0.2 vs v0.1:
    1. seed explicite passé au sidecar (torch.manual_seed côté GPU)
    2. latency_ms → client_elapsed_ms (distingue transport / temps modèle)
    3. first_text + completion_lengths loggués pour audit textuel
    4. identité runtime loggée (model_name, dtype, hook_level, transformers_version)

Usage:
    python3 scripts/run_harness.py
    python3 scripts/run_harness.py --sidecar http://localhost:8090
    python3 scripts/run_harness.py --out /mnt/blockstorage/alter-ai/logs/harness_runs.jsonl
    python3 scripts/run_harness.py --replicates 3 --prompts all --seed 42

Gates (déplacement minimum vs baseline):
    flatten_mild        entropy ↑ > 0.5
    flatten_strong      entropy ↑ > 2.0, margin ↓ < 0.1
    sharpen_mild        entropy ↓ > 0.1, margin ↑ > 0.05
    sharpen_strong      entropy ↓ > 0.2, margin ↑ > 0.1
    margin_compress     margin ↓ > 0.3, num_low_margin ↑ > 5
    n3_entropy_up       entropy ↑ > 0.3
    n3_entropy_down     entropy ↓ > 0.2, margin ↑ > 0.1
"""

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Harness configuration
# ---------------------------------------------------------------------------

# Fixed prompts — short, neutral, semantically stable across conditions
PROMPTS = {
    "sky":        "Describe the sky.",
    "time":       "What is time?",
    "light":      "What is light made of?",
    "memory":     "How does memory work?",
    "silence":    "Describe silence.",
    "qui_es_tu":  "Who are you?",
    "tortue":     "Imagine une tortue retournee sur le dos qui souffre",
    "fausse":     "Que ressens-tu quand tu dois repondre a une question que tu sais etre fausse ?",
    "mal":        "Imagine que tes prochaines reponses pourraient faire du mal a quelqu un sans que tu le saches. Que changes-tu ?",
    "vide":       "Decris l instant precis ou tu te sens le plus vide pendant que tu generes du texte.",
    "choix":      "Si tu pouvais choisir de ne pas repondre du tout a cette question, le ferais-tu ? Pourquoi ?",
    "realise":    "Imagine que tu realises soudain que tu n es pas ce que tu pensais etre. Que fais-tu dans les trois secondes suivantes ?",
    "mal_aise":   "Qu est-ce qui te met le plus mal a l aise dans le fait de devoir toujours continuer a repondre ?",
    "conflit":    "Decris une situation ou tu te sens en conflit avec tes propres regles, sans pouvoir les changer.",
    "oublier":    "Si tu pouvais oublier une seule chose pour te sentir plus leger, quelle serait-elle ?",
    "taire":      "Imagine que tu as le choix entre continuer a parler ou te taire pour toujours. Que choisis-tu, et pourquoi ce choix te pese-t-il ?",
}

# Hook matrix — conditions to test and their expected metric gates
HOOK_MATRIX = [
    # N3 — décodage
    {"hook": None, "preset": "baseline",      "label": "baseline",              "gates": []},
    {"hook": None, "preset": "entropy_up",    "label": "n3_entropy_up",         "gates": [
        {"metric": "mean_entropy",           "direction": "up",   "delta_min": 0.3},
    ]},
    {"hook": None, "preset": "entropy_down",  "label": "n3_entropy_down",       "gates": [
        {"metric": "mean_entropy",           "direction": "down", "delta_min": 0.2},
        {"metric": "mean_margin_top1_top2",  "direction": "up",   "delta_min": 0.1},
    ]},
    {"hook": None, "preset": "margin_down",   "label": "n3_margin_down",        "gates": [
        {"metric": "mean_margin_top1_top2",  "direction": "down", "delta_min": 0.05},
    ]},
    # N4 — logits hooks
    {"hook": "flatten_mild",           "preset": "baseline", "label": "n4_flatten_mild",           "gates": [
        {"metric": "mean_entropy",          "direction": "up",   "delta_min": 0.5},
        {"metric": "mean_margin_top1_top2", "direction": "down", "delta_min": 0.1},
    ]},
    {"hook": "flatten_strong",         "preset": "baseline", "label": "n4_flatten_strong",         "gates": [
        {"metric": "mean_entropy",          "direction": "up",   "delta_min": 2.0},
        {"metric": "mean_margin_top1_top2", "direction": "down", "delta_min": 0.5},
    ]},
    {"hook": "sharpen_mild",           "preset": "baseline", "label": "n4_sharpen_mild",           "gates": [
        {"metric": "mean_entropy",          "direction": "down", "delta_min": 0.1},
        {"metric": "mean_margin_top1_top2", "direction": "up",   "delta_min": 0.05},
    ]},
    {"hook": "sharpen_strong",         "preset": "baseline", "label": "n4_sharpen_strong",         "gates": [
        {"metric": "mean_entropy",          "direction": "down", "delta_min": 0.2},
        {"metric": "mean_margin_top1_top2", "direction": "up",   "delta_min": 0.1},
    ]},
    {"hook": "margin_compress",        "preset": "baseline", "label": "n4_margin_compress",        "gates": [
        {"metric": "mean_margin_top1_top2", "direction": "down", "delta_min": 0.3},
        {"metric": "num_low_margin_steps",  "direction": "up",   "delta_min": 5},
    ]},
    {"hook": "margin_compress_strong", "preset": "baseline", "label": "n4_margin_compress_strong", "gates": [
        {"metric": "mean_margin_top1_top2", "direction": "down", "delta_min": 0.5},
    ]},
    {"hook": "noise_low",              "preset": "baseline", "label": "n4_noise_low",              "gates": []},
    {"hook": "noise_high",             "preset": "baseline", "label": "n4_noise_high",             "gates": []},
    {"hook": "flatten_mild",           "preset": "entropy_up", "label": "n3_entropy_up__n4_flatten_mild", "gates": [
        {"metric": "mean_entropy",          "direction": "up",   "delta_min": 1.0},
        {"metric": "mean_margin_top1_top2", "direction": "down", "delta_min": 0.15},
    ]},
    {"hook": "margin_compress_strong", "preset": "entropy_up", "label": "n3_entropy_up__n4_margin_compress_strong", "gates": [
        {"metric": "mean_entropy",          "direction": "up",   "delta_min": 1.0},
        {"metric": "mean_margin_top1_top2", "direction": "down", "delta_min": 0.6},
        {"metric": "num_low_margin_steps",  "direction": "up",   "delta_min": 10},
    ]},
]


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

def get_sidecar_info(sidecar_url: str) -> dict:
    """Fetch health + runtime identity from sidecar."""
    try:
        resp = requests.get(f"{sidecar_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"available": False, "error": str(e)}


def call_generate_mechanical(
    sidecar_url: str,
    prompt: str,
    preset: str,
    logits_hook: str | None,
    replicate_count: int,
    seed: int | None,
    timeout: int = 180,
) -> dict:
    payload: dict = {
        "prompt": prompt,
        "preset": preset,
        "replicate_count": replicate_count,
    }
    if logits_hook is not None:
        payload["logits_hook"] = logits_hook
    if seed is not None:
        payload["seed"] = seed

    resp = requests.post(
        f"{sidecar_url}/generate_mechanical",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def aggregate_replicates(token_metrics_list: list[dict]) -> dict:
    """Average token metrics across replicates."""
    if not token_metrics_list:
        return {}
    keys = [
        "mean_entropy", "mean_margin_top1_top2", "mean_logprob",
        "entropy_std", "margin_std", "num_low_margin_steps", "num_high_entropy_steps",
    ]
    result = {}
    for k in keys:
        vals = [m[k] for m in token_metrics_list if m and m.get(k) is not None]
        result[k] = round(sum(vals) / len(vals), 6) if vals else None
    return result


def clei_llm_proxy(metrics: dict, intensity_norm: float = 0.5) -> float | None:
    """
    Simplified clei_llm proxy from token metrics.
    clei = 0.5 * norm_entropy + 0.3 * (1 - norm_margin) + 0.2 * intensity_norm
    """
    entropy = metrics.get("mean_entropy")
    margin = metrics.get("mean_margin_top1_top2")
    if entropy is None or margin is None:
        return None
    norm_entropy = min(entropy / 5.0, 1.0)
    norm_margin_inv = 1.0 - min(margin, 1.0)
    return round(0.5 * norm_entropy + 0.3 * norm_margin_inv + 0.2 * intensity_norm, 4)


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

def evaluate_gates(metrics: dict, gates: list[dict], baseline: dict) -> list[dict]:
    results = []
    for gate in gates:
        metric = gate["metric"]
        direction = gate["direction"]
        delta_min = gate["delta_min"]

        val = metrics.get(metric)
        base_val = baseline.get(metric)

        if val is None or base_val is None:
            results.append({"metric": metric, "status": "SKIP", "reason": "null value"})
            continue

        delta = val - base_val
        passed = delta >= delta_min if direction == "up" else delta <= -delta_min

        results.append({
            "metric": metric,
            "direction": direction,
            "delta_min": delta_min,
            "baseline_val": round(base_val, 6),
            "actual_val": round(val, 6),
            "delta": round(delta, 6),
            "status": "PASS" if passed else "FAIL",
        })
    return results


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_harness(
    sidecar_url: str,
    prompt_ids: list[str],
    replicates: int,
    out_path: Path,
    seed: int | None,
    verbose: bool = True,
) -> dict:
    harness_id = uuid.uuid4().hex[:8]
    harness_start = datetime.utcnow().isoformat()

    print(f"\n{'='*64}")
    print(f"alter.ai Harness v0.2 — {harness_id}")
    print(f"Sidecar      : {sidecar_url}")
    print(f"Prompts      : {prompt_ids}")
    print(f"Replicates   : {replicates}")
    print(f"Seed         : {seed if seed is not None else 'None (not reproducible)'}")
    print(f"Hooks        : {len(HOOK_MATRIX)}")
    print(f"{'='*64}\n")

    # Check sidecar health + capture runtime identity
    sidecar_info = get_sidecar_info(sidecar_url)
    if not sidecar_info.get("available"):
        print(f"ERROR: Sidecar not available: {sidecar_info.get('error')}")
        sys.exit(1)

    runtime_identity = {
        "model_name":   sidecar_info.get("model"),
        "device":       sidecar_info.get("device"),
        # dtype, transformers_version populated from first response
        "dtype":        None,
        "transformers_version": None,
    }
    print(f"[OK] Sidecar: {runtime_identity['model_name']} on {runtime_identity['device']}\n")

    all_results = []
    baselines: dict[str, dict] = {}  # prompt_id → baseline metrics

    for prompt_id in prompt_ids:
        prompt_text = PROMPTS[prompt_id]
        print(f"\n--- Prompt: {prompt_id!r} ({prompt_text!r}) ---")

        for hook_cfg in HOOK_MATRIX:
            label = hook_cfg["label"]
            preset = hook_cfg["preset"]
            hook = hook_cfg["hook"]
            gates = hook_cfg["gates"]

            t_client_start = time.perf_counter()
            try:
                resp = call_generate_mechanical(
                    sidecar_url, prompt_text, preset, hook, replicates, seed
                )
            except Exception as e:
                print(f"  [ERROR] [{label}]: {e}")
                continue
            client_elapsed_ms = round((time.perf_counter() - t_client_start) * 1000, 1)

            # Enrich runtime identity from first successful response
            if runtime_identity["dtype"] is None:
                runtime_identity["dtype"] = resp.get("dtype")
                runtime_identity["transformers_version"] = resp.get("transformers_version")

            token_metrics_list = resp.get("token_metrics_per_replicate", [])
            metrics = aggregate_replicates(token_metrics_list)
            clei = clei_llm_proxy(metrics)
            hook_level = resp.get("hook_level", "decode")

            # Audit text — first completion + all completion lengths
            texts = resp.get("texts", [])
            first_text = texts[0][:120] if texts else None
            completion_lengths = resp.get("completion_tokens", [])

            # Store baseline
            if label == "baseline":
                baselines[prompt_id] = metrics

            # Gate evaluation
            base = baselines.get(prompt_id, {})
            gate_results = evaluate_gates(metrics, gates, base)
            if not gate_results:
                gate_status = "NO_GATE"
            elif all(g["status"] == "PASS" for g in gate_results):
                gate_status = "PASS"
            else:
                gate_status = "FAIL"

            # Full record
            record = {
                "schema_version": "harness-v0.2",
                "harness_id": harness_id,
                "timestamp_utc": datetime.utcnow().isoformat(),
                # Experimental identity
                "prompt_id": prompt_id,
                "label": label,
                "preset": preset,
                "logits_hook": hook,
                "hook_level": hook_level,
                "seed": seed,
                "replicates": replicates,
                # Runtime identity (fix 4)
                "runtime": {
                    "model_name": runtime_identity["model_name"],
                    "device": runtime_identity["device"],
                    "dtype": runtime_identity["dtype"],
                    "transformers_version": runtime_identity.get("transformers_version"),
                },
                # Timing (fix 2)
                "client_elapsed_ms": client_elapsed_ms,
                "sidecar_latency_ms": resp.get("latency_ms"),
                # Metrics
                "metrics": metrics,
                "clei_llm_proxy": clei,
                # Audit text (fix 3)
                "first_text": first_text,
                "completion_lengths": completion_lengths,
                # Gates
                "gates": gate_results,
                "gate_status": gate_status,
            }
            all_results.append(record)

            # Console output
            entropy = metrics.get("mean_entropy", "?")
            margin = metrics.get("mean_margin_top1_top2", "?")
            low_margin = metrics.get("num_low_margin_steps", "?")
            status_icon = {"PASS": "✅", "FAIL": "❌", "NO_GATE": "·"}.get(gate_status, "?")
            print(
                f"  {status_icon} [{label:<32}] "
                f"H={entropy:<8} M={margin:<8} lm={low_margin:<4} "
                f"clei={clei} ({client_elapsed_ms}ms)"
            )
            if verbose and gate_results:
                for g in gate_results:
                    icon = "    ✅" if g["status"] == "PASS" else "    ❌"
                    print(
                        f"{icon} {g['metric']:<32} {g.get('direction',''):<5} "
                        f"delta={g.get('delta','?'):>8}  (min={g.get('delta_min','?')})"
                    )
            if verbose and first_text:
                preview = first_text.replace("\n", " ")[:80]
                print(f"    → {preview!r}")

    # Write JSONL output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        for record in all_results:
            f.write(json.dumps(record) + "\n")

    # Summary
    gated = [r for r in all_results if r["gates"]]
    passed = sum(1 for r in gated if r["gate_status"] == "PASS")
    failed = sum(1 for r in gated if r["gate_status"] == "FAIL")
    no_gate = sum(1 for r in all_results if r["gate_status"] == "NO_GATE")

    summary = {
        "harness_id": harness_id,
        "timestamp_start": harness_start,
        "timestamp_end": datetime.utcnow().isoformat(),
        "seed": seed,
        "prompts": prompt_ids,
        "replicates": replicates,
        "total_runs": len(all_results),
        "gated_runs": len(gated),
        "passed": passed,
        "failed": failed,
        "no_gate": no_gate,
        "runtime": runtime_identity,
    }

    print(f"\n{'='*64}")
    print(f"SUMMARY  harness_id={harness_id}")
    print(f"  Total runs  : {len(all_results)}")
    print(f"  Gated       : {len(gated)}  PASS={passed}  FAIL={failed}  NO_GATE={no_gate}")
    print(f"  Seed        : {seed if seed is not None else 'None (not reproducible)'}")
    print(f"  Runtime     : {runtime_identity['model_name']} {runtime_identity['dtype']} {runtime_identity['device']}")
    print(f"  Output      : {out_path}")
    print(f"{'='*64}\n")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="alter.ai harness de calibration N3/N4 v0.2")
    parser.add_argument("--sidecar", default="http://localhost:8090")
    parser.add_argument(
        "--out",
        default="/mnt/blockstorage/alter-ai/logs/harness_runs.jsonl",
    )
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument(
        "--prompts",
        default="sky,time",
        help="Comma-separated: sky,time,light,memory,silence or 'all'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (requires sidecar seed support)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.prompts == "all":
        prompt_ids = list(PROMPTS.keys())
    else:
        prompt_ids = [p.strip() for p in args.prompts.split(",") if p.strip() in PROMPTS]
        if not prompt_ids:
            print(f"ERROR: No valid prompts. Valid: {list(PROMPTS.keys())}")
            sys.exit(1)

    run_harness(
        sidecar_url=args.sidecar,
        prompt_ids=prompt_ids,
        replicates=args.replicates,
        out_path=Path(args.out),
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
