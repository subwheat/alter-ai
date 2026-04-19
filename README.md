# ALTER AI Lab

Research repo for two linked programs:

- **EGO TRIAD** — minimal substrates for emergence / cascade testing
- **ALTER AI** — LLM vulnerability metrology under controlled decoding perturbations

This repository is **not** a product and **not** a SaaS.  
It is an internal research workbench designed to run disciplined experiments, log traces, compute metrics, and produce bounded reports.

## Status

**Experimental / active research repo**

Current reality:

- local serving and instrumentation for **Gemma** and **Qwen**
- ALTER AI API + local sidecar + experimental harness
- JSONL logs and token-level traces
- ongoing metric work on coherence degradation under mechanical perturbations such as `flatten_mild` and `sharpen_mild`

Target direction:

- evolve toward a fuller **ALTER AI Lab** with stricter layering, CLI orchestration, ticket-driven execution, automated audits, and bounded reports

## What this repo is for

This repo exists to make research execution more disciplined.

Main goals:

- run **EGO TRIAD** campaigns on deterministic Python substrates
- run **Q-LLM / ALTER AI** campaigns on local LLMs
- collect traces and compute multi-scale metrics
- compare models under the same prompt bank / hooks / seeds
- keep methodology explicit through tickets, audits, and reproducible reports

## What is in scope today

At repository/runtime level, the useful core is:

- a **local sidecar** for model serving and token-level instrumentation
- an **API server** for ALTER AI runs
- an experimental **harness** for prompt banks / seeds / hook matrices
- traces and metrics used for Q-LLM analysis

This repo is currently the operational base used on the server working tree at `/opt/alter-ai`.

## Research snapshot (April 2026)

### Local Gemma result
On the widened Gemma bench, the local result currently points to **`Q4_32` as the provisional local candidate metric** for coherence vulnerability under perturbation. This came after the earlier `Q4_8` and `Q4_1` comparisons, and after widening the prompt bank to 24 prompts over 3 seeds and 3 hooks.

### Transfer to Qwen
Transfer to Qwen is **not a failure**. The signal remains strong after admissibility filtering and lexical-drift handling.  
However, **`Q4_32` has not yet won the final transfer arbitration against `Q4_8`**. The current status is therefore: **strong Qwen signal, final metric arbitration still open**.

### Immediate consequence
For now, the repo should be read as:
- a live research runtime,
- with a strong local metric story on Gemma,
- and an open final arbitration for transfer on Qwen.

## Repo philosophy

This project is built around a few non-negotiable rules:

- **no silent metric drift**
- **no ad hoc filtering after the fact**
- **no free-form sweep as default mode**
- **ticket first, run second**
- **bounded reports over narrative improvisation**
- **audit before patch**
- **one logical change = one logical commit**

The goal is not to make research slower.  
The goal is to prevent methodological slippage.

## Planned architecture

The broader target architecture is a strict **4-layer lab**:

1. **Substrates**  
   deterministic scientific code for EGO and ALTER AI

2. **Runtime**  
   campaign orchestration, seeds, timestamps, outputs, CLI

3. **Analyzers**  
   statistics, regressions, tables, differentials

4. **LLM helpers**  
   tightly constrained assistance only, never verdict authority

Important: this architecture is the **design target**, not a claim that every layer is already fully implemented today.

## Storage rule

Persistent data must live on **block storage**, not on Scaleway scratch.

Use block storage for:
- models
- logs
- traces
- HF cache / HF home
- campaign outputs

Do **not** rely on `/scratch` for persistence. On this infra, `/scratch` is ephemeral and can be reformatted/remounted at boot. That failure mode has already occurred and is considered a known trap.

## Operational model

Current server-side reality is centered on:

- local repo working tree: `/opt/alter-ai`
- persistent data root: `/mnt/blockstorage/alter-ai`
- model storage under block storage
- sidecar on port `8090`
- API on port `8080`

The current Gemma deployment uses block-storage-backed model path, HF home/cache, and logs directory.

## Principles for contributions

Contributions should preserve:

- explicit hypotheses
- explicit intervention
- explicit ablation inverse
- explicit PASS / FAIL / INCONCLUSIVE criteria
- explicit limits

Avoid:

- magic numbers with no ticket/spec context
- monolithic patches mixing infra + theory + UI + metrics
- hidden data dependencies
- writing persistent outputs into transient locations
- “fixing” results by changing the bench after the fact

## Near-term priorities

The current near-term order is:

1. stabilize and use the Qwen-compatible runtime cleanly
2. finish final arbitration between `Q4_8` and `Q4_32` on the admissible no-drift base
3. continue disciplined metric work, not ad hoc metric proliferation
4. only then widen to new heavy branches (new model families, deeper probes, hybrid ideas)

## Non-goals

This repo is **not** trying to be:

- a commercial product
- a general-purpose LLM platform
- a polished full web app first
- a database-heavy orchestration system
- a cloud-managed pipeline

The center of gravity is still:
**research execution, trace collection, metric comparison, bounded interpretation**.

## Documentation

The repo is expected to grow toward a clearer docs set, including:

- architecture overview
- CLI reference
- first campaign tutorial
- ticket writing guide
- EGO substrate notes
- ALTER AI metric notes
- H100 troubleshooting

## Short version

ALTER AI Lab is a disciplined internal research repo for:

- running EGO TRIAD and ALTER AI campaigns,
- serving and instrumenting local Gemma / Qwen models,
- collecting token-level traces,
- testing coherence-vulnerability metrics under controlled perturbations,
- and enforcing a stricter research workflow than “just run another script”.

It is not a product.  
It is the workbench used to keep the research executable, auditable, and harder to derail.
