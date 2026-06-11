# CICL

Decision-aware context selection and compression for tool-using LLM agents.

CICL studies a simple question: when an agent is about to act, which context is
likely to change the next action rather than merely look semantically similar to
the task? The codebase implements instance context graphs, decision-aware
utility scoring, memory-card compression, baseline selectors, lightweight
rankers, and evaluation utilities for controlled and code-retrieval settings.

## Installation

```bash
git clone https://github.com/stephen-guan-researcher/CICL.git
cd CICL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,plots,graph]"
```

## Quick Checks

```bash
python -m unittest discover -s tests
python -m py_compile $(find cicl_agent tests training/scripts -name '*.py' -print)
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `cicl_agent/` | Core CICL package |
| `cicl_agent/core/` | Shared schemas, I/O helpers, and text utilities |
| `cicl_agent/memory/` | Instance context graph construction and memory curation |
| `cicl_agent/retrieval/` | BM25/hybrid retrieval and budget-aware assembly |
| `cicl_agent/causal/` | Decision utility scoring, LLM judgments, and memory cards |
| `cicl_agent/selectors/` | Baseline, graph, CICL, LLM-CICL, and learned selectors |
| `cicl_agent/runners/` | Experiment and evaluation entry points |
| `cicl_agent/learning/` | Pairwise rankers and lightweight policy training |
| `cicl_agent/integrations/` | Hosted-model client adapters |
| `tests/` | Unit and integration tests |
| `training/scripts/` | QLoRA adapter data/evaluation utilities |

## Minimal Example

The public repository intentionally excludes generated experiment artifacts and
downloaded benchmark data. To run CICL on your own tasks, provide a repository
path and a JSONL task file, then write outputs to a local ignored directory:

```bash
python -m cicl_agent.runners.runner \
  --repo /path/to/repo \
  --tasks /path/to/tasks.jsonl \
  --output artifacts/outputs/local_run \
  --budget 120
```

For hosted LLM judging, set the relevant environment variable before running an
LLM-enabled selector:

```bash
export DASHSCOPE_API_KEY="YOUR_TOKEN"
export ANTHROPIC_API_KEY="YOUR_TOKEN"
```

Do not commit local outputs, downloaded benchmarks, model weights, API tokens,
or paper/submission artifacts.

## Model Artifact

The Qwen3.5-9B QLoRA adapter used for surrogate-judge experiments is released
separately on Hugging Face:

```bash
hf download XinyuGuan/CICL \
  --local-dir artifacts/hf_release/cicl-qwen35-qlora-adapter
```

This is an adapter-only artifact. It must be loaded with the matching
`Qwen/Qwen3.5-9B` base model and should not be interpreted as a standalone
teacher model or a full coding agent.

## Scope

This repository is the source-code release for CICL. Full experiment outputs,
paper PDFs, submission bundles, internal review notes, and benchmark snapshots
are kept outside the GitHub source tree.
