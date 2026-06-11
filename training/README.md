# CICL Qwen QLoRA Training

Workspace for fine-tuning Qwen-family models as the CICL_Distilled student —
i.e. SFT the model to reproduce Opus-generated counterfactual context judgments
(action_shift / necessity / outcome_uplift / negative_transfer / reason).

## Layout

- `configs/`       — training configs (YAML / JSON), one per run
- `data/`          — processed SFT datasets (.jsonl, chat-template format)
- `scripts/`       — one-off scripts (data prep, model inspection, eval)
- `outputs/`       — checkpoints, adapters, eval reports (gitignored)
- `logs/`          — stdout/stderr from training runs (gitignored)

## Pipeline

1. `scripts/build_sft_dataset.py`  — convert Opus judgments → SFT chat format
2. `cicl_agent/learning/train_qwen_judge.py` — QLoRA SFT entry (lives in package)
3. `scripts/eval_judge.py`         — compare trained model vs Opus on val set

## Status

Pipeline smoke-test phase: synthetic data + Qwen3.5-9B + 2 × V100-16GB.
Real Opus-labeled training set is collected elsewhere; this dir validates the
pipeline end-to-end before swapping in the real dataset.
