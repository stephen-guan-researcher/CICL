"""Convert Opus-labeled `llm_examples.jsonl` into Qwen-SFT chat format.

Mirrors `build_synthetic_sft_data.py` but takes the assistant JSON from real
Opus judgments (stored in `metadata` + `features` of each example) instead of
re-running the deterministic simulator.

Output schema matches `build_synthetic_sft_data.py`:
    {"task_id": ..., "context_id": ..., "messages": [...3 turns...]}

so `train_qwen_judge.py` and SFTTrainer can ingest it unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cicl_agent.causal.llm_judge import CounterfactualPromptTemplate
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.runners.runner import load_tasks


SYSTEM_PROMPT = "You are a precise causal critic. Return valid JSON only."

# Same key order as CounterfactualPromptTemplate.render — keeps the SFT target
# byte-identical to what Opus actually emitted.
_JUDGMENT_KEYS = (
    "no_context_action",
    "with_context_action",
    "action_shift",
    "necessity",
    "expected_outcome_uplift",
    "negative_transfer_risk",
    "reason",
    "confidence",
)


def opus_judgment_payload(example: dict) -> dict:
    """Reconstruct the 8-field JSON Opus returned, from the stored fields.

    Backwards-compatible: older checkpoints stored llm_* in ``features``
    (label leakage bug, since fixed). Newer ones store them in ``metadata``.
    Read metadata first, fall back to features.
    """
    md = example.get("metadata", {}) or {}
    feats = example.get("features", {}) or {}

    def pick(key: str, default: float = 0.0) -> float:
        if key in md:
            return float(md[key])
        if key in feats:
            return float(feats[key])
        return float(default)

    return {
        "no_context_action": md.get("no_context_action", ""),
        "with_context_action": md.get("with_context_action", ""),
        "action_shift": pick("llm_action_shift"),
        "necessity": pick("llm_necessity"),
        "expected_outcome_uplift": float(example.get("outcome_delta", 0.0)),
        "negative_transfer_risk": float(example.get("negative_transfer", 0.0)),
        "reason": md.get("reason", ""),
        "confidence": pick("llm_confidence"),
    }


def build(
    examples_path: Path,
    repo: Path,
    tasks_path: Path,
    output_path: Path,
) -> dict:
    tasks = load_tasks(tasks_path)
    if not tasks:
        raise ValueError(f"No tasks in {tasks_path}")

    instance_id = tasks[0].instance_id
    graph = InstanceContextGraph(instance_id)
    graph.build_from_repo(repo)
    for task in tasks:
        if task.split == "base":
            graph.add_task_memory(task)
    graph.apply_conflict_detection()

    tasks_by_id = {t.id: t for t in tasks}
    template = CounterfactualPromptTemplate()

    skipped_missing_task = 0
    skipped_missing_unit = 0
    written = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with examples_path.open() as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            task_id = ex["task_id"]
            context_id = ex["context_id"]

            task = tasks_by_id.get(task_id)
            if task is None:
                skipped_missing_task += 1
                continue
            unit = graph.units.get(context_id)
            if unit is None:
                skipped_missing_unit += 1
                continue

            user_prompt = template.render(task, unit)
            assistant = json.dumps(
                opus_judgment_payload(ex), ensure_ascii=False, indent=2
            )
            record = {
                "task_id": task_id,
                "context_id": context_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant},
                ],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return {
        "examples_path": str(examples_path),
        "output_path": str(output_path),
        "written": written,
        "skipped_missing_task": skipped_missing_task,
        "skipped_missing_unit": skipped_missing_unit,
        "instance_id": instance_id,
        "tasks_loaded": len(tasks),
        "graph_units": len(graph.units),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples",
        default="artifacts/outputs/latest/synthetic_opus_v1/llm_examples.jsonl",
    )
    parser.add_argument("--repo", default="experiments/data/synthetic/v1/repo")
    parser.add_argument("--tasks", default="experiments/data/synthetic/v1/tasks.jsonl")
    parser.add_argument("--output", default="training/data/opus_v1/full.jsonl")
    args = parser.parse_args()

    summary = build(
        examples_path=Path(args.examples),
        repo=Path(args.repo),
        tasks_path=Path(args.tasks),
        output_path=Path(args.output),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
