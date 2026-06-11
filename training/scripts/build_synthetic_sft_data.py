"""Build a tiny synthetic SFT dataset for pipeline smoke-testing.

Uses the deterministic simulator judge (no real LLM) over the toy_repo +
toy_tasks fixture so we can validate the full QLoRA training pipeline before
the real Opus-labeled set is ready.

Output is JSONL of chat-format records:
    {"messages": [{"role": "system", ...},
                  {"role": "user",   ...},
                  {"role": "assistant", ...}]}

The user prompt is rendered by ``CounterfactualPromptTemplate`` so the SFT
distribution matches the prompt the model will see at inference time.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from cicl_agent.causal.llm_judge import (
    CounterfactualPromptTemplate,
    LLMCausalContextJudge,
)
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.retrieval.retrievers import HybridRetriever
from cicl_agent.runners.runner import load_tasks


SYSTEM_PROMPT = "You are a precise causal critic. Return valid JSON only."

# Mirror the key order in CounterfactualPromptTemplate.render so the SFT target
# JSON looks identical to what a real LLM would have produced.
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


def judgment_completion(judgment) -> str:
    payload = {key: getattr(judgment, key) for key in _JUDGMENT_KEYS}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build(
    repo: Path,
    tasks_path: Path,
    output_dir: Path,
    top_k: int = 12,
    val_ratio: float = 0.1,
    seed: int = 17,
    max_examples: int | None = None,
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

    judge = LLMCausalContextJudge()  # no client → deterministic simulator
    template = CounterfactualPromptTemplate()
    retriever = HybridRetriever(list(graph.units.values()))

    records: list[dict] = []
    for task in tasks:
        candidates = retriever.search(task, top_k=top_k)
        for unit, _retrieval_score in candidates:
            judgment = judge.judge(task, unit)
            user_prompt = template.render(task, unit)
            record = {
                "task_id": task.id,
                "context_id": unit.id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": judgment_completion(judgment)},
                ],
            }
            records.append(record)
            if max_examples and len(records) >= max_examples:
                break
        if max_examples and len(records) >= max_examples:
            break

    rng = random.Random(seed)
    rng.shuffle(records)

    val_size = max(1, int(round(len(records) * val_ratio))) if len(records) >= 4 else 0
    val = records[:val_size]
    train = records[val_size:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "synthetic_train.jsonl"
    val_path = output_dir / "synthetic_val.jsonl"

    with train_path.open("w", encoding="utf-8") as fh:
        for record in train:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as fh:
        for record in val:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "train_examples": len(train),
        "val_examples": len(val),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "tasks": len(tasks),
        "instance_id": instance_id,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="experiments/examples/toy_repo")
    parser.add_argument("--tasks", default="experiments/examples/toy_tasks.jsonl")
    parser.add_argument("--output-dir", default="training/data/synthetic_smoke")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    summary = build(
        repo=Path(args.repo),
        tasks_path=Path(args.tasks),
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_examples=args.max_examples,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
