"""Paired bootstrap significance utilities for CICL experiments."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import fields
from pathlib import Path

from cicl_agent.core.io import read_jsonl
from cicl_agent.evaluation.metrics import context_prf
from cicl_agent.runners.runner import load_tasks
from cicl_agent.core.schema import AgentRun


def metric_value(run: AgentRun, gold_context_ids: list[str], metric: str) -> float:
    if metric == "success":
        return float(run.success)
    if metric == "context_f1":
        return context_prf(run.selected_context_ids, gold_context_ids)[2]
    if metric == "context_mrr":
        gold = set(gold_context_ids)
        for idx, context_id in enumerate(run.selected_context_ids, start=1):
            if context_id in gold:
                return 1.0 / idx
        return 0.0
    if metric == "tokens":
        return -float(run.tokens)
    raise ValueError(f"Unsupported metric: {metric}")


def paired_bootstrap(
    runs_path: str | Path,
    tasks_path: str | Path,
    method_a: str,
    method_b: str,
    metric: str = "context_f1",
    samples: int = 2000,
    seed: int = 17,
) -> dict[str, float | str | int]:
    tasks = {task.id: task for task in load_tasks(tasks_path)}
    run_fields = {field.name for field in fields(AgentRun)}
    runs = [AgentRun.from_dict({key: value for key, value in row.items() if key in run_fields}) for row in read_jsonl(runs_path)]
    by_task_method = {(run.task_id, run.method): run for run in runs}
    task_ids = sorted(
        task_id
        for task_id in tasks
        if (task_id, method_a) in by_task_method and (task_id, method_b) in by_task_method
    )
    if not task_ids:
        raise ValueError(f"No paired runs found for {method_a} and {method_b}")
    deltas = []
    for task_id in task_ids:
        task = tasks[task_id]
        a = metric_value(by_task_method[(task_id, method_a)], task.gold_context_ids, metric)
        b = metric_value(by_task_method[(task_id, method_b)], task.gold_context_ids, metric)
        deltas.append(a - b)
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    boot = []
    for _ in range(samples):
        sample = [rng.choice(deltas) for _ in deltas]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    low = boot[int(0.025 * (samples - 1))]
    high = boot[int(0.975 * (samples - 1))]
    p_value = sum(1 for value in boot if value <= 0) / samples if observed >= 0 else sum(1 for value in boot if value >= 0) / samples
    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "n_pairs": len(deltas),
        "delta_mean": round(observed, 6),
        "ci95_low": round(low, 6),
        "ci95_high": round(high, 6),
        "bootstrap_p": round(p_value, 6),
        "samples": samples,
    }


def write_rows(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired bootstrap comparisons.")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--method-a", required=True)
    parser.add_argument("--method-b", action="append", required=True)
    parser.add_argument("--metric", default="context_f1", choices=["success", "context_f1", "context_mrr", "tokens"])
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [
        paired_bootstrap(args.runs, args.tasks, args.method_a, method_b, args.metric, args.samples)
        for method_b in args.method_b
    ]
    write_rows(args.output, rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
