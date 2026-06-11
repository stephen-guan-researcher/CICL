"""Evaluation metrics for context learning experiments."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from cicl_agent.core.schema import AgentRun, Task


def context_prf(selected: list[str], gold: list[str]) -> tuple[float, float, float]:
    selected_set = set(selected)
    gold_set = set(gold)
    if not selected_set and not gold_set:
        return 1.0, 1.0, 1.0
    precision = len(selected_set & gold_set) / max(1, len(selected_set))
    recall = len(selected_set & gold_set) / max(1, len(gold_set))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    variance = sum((value - m) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def mrr(selected: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for idx, context_id in enumerate(selected, start=1):
        if context_id in gold_set:
            return 1.0 / idx
    return 0.0


def summarize_runs(runs: list[AgentRun], tasks: dict[str, Task]) -> list[dict[str, float | str | int]]:
    grouped: dict[str, list[AgentRun]] = defaultdict(list)
    for run in runs:
        grouped[run.method].append(run)
    rows: list[dict[str, float | str | int]] = []
    for method, method_runs in sorted(grouped.items()):
        success = [float(run.success) for run in method_runs]
        precision: list[float] = []
        recall: list[float] = []
        f1: list[float] = []
        rr: list[float] = []
        for run in method_runs:
            task = tasks[run.task_id]
            p, r, f = context_prf(run.selected_context_ids, task.gold_context_ids)
            precision.append(p)
            recall.append(r)
            f1.append(f)
            rr.append(mrr(run.selected_context_ids, task.gold_context_ids))
        rows.append(
            {
                "method": method,
                "n": len(method_runs),
                "success_rate": round(mean(success), 6),
                "success_sem": round(sem(success), 6),
                "context_precision": round(mean(precision), 6),
                "context_recall": round(mean(recall), 6),
                "context_f1": round(mean(f1), 6),
                "context_mrr": round(mean(rr), 6),
                "avg_tokens": round(mean([run.tokens for run in method_runs]), 3),
                "avg_tool_calls": round(mean([run.tool_calls for run in method_runs]), 3),
                "avg_runtime_seconds": round(mean([run.runtime_seconds for run in method_runs]), 6),
            }
        )
    return rows


def write_metrics_csv(path: str | Path, rows: list[dict[str, float | str | int]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["method"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
