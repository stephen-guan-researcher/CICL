"""Executable toy patch pilot for workshop-stage evidence."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean

from cicl_agent.agents.toy_patch import ToyPatchAgent
from cicl_agent.core.io import write_jsonl
from cicl_agent.core.schema import AgentRun, Task
from cicl_agent.evaluation.metrics import context_prf
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.runners.runner import load_tasks
from cicl_agent.selectors.baselines import (
    AutoContextKGSelector,
    NoContextSelector,
    OracleGoldContextSelector,
    SummaryMemorySelector,
    VanillaRAGSelector,
)
from cicl_agent.selectors.cicl import CICLSelector


def run_toy_patch_pilot(
    repo: str | Path,
    tasks_path: str | Path,
    output: str | Path,
    budget: int = 80,
) -> dict[str, list[dict]]:
    tasks = load_tasks(tasks_path)
    graph = InstanceContextGraph(tasks[0].instance_id)
    graph.build_from_repo(repo)
    for task in tasks:
        if task.split == "base":
            graph.add_task_memory(task)
    graph.apply_conflict_detection()

    selectors = [
        NoContextSelector(graph),
        VanillaRAGSelector(graph),
        SummaryMemorySelector(graph),
        AutoContextKGSelector(graph),
        CICLSelector(graph),
        OracleGoldContextSelector(graph),
    ]
    agent = ToyPatchAgent(repo)
    output = Path(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    graph.save(output)

    eval_tasks = [task for task in tasks if task.split != "base"]
    runs: list[AgentRun] = []
    rows: list[dict] = []
    for task in eval_tasks:
        history: list[AgentRun] = []
        for selector in selectors:
            selected_units = selector.select(task, budget=budget, history=history)
            run = agent.run(task, selector.name, selected_units)
            runs.append(run)
            history.append(run)
            precision, recall, f1 = context_prf(run.selected_context_ids, task.gold_context_ids)
            rows.append(
                {
                    "task_id": task.id,
                    "method": selector.name,
                    "success": int(run.success),
                    "patch_applied": int(bool(run.metadata.get("patch_applied"))),
                    "pre_validation_success": int(bool(run.metadata.get("pre_validation_success"))),
                    "post_validation_success": int(bool(run.metadata.get("post_validation_success"))),
                    "selected_count": len(run.selected_context_ids),
                    "tokens": run.tokens,
                    "context_precision": round(precision, 6),
                    "context_recall": round(recall, 6),
                    "context_f1": round(f1, 6),
                    "actions": " | ".join(run.actions),
                    "selected_context_ids": "|".join(run.selected_context_ids),
                }
            )

    summary_rows = summarize_toy_patch(rows)
    write_jsonl(output / "toy_patch_runs.jsonl", (run.to_dict() for run in runs))
    _write_csv(output / "toy_patch_metrics.csv", rows)
    _write_csv(output / "toy_patch_summary.csv", summary_rows)
    write_toy_patch_report(output / "report.md", summary_rows, rows)
    return {"summary": summary_rows, "metrics": rows}


def summarize_toy_patch(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summary: list[dict] = []
    for method, items in sorted(grouped.items()):
        summary.append(
            {
                "method": method,
                "n": len(items),
                "patch_success_rate": _avg(items, "success"),
                "patch_applied_rate": _avg(items, "patch_applied"),
                "post_validation_rate": _avg(items, "post_validation_success"),
                "context_recall": _avg(items, "context_recall"),
                "context_f1": _avg(items, "context_f1"),
                "avg_tokens": round(mean(float(item["tokens"]) for item in items), 3),
            }
        )
    return summary


def write_toy_patch_report(path: str | Path, summary_rows: list[dict], metric_rows: list[dict]) -> None:
    path = Path(path)
    lines = [
        "# Toy Patch Execution Pilot",
        "",
        "This is a deterministic executable pilot. It injects known bugs into a temporary copy of the toy repo, applies a local patch only when selected context supports the relevant file/rule, and validates the patch with Python assertions.",
        "",
        "It is real local execution, but it is not an LLM coding-agent result.",
        "",
        "## Summary",
        "",
    ]
    lines.extend(_markdown_table(summary_rows))
    lines.extend(["", "## Per-Task Rows", ""])
    lines.extend(_markdown_table(metric_rows))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(row.get(header, "")) for header in headers) + " |")
    return lines


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _avg(rows: list[dict], key: str) -> float:
    return round(mean(float(row[key]) for row in rows), 6) if rows else 0.0


def _write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run executable toy patch pilot.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, default=80)
    args = parser.parse_args()

    result = run_toy_patch_pilot(args.repo, args.tasks, args.output, budget=args.budget)
    for row in result["summary"]:
        print(
            f"{row['method']}: patch_success={row['patch_success_rate']} "
            f"recall={row['context_recall']} tokens={row['avg_tokens']}"
        )


if __name__ == "__main__":
    main()
