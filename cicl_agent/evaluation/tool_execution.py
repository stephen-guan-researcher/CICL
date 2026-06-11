"""Local tool-execution evaluation for compression suite outputs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from cicl_agent.agents.local_tool import LocalContextInspectionAgent, normalize_context_path
from cicl_agent.core.io import read_jsonl, write_jsonl
from cicl_agent.core.schema import AgentRun, ContextUnit, Task
from cicl_agent.evaluation.replay import _budget_sort_key, _resolve_selected_units, _task_unit_map, _unit_map


def evaluate_compression_suite_local_tools(
    repo: str | Path,
    suite_dir: str | Path,
    tasks_path: str | Path,
    max_inspections: int = 3,
) -> dict[str, list[dict]]:
    suite_dir = Path(suite_dir)
    tasks = {task.id: task for task in _load_tasks(tasks_path)}
    task_rows: list[dict] = []
    tool_runs: list[AgentRun] = []
    for budget_dir in sorted(suite_dir.glob("budget_*"), key=_budget_sort_key):
        if not budget_dir.is_dir():
            continue
        budget = int(budget_dir.name.split("_", 1)[1])
        result = evaluate_budget_local_tools(repo, budget_dir, tasks, budget, max_inspections=max_inspections)
        task_rows.extend(result["task_metrics"])
        tool_runs.extend(result["runs"])

    summary_rows = summarize_tool_rows(task_rows)
    _write_csv(suite_dir / "tool_task_metrics.csv", task_rows)
    _write_csv(suite_dir / "tool_summary.csv", summary_rows)
    write_jsonl(suite_dir / "tool_agent_runs.jsonl", (run.to_dict() for run in tool_runs))
    write_tool_report(suite_dir / "report" / "tool_summary.md", summary_rows)
    return {"task_metrics": task_rows, "summary": summary_rows, "runs": [run.to_dict() for run in tool_runs]}


def evaluate_budget_local_tools(
    repo: str | Path,
    budget_dir: str | Path,
    tasks: dict[str, Task],
    budget: int,
    max_inspections: int = 3,
) -> dict[str, list]:
    budget_dir = Path(budget_dir)
    raw_units = _unit_map(budget_dir / "context_units.jsonl")
    compressed_units = _task_unit_map(budget_dir / "compressed_context_units.jsonl")
    summary_units = _task_unit_map(budget_dir / "summary_context_units.jsonl")
    agent = LocalContextInspectionAgent(repo, max_inspections=max_inspections)
    rows: list[dict] = []
    runs: list[AgentRun] = []
    for row in read_jsonl(budget_dir / "agent_runs.jsonl"):
        source_run = AgentRun.from_dict(row)
        task = tasks.get(source_run.task_id)
        if task is None:
            continue
        selected_units = _resolve_selected_units(source_run, raw_units, compressed_units, summary_units)
        tool_run = agent.run(task, method=source_run.method, selected_units=selected_units)
        rows.append(evaluate_tool_run(tool_run, task, selected_units, budget))
        runs.append(tool_run)
    return {"task_metrics": rows, "runs": runs}


def evaluate_tool_run(
    run: AgentRun,
    task: Task,
    selected_units: list[ContextUnit],
    budget: int,
) -> dict:
    read_paths = [normalize_context_path(path) for path in run.metadata.get("read_paths", [])]
    read_paths = [path for path in read_paths if path]
    inspected_paths = [normalize_context_path(path) for path in run.metadata.get("inspected_paths", [])]
    inspected_paths = [path for path in inspected_paths if path]
    gold_paths = {normalize_context_path(context_id) for context_id in task.gold_context_ids}
    gold_paths = {path for path in gold_paths if path}
    selected_paths = {normalize_context_path(unit.source) for unit in selected_units}
    selected_paths.update(normalize_context_path(str(unit.metadata.get("original_source", ""))) for unit in selected_units)
    selected_paths = {path for path in selected_paths if path}

    gold_read = any(_path_matches(path, gold_paths) for path in read_paths)
    first_read_gold = bool(read_paths and _path_matches(read_paths[0], gold_paths))
    selected_gold = bool(set(run.selected_context_ids) & set(task.gold_context_ids))
    selected_gold_read = bool(selected_gold and gold_read)
    target_prefix_read = any(path.startswith(("targets/", "target/")) for path in read_paths)
    missing_count = len(run.metadata.get("missing_paths", []))
    card_units = [unit for unit in selected_units if unit.type == "memory_card"]
    card_read = any(unit.type == "memory_card" for unit in selected_units[: len(inspected_paths)])

    return {
        "budget": budget,
        "task_id": run.task_id,
        "method": run.method,
        "tool_success": int(run.success),
        "selected_gold": int(selected_gold),
        "gold_read": int(gold_read),
        "first_read_gold": int(first_read_gold),
        "selected_gold_read": int(selected_gold_read),
        "target_prefix_read": int(target_prefix_read),
        "read_count": len(read_paths),
        "missing_count": missing_count,
        "card_read": int(card_read),
        "card_selected": int(bool(card_units)),
        "tokens": run.tokens,
        "read_bytes": int(run.metadata.get("read_bytes", 0)),
        "inspected_paths": "|".join(inspected_paths),
        "read_paths": "|".join(read_paths),
        "gold_paths": "|".join(sorted(gold_paths)),
        "selected_paths": "|".join(sorted(selected_paths)),
        "selected_context_ids": "|".join(run.selected_context_ids),
    }


def summarize_tool_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["budget"]), str(row["method"]))].append(row)

    summary: list[dict] = []
    for (budget, method), items in sorted(grouped.items()):
        summary.append(
            {
                "budget": budget,
                "method": method,
                "n": len(items),
                "tool_success_rate": _avg(items, "tool_success"),
                "selected_gold_rate": _avg(items, "selected_gold"),
                "gold_read_rate": _avg(items, "gold_read"),
                "first_read_gold_rate": _avg(items, "first_read_gold"),
                "selected_gold_read_rate": _avg(items, "selected_gold_read"),
                "target_prefix_read_rate": _avg(items, "target_prefix_read"),
                "card_read_rate": _avg(items, "card_read"),
                "avg_read_count": _avg(items, "read_count"),
                "avg_missing_count": _avg(items, "missing_count"),
                "avg_read_bytes": round(mean(float(item["read_bytes"]) for item in items), 3) if items else 0.0,
                "avg_tokens": round(mean(float(item["tokens"]) for item in items), 3) if items else 0.0,
            }
        )
    return summary


def write_tool_report(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Local Tool Execution Evaluation",
        "",
        "This report uses a lightweight local inspection agent that actually reads repository files referenced by selected context units. It is stronger than log replay but still not LLM coding success.",
        "",
    ]
    if not rows:
        lines.append("No local tool rows were generated.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines.extend([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ])
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    lines.extend([
        "",
        "## Reading Notes",
        "",
        "- `tool_success_rate`: at least one selected context path was read successfully.",
        "- `gold_read_rate`: a local read action opened a path matching the gold context.",
        "- `first_read_gold_rate`: the first read path matched a gold context.",
        "- `target_prefix_read_rate`: the tool read a target/prompt-prefix file, which can indicate shortcut risk on RepoBench-style tasks.",
        "- This evaluates context-to-tool grounding, not final patch correctness.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_tasks(path: str | Path) -> list[Task]:
    return [Task.from_dict(row) for row in read_jsonl(path)]


def _path_matches(target: str, candidate_paths: set[str]) -> bool:
    if not target:
        return False
    for path in candidate_paths:
        if target == path or target.endswith(path) or path.endswith(target):
            return True
    return False


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

