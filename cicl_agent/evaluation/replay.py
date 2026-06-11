"""Replay-style evaluation for selected context and recorded agent actions.

This module does not execute code or call a real LLM. It replays logged
`AgentRun` traces and asks a stricter question than context recall: did the
recorded inspect/action path actually point at selected and gold context?
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from cicl_agent.core.io import read_jsonl
from cicl_agent.core.schema import AgentRun, ContextUnit, Task


INSPECT_RE = re.compile(r"^\s*inspect\s+(.+?)\s*$", re.IGNORECASE)
PATH_RE = re.compile(r"([A-Za-z0-9_./-]+\.(?:py|md|json|yaml|yml|toml|txt))")
TARGET_PREFIXES = ("targets/", "target/")


def evaluate_compression_suite_replay(
    suite_dir: str | Path,
    tasks_path: str | Path,
) -> dict[str, list[dict]]:
    """Evaluate all `budget_*` runs in a compression suite directory."""

    suite_dir = Path(suite_dir)
    tasks = {task.id: task for task in _load_tasks(tasks_path)}
    task_rows: list[dict] = []
    for budget_dir in sorted(suite_dir.glob("budget_*"), key=_budget_sort_key):
        if not budget_dir.is_dir():
            continue
        budget = int(budget_dir.name.split("_", 1)[1])
        task_rows.extend(evaluate_budget_replay(budget_dir, tasks, budget))

    summary_rows = summarize_replay_rows(task_rows)
    _write_csv(suite_dir / "replay_task_metrics.csv", task_rows)
    _write_csv(suite_dir / "replay_summary.csv", summary_rows)
    write_replay_report(suite_dir / "report" / "replay_summary.md", summary_rows)
    return {"task_metrics": task_rows, "summary": summary_rows}


def evaluate_budget_replay(
    budget_dir: str | Path,
    tasks: dict[str, Task],
    budget: int,
) -> list[dict]:
    budget_dir = Path(budget_dir)
    raw_units = _unit_map(budget_dir / "context_units.jsonl")
    compressed_units = _task_unit_map(budget_dir / "compressed_context_units.jsonl")
    summary_units = _task_unit_map(budget_dir / "summary_context_units.jsonl")
    rows: list[dict] = []
    for row in read_jsonl(budget_dir / "agent_runs.jsonl"):
        run = AgentRun.from_dict(row)
        task = tasks.get(run.task_id)
        if task is None:
            continue
        selected_units = _resolve_selected_units(run, raw_units, compressed_units, summary_units)
        rows.append(evaluate_run_replay(run, task, selected_units, budget))
    return rows


def evaluate_run_replay(
    run: AgentRun,
    task: Task,
    selected_units: list[ContextUnit],
    budget: int,
) -> dict:
    selected_ids = set(run.selected_context_ids)
    gold_ids = set(task.gold_context_ids)
    selected_gold = bool(selected_ids & gold_ids)

    inspect_targets = [_normalize_path(target) for target in _inspect_targets(run.actions)]
    inspect_targets = [target for target in inspect_targets if target]
    selected_paths = _paths_for_units(selected_units)
    gold_paths = {_normalize_path(context_id) for context_id in task.gold_context_ids}
    gold_paths = {path for path in gold_paths if path}

    action_supported = any(_path_matches(target, selected_paths) for target in inspect_targets)
    gold_action_supported = any(_path_matches(target, gold_paths) for target in inspect_targets)
    first_action_gold = bool(inspect_targets and _path_matches(inspect_targets[0], gold_paths))
    expected_action_supported = _expected_action_supported(run.actions, task.expected_actions)
    easy_without_context = task.metadata.get("requires_context") is False
    replay_success = bool(gold_action_supported or expected_action_supported or easy_without_context)

    target_prefix_selected = any(
        path.startswith(TARGET_PREFIXES) for path in selected_paths
    )
    card_units = [unit for unit in selected_units if unit.type == "memory_card"]
    card_actionable = [
        unit for unit in card_units
        if "Action hint:" in unit.content and "Scope:" in unit.content
    ]
    evidence_units = [
        unit for unit in selected_units
        if "Evidence:" in unit.content or "Summary:" in unit.content
    ]

    return {
        "budget": budget,
        "task_id": run.task_id,
        "method": run.method,
        "selected_gold": int(selected_gold),
        "action_supported": int(action_supported),
        "gold_action_supported": int(gold_action_supported),
        "first_action_gold": int(first_action_gold),
        "expected_action_supported": int(expected_action_supported),
        "replay_success": int(replay_success),
        "selected_but_unused_gold": int(selected_gold and not gold_action_supported),
        "target_prefix_selected": int(target_prefix_selected),
        "card_actionability": round(len(card_actionable) / max(1, len(card_units)), 6) if card_units else "",
        "evidence_marker_rate": round(len(evidence_units) / max(1, len(selected_units)), 6),
        "tokens": run.tokens,
        "selected_count": len(run.selected_context_ids),
        "inspect_targets": "|".join(inspect_targets),
        "selected_paths": "|".join(sorted(selected_paths)),
        "gold_paths": "|".join(sorted(gold_paths)),
        "selected_context_ids": "|".join(run.selected_context_ids),
    }


def summarize_replay_rows(rows: list[dict]) -> list[dict]:
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
                "replay_success_rate": _avg(items, "replay_success"),
                "selected_gold_rate": _avg(items, "selected_gold"),
                "gold_action_rate": _avg(items, "gold_action_supported"),
                "first_action_gold_rate": _avg(items, "first_action_gold"),
                "selected_unused_gold_rate": _avg(items, "selected_but_unused_gold"),
                "action_supported_rate": _avg(items, "action_supported"),
                "target_prefix_selection_rate": _avg(items, "target_prefix_selected"),
                "avg_evidence_marker_rate": _avg(items, "evidence_marker_rate"),
                "avg_card_actionability": _optional_avg(items, "card_actionability"),
                "avg_tokens": round(mean(float(item["tokens"]) for item in items), 3),
            }
        )
    return summary


def write_replay_report(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Replay Execution Evaluation",
        "",
        "This report replays logged actions. It is stricter than context recall: a run only gets `gold_action_rate` credit when an inspect action points to a gold context path.",
        "",
    ]
    if not rows:
        lines.append("No replay rows were generated.")
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
        "- `selected_gold_rate`: a gold context id was selected somewhere in the context block.",
        "- `gold_action_rate`: the replayed inspect action pointed to a gold context path.",
        "- `selected_unused_gold_rate`: gold context was selected but not actioned in the trace.",
        "- `target_prefix_selection_rate`: the method selected prompt/target-prefix files as context, a possible shortcut risk for RepoBench-style retrieval tasks.",
        "- This is still replay evidence, not real code execution.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_selected_units(
    run: AgentRun,
    raw_units: dict[str, ContextUnit],
    compressed_units: dict[tuple[str, str], ContextUnit],
    summary_units: dict[tuple[str, str], ContextUnit],
) -> list[ContextUnit]:
    if "Summary" in run.method:
        task_units = summary_units
    elif "Compressed" in run.method:
        task_units = compressed_units
    else:
        task_units = {}
    selected: list[ContextUnit] = []
    for context_id in run.selected_context_ids:
        unit = task_units.get((run.task_id, context_id)) or raw_units.get(context_id)
        if unit is not None:
            selected.append(unit)
    return selected


def _unit_map(path: Path) -> dict[str, ContextUnit]:
    return {row["id"]: ContextUnit.from_dict(row) for row in read_jsonl(path)}


def _task_unit_map(path: Path) -> dict[tuple[str, str], ContextUnit]:
    units: dict[tuple[str, str], ContextUnit] = {}
    for row in read_jsonl(path):
        unit = ContextUnit.from_dict(row)
        task_id = str(unit.metadata.get("task_id", ""))
        if task_id:
            units[(task_id, unit.id)] = unit
    return units


def _load_tasks(path: str | Path) -> list[Task]:
    return [Task.from_dict(row) for row in read_jsonl(path)]


def _inspect_targets(actions: list[str]) -> list[str]:
    targets = []
    for action in actions:
        match = INSPECT_RE.match(action)
        if match:
            targets.append(match.group(1))
    return targets


def _paths_for_units(units: list[ContextUnit]) -> set[str]:
    paths: set[str] = set()
    for unit in units:
        for value in [
            unit.id,
            unit.source,
            str(unit.metadata.get("original_source", "")),
            str(unit.metadata.get("path", "")),
        ]:
            path = _normalize_path(value)
            if path:
                paths.add(path)
        for match in PATH_RE.finditer(unit.content):
            paths.add(match.group(1))
    return paths


def _normalize_path(value: str) -> str:
    value = value.strip().strip("`'\"")
    if not value:
        return ""
    if value.startswith("inspect "):
        value = value.split(" ", 1)[1]
    for prefix in ["compressed:", "summary:", "file:", "symbol:", "memory:"]:
        if value.startswith(prefix):
            value = value[len(prefix):]
    match = PATH_RE.search(value)
    if match:
        return match.group(1)
    if ":" in value and value.rsplit(":", 1)[1].isdigit():
        value = value.rsplit(":", 1)[0]
    return value


def _path_matches(target: str, candidate_paths: set[str]) -> bool:
    if not target:
        return False
    for path in candidate_paths:
        if target == path or target.endswith(path) or path.endswith(target):
            return True
    return False


def _expected_action_supported(actions: list[str], expected_actions: list[str]) -> bool:
    if not expected_actions:
        return False
    action_text = " ".join(actions).lower()
    for expected in expected_actions:
        terms = [term for term in re.split(r"[^a-zA-Z0-9_./-]+", expected.lower()) if term]
        if terms and sum(1 for term in terms if term in action_text) / len(terms) >= 0.5:
            return True
    return False


def _avg(rows: list[dict], key: str) -> float:
    return round(mean(float(row[key]) for row in rows), 6) if rows else 0.0


def _optional_avg(rows: list[dict], key: str) -> str | float:
    values = [float(row[key]) for row in rows if row.get(key) != ""]
    return round(mean(values), 6) if values else ""


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


def _budget_sort_key(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0
