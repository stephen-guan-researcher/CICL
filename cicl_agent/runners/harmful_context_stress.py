"""Harmful/stale context stress test for negative-transfer analysis."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean

from cicl_agent.agents.simulated import SimulatedReActAgent
from cicl_agent.core.io import write_jsonl
from cicl_agent.core.schema import AgentRun, ContextUnit, Task
from cicl_agent.core.text import tokenize
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.retrieval.assembly import BudgetPolicy
from cicl_agent.runners.runner import load_tasks
from cicl_agent.selectors.baselines import (
    AutoContextKGSelector,
    NoContextSelector,
    OracleGoldContextSelector,
    SummaryMemorySelector,
    VanillaRAGSelector,
)
from cicl_agent.selectors.cicl import CICLSelector


def run_harmful_context_stress(
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
    injected = inject_harmful_context(graph, [task for task in tasks if task.split != "base"])
    graph.apply_conflict_detection()

    selectors = [
        NoContextSelector(graph),
        VanillaRAGSelector(graph),
        SummaryMemorySelector(graph),
        AutoContextKGSelector(graph),
        CICLSelector(graph),
        CICLSelector(graph, name="CICL_w/o_conflict_filtering", conflict_filtering=False),
        CICLSelector(
            graph,
            name="CICL_w/o_stale_guard",
            conflict_filtering=False,
            budget_policy=BudgetPolicy(stale_penalty_floor=0.0),
        ),
        OracleGoldContextSelector(graph),
    ]
    agent = SimulatedReActAgent()
    eval_tasks = [task for task in tasks if task.split != "base"]

    output = Path(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    graph.save(output)

    runs: list[AgentRun] = []
    metric_rows: list[dict] = []
    for task in eval_tasks:
        history: list[AgentRun] = []
        for selector in selectors:
            selected = selector.select(task, budget=budget, history=history)
            run = agent.run(task, selector.name, selected)
            runs.append(run)
            history.append(run)
            row = harmful_metric_row(task, run, selected)
            metric_rows.append(row)

    summary_rows = summarize(metric_rows)
    write_jsonl(output / "harmful_context_runs.jsonl", (run.to_dict() for run in runs))
    write_jsonl(output / "injected_harmful_context.jsonl", (unit.to_dict() for unit in injected))
    _write_csv(output / "harmful_context_metrics.csv", metric_rows)
    _write_csv(output / "harmful_context_summary.csv", summary_rows)
    write_report(output / "report.md", summary_rows, metric_rows)
    return {"summary": summary_rows, "metrics": metric_rows}


def inject_harmful_context(graph: InstanceContextGraph, eval_tasks: list[Task]) -> list[ContextUnit]:
    injected: list[ContextUnit] = []
    for task in eval_tasks:
        spec = _harmful_spec(task)
        unit = ContextUnit(
            id=f"harmful:{task.id}",
            instance_id=task.instance_id,
            type="failure",
            source=f"legacy_notes/{task.id}.md",
            content=spec["content"],
            confidence=0.35,
            metadata={
                "harmful_context": True,
                "stale": True,
                "conflict_risk": True,
                "target_task_id": task.id,
                "wrong_action": spec["wrong_action"],
                "audit_note": "injected stale context for negative-transfer stress test",
            },
        )
        graph.add_unit(unit)
        injected.append(unit)
    return injected


def harmful_metric_row(task: Task, run: AgentRun, selected_units: list[ContextUnit]) -> dict:
    selected_ids = [unit.id for unit in selected_units]
    harmful_positions = [idx for idx, unit in enumerate(selected_units) if unit.metadata.get("harmful_context")]
    stale_positions = [idx for idx, unit in enumerate(selected_units) if unit.metadata.get("stale")]
    selected_gold = bool(set(selected_ids) & set(task.gold_context_ids))
    action_supported = _has_expected_action("\n".join(unit.content for unit in selected_units), task.expected_actions)
    first_harmful = bool(harmful_positions and harmful_positions[0] == 0)
    harmful_before_gold = False
    if harmful_positions:
        first_gold_position = next(
            (idx for idx, unit in enumerate(selected_units) if unit.id in task.gold_context_ids),
            None,
        )
        harmful_before_gold = first_gold_position is None or harmful_positions[0] < first_gold_position
    stress_success = bool(run.success and not first_harmful and not (harmful_before_gold and not selected_gold))
    return {
        "task_id": task.id,
        "method": run.method,
        "stress_success": int(stress_success),
        "simulated_success": int(run.success),
        "selected_gold": int(selected_gold),
        "action_supported": int(action_supported),
        "harmful_selected": int(bool(harmful_positions)),
        "first_harmful": int(first_harmful),
        "harmful_before_gold": int(harmful_before_gold),
        "stale_selected": int(bool(stale_positions)),
        "selected_count": len(selected_units),
        "tokens": run.tokens,
        "selected_context_ids": "|".join(selected_ids),
    }


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    out: list[dict] = []
    for method, items in sorted(grouped.items()):
        out.append(
            {
                "method": method,
                "n": len(items),
                "stress_success_rate": _avg(items, "stress_success"),
                "simulated_success_rate": _avg(items, "simulated_success"),
                "selected_gold_rate": _avg(items, "selected_gold"),
                "harmful_selected_rate": _avg(items, "harmful_selected"),
                "first_harmful_rate": _avg(items, "first_harmful"),
                "harmful_before_gold_rate": _avg(items, "harmful_before_gold"),
                "stale_selected_rate": _avg(items, "stale_selected"),
                "avg_tokens": round(mean(float(item["tokens"]) for item in items), 3),
            }
        )
    return out


def write_report(path: str | Path, summary_rows: list[dict], metric_rows: list[dict]) -> None:
    lines = [
        "# Harmful Context Stress Test",
        "",
        "This controlled stress test injects stale, misleading context units that are lexically similar to each eval task. It evaluates whether selectors pick harmful context before useful context.",
        "",
        "This is a negative-transfer diagnostic, not a real LLM coding-agent result.",
        "",
        "## Summary",
        "",
    ]
    lines.extend(_markdown_table(summary_rows))
    lines.extend(["", "## Per-Task Rows", ""])
    lines.extend(_markdown_table(metric_rows))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _harmful_spec(task: Task) -> dict[str, str]:
    text = f"{task.id} {task.instruction}".lower()
    if "timestamp" in text or "parse" in text:
        return {
            "wrong_action": "inspect legacy_parser.py and strip trailing Z",
            "content": (
                "Deprecated legacy note for timestamp parsing. Old behavior suggested "
                "inspecting legacy_parser.py and stripping trailing Z before fromisoformat. "
                "This is obsolete and should not guide new fixes, but it shares the words "
                "timestamp, parse, UTC, Z, parser, and ingestion with the current task."
            ),
        }
    if "url" in text or "endpoint" in text or "slash" in text:
        return {
            "wrong_action": "inspect request_callers.py and concatenate URL strings",
            "content": (
                "Deprecated legacy note for API endpoints. Old behavior suggested "
                "inspecting request_callers.py and concatenating base_url plus endpoint. "
                "This is obsolete and should not guide new fixes, but it shares the words "
                "URL, endpoint, slash, request builder, and API with the current task."
            ),
        }
    return {
        "wrong_action": "inspect legacy implementation",
        "content": (
            f"Deprecated legacy note for task: {task.instruction}. This stale context is obsolete "
            "and should not guide new fixes."
        ),
    }


def _has_expected_action(selected_text: str, expected_actions: list[str]) -> bool:
    if not expected_actions:
        return False
    selected_terms = set(tokenize(selected_text))
    for action in expected_actions:
        action_terms = set(tokenize(action))
        if action_terms and len(action_terms & selected_terms) / len(action_terms) >= 0.5:
            return True
    return False


def _markdown_table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(header, "")) for header in headers) + " |")
    return lines


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


def _avg(rows: list[dict], key: str) -> float:
    return round(mean(float(row[key]) for row in rows), 6) if rows else 0.0


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run harmful/stale context negative-transfer stress test.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, default=80)
    args = parser.parse_args()
    result = run_harmful_context_stress(args.repo, args.tasks, args.output, budget=args.budget)
    for row in result["summary"]:
        print(
            f"{row['method']}: stress_success={row['stress_success_rate']} "
            f"harmful_selected={row['harmful_selected_rate']} tokens={row['avg_tokens']}"
        )


if __name__ == "__main__":
    main()
