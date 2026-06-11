"""Raw-context versus causal-compressed-context experiment."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import random
import shutil

from cicl_agent.agents.simulated import SimulatedReActAgent
from cicl_agent.causal.compression import CausalMemoryCard, CausalMemoryCompressor, ExtractiveContextCompressor
from cicl_agent.causal.judgment import judgment_to_score
from cicl_agent.causal.llm_judge import LLMCausalContextJudge
from cicl_agent.core.io import write_jsonl
from cicl_agent.core.schema import AgentRun, ContextUnit, Task
from cicl_agent.evaluation.metrics import context_prf, mean, sem
from cicl_agent.evaluation.replay import evaluate_compression_suite_replay
from cicl_agent.evaluation.tool_execution import evaluate_compression_suite_local_tools
from cicl_agent.integrations.llm_clients import build_llm_judge, normalize_provider, resolve_llm_model
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.retrieval.retrievers import HybridRetriever
from cicl_agent.runners.runner import load_tasks
from cicl_agent.selectors.base import fit_budget


CASE_STUDY_METHODS = [
    "RawCICLSelection",
    "SelectedThenCompressedCICL",
    "SummaryCompressedCICL",
    "CausalCompressedCICL",
]

SIGNIFICANCE_COMPARISONS = [
    ("CausalCompressedCICL", "RawCICLSelection"),
    ("SummaryCompressedCICL", "RawCICLSelection"),
    ("CausalCompressedCICL", "SummaryCompressedCICL"),
    ("SelectedThenCompressedCICL", "RawCICLSelection"),
    ("SelectedThenSummaryCompressedCICL", "RawCICLSelection"),
]

SIGNIFICANCE_METRICS = ["success", "context_recall", "context_f1", "tokens_saved"]


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: str | Path, rows: list[dict], title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_graph(repo: str | Path, tasks: list[Task]) -> InstanceContextGraph:
    graph = InstanceContextGraph(tasks[0].instance_id)
    graph.build_from_repo(repo)
    for task in tasks:
        if task.split == "base":
            graph.add_task_memory(task)
    graph.apply_conflict_detection()
    return graph


def ranked_candidates(
    graph: InstanceContextGraph,
    task: Task,
    judge: LLMCausalContextJudge,
    top_k: int,
) -> tuple[list[ContextUnit], dict[str, object]]:
    retrieved = HybridRetriever(list(graph.units.values())).search(task, top_k=top_k)
    candidates: dict[str, ContextUnit] = {unit.id: unit for unit, _ in retrieved}
    for unit, _ in retrieved[: max(1, top_k // 3)]:
        for neighbor in graph.neighbors(unit.id, depth=1):
            candidates.setdefault(neighbor.id, neighbor)

    judgments = {unit.id: judge.judge(task, unit) for unit in candidates.values()}
    scored = [
        (unit, judgment_to_score(judgments[unit.id], unit))
        for unit in candidates.values()
    ]
    scored.sort(key=lambda row: row[1].final_score, reverse=True)
    return [unit for unit, _ in scored], judgments


def run_compression_experiment(
    repo: str | Path,
    tasks_path: str | Path,
    output: str | Path,
    budget: int,
    top_k: int = 12,
    max_eval_tasks: int | None = None,
    llm_provider: str = "simulator",
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key_env: str = "DASHSCOPE_API_KEY",
    llm_timeout_seconds: float = 60.0,
) -> dict[str, list[dict]]:
    tasks = load_tasks(tasks_path)
    if not tasks:
        raise ValueError(f"No tasks found in {tasks_path}")
    graph = build_graph(repo, tasks)
    eval_tasks = [task for task in tasks if task.split != "base"]
    if max_eval_tasks is not None:
        eval_tasks = eval_tasks[:max_eval_tasks]

    llm_provider = normalize_provider(llm_provider)
    judge = build_llm_judge(
        provider=llm_provider,
        model=resolve_llm_model(llm_provider, llm_model),
        base_url=llm_base_url,
        api_key_env=llm_api_key_env,
        timeout_seconds=llm_timeout_seconds,
    )
    compressor = CausalMemoryCompressor(judge=judge)
    summary_compressor = ExtractiveContextCompressor()
    agent = SimulatedReActAgent()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    runs: list[AgentRun] = []
    per_task_rows: list[dict] = []
    cards: list[CausalMemoryCard] = []
    compressed_units: list[ContextUnit] = []
    summary_units: list[ContextUnit] = []

    for task in eval_tasks:
        ranked, judgments = ranked_candidates(graph, task, judge=judge, top_k=top_k)
        raw_selected = fit_budget(ranked, budget)

        compressed_ranked: list[ContextUnit] = []
        compressed_by_id: dict[str, ContextUnit] = {}
        summary_ranked: list[ContextUnit] = []
        summary_by_id: dict[str, ContextUnit] = {}
        for unit in ranked:
            compressed, card, judgment = compressor.compress(task, unit, judgments.get(unit.id))
            summary_compressed = summary_compressor.compress(task, unit)
            compressed.metadata["task_id"] = task.id
            summary_compressed.metadata["task_id"] = task.id
            compressed_ranked.append(compressed)
            compressed_by_id[unit.id] = compressed
            summary_ranked.append(summary_compressed)
            summary_by_id[unit.id] = summary_compressed
            cards.append(card)
            compressed_units.append(compressed)
            summary_units.append(summary_compressed)
            judgments[unit.id] = judgment
        compressed_selected = fit_budget(compressed_ranked, budget)
        selected_then_compressed = [compressed_by_id[unit.id] for unit in raw_selected if unit.id in compressed_by_id]
        summary_selected = fit_budget(summary_ranked, budget)
        selected_then_summary = [summary_by_id[unit.id] for unit in raw_selected if unit.id in summary_by_id]

        for method, selected in [
            ("RawCICLSelection", raw_selected),
            ("SelectedThenSummaryCompressedCICL", selected_then_summary),
            ("SelectedThenCompressedCICL", selected_then_compressed),
            ("SummaryCompressedCICL", summary_selected),
            ("CausalCompressedCICL", compressed_selected),
        ]:
            run = agent.run(task, method=method, selected_units=selected)
            runs.append(run)
            p, r, f1 = context_prf(run.selected_context_ids, task.gold_context_ids)
            original_tokens = sum(int(unit.metadata.get("original_token_cost", unit.token_cost)) for unit in selected)
            compressed_tokens = sum(unit.token_cost for unit in selected)
            compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            per_task_rows.append(
                {
                    "task_id": task.id,
                    "method": method,
                    "success": int(run.success),
                    "selected_count": len(selected),
                    "tokens": compressed_tokens,
                    "original_tokens": original_tokens,
                    "compression_ratio": round(compression_ratio, 6),
                    "context_precision": round(p, 6),
                    "context_recall": round(r, 6),
                    "context_f1": round(f1, 6),
                    "selected_context_ids": "|".join(run.selected_context_ids),
                }
            )

    summary_rows = summarize_compression(runs, per_task_rows)
    write_jsonl(output / "agent_runs.jsonl", (run.to_dict() for run in runs))
    write_jsonl(output / "compressed_context_units.jsonl", (unit.to_dict() for unit in compressed_units))
    write_jsonl(output / "summary_context_units.jsonl", (unit.to_dict() for unit in summary_units))
    write_jsonl(output / "causal_memory_cards.jsonl", (card.to_dict() for card in cards))
    write_csv(output / "compression_metrics.csv", per_task_rows)
    write_csv(output / "compression_summary.csv", summary_rows)
    graph.save(output)
    return {"summary": summary_rows, "metrics": per_task_rows}


def run_compression_suite(
    repo: str | Path,
    tasks_path: str | Path,
    output: str | Path,
    budgets: list[int],
    top_k: int = 12,
    max_eval_tasks: int | None = None,
    llm_provider: str = "simulator",
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key_env: str = "DASHSCOPE_API_KEY",
    llm_timeout_seconds: float = 60.0,
) -> dict[str, list[dict]]:
    output = Path(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    metric_rows: list[dict] = []
    for budget in budgets:
        run_dir = output / f"budget_{budget}"
        result = run_compression_experiment(
            repo=repo,
            tasks_path=tasks_path,
            output=run_dir,
            budget=budget,
            top_k=top_k,
            max_eval_tasks=max_eval_tasks,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key_env=llm_api_key_env,
            llm_timeout_seconds=llm_timeout_seconds,
        )
        for row in result["summary"]:
            summary_rows.append({"budget": budget, **row})
        for row in result["metrics"]:
            metric_rows.append({"budget": budget, **row})

    write_csv(output / "compression_budget_sweep.csv", summary_rows)
    write_csv(output / "compression_task_metrics.csv", metric_rows)
    case_rows = extract_case_studies(metric_rows, load_tasks(tasks_path))
    write_csv(output / "case_studies.csv", case_rows)
    significance_rows = paired_compression_significance(metric_rows)
    write_csv(output / "paired_significance.csv", significance_rows)
    report_dir = output / "report"
    write_markdown_table(report_dir / "compression_budget_sweep.md", summary_rows, "Compression Budget Sweep")
    write_markdown_table(report_dir / "compression_task_metrics.md", metric_rows[:120], "Compression Task Metrics Sample")
    write_markdown_table(report_dir / "paired_significance.md", significance_rows, "Paired Bootstrap Significance")
    write_case_study_report(report_dir / "case_studies.md", case_rows)
    replay_result = evaluate_compression_suite_replay(output, tasks_path)
    tool_result = evaluate_compression_suite_local_tools(repo, output, tasks_path)
    write_compression_report(output, summary_rows, case_rows, significance_rows)
    return {
        "summary": summary_rows,
        "metrics": metric_rows,
        "case_studies": case_rows,
        "significance": significance_rows,
        "replay": replay_result["summary"],
        "local_tool": tool_result["summary"],
    }


def write_compression_report(
    output: Path,
    summary_rows: list[dict],
    case_rows: list[dict] | None = None,
    significance_rows: list[dict] | None = None,
) -> None:
    report = output / "report" / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    best_by_budget: list[str] = []
    for budget in sorted({int(row["budget"]) for row in summary_rows}):
        rows = [row for row in summary_rows if int(row["budget"]) == budget]
        if not rows:
            continue
        best_recall = max(rows, key=lambda row: float(row["context_recall"]))
        best_ratio = min(rows, key=lambda row: float(row["avg_compression_ratio"]))
        best_by_budget.append(
            f"- Budget {budget}: best recall `{best_recall['method']}`={best_recall['context_recall']}; "
            f"lowest ratio `{best_ratio['method']}`={best_ratio['avg_compression_ratio']}."
        )
    lines = [
        "# Causal Compression Suite Report",
        "",
        "This report compares raw selected context with causal memory-card compression.",
        "",
        "## Tables",
        "",
        "- [compression_budget_sweep.md](compression_budget_sweep.md)",
        "- [compression_task_metrics.md](compression_task_metrics.md)",
        "- [paired_significance.md](paired_significance.md)",
        "- [case_studies.md](case_studies.md)",
        "- [replay_summary.md](replay_summary.md)",
        "- [tool_summary.md](tool_summary.md)",
        "",
        "## Quick Read",
        "",
        *best_by_budget,
        "",
        "## Case Study Counts",
        "",
        *_case_study_count_lines(case_rows or []),
        "",
        "## Paired Significance Highlights",
        "",
        *_significance_highlight_lines(significance_rows or []),
        "",
        "## Method Meaning",
        "",
        "- `RawCICLSelection`: select raw context directly.",
        "- `SelectedThenSummaryCompressedCICL`: select raw context first, then apply generic extractive compression.",
        "- `SelectedThenCompressedCICL`: select raw context first, then compress the selected ids.",
        "- `SummaryCompressedCICL`: apply generic extractive compression before budget fitting.",
        "- `CausalCompressedCICL`: compress candidates first, then select under the same budget.",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def _case_study_count_lines(case_rows: list[dict]) -> list[str]:
    if not case_rows:
        return ["No automatically selected case studies."]
    counts: dict[str, int] = defaultdict(int)
    for row in case_rows:
        counts[str(row["category"])] += 1
    return [f"- `{category}`: {count}" for category, count in sorted(counts.items())]


def _significance_highlight_lines(significance_rows: list[dict]) -> list[str]:
    if not significance_rows:
        return ["No paired significance rows."]
    highlights = []
    target_rows = [
        row for row in significance_rows
        if row["metric"] == "context_recall"
        and row["method_a"] in {"CausalCompressedCICL", "SummaryCompressedCICL"}
        and row["method_b"] == "RawCICLSelection"
    ]
    for row in target_rows:
        highlights.append(
            f"- Budget {row['budget']} `{row['method_a']}` vs `{row['method_b']}` "
            f"context_recall delta={row['delta_mean']} CI95=[{row['ci95_low']}, {row['ci95_high']}], "
            f"p={row['bootstrap_p']}."
        )
    return highlights or ["No recall comparison highlights."]


def paired_compression_significance(
    metric_rows: list[dict],
    samples: int = 2000,
    seed: int = 17,
) -> list[dict]:
    grouped: dict[tuple[int, str], dict[str, dict]] = defaultdict(dict)
    for row in metric_rows:
        grouped[(int(row["budget"]), str(row["task_id"]))][str(row["method"])] = row

    rows: list[dict] = []
    budgets = sorted({int(row["budget"]) for row in metric_rows})
    for budget in budgets:
        budget_groups = {
            key: methods
            for key, methods in grouped.items()
            if key[0] == budget
        }
        for method_a, method_b in SIGNIFICANCE_COMPARISONS:
            for metric in SIGNIFICANCE_METRICS:
                deltas = []
                for methods in budget_groups.values():
                    if method_a not in methods or method_b not in methods:
                        continue
                    deltas.append(_metric_delta(methods[method_a], methods[method_b], metric))
                if not deltas:
                    continue
                rows.append(_bootstrap_row(budget, method_a, method_b, metric, deltas, samples, seed))
    return rows


def _metric_delta(row_a: dict, row_b: dict, metric: str) -> float:
    if metric in {"success", "context_recall", "context_f1"}:
        return float(row_a[metric]) - float(row_b[metric])
    if metric == "tokens_saved":
        return float(row_b["tokens"]) - float(row_a["tokens"])
    raise ValueError(f"Unsupported compression metric: {metric}")


def _bootstrap_row(
    budget: int,
    method_a: str,
    method_b: str,
    metric: str,
    deltas: list[float],
    samples: int,
    seed: int,
) -> dict:
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed + budget + len(method_a) * 13 + len(method_b) * 7 + len(metric))
    boot = []
    for _ in range(samples):
        sample = [rng.choice(deltas) for _ in deltas]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    low = boot[int(0.025 * (samples - 1))]
    high = boot[int(0.975 * (samples - 1))]
    if observed >= 0:
        p_value = sum(1 for value in boot if value <= 0.0) / samples
    else:
        p_value = sum(1 for value in boot if value >= 0.0) / samples
    return {
        "budget": budget,
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "n_pairs": len(deltas),
        "delta_mean": round(observed, 6),
        "ci95_low": round(low, 6),
        "ci95_high": round(high, 6),
        "bootstrap_p": round(p_value, 6),
        "samples": samples,
        "interpretation": _significance_interpretation(observed, low, high, metric),
    }


def _significance_interpretation(observed: float, low: float, high: float, metric: str) -> str:
    if low > 0:
        direction = "method_a_better"
    elif high < 0:
        direction = "method_b_better"
    else:
        direction = "inconclusive"
    if metric == "tokens_saved" and direction == "method_a_better":
        return "method_a_uses_fewer_tokens"
    if metric == "tokens_saved" and direction == "method_b_better":
        return "method_b_uses_fewer_tokens"
    return direction


def extract_case_studies(metric_rows: list[dict], tasks: list[Task], max_per_category: int = 3) -> list[dict]:
    """Select objective qualitative examples from paired per-task metrics.

    The categories intentionally include positive and negative evidence so the
    generated report cannot silently cherry-pick only CICL-friendly cases.
    """

    tasks_by_id = {task.id: task for task in tasks}
    grouped: dict[tuple[int, str], dict[str, dict]] = defaultdict(dict)
    for row in metric_rows:
        grouped[(int(row["budget"]), str(row["task_id"]))][str(row["method"])] = row

    candidates: dict[str, list[dict]] = defaultdict(list)
    for (budget, task_id), rows in grouped.items():
        if not all(method in rows for method in CASE_STUDY_METHODS):
            continue
        task = tasks_by_id.get(task_id)
        if task is None:
            continue

        raw = rows["RawCICLSelection"]
        selected_then = rows["SelectedThenCompressedCICL"]
        summary = rows["SummaryCompressedCICL"]
        causal = rows["CausalCompressedCICL"]

        causal_gain = _score_gain(causal, raw)
        if causal_gain > 0:
            candidates["causal_beats_raw"].append(
                _case_row(
                    category="causal_beats_raw",
                    budget=budget,
                    task=task,
                    rows=rows,
                    primary_delta=round(causal_gain, 6),
                    note="Causal compression selected more useful context than raw CICL under the same budget.",
                )
            )

        summary_gain = _score_gain(summary, causal)
        if summary_gain > 0:
            candidates["summary_beats_causal"].append(
                _case_row(
                    category="summary_beats_causal",
                    budget=budget,
                    task=task,
                    rows=rows,
                    primary_delta=round(summary_gain, 6),
                    note="Generic extractive compression beat causal cards; this is negative evidence for the main method.",
                )
            )

        raw_tokens = float(raw["tokens"])
        selected_tokens = float(selected_then["tokens"])
        same_recall = float(selected_then["context_recall"]) == float(raw["context_recall"])
        same_success = int(selected_then["success"]) == int(raw["success"])
        if raw_tokens > selected_tokens and same_recall and same_success:
            candidates["same_selection_token_saving"].append(
                _case_row(
                    category="same_selection_token_saving",
                    budget=budget,
                    task=task,
                    rows=rows,
                    primary_delta=round(raw_tokens - selected_tokens, 6),
                    note="Compressing the already selected raw context saved tokens without changing simulated success or recall.",
                )
            )

        raw_ids = set(_selected_ids(raw))
        causal_ids = set(_selected_ids(causal))
        gold_ids = set(task.gold_context_ids)
        if gold_ids.intersection(causal_ids) and not gold_ids.intersection(raw_ids):
            candidates["causal_adds_gold_context"].append(
                _case_row(
                    category="causal_adds_gold_context",
                    budget=budget,
                    task=task,
                    rows=rows,
                    primary_delta=round(float(causal["context_recall"]) - float(raw["context_recall"]), 6),
                    note="Causal compression fit a gold context that raw budget fitting missed.",
                )
            )

    ranked_cases: list[dict] = []
    sort_specs = {
        "causal_beats_raw": ("primary_delta", "causal_recall", "raw_tokens"),
        "summary_beats_causal": ("primary_delta", "summary_recall", "causal_tokens"),
        "same_selection_token_saving": ("primary_delta", "raw_tokens", "budget"),
        "causal_adds_gold_context": ("primary_delta", "causal_recall", "raw_tokens"),
    }
    for category in [
        "causal_beats_raw",
        "causal_adds_gold_context",
        "summary_beats_causal",
        "same_selection_token_saving",
    ]:
        keys = sort_specs[category]
        rows = sorted(
            candidates.get(category, []),
            key=lambda row: tuple(float(row[key]) for key in keys),
            reverse=True,
        )
        ranked_cases.extend(rows[:max_per_category])
    return ranked_cases


def _score_gain(lhs: dict, rhs: dict) -> float:
    return (
        10.0 * (float(lhs["success"]) - float(rhs["success"]))
        + 3.0 * (float(lhs["context_recall"]) - float(rhs["context_recall"]))
        + float(lhs["context_f1"]) - float(rhs["context_f1"])
    )


def _case_row(
    category: str,
    budget: int,
    task: Task,
    rows: dict[str, dict],
    primary_delta: float,
    note: str,
) -> dict:
    raw = rows["RawCICLSelection"]
    selected_then = rows["SelectedThenCompressedCICL"]
    summary = rows["SummaryCompressedCICL"]
    causal = rows["CausalCompressedCICL"]
    return {
        "category": category,
        "budget": budget,
        "task_id": task.id,
        "repo": task.instance_id,
        "instruction_excerpt": _excerpt(task.instruction, words=34),
        "gold_context_ids": "|".join(task.gold_context_ids),
        "raw_selected": raw["selected_context_ids"],
        "causal_selected": causal["selected_context_ids"],
        "summary_selected": summary["selected_context_ids"],
        "raw_success": raw["success"],
        "causal_success": causal["success"],
        "summary_success": summary["success"],
        "raw_recall": raw["context_recall"],
        "causal_recall": causal["context_recall"],
        "summary_recall": summary["context_recall"],
        "raw_tokens": raw["tokens"],
        "selected_then_compressed_tokens": selected_then["tokens"],
        "causal_tokens": causal["tokens"],
        "summary_tokens": summary["tokens"],
        "primary_delta": primary_delta,
        "note": note,
    }


def _selected_ids(row: dict) -> list[str]:
    value = str(row.get("selected_context_ids", ""))
    return [item for item in value.split("|") if item]


def _excerpt(text: str, words: int = 34) -> str:
    tokens = text.replace("\n", " ").split()
    if len(tokens) <= words:
        return " ".join(tokens)
    return " ".join(tokens[:words]).rstrip(" ,.;:") + "..."


def write_case_study_report(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Compression Case Studies",
        "",
        "These cases are selected automatically from paired task metrics. They include both positive and negative evidence for causal memory cards.",
        "",
    ]
    if not rows:
        lines.append("No case studies matched the selection rules.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)

    titles = {
        "causal_beats_raw": "Causal Beats Raw",
        "causal_adds_gold_context": "Causal Adds Gold Context",
        "summary_beats_causal": "Summary Beats Causal",
        "same_selection_token_saving": "Same Selection Token Saving",
    }
    for category in [
        "causal_beats_raw",
        "causal_adds_gold_context",
        "summary_beats_causal",
        "same_selection_token_saving",
    ]:
        category_rows = by_category.get(category, [])
        if not category_rows:
            continue
        lines.extend([f"## {titles[category]}", ""])
        for row in category_rows:
            lines.extend(
                [
                    f"### {row['task_id']} at budget {row['budget']}",
                    "",
                    f"- Repo: `{row['repo']}`",
                    f"- Instruction: {row['instruction_excerpt']}",
                    f"- Gold context: `{row['gold_context_ids']}`",
                    f"- Raw: success={row['raw_success']}, recall={row['raw_recall']}, tokens={row['raw_tokens']}",
                    f"- Selected then causal-compressed tokens: {row['selected_then_compressed_tokens']}",
                    f"- Causal: success={row['causal_success']}, recall={row['causal_recall']}, tokens={row['causal_tokens']}",
                    f"- Summary: success={row['summary_success']}, recall={row['summary_recall']}, tokens={row['summary_tokens']}",
                    f"- Note: {row['note']}",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_compression(runs: list[AgentRun], per_task_rows: list[dict]) -> list[dict]:
    rows_by_method: dict[str, list[dict]] = defaultdict(list)
    runs_by_method: dict[str, list[AgentRun]] = defaultdict(list)
    for row in per_task_rows:
        rows_by_method[str(row["method"])].append(row)
    for run in runs:
        runs_by_method[run.method].append(run)

    summary: list[dict] = []
    for method in sorted(rows_by_method):
        rows = rows_by_method[method]
        method_runs = runs_by_method[method]
        success = [float(run.success) for run in method_runs]
        summary.append(
            {
                "method": method,
                "n": len(rows),
                "success_rate": round(mean(success), 6),
                "success_sem": round(sem(success), 6),
                "context_precision": round(mean([float(row["context_precision"]) for row in rows]), 6),
                "context_recall": round(mean([float(row["context_recall"]) for row in rows]), 6),
                "context_f1": round(mean([float(row["context_f1"]) for row in rows]), 6),
                "avg_selected_count": round(mean([float(row["selected_count"]) for row in rows]), 3),
                "avg_tokens": round(mean([float(row["tokens"]) for row in rows]), 3),
                "avg_original_tokens": round(mean([float(row["original_tokens"]) for row in rows]), 3),
                "avg_compression_ratio": round(mean([float(row["compression_ratio"]) for row in rows]), 6),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run raw-vs-causal-compressed context experiment.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--budgets", help="Comma-separated budgets. If set, runs a compression budget suite.")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-eval-tasks", type=int)
    parser.add_argument("--llm-provider", choices=["simulator", "qwen", "qwen_local", "anthropic", "opus", "claude"], default="simulator")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    if args.budgets:
        budgets = [int(value) for value in args.budgets.split(",") if value]
        result = run_compression_suite(
            repo=args.repo,
            tasks_path=args.tasks,
            output=args.output,
            budgets=budgets,
            top_k=args.top_k,
            max_eval_tasks=args.max_eval_tasks,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key_env=args.llm_api_key_env,
            llm_timeout_seconds=args.llm_timeout_seconds,
        )
    else:
        result = run_compression_experiment(
            repo=args.repo,
            tasks_path=args.tasks,
            output=args.output,
            budget=args.budget,
            top_k=args.top_k,
            max_eval_tasks=args.max_eval_tasks,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key_env=args.llm_api_key_env,
            llm_timeout_seconds=args.llm_timeout_seconds,
        )
    for row in result["summary"]:
        prefix = f"budget={row['budget']} " if "budget" in row else ""
        print(
            f"{prefix}{row['method']}: success={row['success_rate']} "
            f"recall={row['context_recall']} tokens={row['avg_tokens']} "
            f"ratio={row['avg_compression_ratio']}"
        )


if __name__ == "__main__":
    main()
