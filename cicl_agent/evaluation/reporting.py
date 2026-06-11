"""Generate paper-style tables, figures, and a compact experiment report."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from cicl_agent.evaluation.figures import (
    svg_bar_chart,
    svg_causal_components,
    svg_grouped_precision_recall,
)
from cicl_agent.evaluation.tables import (
    read_csv_dicts,
    read_jsonl,
    write_csv,
    write_markdown_table,
)


MAIN_METHODS = {
    "NoContext",
    "FullContext",
    "VanillaRAG",
    "SummaryMemory",
    "GraphMemory",
    "AutoContextKG",
    "SelfGeneratedExamples",
    "CICL",
    "OracleGoldContext",
}


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return 0.0


def rounded(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def main_results_rows(metrics: list[dict[str, str]]) -> list[dict]:
    rows = [row for row in metrics if row["method"] in MAIN_METHODS]
    rows.sort(key=lambda row: (row["method"] != "CICL", -as_float(row, "success_rate"), -as_float(row, "context_f1"), row["method"]))
    return [
        {
            "method": row["method"],
            "n": row["n"],
            "success": rounded(as_float(row, "success_rate")),
            "context_f1": rounded(as_float(row, "context_f1")),
            "precision": rounded(as_float(row, "context_precision")),
            "recall": rounded(as_float(row, "context_recall")),
            "mrr": rounded(as_float(row, "context_mrr")),
            "tokens": rounded(as_float(row, "avg_tokens"), 1),
            "tool_calls": rounded(as_float(row, "avg_tool_calls"), 1),
        }
        for row in rows
    ]


def cost_rows(metrics: list[dict[str, str]]) -> list[dict]:
    rows = [row for row in metrics if row["method"] in MAIN_METHODS]
    rows.sort(key=lambda row: as_float(row, "avg_tokens"))
    return [
        {
            "method": row["method"],
            "avg_tokens": rounded(as_float(row, "avg_tokens"), 1),
            "avg_tool_calls": rounded(as_float(row, "avg_tool_calls"), 1),
            "avg_runtime_ms": rounded(as_float(row, "avg_runtime_seconds") * 1000, 4),
            "success": rounded(as_float(row, "success_rate")),
            "context_f1": rounded(as_float(row, "context_f1")),
        }
        for row in rows
    ]


def ablation_rows(metrics: list[dict[str, str]]) -> list[dict]:
    base = next((row for row in metrics if row["method"] == "CICL"), None)
    base_success = as_float(base, "success_rate") if base else 0.0
    base_f1 = as_float(base, "context_f1") if base else 0.0
    base_tokens = as_float(base, "avg_tokens") if base else 0.0
    rows = [row for row in metrics if row["method"].startswith("CICL_") or row["method"] == "CICL"]
    rows.sort(key=lambda row: (row["method"] != "CICL", row["method"]))
    return [
        {
            "variant": row["method"],
            "success": rounded(as_float(row, "success_rate")),
            "delta_success": rounded(as_float(row, "success_rate") - base_success),
            "context_f1": rounded(as_float(row, "context_f1")),
            "delta_f1": rounded(as_float(row, "context_f1") - base_f1),
            "tokens": rounded(as_float(row, "avg_tokens"), 1),
            "delta_tokens": rounded(as_float(row, "avg_tokens") - base_tokens, 1),
        }
        for row in rows
    ]


def causal_summary_rows(scores: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for score in scores:
        grouped[score["task_id"]].append(score)
    rows: list[dict] = []
    for task_id, task_scores in sorted(grouped.items()):
        top = max(task_scores, key=lambda row: row["final_score"])
        rows.append(
            {
                "task_id": task_id,
                "n_scores": len(task_scores),
                "top_context_id": top["context_id"],
                "top_final": rounded(top["final_score"]),
                "mean_final": rounded(mean(row["final_score"] for row in task_scores)),
                "mean_action_delta": rounded(mean(row["action_delta"] for row in task_scores)),
                "mean_outcome_uplift": rounded(mean(row["outcome_uplift"] for row in task_scores)),
                "mean_necessity": rounded(mean(row["necessity_score"] for row in task_scores)),
            }
        )
    return rows


def write_report(path: str | Path, tables: dict[str, Path], figures: dict[str, Path], metrics: list[dict]) -> None:
    path = Path(path)
    cicl = next((row for row in metrics if row.get("method") == "CICL"), None)
    summary = ""
    if cicl:
        summary = (
            f"CICL success={float(cicl['success_rate']):.3f}, "
            f"context_f1={float(cicl['context_f1']):.3f}, "
            f"avg_tokens={float(cicl['avg_tokens']):.1f}."
        )
    lines = [
        "# CICL Experiment Report",
        "",
        summary or "CICL metrics were not found.",
        "",
        "## Tables",
        "",
    ]
    for name, table_path in tables.items():
        rel = table_path.relative_to(path.parent)
        lines.append(f"- [{name}]({rel.as_posix()})")
    lines.extend(["", "## Figures", ""])
    for name, figure_path in figures.items():
        rel = figure_path.relative_to(path.parent)
        lines.append(f"- [{name}]({rel.as_posix()})")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This report is generated from the current experiment outputs.",
            "- Toy outputs verify the pipeline; paper claims require real benchmark runs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_report(input_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path / "report"
    tables_path = output_path / "tables"
    figures_path = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = read_csv_dicts(input_path / "metrics.csv")
    causal_scores = read_jsonl(input_path / "causal_scores.jsonl")

    main_rows = main_results_rows(metrics)
    cost_table_rows = cost_rows(metrics)
    ablation_table_rows = ablation_rows(metrics)
    causal_rows = causal_summary_rows(causal_scores)

    tables = {
        "main_results.md": tables_path / "main_results.md",
        "cost_efficiency.md": tables_path / "cost_efficiency.md",
        "ablation_results.md": tables_path / "ablation_results.md",
        "causal_score_summary.md": tables_path / "causal_score_summary.md",
    }
    write_csv(tables_path / "main_results.csv", main_rows)
    write_csv(tables_path / "cost_efficiency.csv", cost_table_rows)
    write_csv(tables_path / "ablation_results.csv", ablation_table_rows)
    write_csv(tables_path / "causal_score_summary.csv", causal_rows)
    write_markdown_table(tables["main_results.md"], main_rows, "Main Results")
    write_markdown_table(tables["cost_efficiency.md"], cost_table_rows, "Cost and Efficiency")
    write_markdown_table(tables["ablation_results.md"], ablation_table_rows, "CICL Ablation Results")
    write_markdown_table(tables["causal_score_summary.md"], causal_rows, "Causal Score Summary")

    figures = {
        "success_rate_by_method.svg": figures_path / "success_rate_by_method.svg",
        "context_f1_by_method.svg": figures_path / "context_f1_by_method.svg",
        "avg_tokens_by_method.svg": figures_path / "avg_tokens_by_method.svg",
        "precision_recall_by_method.svg": figures_path / "precision_recall_by_method.svg",
        "ablation_delta_f1.svg": figures_path / "ablation_delta_f1.svg",
        "causal_score_components.svg": figures_path / "causal_score_components.svg",
    }
    svg_bar_chart(figures["success_rate_by_method.svg"], main_rows, "method", "success", "Success Rate by Method", "Success Rate")
    svg_bar_chart(figures["context_f1_by_method.svg"], main_rows, "method", "context_f1", "Context F1 by Method", "Context F1")
    svg_bar_chart(figures["avg_tokens_by_method.svg"], main_rows, "method", "tokens", "Average Token Cost by Method", "Tokens")
    svg_grouped_precision_recall(figures["precision_recall_by_method.svg"], main_rows)
    svg_bar_chart(figures["ablation_delta_f1.svg"], ablation_table_rows, "variant", "delta_f1", "CICL Ablation Delta F1", "Delta F1")
    svg_causal_components(figures["causal_score_components.svg"], causal_scores)

    write_report(output_path / "report.md", tables, figures, metrics)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CICL paper tables and SVG figures.")
    parser.add_argument("--input", required=True, help="Experiment output directory containing metrics.csv.")
    parser.add_argument("--output", help="Report output directory. Defaults to <input>/report.")
    args = parser.parse_args()
    output = generate_report(args.input, args.output)
    print(output)


if __name__ == "__main__":
    main()
