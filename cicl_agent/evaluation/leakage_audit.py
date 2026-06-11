"""Audit gold-label and oracle leakage risks in CICL artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from cicl_agent.core.io import read_jsonl
from cicl_agent.core.schema import ContextEdge, ContextUnit, Task


@dataclass
class LeakageItem:
    category: str
    target: str
    status: str
    detail: str


FORBIDDEN_UNIT_MARKERS = [
    "gold_context_ids",
    "gold context ids",
    "solution patch",
    "test patch",
    "oracle answer",
]

ALLOWED_GOLD_SOURCE_PATHS = {
    "cicl_agent/benchmarks/adapters.py",
    "cicl_agent/benchmarks/download.py",
    "cicl_agent/benchmarks/synthetic.py",
    "cicl_agent/core/schema.py",
    "cicl_agent/evaluation/leakage_audit.py",
    "cicl_agent/evaluation/metrics.py",
    "cicl_agent/evaluation/paper_cases.py",
    "cicl_agent/evaluation/replay.py",
    "cicl_agent/evaluation/significance.py",
    "cicl_agent/evaluation/tool_execution.py",
    "cicl_agent/selectors/baselines.py",
}

WARN_GOLD_SOURCE_PATHS = {
    "cicl_agent/agents/simulated.py",
    "cicl_agent/causal/scoring.py",
    "cicl_agent/memory/graph.py",
    "cicl_agent/runners/compression_experiment.py",
    "cicl_agent/runners/experiment_suite.py",
    "cicl_agent/runners/harmful_context_stress.py",
    "cicl_agent/runners/swebench_rr_retrieval.py",
    "cicl_agent/runners/toy_patch_pilot.py",
}

WARN_GOLD_SOURCE_DETAILS = {
    "cicl_agent/agents/simulated.py": "simulated success metric 使用 gold overlap；需披露为 simulator-only",
    "cicl_agent/causal/scoring.py": "包含显式 allow_gold_labels debug hook；默认关闭",
    "cicl_agent/memory/graph.py": "包含显式 link_gold_context opt-in hook；默认关闭",
    "cicl_agent/runners/compression_experiment.py": "offline metrics/case-study reporting 使用 gold labels",
    "cicl_agent/runners/experiment_suite.py": "offline metrics 使用 gold labels",
    "cicl_agent/runners/harmful_context_stress.py": "offline stress metrics 使用 gold labels",
    "cicl_agent/runners/swebench_rr_retrieval.py": "offline retrieval metrics 和 oracle upper bound 使用 gold labels；不能作为 method-facing context",
    "cicl_agent/runners/toy_patch_pilot.py": "offline context precision/recall metrics 使用 gold labels",
}


def audit_leakage(
    *,
    root: str | Path = ".",
    tasks_path: str | Path | None = None,
    artifact_dirs: list[str | Path] | None = None,
) -> list[LeakageItem]:
    root_path = Path(root)
    items: list[LeakageItem] = []
    tasks = _load_tasks(root_path / tasks_path) if tasks_path else []
    eval_gold = {
        gold_id
        for task in tasks
        if task.split != "base"
        for gold_id in task.gold_context_ids
    }
    if tasks_path:
        items.append(
            LeakageItem(
                "task-gold-labels",
                str(tasks_path),
                "PASS",
                f"加载 {len(tasks)} 个 tasks；eval gold ids 仅作为 evaluation-only inputs",
            )
        )

    for artifact_dir in artifact_dirs or []:
        path = root_path / artifact_dir
        items.extend(_audit_artifact_dir(path, eval_gold))

    items.extend(_audit_source_gold_usage(root_path))
    return items


def write_leakage_report(items: list[LeakageItem], output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status = "PASS"
    if any(item.status == "FAIL" for item in items):
        status = "FAIL"
    elif any(item.status == "WARN" for item in items):
        status = "WARN"
    lines = [
        "# Gold Label 泄漏审计",
        "",
        f"Overall status: **{status}**",
        "",
        "| 类别 | 目标 | 状态 | 细节 |",
        "| --- | --- | --- | --- |",
    ]
    for item in items:
        lines.append(f"| {_md(item.category)} | {_md(item.target)} | {_md(item.status)} | {_md(item.detail)} |")
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- `FAIL`：gold label 或 oracle-only artifact 出现在 method-facing context artifact 中。",
            "- `WARN`：代码只在 simulator/evaluation/debug 路径使用 gold labels，但必须披露。",
            "- `PASS`：本审计未发现直接泄漏模式。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _audit_artifact_dir(path: Path, eval_gold: set[str]) -> list[LeakageItem]:
    items: list[LeakageItem] = []
    if not path.exists():
        return [LeakageItem("artifact", str(path), "FAIL", "缺失 artifact directory")]
    units_path = path / "context_units.jsonl"
    edges_path = path / "context_edges.jsonl"
    if units_path.exists():
        items.extend(_audit_units(units_path))
    else:
        items.append(LeakageItem("context-units", str(units_path), "WARN", "未找到 context_units.jsonl"))
    if edges_path.exists():
        items.extend(_audit_edges(edges_path, eval_gold))
    else:
        items.append(LeakageItem("context-edges", str(edges_path), "WARN", "未找到 context_edges.jsonl"))
    return items


def _audit_units(path: Path) -> list[LeakageItem]:
    rows = list(read_jsonl(path))
    failures: list[str] = []
    for index, row in enumerate(rows, start=1):
        unit = ContextUnit.from_dict(row)
        metadata_text = json.dumps(unit.metadata, sort_keys=True)
        haystack = f"{unit.id}\n{unit.source}\n{unit.content}\n{metadata_text}".lower()
        for marker in FORBIDDEN_UNIT_MARKERS:
            if marker in haystack:
                failures.append(f"line {index} marker `{marker}` in {unit.id}")
                break
    if failures:
        return [LeakageItem("context-units", str(path), "FAIL", "; ".join(failures[:5]))]
    return [LeakageItem("context-units", str(path), "PASS", f"已检查 {len(rows)} 个 units")]


def _audit_edges(path: Path, eval_gold: set[str]) -> list[LeakageItem]:
    rows = list(read_jsonl(path))
    failures: list[str] = []
    for index, row in enumerate(rows, start=1):
        edge = ContextEdge.from_dict(row)
        relation = edge.relation.lower()
        if relation in {"derived-from", "gold", "oracle", "gold-context"}:
            failures.append(f"line {index} relation `{edge.relation}`: {edge.source_id}->{edge.target_id}")
    if failures:
        return [LeakageItem("context-edges", str(path), "FAIL", "; ".join(failures[:5]))]
    detail = f"已检查 {len(rows)} 条 edges；普通 file/symbol edges 可能涉及 gold ids，因为 gold contexts 也是普通 repo units"
    return [LeakageItem("context-edges", str(path), "PASS", detail)]


def _audit_source_gold_usage(root: Path) -> list[LeakageItem]:
    items: list[LeakageItem] = []
    for path in sorted((root / "cicl_agent").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        if "gold_context_ids" not in text and "allow_gold_labels" not in text:
            continue
        if rel in ALLOWED_GOLD_SOURCE_PATHS:
            status = "PASS"
            detail = "允许的 evaluation/oracle/data 路径"
        elif rel in WARN_GOLD_SOURCE_PATHS:
            status = "WARN"
            detail = WARN_GOLD_SOURCE_DETAILS.get(rel, "gold-label use 已披露，但必须保持 non-method-facing")
        else:
            status = "FAIL"
            detail = "source 中出现未预期的 gold-label reference"
        items.append(LeakageItem("source-gold-usage", rel, status, detail))
    return items


def _load_tasks(path: Path) -> list[Task]:
    return [Task.from_dict(row) for row in read_jsonl(path)]


def _md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit gold-label leakage in CICL artifacts.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--tasks")
    parser.add_argument("--artifact-dir", action="append", default=[])
    parser.add_argument("--output", default="reports/gold_leakage_audit.md")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    items = audit_leakage(root=args.root, tasks_path=args.tasks, artifact_dirs=args.artifact_dir)
    report = write_leakage_report(items, Path(args.root) / args.output)
    print(report)
    if args.strict and any(item.status == "FAIL" for item in items):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
