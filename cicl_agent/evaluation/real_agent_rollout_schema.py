"""Schema audit for a formal non-toy LLM coding-agent rollout artifact."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_ROLLOUT_SUMMARY = "artifacts/outputs/latest/real_agent_rollout/summary.json"
REQUIRED_METHODS = {"NoContext", "CICL"}
BASELINE_METHODS = {"BM25", "VanillaRAG", "HybridRAG"}


def audit_real_agent_rollout_schema(
    root: str | Path = ".",
    rollout_summary: str | Path = DEFAULT_ROLLOUT_SUMMARY,
) -> dict[str, object]:
    """Validate the evidence contract for a formal real-agent rollout.

    The validator is intentionally stricter than the current preflight. Missing
    artifacts are marked DEFERRED because the manuscript already excludes this
    claim; malformed artifacts are FAIL because they would be unsafe evidence.
    """

    root_path = Path(root)
    summary_path = root_path / rollout_summary
    checks: list[dict[str, str]] = []

    if not summary_path.is_file():
        _add_check(checks, "formal_rollout_artifact", "DEFERRED", f"missing {summary_path.relative_to(root_path)}")
        return _build_report(root_path, summary_path, checks)

    _add_check(checks, "formal_rollout_artifact", "PASS", str(summary_path.relative_to(root_path)))
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _add_check(checks, "json_readable", "FAIL", f"{type(exc).__name__}: {exc}")
        return _build_report(root_path, summary_path, checks)

    if not isinstance(data, dict):
        _add_check(checks, "json_object", "FAIL", "top-level JSON value must be an object")
        return _build_report(root_path, summary_path, checks)
    _add_check(checks, "json_object", "PASS", "top-level JSON object")

    n_tasks = _positive_int(data.get("n_tasks"))
    _add_check(
        checks,
        "n_tasks",
        "PASS" if n_tasks else "FAIL",
        f"n_tasks={data.get('n_tasks')!r}" if n_tasks else "n_tasks must be a positive integer",
    )

    methods = _method_names(data)
    missing_required = sorted(REQUIRED_METHODS - methods)
    baseline_methods = sorted(BASELINE_METHODS & methods)
    method_problems: list[str] = []
    if missing_required:
        method_problems.append("missing required methods=" + ",".join(missing_required))
    if not baseline_methods:
        method_problems.append("missing BM25/VanillaRAG/HybridRAG baseline")
    _add_check(
        checks,
        "method_set",
        "PASS" if not method_problems else "FAIL",
        "methods=" + ",".join(sorted(methods)) if not method_problems else "; ".join(method_problems),
    )

    benchmark = str(data.get("benchmark") or data.get("dataset") or data.get("task_source") or "")
    benchmark_ok = bool(re.search(r"\bswe[-_ ]?bench\b", benchmark, flags=re.IGNORECASE))
    _add_check(
        checks,
        "non_toy_benchmark",
        "PASS" if benchmark_ok else "FAIL",
        benchmark or "benchmark/dataset/task_source must identify SWE-bench or equivalent non-toy coding tasks",
    )

    task_ids = data.get("task_ids")
    task_source = str(data.get("task_source") or "")
    task_ids_ok = isinstance(task_ids, list) and bool(n_tasks) and len(task_ids) == n_tasks and all(isinstance(item, str) and item for item in task_ids)
    task_source_ok = bool(task_source)
    _add_check(
        checks,
        "task_identity",
        "PASS" if task_ids_ok or task_source_ok else "FAIL",
        f"task_ids={len(task_ids) if isinstance(task_ids, list) else 'missing'}; task_source={task_source or 'missing'}",
    )

    regression_ok = data.get("official_or_equivalent_regression_tests") is True
    harness = str(data.get("test_harness") or data.get("regression_test_harness") or "")
    harness_ok = bool(harness.strip())
    _add_check(
        checks,
        "regression_test_harness",
        "PASS" if regression_ok and harness_ok else "FAIL",
        f"official_or_equivalent_regression_tests={data.get('official_or_equivalent_regression_tests')!r}; harness={harness or 'missing'}",
    )

    budget_ok, budget_detail = _budget_detail(data)
    _add_check(checks, "identical_budget", "PASS" if budget_ok else "FAIL", budget_detail)

    gold_ok = data.get("gold_patch_visible_to_agent") is False
    _add_check(
        checks,
        "gold_patch_hidden",
        "PASS" if gold_ok else "FAIL",
        f"gold_patch_visible_to_agent={data.get('gold_patch_visible_to_agent')!r}",
    )

    provider = str(data.get("provider") or data.get("llm_provider") or "")
    model = str(data.get("model") or data.get("llm_model") or "")
    _add_check(
        checks,
        "model_provenance",
        "PASS" if provider and model else "FAIL",
        f"provider={provider or 'missing'}; model={model or 'missing'}",
    )

    metrics = _metrics_by_method(data)
    metric_problems = _metric_problems(methods, metrics, n_tasks)
    _add_check(
        checks,
        "method_metrics",
        "PASS" if not metric_problems else "FAIL",
        f"{len(metrics)} method metric rows" if not metric_problems else "; ".join(metric_problems[:8]),
    )

    boundary_ok, boundary_detail = _claim_boundary_detail(str(data.get("claim_boundary") or ""))
    _add_check(checks, "claim_boundary", "PASS" if boundary_ok else "FAIL", boundary_detail)

    return _build_report(root_path, summary_path, checks)


def write_json_report(report: dict[str, object], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_markdown_report(report: dict[str, object], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Real-Agent Rollout Schema Audit",
        "",
        f"Overall status: **{report['status']}**",
        "",
        "This audit validates the minimum evidence contract for a formal non-toy",
        "LLM coding-agent rollout. A missing artifact remains a deferred research",
        "gap; a malformed artifact is a submission-blocking failure.",
        "",
        f"- Expected rollout summary: `{report['rollout_summary']}`",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for row in report["checks"]:
        lines.append(f"| {row['name']} | {row['status']} | {_md(row['detail'])} |")
    if report["blocking_checks"]:
        lines.extend(["", "## Blocking Checks", ""])
        for row in report["blocking_checks"]:
            lines.append(f"- `{row['name']}`: {_md(row['detail'])}")
    lines.extend(["", "## Claim Boundary", "", str(report["claim_boundary"]), ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _build_report(root: Path, summary_path: Path, checks: list[dict[str, str]]) -> dict[str, object]:
    blocking = [row for row in checks if row["status"] != "PASS"]
    statuses = {row["status"] for row in blocking}
    if "FAIL" in statuses:
        status = "FAIL"
    elif "DEFERRED" in statuses:
        status = "DEFERRED"
    else:
        status = "PASS"
    return {
        "status": status,
        "rollout_summary": str(summary_path.relative_to(root)),
        "checks": checks,
        "blocking_checks": blocking,
        "claim_boundary": (
            "PASS means the summary artifact is structurally suitable evidence "
            "for a non-toy LLM coding-agent rollout. It still needs manuscript "
            "wording and numeric table checks before becoming a main claim. "
            "DEFERRED means no formal rollout artifact is present; FAIL means an "
            "artifact exists but is not safe evidence."
        ),
    }


def _add_check(checks: list[dict[str, str]], name: str, status: str, detail: str) -> None:
    checks.append({"name": name, "status": status, "detail": detail})


def _method_names(data: dict[str, Any]) -> set[str]:
    methods: set[str] = set()
    raw_methods = data.get("methods")
    if isinstance(raw_methods, dict):
        methods.update(str(name) for name in raw_methods if str(name))
    elif isinstance(raw_methods, list):
        for item in raw_methods:
            if isinstance(item, str) and item:
                methods.add(item)
            elif isinstance(item, dict):
                name = item.get("method") or item.get("name")
                if name:
                    methods.add(str(name))
    methods.update(_metrics_by_method(data))
    return methods


def _metrics_by_method(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    for key in ["metrics_by_method", "method_metrics", "results_by_method"]:
        raw = data.get(key)
        if isinstance(raw, dict):
            return {
                str(method): row
                for method, row in raw.items()
                if str(method) and isinstance(row, dict)
            }

    metrics: dict[str, dict[str, Any]] = {}
    for key in ["results", "method_results", "methods"]:
        raw_rows = data.get(key)
        if not isinstance(raw_rows, list):
            continue
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            method = row.get("method") or row.get("name")
            if not method:
                continue
            nested = row.get("metrics")
            metrics[str(method)] = nested if isinstance(nested, dict) else row
        if metrics:
            return metrics
    return metrics


def _metric_problems(methods: set[str], metrics: dict[str, dict[str, Any]], n_tasks: int | None) -> list[str]:
    problems: list[str] = []
    if not methods:
        problems.append("no methods declared")
        return problems
    missing = sorted(method for method in methods if method not in metrics)
    if missing:
        problems.append("missing metrics for methods=" + ",".join(missing))
    for method in sorted(methods & set(metrics)):
        problems.extend(f"{method}: {problem}" for problem in _single_method_metric_problems(metrics[method], n_tasks))
    return problems


def _single_method_metric_problems(row: dict[str, Any], n_tasks: int | None) -> list[str]:
    problems: list[str] = []
    attempted = _metric_int(row, ["n_attempted", "attempted"])
    applied = _metric_int(row, ["n_applied", "applied"])
    executable = _metric_int(row, ["n_executable", "executable", "n_executable_tests", "executable_tests"])
    passed = _metric_int(row, ["n_passed", "passed"])
    apply_rate = _metric_rate(row, ["patch_apply_rate", "apply_rate"])
    executable_rate = _metric_rate(row, ["executable_test_rate", "executable_rate"])
    pass_rate = _metric_rate(row, ["pass_rate"])

    if attempted is None or attempted <= 0:
        problems.append("attempted/n_attempted must be positive")
    elif n_tasks is not None and attempted != n_tasks:
        problems.append(f"attempted={attempted} does not match n_tasks={n_tasks}")

    if applied is None and apply_rate is None:
        problems.append("missing applied count or patch_apply_rate")
    if executable is None and executable_rate is None:
        problems.append("missing executable count or executable_test_rate")
    if passed is None and pass_rate is None:
        problems.append("missing passed count or pass_rate")

    counts = [count for count in [attempted, applied, executable, passed] if count is not None]
    if any(count < 0 for count in counts):
        problems.append("counts must be non-negative")
    if attempted is not None:
        for label, count in [("applied", applied), ("executable", executable), ("passed", passed)]:
            if count is not None and count > attempted:
                problems.append(f"{label}={count} exceeds attempted={attempted}")
    if None not in (attempted, applied, executable, passed) and not (passed <= executable <= applied <= attempted):
        problems.append("counts must satisfy passed <= executable <= applied <= attempted")

    for label, rate in [
        ("patch_apply_rate", apply_rate),
        ("executable_test_rate", executable_rate),
        ("pass_rate", pass_rate),
    ]:
        if rate is not None and not 0.0 <= rate <= 1.0:
            problems.append(f"{label}={rate} outside [0, 1]")

    return problems


def _metric_int(row: dict[str, Any], keys: list[str]) -> int | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _metric_rate(row: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    return None


def _budget_detail(data: dict[str, Any]) -> tuple[bool, str]:
    same_budget = data.get("same_budget_across_methods") is True
    budget = _positive_int(data.get("context_budget")) or _positive_int(data.get("token_budget")) or _positive_int(data.get("context_token_budget"))
    budgets_by_method = data.get("budgets_by_method")
    if budget:
        return same_budget, f"same_budget_across_methods={data.get('same_budget_across_methods')!r}; budget={budget}"
    if isinstance(budgets_by_method, dict):
        budgets = [
            _positive_int(value)
            for value in budgets_by_method.values()
        ]
        if budgets and all(value is not None for value in budgets) and len(set(budgets)) == 1:
            return same_budget, f"same_budget_across_methods={data.get('same_budget_across_methods')!r}; per-method budget={budgets[0]}"
    return False, "same_budget_across_methods must be true and a positive shared context/token budget must be recorded"


def _claim_boundary_detail(boundary: str) -> tuple[bool, str]:
    if not boundary.strip():
        return False, "claim_boundary is missing"
    lower = boundary.lower()
    required = [
        ("official/equivalent regression tests", ("official", "equivalent", "regression")),
        ("non-toy or not toy", ("non-toy", "not toy", "not a toy", "swe-bench")),
    ]
    missing: list[str] = []
    if not (("official" in lower or "equivalent" in lower) and "regression" in lower):
        missing.append(required[0][0])
    if not any(phrase in lower for phrase in required[1][1]):
        missing.append(required[1][0])
    overstrong = [
        pattern
        for pattern in [
            r"state[- ]of[- ]the[- ]art",
            r"\bproves?\b",
            r"production repair performance",
            r"official swe-bench success",
        ]
        if re.search(pattern, lower)
    ]
    if missing or overstrong:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if overstrong:
            details.append("overstrong=" + ",".join(overstrong))
        return False, "; ".join(details)
    return True, "boundary names official/equivalent regression tests and non-toy scope without overclaiming"


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the formal real-agent rollout summary schema.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--rollout-summary", default=DEFAULT_ROLLOUT_SUMMARY)
    parser.add_argument("--output", default="reports/real_agent_rollout_schema_audit.json")
    parser.add_argument("--markdown-output", default="reports/real_agent_rollout_schema_audit.md")
    args = parser.parse_args()

    root = Path(args.root)
    report = audit_real_agent_rollout_schema(root=root, rollout_summary=args.rollout_summary)
    json_path = write_json_report(report, root / args.output)
    md_path = write_markdown_report(report, root / args.markdown_output)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "status": report["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
