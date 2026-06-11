"""Preflight checks for a non-toy LLM coding-agent rollout."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
from pathlib import Path

from cicl_agent.core.io import read_jsonl
from cicl_agent.evaluation.real_agent_rollout_schema import audit_real_agent_rollout_schema


DEFAULT_MODEL_KEY_ENVS = (
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "DASHSCOPE_API_KEY",
    "OPENAI_API_KEY",
)


def check_real_agent_rollout_preflight(
    root: str | Path = ".",
    tasks: str | Path = "experiments/data/raw/swebench_verified_rr/corpus_sample_50.jsonl",
    retrieval_summary: str | Path = "artifacts/outputs/latest/swebench_verified_rr_retrieval/retrieval_summary.csv",
    patch_smoke_summary: str | Path = "artifacts/outputs/latest/swebench_opus_patch_smoke_n3/summary.json",
    patch_smoke_exec: str | Path = "artifacts/outputs/latest/swebench_opus_patch_smoke_n3/executable_validation.json",
    rollout_summary: str | Path = "artifacts/outputs/latest/real_agent_rollout/summary.json",
    min_free_mb: int = 800,
    model_key_envs: tuple[str, ...] = DEFAULT_MODEL_KEY_ENVS,
) -> dict[str, object]:
    """Return a structured readiness report without running an LLM agent.

    PASS would mean this machine has the basic artifacts and execution
    environment needed for a formal non-toy rollout. It would still not be the
    rollout result itself; the rollout summary is a separate required artifact.
    """

    root_path = Path(root)
    checks: list[dict[str, str]] = []
    tasks_path = root_path / tasks
    retrieval_path = root_path / retrieval_summary
    patch_summary_path = root_path / patch_smoke_summary
    patch_exec_path = root_path / patch_smoke_exec
    rollout_path = root_path / rollout_summary

    _add_check(checks, "non_toy_tasks_file", tasks_path.is_file(), str(tasks_path.relative_to(root_path)))
    task_count = _count_jsonl(tasks_path) if tasks_path.exists() else 0
    _add_check(checks, "non_toy_tasks_nonempty", task_count > 0, f"{task_count} rows")

    retrieval_ok, retrieval_detail = _retrieval_detail(retrieval_path)
    _add_check(checks, "retrieval_baseline_artifact", retrieval_ok, retrieval_detail)

    patch_ok, patch_detail = _patch_smoke_detail(patch_summary_path, patch_exec_path)
    _add_check(checks, "patch_smoke_artifact", patch_ok, patch_detail)

    free_mb = shutil.disk_usage(root_path).free // (1024 * 1024)
    _add_check(checks, "disk_free_for_rollout", free_mb >= min_free_mb, f"{free_mb} MiB available; threshold {min_free_mb} MiB")

    available_key_envs = [name for name in model_key_envs if os.getenv(name)]
    _add_check(
        checks,
        "model_credentials",
        bool(available_key_envs),
        "set=" + ",".join(available_key_envs) if available_key_envs else "no model API key env is set",
    )

    docker = shutil.which("docker")
    _add_check(checks, "docker_available", bool(docker), docker or "docker not found")
    pytest = shutil.which("pytest")
    pytest_module = importlib.util.find_spec("pytest")
    _add_check(
        checks,
        "pytest_available",
        bool(pytest or pytest_module),
        pytest or ("python module pytest available" if pytest_module else "pytest not found"),
    )

    rollout_ok, rollout_detail = _formal_rollout_detail(rollout_path)
    _add_check(checks, "formal_rollout_artifact", rollout_ok, rollout_detail)

    blocking = [row for row in checks if row["status"] != "PASS"]
    status = "PASS" if checks and not blocking else "FAIL"
    return {
        "status": status,
        "tasks": str(tasks_path.relative_to(root_path)),
        "task_count": task_count,
        "retrieval_summary": str(retrieval_path.relative_to(root_path)),
        "patch_smoke_summary": str(patch_summary_path.relative_to(root_path)),
        "patch_smoke_executable_validation": str(patch_exec_path.relative_to(root_path)),
        "rollout_summary": str(rollout_path.relative_to(root_path)),
        "min_free_mb": min_free_mb,
        "model_key_envs": list(model_key_envs),
        "checks": checks,
        "blocking_checks": blocking,
        "claim_boundary": (
            "PASS means prerequisites and a formal rollout artifact are present. "
            "FAIL means the paper must keep real non-toy LLM coding-agent success "
            "as deferred. Patch-generation smokes are not official or equivalent "
            "coding-agent success evidence."
        ),
    }


def write_json_report(report: dict[str, object], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_markdown_report(report: dict[str, object], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Real-Agent Rollout Preflight",
        "",
        f"Overall status: **{report['status']}**",
        "",
        "This preflight checks whether the current machine and artifacts can support",
        "a non-toy LLM coding-agent rollout with official or equivalent regression",
        "tests. It does not run an LLM agent and is not success evidence by itself.",
        "",
        f"- Tasks: `{report['tasks']}` ({report['task_count']} rows)",
        f"- Retrieval baseline: `{report['retrieval_summary']}`",
        f"- Patch smoke: `{report['patch_smoke_summary']}`",
        f"- Expected formal rollout: `{report['rollout_summary']}`",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for row in report["checks"]:
        lines.append(f"| {row['name']} | {row['status']} | {_md(row['detail'])} |")
    lines.extend(["", "## Claim Boundary", "", str(report["claim_boundary"]), ""])
    if report["blocking_checks"]:
        lines.extend(["## Blocking Checks", ""])
        for row in report["blocking_checks"]:
            lines.append(f"- `{row['name']}`: {_md(row['detail'])}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _add_check(checks: list[dict[str, str]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})


def _count_jsonl(path: Path) -> int:
    try:
        return sum(1 for _ in read_jsonl(path))
    except Exception:
        return 0


def _retrieval_detail(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing {path}"
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        return False, f"unreadable retrieval summary: {exc}"
    methods = {row.get("method", "") for row in rows}
    has_baseline = bool({"BM25", "HybridRAG", "VanillaRAG"} & methods)
    has_cicl = any("Causal" in method or method == "CICL" for method in methods)
    return has_baseline and has_cicl, f"{len(rows)} rows; methods={sorted(method for method in methods if method)}"


def _patch_smoke_detail(summary_path: Path, exec_path: Path) -> tuple[bool, str]:
    if not summary_path.is_file():
        return False, f"missing {summary_path}"
    if not exec_path.is_file():
        return False, f"missing {exec_path}"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        executable = json.loads(exec_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"unreadable patch-smoke artifact: {exc}"
    n_tasks = int(summary.get("n_tasks", 0))
    official = str(executable.get("official_swebench_harness", summary.get("official_swebench_harness", "not run")))
    n_passed = executable.get("n_passed", summary.get("n_success", 0))
    ok = n_tasks > 0 and "not run" in official.lower()
    return ok, f"n_tasks={n_tasks}; targeted_passed={n_passed}; official_harness={official}"


def _formal_rollout_detail(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing {path}"
    report = audit_real_agent_rollout_schema(root=path.parent, rollout_summary=path.name)
    if report.get("status") == "PASS":
        return True, "formal rollout schema PASS"
    blocking = [
        str(row.get("name", ""))
        for row in report.get("blocking_checks", [])
        if isinstance(row, dict)
    ]
    detail = ",".join(blocking) if blocking else str(report.get("status", "unknown"))
    return False, f"formal rollout schema {report.get('status')}: {detail}"


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight a real non-toy LLM coding-agent rollout.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--tasks", default="experiments/data/raw/swebench_verified_rr/corpus_sample_50.jsonl")
    parser.add_argument("--retrieval-summary", default="artifacts/outputs/latest/swebench_verified_rr_retrieval/retrieval_summary.csv")
    parser.add_argument("--patch-smoke-summary", default="artifacts/outputs/latest/swebench_opus_patch_smoke_n3/summary.json")
    parser.add_argument("--patch-smoke-exec", default="artifacts/outputs/latest/swebench_opus_patch_smoke_n3/executable_validation.json")
    parser.add_argument("--rollout-summary", default="artifacts/outputs/latest/real_agent_rollout/summary.json")
    parser.add_argument("--min-free-mb", type=int, default=800)
    parser.add_argument("--output", default="reports/real_agent_rollout_preflight.json")
    parser.add_argument("--markdown-output", default="reports/real_agent_rollout_preflight.md")
    args = parser.parse_args()

    root = Path(args.root)
    report = check_real_agent_rollout_preflight(
        root=root,
        tasks=args.tasks,
        retrieval_summary=args.retrieval_summary,
        patch_smoke_summary=args.patch_smoke_summary,
        patch_smoke_exec=args.patch_smoke_exec,
        rollout_summary=args.rollout_summary,
        min_free_mb=args.min_free_mb,
    )
    json_path = write_json_report(report, root / args.output)
    md_path = write_markdown_report(report, root / args.markdown_output)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "status": report["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
