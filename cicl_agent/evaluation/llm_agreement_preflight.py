"""Preflight checks for real LLM agreement runs.

The agreement evaluator can use either a hosted Qwen/DashScope endpoint or a
local Qwen base model plus LoRA adapter. This module records whether the current
machine can run the real judge, without sending any model requests.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from cicl_agent.core.io import read_jsonl
from cicl_agent.integrations.llm_clients import (
    DEFAULT_QWEN_LOCAL_ADAPTER,
    DEFAULT_QWEN_LOCAL_BASE,
    normalize_provider,
)


def check_llm_agreement_preflight(
    repo: str | Path,
    tasks: str | Path,
    teacher_examples: str | Path,
    llm_provider: str = "qwen",
    llm_api_key_env: str = "DASHSCOPE_API_KEY",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    min_free_mb: int = 200,
) -> dict:
    """Return a structured readiness report for a real agreement run."""

    provider = normalize_provider(llm_provider)
    checks: list[dict] = []
    repo_path = Path(repo)
    tasks_path = Path(tasks)
    teacher_path = Path(teacher_examples)

    _add_check(checks, "repo_dir", repo_path.is_dir(), str(repo_path))
    _add_file_check(checks, "tasks_file", tasks_path)
    _add_file_check(checks, "teacher_examples", teacher_path)

    task_count = _count_jsonl(tasks_path) if tasks_path.exists() else 0
    teacher_count = _count_jsonl(teacher_path) if teacher_path.exists() else 0
    _add_check(checks, "tasks_nonempty", task_count > 0, f"{task_count} tasks")
    _add_check(
        checks,
        "teacher_examples_nonempty",
        teacher_count > 0,
        f"{teacher_count} teacher examples",
    )

    free_mb = _free_disk_mb(Path.cwd())
    _add_check(
        checks,
        "disk_free",
        free_mb >= min_free_mb,
        f"{free_mb} MiB available; threshold {min_free_mb} MiB",
    )

    if provider == "qwen":
        key_set = bool(os.getenv(llm_api_key_env))
        _add_check(
            checks,
            "qwen_api_key",
            key_set,
            f"{llm_api_key_env} is {'set' if key_set else 'unset'}",
        )
    elif provider == "qwen_local":
        base = llm_base_url or os.getenv("QWEN_LOCAL_BASE", DEFAULT_QWEN_LOCAL_BASE)
        adapter = llm_model or os.getenv("QWEN_LOCAL_ADAPTER", DEFAULT_QWEN_LOCAL_ADAPTER)
        _add_model_ref_check(checks, "qwen_local_base", base)
        _add_model_ref_check(checks, "qwen_local_adapter", adapter)
    elif provider in {"anthropic", "simulator"}:
        _add_check(
            checks,
            "provider_supported",
            True,
            f"{provider} does not prove Qwen-vs-Opus agreement",
        )
    else:
        _add_check(checks, "provider_supported", False, f"unsupported provider: {provider}")

    status = "PASS" if checks and all(row["status"] == "PASS" for row in checks) else "FAIL"
    blocking = [row for row in checks if row["status"] != "PASS"]
    return {
        "status": status,
        "provider": provider,
        "repo": str(repo_path),
        "tasks": str(tasks_path),
        "teacher_examples": str(teacher_path),
        "task_count": task_count,
        "teacher_example_count": teacher_count,
        "min_free_mb": min_free_mb,
        "checks": checks,
        "blocking_checks": blocking,
        "claim_boundary": (
            "PASS means the machine is ready to run the real agreement evaluator; "
            "it is not itself agreement evidence. FAIL means the paper must keep "
            "real-code Qwen-vs-Opus agreement as deferred."
        ),
    }


def write_json_report(report: dict, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_markdown_report(report: dict, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Real-Code Qwen Agreement Preflight",
        "",
        f"Overall status: **{report['status']}**",
        "",
        "This preflight checks whether the current machine can run the real-code",
        "Qwen-vs-Opus agreement evaluator. It does not call a model and is not",
        "agreement evidence by itself.",
        "",
        f"- Provider: `{report['provider']}`",
        f"- Tasks: `{report['tasks']}` ({report['task_count']} rows)",
        f"- Teacher examples: `{report['teacher_examples']}` ({report['teacher_example_count']} rows)",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for row in report["checks"]:
        lines.append(f"| {row['name']} | {row['status']} | {row['detail']} |")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    if report["blocking_checks"]:
        lines.extend(["## Blocking Checks", ""])
        for row in report["blocking_checks"]:
            lines.append(f"- `{row['name']}`: {row['detail']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _add_file_check(checks: list[dict], name: str, path: Path) -> None:
    _add_check(checks, name, path.is_file(), str(path))


def _add_model_ref_check(checks: list[dict], name: str, value: str) -> None:
    path = Path(value)
    if path.exists():
        _add_check(checks, name, True, f"local path: {value}")
        return
    if _looks_like_hf_repo_id(value):
        _add_check(checks, name, True, f"Hugging Face repo id: {value}")
        return
    _add_check(checks, name, False, f"missing local path or invalid HF repo id: {value}")


def _add_check(checks: list[dict], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})


def _looks_like_hf_repo_id(value: str) -> bool:
    if not value or value.startswith(("/", "./", "../", "~")):
        return False
    parts = value.split("/")
    if len(parts) != 2:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return all(part and set(part) <= allowed for part in parts)


def _count_jsonl(path: Path) -> int:
    try:
        return sum(1 for _ in read_jsonl(path))
    except Exception:
        return 0


def _free_disk_mb(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free // (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight real LLM agreement requirements.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--teacher-examples", required=True)
    parser.add_argument("--output", default="reports/qwen_realcode_agreement_preflight.json")
    parser.add_argument("--markdown-output", default="reports/qwen_realcode_agreement_preflight.md")
    parser.add_argument("--llm-provider", default="qwen")
    parser.add_argument("--llm-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-model")
    parser.add_argument("--min-free-mb", type=int, default=200)
    args = parser.parse_args()

    report = check_llm_agreement_preflight(
        repo=args.repo,
        tasks=args.tasks,
        teacher_examples=args.teacher_examples,
        llm_provider=args.llm_provider,
        llm_api_key_env=args.llm_api_key_env,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        min_free_mb=args.min_free_mb,
    )
    write_json_report(report, args.output)
    write_markdown_report(report, args.markdown_output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
