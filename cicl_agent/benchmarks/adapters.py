"""Dataset adapter interfaces.

Real paper experiments should convert benchmark records into the shared `Task`
schema here, while keeping the rest of the framework unchanged.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from cicl_agent.core.io import read_jsonl
from cicl_agent.core.schema import Task


DIFF_PATH_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def patch_file_context_ids(diff_text: str) -> list[str]:
    """Return file-level context ids touched by a unified git diff."""
    paths: set[str] = set()
    for match in DIFF_PATH_RE.finditer(diff_text or ""):
        before, after = match.groups()
        for path in (before, after):
            if path != "/dev/null":
                paths.add(path)
    return [f"file:{path}" for path in sorted(paths)]


def swebench_row_to_task(row: dict[str, Any], split: str = "eval") -> Task:
    """Convert a raw SWE-bench row to a non-leaking CICL task.

    The solution patch is used only to derive file-level gold context ids for
    offline retrieval metrics. The patch itself is intentionally excluded from
    the task metadata and instruction.
    """

    instance_id = str(row.get("repo", "swebench"))
    task_id = str(row.get("instance_id", row.get("id", "")))
    problem = str(row.get("problem_statement", "")).strip()
    hints = str(row.get("hints_text", "")).strip()
    instruction = problem if not hints else f"{problem}\n\nHints from issue discussion:\n{hints}"
    return Task(
        id=task_id,
        instance_id=instance_id,
        instruction=instruction,
        split=split,
        gold_context_ids=patch_file_context_ids(str(row.get("patch", ""))),
        expected_actions=[],
        difficulty=str(row.get("difficulty", "hard")),
        metadata={
            "source_dataset": str(row.get("source_dataset", "swebench")),
            "repo": instance_id,
            "base_commit": str(row.get("base_commit", "")),
            "created_at": str(row.get("created_at", "")),
            "version": str(row.get("version", "")),
            "fail_to_pass_count": len(row.get("FAIL_TO_PASS", []) or []),
            "pass_to_pass_count": len(row.get("PASS_TO_PASS", []) or []),
            "requires_context": True,
        },
    )


def repobench_row_to_task(row: dict[str, Any], index: int, split: str = "eval") -> Task:
    """Convert a RepoBench-R retrieval/completion record to a CICL task."""

    repo_name = str(row.get("repo_name", "repobench-r"))
    file_path = str(row.get("file_path", f"sample_{index}.py"))
    gold_index = row.get("golden_snippet_index")
    gold_ids: list[str] = []
    if isinstance(gold_index, int):
        gold_ids.append(f"file:contexts/{index:05d}_{gold_index:02d}.py")
    prefix = str(row.get("code", "")).strip()
    imports = str(row.get("import_statement", "")).strip()
    prompt_parts = [
        f"Complete the next line in `{file_path}`.",
        "Use repository context snippets when they causally determine the completion.",
    ]
    if imports:
        prompt_parts.append(f"Imports:\n{imports}")
    if prefix:
        prompt_parts.append(f"Code prefix:\n{prefix[-4000:]}")
    return Task(
        id=f"repobench-r-{index:05d}",
        instance_id=repo_name,
        instruction="\n\n".join(prompt_parts),
        split=split,
        gold_context_ids=gold_ids,
        expected_actions=[],
        difficulty=str(row.get("difficulty", "medium")),
        metadata={
            "source_dataset": "repobench-r",
            "repo": repo_name,
            "file_path": file_path,
            "language": "python",
            "task_type": "repository_context_retrieval",
            "requires_context": True,
        },
    )


def swebench_rr_query_to_task(query: dict[str, Any], gold_ids: list[str], split: str = "eval") -> Task:
    """Convert a SWEbenchVerifiedRR query row to a retrieval task."""

    query_id = str(query["id"])
    return Task(
        id=query_id,
        instance_id="swebench_verified_rr",
        instruction=str(query.get("text", "")),
        split=split,
        gold_context_ids=[f"doc:{doc_id}" for doc_id in gold_ids],
        expected_actions=[],
        difficulty="medium",
        metadata={
            "source_dataset": "mteb/SWEbenchVerifiedRR",
            "task_type": "context_retrieval",
            "requires_context": True,
        },
    )


class BenchmarkAdapter(ABC):
    @abstractmethod
    def load_tasks(self) -> list[Task]:
        """Return tasks in CICL's shared schema."""


class JsonlTaskAdapter(BenchmarkAdapter):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load_tasks(self) -> list[Task]:
        return [Task.from_dict(row) for row in read_jsonl(self.path)]


class SWEContextBenchAdapter(BenchmarkAdapter):
    """Adapter skeleton for SWE-ContextBench style records.

    Expected normalized input fields:
    - task_id
    - repo
    - instruction
    - split
    - gold_context_ids
    - prior_experience_summary
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load_tasks(self) -> list[Task]:
        tasks: list[Task] = []
        for row in read_jsonl(self.path):
            tasks.append(
                Task(
                    id=str(row["task_id"]),
                    instance_id=str(row["repo"]),
                    instruction=str(row["instruction"]),
                    split=str(row.get("split", "eval")),
                    gold_context_ids=list(row.get("gold_context_ids", [])),
                    expected_actions=list(row.get("expected_actions", [])),
                    memory=str(row.get("prior_experience_summary", "")),
                    difficulty=str(row.get("difficulty", "medium")),
                    metadata={key: value for key, value in row.items() if key not in {
                        "task_id",
                        "repo",
                        "instruction",
                        "split",
                        "gold_context_ids",
                        "expected_actions",
                        "prior_experience_summary",
                        "difficulty",
                    }},
                )
            )
        return tasks


class ContextBenchAdapter(BenchmarkAdapter):
    """Adapter skeleton for ContextBench style gold-context records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load_tasks(self) -> list[Task]:
        tasks: list[Task] = []
        for row in read_jsonl(self.path):
            tasks.append(
                Task(
                    id=str(row["issue_id"]),
                    instance_id=str(row["repo"]),
                    instruction=str(row.get("issue_text", row.get("instruction", ""))),
                    split=str(row.get("split", "eval")),
                    gold_context_ids=list(row.get("gold_context_ids", row.get("gold_files", []))),
                    expected_actions=list(row.get("expected_actions", [])),
                    difficulty=str(row.get("difficulty", "medium")),
                    metadata={"language": row.get("language"), "raw": row},
                )
            )
        return tasks


class SWEBenchIssueAdapter(BenchmarkAdapter):
    """Adapter for raw rows from SWE-bench Lite/Verified JSONL samples."""

    def __init__(self, path: str | Path, split: str = "eval") -> None:
        self.path = Path(path)
        self.split = split

    def load_tasks(self) -> list[Task]:
        return [swebench_row_to_task(row, split=self.split) for row in read_jsonl(self.path)]


class RepoBenchRAdapter(BenchmarkAdapter):
    """Adapter for normalized RepoBench-R JSONL samples emitted by the downloader."""

    def __init__(self, path: str | Path, split: str = "eval") -> None:
        self.path = Path(path)
        self.split = split

    def load_tasks(self) -> list[Task]:
        return [repobench_row_to_task(row, index=i, split=self.split) for i, row in enumerate(read_jsonl(self.path))]


class SWEbenchVerifiedRRAdapter(BenchmarkAdapter):
    """Adapter for processed SWEbenchVerifiedRR query task JSONL files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load_tasks(self) -> list[Task]:
        return [Task.from_dict(row) for row in read_jsonl(self.path)]
