"""Lightweight local tool agent for context-inspection experiments.

The agent does not solve coding tasks. It executes a small, deterministic
inspect/read policy over selected context units so experiments can distinguish
"selected context" from "context that a tool action actually read".
"""

from __future__ import annotations

import time
from pathlib import Path

from cicl_agent.core.schema import AgentRun, ContextUnit, Task


class LocalContextInspectionAgent:
    """Read files referenced by selected context units.

    This is a non-LLM execution adapter. Its `success` field means the local
    read tools executed successfully, not that the coding task was solved.
    Gold-context alignment is computed by the evaluator, outside the agent.
    """

    def __init__(self, repo: str | Path, max_inspections: int = 3) -> None:
        self.repo = Path(repo)
        self.max_inspections = max(1, max_inspections)

    def run(self, task: Task, method: str, selected_units: list[ContextUnit]) -> AgentRun:
        start = time.perf_counter()
        actions: list[str] = []
        observations: list[str] = []
        inspected_paths: list[str] = []
        read_paths: list[str] = []
        missing_paths: list[str] = []
        read_bytes = 0

        for unit in selected_units[: self.max_inspections]:
            inspect_path = context_unit_path(unit)
            if not inspect_path:
                observations.append(f"no_path:{unit.id}")
                continue
            actions.append(f"inspect {inspect_path}")
            inspected_paths.append(inspect_path)
            content = self._read_repo_file(inspect_path)
            if content is None:
                observations.append(f"missing:{inspect_path}")
                missing_paths.append(inspect_path)
                continue
            read_paths.append(inspect_path)
            read_bytes += len(content.encode("utf-8"))
            observations.append(f"read:{inspect_path}:bytes={len(content.encode('utf-8'))}")

        if not actions:
            actions.append("inspect task")
            observations.append("no_context_to_inspect")

        runtime = time.perf_counter() - start
        return AgentRun(
            task_id=task.id,
            method=method,
            selected_context_ids=[unit.id for unit in selected_units],
            candidate_context_ids=[unit.id for unit in selected_units],
            actions=actions,
            observations=observations,
            success=bool(read_paths),
            tokens=sum(unit.token_cost for unit in selected_units),
            runtime_seconds=runtime,
            tool_calls=len(actions),
            instance_id=task.instance_id,
            metadata={
                "adapter": "local_context_inspection",
                "max_inspections": self.max_inspections,
                "inspected_paths": inspected_paths,
                "read_paths": read_paths,
                "missing_paths": missing_paths,
                "read_bytes": read_bytes,
            },
        )

    def _read_repo_file(self, path: str) -> str | None:
        normalized = _strip_line_suffix(path)
        candidate = (self.repo / normalized).resolve()
        repo_root = self.repo.resolve()
        try:
            candidate.relative_to(repo_root)
        except ValueError:
            return None
        if not candidate.is_file():
            return None
        try:
            return candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None


def context_unit_path(unit: ContextUnit) -> str:
    """Return the best executable inspect path for a context unit."""

    for value in [
        unit.source,
        str(unit.metadata.get("original_source", "")),
        str(unit.metadata.get("path", "")),
        unit.id,
    ]:
        path = normalize_context_path(value)
        if path:
            return path
    return ""


def normalize_context_path(value: str) -> str:
    value = value.strip().strip("`'\"")
    if not value:
        return ""
    if value.startswith("inspect "):
        value = value.split(" ", 1)[1]
    for prefix in ["compressed:", "summary:", "file:", "symbol:", "memory:"]:
        if value.startswith(prefix):
            value = value[len(prefix):]
    return _strip_line_suffix(value)


def _strip_line_suffix(value: str) -> str:
    parts = value.split(":")
    if len(parts) > 1 and parts[-1].isdigit():
        return ":".join(parts[:-1])
    return value

