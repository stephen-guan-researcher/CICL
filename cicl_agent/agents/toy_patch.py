"""Executable toy patch agent.

This agent is intentionally small and deterministic. It creates a temporary
copy of the toy repository, injects a known bug for each toy task, attempts a
local patch only when selected context supports the relevant file/rule, and
runs a Python validation command.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from cicl_agent.core.schema import AgentRun, ContextUnit, Task


class ToyPatchAgent:
    """A minimal executable patch pilot for the toy benchmark."""

    def __init__(self, repo_template: str | Path) -> None:
        self.repo_template = Path(repo_template)

    def run(self, task: Task, method: str, selected_units: list[ContextUnit]) -> AgentRun:
        start = time.perf_counter()
        actions: list[str] = []
        observations: list[str] = []
        selected_text = "\n".join(
            "\n".join([unit.id, unit.source, unit.content, str(unit.metadata)])
            for unit in selected_units
        ).lower()

        with tempfile.TemporaryDirectory(prefix=f"cicl-toy-{task.id}-") as tmpdir:
            workspace = Path(tmpdir) / "repo"
            shutil.copytree(self.repo_template, workspace)
            mutation = self._mutation_for_task(task)
            if mutation == "timestamp":
                self._inject_timestamp_bug(workspace)
                actions.append("mutate parser.py")
            elif mutation == "api_path":
                self._inject_api_path_bug(workspace)
                actions.append("mutate client.py")
            else:
                observations.append("unknown_task_mutation")

            pre_success, pre_output = self._run_validation(workspace, mutation)
            observations.append(f"pre_validation={'pass' if pre_success else 'fail'}")

            patch_applied = False
            if mutation == "timestamp":
                actions.append("inspect parser.py")
                if _has_any(selected_text, ["parser.py", "parse_iso8601_timestamp", "timestamp", "timezone"]):
                    self._patch_timestamp(workspace)
                    patch_applied = True
                    actions.append("apply patch parser.py")
                else:
                    actions.append("skip patch parser.py")
            elif mutation == "api_path":
                actions.append("inspect client.py")
                if _has_any(selected_text, ["client.py", "build_url", "api path", "leading slash"]):
                    self._patch_api_path(workspace)
                    patch_applied = True
                    actions.append("apply patch client.py")
                else:
                    actions.append("skip patch client.py")

            post_success, post_output = self._run_validation(workspace, mutation)
            observations.append(f"post_validation={'pass' if post_success else 'fail'}")
            if post_output:
                observations.append("validation_output:" + _compact(post_output))

        runtime = time.perf_counter() - start
        return AgentRun(
            task_id=task.id,
            method=method,
            selected_context_ids=[unit.id for unit in selected_units],
            candidate_context_ids=[unit.id for unit in selected_units],
            actions=actions,
            observations=observations,
            success=bool(post_success and not pre_success),
            tokens=sum(unit.token_cost for unit in selected_units),
            runtime_seconds=runtime,
            tool_calls=len(actions) + 2,
            instance_id=task.instance_id,
            metadata={
                "adapter": "toy_patch_agent",
                "mutation": mutation,
                "pre_validation_success": pre_success,
                "post_validation_success": post_success,
                "patch_applied": patch_applied,
                "pre_output": _compact(pre_output),
                "post_output": _compact(post_output),
            },
        )

    @staticmethod
    def _mutation_for_task(task: Task) -> str:
        text = f"{task.id} {task.instruction}".lower()
        if "timestamp" in text or "parse" in text:
            return "timestamp"
        if "url" in text or "endpoint" in text or "slash" in text:
            return "api_path"
        return "unknown"

    @staticmethod
    def _inject_timestamp_bug(workspace: Path) -> None:
        path = workspace / "parser.py"
        text = path.read_text(encoding="utf-8")
        start = text.index("def parse_iso8601_timestamp")
        end = text.index("\n\ndef normalize_user_id")
        broken = '''def parse_iso8601_timestamp(value: str) -> datetime:
    """Parse timestamps used by the ingestion pipeline.

    BUG: trailing Z is stripped, producing a naive datetime.
    """
    if value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value)
'''
        path.write_text(text[:start] + broken + text[end:], encoding="utf-8")

    @staticmethod
    def _patch_timestamp(workspace: Path) -> None:
        path = workspace / "parser.py"
        text = path.read_text(encoding="utf-8")
        start = text.index("def parse_iso8601_timestamp")
        end = text.index("\n\ndef normalize_user_id")
        fixed = '''def parse_iso8601_timestamp(value: str) -> datetime:
    """Parse timestamps used by the ingestion pipeline.

    Local project rule: timestamps ending with Z must be interpreted as UTC.
    """
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
'''
        path.write_text(text[:start] + fixed + text[end:], encoding="utf-8")

    @staticmethod
    def _inject_api_path_bug(workspace: Path) -> None:
        path = workspace / "client.py"
        text = path.read_text(encoding="utf-8")
        start = text.index("    def build_url")
        end = text.index("\n\n\ndef retryable_status")
        broken = '''    def build_url(self, path: str) -> str:
        return self.base_url + path
'''
        path.write_text(text[:start] + broken + text[end:], encoding="utf-8")

    @staticmethod
    def _patch_api_path(workspace: Path) -> None:
        path = workspace / "client.py"
        text = path.read_text(encoding="utf-8")
        start = text.index("    def build_url")
        end = text.index("\n\n\ndef retryable_status")
        fixed = '''    def build_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path
'''
        path.write_text(text[:start] + fixed + text[end:], encoding="utf-8")

    @staticmethod
    def _run_validation(workspace: Path, mutation: str) -> tuple[bool, str]:
        if mutation == "timestamp":
            code = (
                "from parser import parse_iso8601_timestamp;"
                "dt=parse_iso8601_timestamp('2024-01-01T00:00:00Z');"
                "assert dt.tzinfo is not None;"
                "assert dt.utcoffset().total_seconds()==0"
            )
        elif mutation == "api_path":
            code = (
                "from client import ApiClient;"
                "c=ApiClient('https://example.com/');"
                "assert c.build_url('v1')=='https://example.com/v1';"
                "assert c.build_url('/v2')=='https://example.com/v2'"
            )
        else:
            return False, "unknown mutation"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=workspace,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output


def _has_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _compact(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text[:limit]
