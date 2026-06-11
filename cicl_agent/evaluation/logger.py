"""Trajectory logging for agent runs."""

from __future__ import annotations

from pathlib import Path

from cicl_agent.core.io import append_jsonl
from cicl_agent.core.schema import AgentRun, ContextUnit, Task


class TrajectoryLogger:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.runs_path = self.output_dir / "agent_runs.jsonl"

    def log_run(self, run: AgentRun, task: Task, selected_units: list[ContextUnit]) -> None:
        row = run.to_dict()
        row["task_instruction"] = task.instruction
        row["selected_context_preview"] = [
            {
                "id": unit.id,
                "type": unit.type,
                "source": unit.source,
                "token_cost": unit.token_cost,
            }
            for unit in selected_units
        ]
        append_jsonl(self.runs_path, row)


__all__ = ["TrajectoryLogger"]
