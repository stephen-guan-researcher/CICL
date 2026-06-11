"""Cross-cutting structural protocols.

Centralising Protocols here keeps low-level modules (causal, learning, runners)
from importing concrete implementations of each other just to share a type.
"""

from __future__ import annotations

from typing import Protocol

from cicl_agent.core.schema import AgentRun, ContextUnit, Task


class AgentAdapter(Protocol):
    """Minimal contract for any agent that can be driven by a context selector."""

    def run(self, task: Task, method: str, selected_units: list[ContextUnit]) -> AgentRun:
        ...
