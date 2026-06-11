"""Selector protocol and the shared budget-fitting helper."""

from __future__ import annotations

from typing import Protocol

from cicl_agent.core.schema import AgentRun, ContextUnit, Task


class ContextSelector(Protocol):
    name: str

    def select(self, task: Task, budget: int, history: list[AgentRun]) -> list[ContextUnit]:
        ...


def fit_budget(units: list[ContextUnit], budget: int) -> list[ContextUnit]:
    """Greedy first-fit packing: keep units while their token cost fits the budget."""

    selected: list[ContextUnit] = []
    total = 0
    for unit in units:
        if total + unit.token_cost > budget:
            continue
        selected.append(unit)
        total += unit.token_cost
    return selected


__all__ = ["ContextSelector", "fit_budget"]
