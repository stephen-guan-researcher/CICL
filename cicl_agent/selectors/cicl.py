"""Main CICL selector and the no-instance-graph ablation."""

from __future__ import annotations

from dataclasses import dataclass, field

from cicl_agent.causal.scoring import ScoringWeights
from cicl_agent.core.schema import AgentRun, ContextUnit, Task
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.retrieval.assembly import BudgetAwareContextAssembler, BudgetPolicy


@dataclass(slots=True)
class CICLSelector:
    graph: InstanceContextGraph
    name: str = "CICL"
    weights: ScoringWeights | None = None
    conflict_filtering: bool = True
    max_units: int | None = 5
    budget_policy: BudgetPolicy | None = None
    last_scores: list = field(default_factory=list)

    def select(self, task: Task, budget: int, history: list[AgentRun]) -> list[ContextUnit]:
        policy = self.budget_policy or BudgetPolicy(max_units=self.max_units)
        selected, scores = BudgetAwareContextAssembler(
            self.graph,
            history=history,
            weights=self.weights,
            conflict_filtering=self.conflict_filtering,
            max_units=self.max_units,
            budget_policy=policy,
        ).assemble(task, budget)
        self.last_scores = scores
        return selected


@dataclass(slots=True)
class CICLNoInstanceGraphSelector:
    graph: InstanceContextGraph
    name: str = "CICL_w/o_instance_graph"
    max_units: int | None = 5
    last_scores: list = field(default_factory=list)

    def select(self, task: Task, budget: int, history: list[AgentRun]) -> list[ContextUnit]:
        memory_graph = InstanceContextGraph(task.instance_id)
        for unit in self.graph.units.values():
            if unit.type in {"rule", "strategy", "trajectory", "failure"}:
                memory_graph.add_unit(unit)
        selected, scores = BudgetAwareContextAssembler(
            memory_graph,
            history=history,
            max_units=self.max_units,
        ).assemble(task, budget)
        self.last_scores = scores
        return selected


__all__ = ["CICLNoInstanceGraphSelector", "CICLSelector"]
