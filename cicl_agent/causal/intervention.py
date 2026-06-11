"""Interventional diagnostics for context utility.

This module implements a lightweight CMI-style ablation probe. It runs the same
task with no context and with one candidate context, then measures whether the
candidate changes the agent's actions or outcome.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cicl_agent.agents.simulated import SimulatedReActAgent
from cicl_agent.core.protocols import AgentAdapter
from cicl_agent.core.schema import AgentRun, CausalScore, ContextUnit, Task


@dataclass(slots=True)
class InterventionResult:
    task_id: str
    context_id: str
    no_context_success: bool
    with_context_success: bool
    action_delta: float
    success_delta: float
    no_context_actions: list[str]
    with_context_actions: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class InterventionScorer:
    def __init__(self, agent: AgentAdapter | None = None) -> None:
        self.agent = agent or SimulatedReActAgent()

    def score_candidates(self, task: Task, candidates: list[ContextUnit]) -> list[InterventionResult]:
        base = self.agent.run(task, method="intervention:no_context", selected_units=[])
        rows: list[InterventionResult] = []
        for unit in candidates:
            with_context = self.agent.run(
                task,
                method=f"intervention:{unit.id}",
                selected_units=[unit],
            )
            rows.append(self._compare(task.id, unit.id, base, with_context))
        return rows

    def score_as_causal_scores(self, task: Task, candidates: list[ContextUnit]) -> list[CausalScore]:
        scores: list[CausalScore] = []
        for row in self.score_candidates(task, candidates):
            final = 0.7 * row.action_delta + 0.3 * max(0.0, row.success_delta)
            scores.append(
                CausalScore(
                    context_id=row.context_id,
                    task_id=row.task_id,
                    action_delta=round(row.action_delta, 6),
                    outcome_uplift=round(max(0.0, row.success_delta), 6),
                    necessity_score=round(float(row.with_context_success and not row.no_context_success), 6),
                    cost_penalty=0.0,
                    final_score=round(final, 6),
                    metadata={"probe": "single_context_intervention"},
                )
            )
        return scores

    @staticmethod
    def _compare(
        task_id: str,
        context_id: str,
        base: AgentRun,
        with_context: AgentRun,
    ) -> InterventionResult:
        base_actions = set(base.actions)
        context_actions = set(with_context.actions)
        union = base_actions | context_actions
        if not union:
            action_delta = 0.0
        else:
            action_delta = 1.0 - (len(base_actions & context_actions) / len(union))
        return InterventionResult(
            task_id=task_id,
            context_id=context_id,
            no_context_success=base.success,
            with_context_success=with_context.success,
            action_delta=action_delta,
            success_delta=float(with_context.success) - float(base.success),
            no_context_actions=list(base.actions),
            with_context_actions=list(with_context.actions),
        )
