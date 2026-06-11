"""Counterfactual-inspired causal utility scoring."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from cicl_agent.core.schema import AgentRun, CausalScore, ContextUnit, Task
from cicl_agent.core.text import jaccard, tokenize


@dataclass(slots=True)
class ScoringWeights:
    action_delta: float = 0.35
    outcome_uplift: float = 0.35
    necessity: float = 0.20
    cost: float = 0.10


class CausalUtilityScorer:
    """Scores context by expected usefulness for an agent decision.

    The default implementation is deterministic and model-free. Production
    experiments can replace `necessity_judge` with an LLM judge or replace
    `estimate_action_delta` with paired agent rollouts.
    """

    def __init__(
        self,
        history: list[AgentRun] | None = None,
        weights: ScoringWeights | None = None,
        necessity_judge: callable | None = None,
        allow_gold_labels: bool = False,
    ) -> None:
        self.history = history or []
        self.weights = weights or ScoringWeights()
        self.necessity_judge = necessity_judge
        self.allow_gold_labels = allow_gold_labels

    def score(self, unit: ContextUnit, task: Task, retrieval_score: float = 0.0) -> CausalScore:
        action_delta = self.estimate_action_delta(unit, task)
        outcome_uplift = self.estimate_outcome_uplift(unit, task)
        necessity = self.estimate_necessity(unit, task)
        cost_penalty = self.estimate_cost_penalty(unit)
        final = (
            self.weights.action_delta * action_delta
            + self.weights.outcome_uplift * outcome_uplift
            + self.weights.necessity * necessity
            - self.weights.cost * cost_penalty
        )
        final += 0.05 * min(1.0, retrieval_score)
        return CausalScore(
            context_id=unit.id,
            task_id=task.id,
            action_delta=round(action_delta, 6),
            outcome_uplift=round(outcome_uplift, 6),
            necessity_score=round(necessity, 6),
            cost_penalty=round(cost_penalty, 6),
            final_score=round(final, 6),
            metadata={"unit_type": unit.type, "retrieval_score": retrieval_score},
        )

    def estimate_action_delta(self, unit: ContextUnit, task: Task) -> float:
        expected = " ".join(task.expected_actions) or task.instruction
        action_terms = set(tokenize(expected))
        if not action_terms:
            return 0.0
        context_terms = set(tokenize(unit.content + " " + unit.source))
        instruction_terms = set(tokenize(task.instruction))
        new_terms = action_terms & (context_terms - instruction_terms)
        direct_terms = action_terms & context_terms
        return min(1.0, 0.65 * len(direct_terms) / len(action_terms) + 0.35 * len(new_terms) / len(action_terms))

    def estimate_outcome_uplift(self, unit: ContextUnit, task: Task) -> float:
        if not self.history:
            return 0.0
        task_terms = set(tokenize(task.instruction))
        used_success: list[float] = []
        global_success: list[float] = []
        for run in self.history:
            similarity = jaccard(" ".join(run.actions + run.observations), task.instruction)
            weight = max(0.1, similarity) if task_terms else 0.1
            global_success.append(weight * float(run.success))
            if unit.id in run.selected_context_ids:
                used_success.append(weight * float(run.success))
        if not used_success:
            return 0.0
        used_rate = sum(used_success) / max(1e-9, len(used_success))
        global_rate = sum(global_success) / max(1e-9, len(global_success))
        return max(0.0, min(1.0, used_rate - global_rate + 0.5))

    def estimate_necessity(self, unit: ContextUnit, task: Task) -> float:
        if self.necessity_judge:
            return float(self.necessity_judge(unit, task))
        if self.allow_gold_labels and unit.id in task.gold_context_ids:
            return 1.0
        relevance = jaccard(unit.content + " " + unit.source, task.instruction)
        action_relevance = jaccard(unit.content, " ".join(task.expected_actions))
        type_bonus = 0.15 if unit.type in {"rule", "strategy", "failure", "trajectory"} else 0.0
        return min(1.0, relevance + action_relevance + type_bonus)

    @staticmethod
    def estimate_cost_penalty(unit: ContextUnit) -> float:
        token_penalty = min(1.0, unit.token_cost / 1000)
        confidence_penalty = max(0.0, 1.0 - unit.confidence)
        stale_penalty = 0.15 if unit.metadata.get("stale") else 0.0
        conflict_penalty = 0.2 if unit.metadata.get("conflict_risk") else 0.0
        return min(1.0, token_penalty + confidence_penalty + stale_penalty + conflict_penalty)
