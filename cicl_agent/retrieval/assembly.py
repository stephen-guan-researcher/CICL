"""Budget-aware context assembly for CICL.

The `BudgetPolicy` dataclass and the `BudgetAwareContextAssembler` co-locate:
the policy is purely a budget guardrail, while the assembler turns retrieved
candidates plus a policy into a final context bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from cicl_agent.causal.scoring import CausalUtilityScorer, ScoringWeights
from cicl_agent.core.schema import AgentRun, CausalScore, ContextUnit, Task
from cicl_agent.core.text import jaccard
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.retrieval.retrievers import HybridRetriever


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """Adaptive budget policy: ACON-style budget competition with stale-aware floors."""

    max_units: int | None = 5
    min_final_score: float = 0.02
    min_score_after_first: float = 0.06
    reserve_tokens: int = 0
    stale_penalty_floor: float = 0.12

    def allow(
        self,
        unit: ContextUnit,
        score: CausalScore,
        selected: list[ContextUnit],
        used_tokens: int,
        token_budget: int,
    ) -> bool:
        if self.max_units is not None and len(selected) >= self.max_units:
            return False
        if used_tokens + unit.token_cost > max(0, token_budget - self.reserve_tokens):
            return False
        threshold = self.min_final_score if not selected else self.min_score_after_first
        if unit.metadata.get("stale"):
            threshold += self.stale_penalty_floor
        return score.final_score >= threshold

    def after_failure(self, missing_gold_context: bool = False) -> "BudgetPolicy":
        if missing_gold_context:
            return replace(
                self,
                max_units=None if self.max_units is None else min(self.max_units + 1, 10),
                min_score_after_first=max(0.0, self.min_score_after_first - 0.02),
            )
        return replace(
            self,
            min_final_score=self.min_final_score + 0.02,
            min_score_after_first=self.min_score_after_first + 0.02,
        )


class BudgetAwareContextAssembler:
    def __init__(
        self,
        graph: InstanceContextGraph,
        history: list[AgentRun] | None = None,
        top_k: int = 30,
        neighbor_depth: int = 1,
        redundancy_threshold: float = 0.82,
        weights: ScoringWeights | None = None,
        conflict_filtering: bool = True,
        max_units: int | None = 5,
        budget_policy: BudgetPolicy | None = None,
    ) -> None:
        self.graph = graph
        self.history = history or []
        self.top_k = top_k
        self.neighbor_depth = neighbor_depth
        self.redundancy_threshold = redundancy_threshold
        self.weights = weights
        self.conflict_filtering = conflict_filtering
        self.max_units = max_units
        self.budget_policy = budget_policy or BudgetPolicy(max_units=max_units)

    def assemble(self, task: Task, token_budget: int) -> tuple[list[ContextUnit], list[CausalScore]]:
        units = list(self.graph.units.values())
        retriever = HybridRetriever(units)
        retrieved = retriever.search(task, top_k=self.top_k)
        candidates: dict[str, tuple[ContextUnit, float]] = {unit.id: (unit, score) for unit, score in retrieved}
        for unit, score in retrieved[: max(1, self.top_k // 3)]:
            for neighbor in self.graph.neighbors(unit.id, depth=self.neighbor_depth):
                candidates.setdefault(neighbor.id, (neighbor, score * 0.6))

        scorer = CausalUtilityScorer(history=self.history, weights=self.weights)
        scored = [scorer.score(unit, task, retrieval_score=score) for unit, score in candidates.values()]
        score_by_id = {score.context_id: score for score in scored}
        ranked = sorted(candidates.values(), key=lambda row: score_by_id[row[0].id].final_score, reverse=True)

        selected: list[ContextUnit] = []
        used_tokens = 0
        blocked: set[str] = set()
        for unit, _ in ranked:
            if unit.id in blocked:
                continue
            if self._is_redundant(unit, selected):
                continue
            score = score_by_id[unit.id]
            if not self.budget_policy.allow(unit, score, selected, used_tokens, token_budget):
                if self.budget_policy.max_units is not None and len(selected) >= self.budget_policy.max_units:
                    break
                continue
            selected.append(unit)
            used_tokens += unit.token_cost
            if self.conflict_filtering:
                blocked.update(self.graph.conflicts(unit.id))

        return selected, [score_by_id[unit.id] for unit in selected if unit.id in score_by_id]

    def _is_redundant(self, unit: ContextUnit, selected: list[ContextUnit]) -> bool:
        return any(jaccard(unit.content, prior.content) >= self.redundancy_threshold for prior in selected)
