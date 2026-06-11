"""LearnedCICL: bandit-shaped pairwise ranker over candidate context."""

from __future__ import annotations

from dataclasses import dataclass, field

from cicl_agent.core.schema import AgentRun, CausalScore, ContextUnit, Task
from cicl_agent.learning.bandit import CausalContextBanditPolicy
from cicl_agent.learning.features import ContextFeatureExtractor
from cicl_agent.learning.ranker import PairwiseContextRanker
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.retrieval.assembly import BudgetPolicy
from cicl_agent.retrieval.retrievers import HybridRetriever


@dataclass(slots=True)
class LearnedCICLSelector:
    graph: InstanceContextGraph
    ranker: PairwiseContextRanker
    name: str = "LearnedCICL"
    max_units: int | None = 5
    conflict_filtering: bool = True
    exploration: float = 0.04
    last_scores: list = field(default_factory=list)
    _bandit: CausalContextBanditPolicy | None = None

    def __post_init__(self) -> None:
        self._bandit = CausalContextBanditPolicy(self.ranker, exploration=self.exploration)

    def select(self, task: Task, budget: int, history: list[AgentRun]) -> list[ContextUnit]:
        retrieved = HybridRetriever(list(self.graph.units.values())).search(task, top_k=30)
        candidates: dict[str, tuple[ContextUnit, float]] = {unit.id: (unit, score) for unit, score in retrieved}
        for unit, score in retrieved[:10]:
            for neighbor in self.graph.neighbors(unit.id, depth=1):
                candidates.setdefault(neighbor.id, (neighbor, score * 0.6))

        extractor = ContextFeatureExtractor(self.graph, history=history)
        scored_rows: list[tuple[ContextUnit, CausalScore]] = []
        for unit, retrieval_score in candidates.values():
            features = extractor.features(task, unit, retrieval_score=retrieval_score)
            assert self._bandit is not None
            learned_score = self._bandit.score(features, unit.id)
            cost_penalty = min(1.0, unit.token_cost / 1000) + (0.15 if unit.metadata.get("stale") else 0.0)
            scored_rows.append(
                (
                    unit,
                    CausalScore(
                        context_id=unit.id,
                        task_id=task.id,
                        action_delta=round(features.get("heuristic_action_delta", 0.0), 6),
                        outcome_uplift=round(features.get("history_used_success", 0.0), 6),
                        necessity_score=round(self.ranker.score(features), 6),
                        cost_penalty=round(min(1.0, cost_penalty), 6),
                        final_score=round(learned_score, 6),
                        metadata={"selector": self.name, "retrieval_score": retrieval_score},
                    ),
                )
            )
        scored_rows.sort(key=lambda row: row[1].final_score, reverse=True)

        policy = BudgetPolicy(max_units=self.max_units, min_final_score=0.05, min_score_after_first=0.05)
        selected: list[ContextUnit] = []
        used_tokens = 0
        blocked: set[str] = set()
        for unit, score in scored_rows:
            if unit.id in blocked:
                continue
            if not policy.allow(unit, score, selected, used_tokens, budget):
                if policy.max_units is not None and len(selected) >= policy.max_units:
                    break
                continue
            selected.append(unit)
            used_tokens += unit.token_cost
            if self.conflict_filtering:
                blocked.update(self.graph.conflicts(unit.id))
        selected_ids = {unit.id for unit in selected}
        self.last_scores = [score for unit, score in scored_rows if unit.id in selected_ids]
        return selected

    def update(self, task: Task, selected_units: list[ContextUnit], run: AgentRun) -> None:
        reward = float(run.success) - 0.001 * run.tokens - 0.02 * run.tool_calls
        assert self._bandit is not None
        self._bandit.update([unit.id for unit in selected_units], reward)


__all__ = ["LearnedCICLSelector"]
