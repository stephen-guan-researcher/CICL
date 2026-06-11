"""LLM-CICL selector: counterfactual judging drives selection and graph edges."""

from __future__ import annotations

from dataclasses import dataclass, field

from cicl_agent.causal.judgment import judgment_to_graph_artifact, judgment_to_score
from cicl_agent.causal.llm_judge import LLMCausalContextJudge
from cicl_agent.core.schema import AgentRun, ContextUnit, Task
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.retrieval.assembly import BudgetPolicy
from cicl_agent.retrieval.retrievers import HybridRetriever


@dataclass(slots=True)
class LLMCICLSelector:
    graph: InstanceContextGraph
    name: str = "LLM-CICL"
    max_units: int | None = 5
    conflict_filtering: bool = True
    record_policy_edges: bool = False
    last_scores: list = field(default_factory=list)
    judge: LLMCausalContextJudge = field(default_factory=LLMCausalContextJudge)

    def select(self, task: Task, budget: int, history: list[AgentRun]) -> list[ContextUnit]:
        retrieved = HybridRetriever(list(self.graph.units.values())).search(task, top_k=24)
        candidates: dict[str, tuple[ContextUnit, float]] = {unit.id: (unit, score) for unit, score in retrieved}
        for unit, score in retrieved[:8]:
            for neighbor in self.graph.neighbors(unit.id, depth=1):
                candidates.setdefault(neighbor.id, (neighbor, score * 0.6))

        judgments = {unit.id: self.judge.judge(task, unit) for unit, _ in candidates.values()}
        scored_rows = [
            (unit, judgment_to_score(judgments[unit.id], unit))
            for unit, _ in candidates.values()
        ]
        scored_rows.sort(key=lambda row: row[1].final_score, reverse=True)

        policy = BudgetPolicy(max_units=self.max_units, min_final_score=0.02, min_score_after_first=0.04)
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
        if self.record_policy_edges:
            for unit in selected:
                artifact, edge = judgment_to_graph_artifact(task, unit, judgments[unit.id])
                self.graph.add_unit(artifact)
                self.graph.add_edge(edge)
        return selected


__all__ = ["LLMCICLSelector"]
