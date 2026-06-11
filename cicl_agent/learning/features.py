"""Deterministic context-utility feature extractor.

The pairwise ranker and bandit are trained on top of these features. Keeping
them in their own module makes feature engineering decoupled from training and
reusable from selectors.
"""

from __future__ import annotations

from cicl_agent.causal.scoring import CausalUtilityScorer
from cicl_agent.core.schema import AgentRun, ContextUnit, Task
from cicl_agent.core.text import jaccard, tokenize
from cicl_agent.memory.graph import InstanceContextGraph


class ContextFeatureExtractor:
    """Create deterministic task-context features for learned ranking."""

    TYPE_NAMES = ["file", "symbol", "class", "rule", "strategy", "trajectory", "failure", "api"]

    def __init__(self, graph: InstanceContextGraph, history: list[AgentRun] | None = None) -> None:
        self.graph = graph
        self.history = history or []
        self.heuristic = CausalUtilityScorer(history=self.history)

    def features(self, task: Task, unit: ContextUnit, retrieval_score: float = 0.0) -> dict[str, float]:
        text = f"{unit.content} {unit.source}"
        expected = " ".join(task.expected_actions)
        task_tokens = set(tokenize(task.instruction))
        unit_tokens = set(tokenize(text))
        expected_tokens = set(tokenize(expected))
        overlap = len(task_tokens & unit_tokens) / max(1, len(task_tokens | unit_tokens))
        expected_overlap = len(expected_tokens & unit_tokens) / max(1, len(expected_tokens))
        conflict_degree = len(self.graph.conflicts(unit.id))
        used_success, used_failure = self._history_rates(unit.id)
        score = self.heuristic.score(unit, task, retrieval_score=retrieval_score)
        features: dict[str, float] = {
            "bias": 1.0,
            # BM25-based hybrid score is unbounded; squash to [0,1) without
            # saturating like min(1.0, ...) would (BM25 scores routinely
            # exceed 1.0 so a hard clamp killed all training-set variance).
            "retrieval_score": retrieval_score / (1.0 + max(0.0, retrieval_score)),
            "task_context_jaccard": overlap,
            "expected_action_overlap": expected_overlap,
            "source_jaccard": jaccard(unit.source, task.instruction),
            "heuristic_action_delta": score.action_delta,
            "heuristic_necessity": score.necessity_score,
            "heuristic_cost": score.cost_penalty,
            "heuristic_final": score.final_score,
            "token_cost_norm": min(1.0, unit.token_cost / 1000),
            "confidence": unit.confidence,
            "stale": float(bool(unit.metadata.get("stale"))),
            "conflict_risk": float(bool(unit.metadata.get("conflict_risk"))),
            "conflict_degree_norm": min(1.0, conflict_degree / 5),
            "history_used_success": used_success,
            "history_used_failure": used_failure,
            "is_memory": float(unit.type in {"rule", "strategy", "trajectory", "failure"}),
        }
        for type_name in self.TYPE_NAMES:
            features[f"type_{type_name}"] = float(unit.type == type_name)
        return features

    def _history_rates(self, context_id: str) -> tuple[float, float]:
        used = [run for run in self.history if context_id in run.selected_context_ids]
        if not used:
            return 0.0, 0.0
        success = sum(1 for run in used if run.success) / len(used)
        return success, 1.0 - success


__all__ = ["ContextFeatureExtractor"]
