"""UCB-style bandit wrapper around a learned context ranker."""

from __future__ import annotations

import math
from collections import defaultdict

from cicl_agent.learning.ranker import PairwiseContextRanker


class CausalContextBanditPolicy:
    """UCB-style wrapper around a learned context ranker."""

    def __init__(self, ranker: PairwiseContextRanker, exploration: float = 0.08) -> None:
        self.ranker = ranker
        self.exploration = exploration
        self.counts: dict[str, int] = defaultdict(int)
        self.values: dict[str, float] = defaultdict(float)
        self.total_updates = 0

    def score(self, features: dict[str, float], context_id: str) -> float:
        learned = self.ranker.score(features)
        count = self.counts[context_id]
        bonus = self.exploration * math.sqrt(math.log(self.total_updates + 2) / (count + 1))
        return learned + bonus

    def update(self, selected_context_ids: list[str], reward: float) -> None:
        self.total_updates += 1
        for context_id in selected_context_ids:
            count = self.counts[context_id]
            old = self.values[context_id]
            self.counts[context_id] = count + 1
            self.values[context_id] = old + (reward - old) / (count + 1)


__all__ = ["CausalContextBanditPolicy"]
