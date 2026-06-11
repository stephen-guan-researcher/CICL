"""Comparable baselines for context selection.

Each selector matches a published context-selection strategy in spirit, so
ablations and the main CICL selector all share the same interface.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from cicl_agent.core.schema import AgentRun, ContextUnit, Task
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.retrieval.retrievers import BM25Retriever, HybridRetriever
from cicl_agent.selectors.base import fit_budget


@dataclass(slots=True)
class NoContextSelector:
    graph: InstanceContextGraph
    name: str = "NoContext"

    def select(self, task: Task, budget: int, history: list[AgentRun]) -> list[ContextUnit]:
        return []


@dataclass(slots=True)
class FullContextSelector:
    graph: InstanceContextGraph
    name: str = "FullContext"

    def select(self, task: Task, budget: int, history: list[AgentRun]) -> list[ContextUnit]:
        units = sorted(
            self.graph.units.values(),
            key=lambda unit: (unit.type not in {"rule", "strategy"}, unit.token_cost),
        )
        return fit_budget(units, budget)


@dataclass(slots=True)
class VanillaRAGSelector:
    graph: InstanceContextGraph
    name: str = "VanillaRAG"

    def select(self, task: Task, budget: int, history: list[AgentRun]) -> list[ContextUnit]:
        retrieved = HybridRetriever(list(self.graph.units.values())).search(task, top_k=40)
        return fit_budget([unit for unit, _ in retrieved], budget)


@dataclass(slots=True)
class SummaryMemorySelector:
    graph: InstanceContextGraph
    name: str = "SummaryMemory"

    def select(self, task: Task, budget: int, history: list[AgentRun]) -> list[ContextUnit]:
        summaries = [
            unit
            for unit in self.graph.units.values()
            if unit.type in {"rule", "strategy", "trajectory", "failure"}
        ]
        ranked = BM25Retriever(summaries).search(task.instruction, top_k=20) if summaries else []
        return fit_budget([unit for unit, _ in ranked], budget)


@dataclass(slots=True)
class GraphMemorySelector:
    graph: InstanceContextGraph
    name: str = "GraphMemory"

    def select(self, task: Task, budget: int, history: list[AgentRun]) -> list[ContextUnit]:
        retrieved = HybridRetriever(list(self.graph.units.values())).search(task, top_k=12)
        expanded: list[ContextUnit] = []
        seen: set[str] = set()
        for unit, _ in retrieved:
            for candidate in [unit, *self.graph.neighbors(unit.id, depth=1)]:
                if candidate.id in seen:
                    continue
                seen.add(candidate.id)
                expanded.append(candidate)
        return fit_budget(expanded, budget)


@dataclass(slots=True)
class AutoContextKGSelector:
    graph: InstanceContextGraph
    name: str = "AutoContextKG"

    def select(self, task: Task, budget: int, history: list[AgentRun]) -> list[ContextUnit]:
        retrieved = HybridRetriever(list(self.graph.units.values())).search(task, top_k=20)
        ranked = sorted(
            (unit for unit, _ in retrieved),
            key=lambda unit: unit.type not in {"file", "symbol", "rule"},
        )
        return fit_budget(ranked, budget)


@dataclass(slots=True)
class SelfGeneratedExamplesSelector:
    graph: InstanceContextGraph
    name: str = "SelfGeneratedExamples"

    def select(self, task: Task, budget: int, history: list[AgentRun]) -> list[ContextUnit]:
        successes = [
            unit
            for unit in self.graph.units.values()
            if unit.type in {"rule", "trajectory", "strategy"} and unit.confidence >= 0.75
        ]
        ranked = BM25Retriever(successes).search(task.instruction, top_k=20) if successes else []
        return fit_budget([unit for unit, _ in ranked], budget)


@dataclass(slots=True)
class OracleGoldContextSelector:
    graph: InstanceContextGraph
    name: str = "OracleGoldContext"

    def select(self, task: Task, budget: int, history: list[AgentRun]) -> list[ContextUnit]:
        gold = [self.graph.units[unit_id] for unit_id in task.gold_context_ids if unit_id in self.graph.units]
        return fit_budget(gold, budget)


@dataclass(slots=True)
class CICLRandomContextSelector:
    graph: InstanceContextGraph
    seed: int = 13
    name: str = "CICL_random_context"

    def select(self, task: Task, budget: int, history: list[AgentRun]) -> list[ContextUnit]:
        units = list(self.graph.units.values())
        rng = random.Random(f"{self.seed}:{task.id}")
        rng.shuffle(units)
        return fit_budget(units, budget)


__all__ = [
    "AutoContextKGSelector",
    "CICLRandomContextSelector",
    "FullContextSelector",
    "GraphMemorySelector",
    "NoContextSelector",
    "OracleGoldContextSelector",
    "SelfGeneratedExamplesSelector",
    "SummaryMemorySelector",
    "VanillaRAGSelector",
]
