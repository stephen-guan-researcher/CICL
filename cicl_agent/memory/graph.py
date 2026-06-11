"""Instance context graph storage and traversal.

This module owns the data structure only. Repo-walking lives in
[cicl_agent.memory.builder][] and conflict-marking lives in
[cicl_agent.memory.conflict][].
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from cicl_agent.core.io import read_jsonl, write_jsonl
from cicl_agent.core.schema import AgentRun, ContextEdge, ContextUnit, Task
from cicl_agent.memory.builder import populate_from_repo
from cicl_agent.memory.conflict import ConflictDetector
from cicl_agent.memory.curation import ExperienceCurator


class InstanceContextGraph:
    """A lightweight graph for file, symbol, rule, and trajectory context."""

    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id
        self.units: dict[str, ContextUnit] = {}
        self.edges: list[ContextEdge] = []
        self._adjacency: dict[str, list[ContextEdge]] = defaultdict(list)

    def add_unit(self, unit: ContextUnit) -> ContextUnit:
        self.units[unit.id] = unit
        return unit

    def add_edge(self, edge: ContextEdge) -> ContextEdge:
        self.edges.append(edge)
        self._adjacency[edge.source_id].append(edge)
        self._adjacency[edge.target_id].append(
            ContextEdge(edge.target_id, edge.source_id, edge.relation, edge.weight, edge.metadata)
        )
        return edge

    def neighbors(self, unit_id: str, relations: set[str] | None = None, depth: int = 1) -> list[ContextUnit]:
        seen = {unit_id}
        frontier = [unit_id]
        out: list[ContextUnit] = []
        for _ in range(depth):
            next_frontier: list[str] = []
            for current in frontier:
                for edge in self._adjacency.get(current, []):
                    if relations and edge.relation not in relations:
                        continue
                    if edge.target_id in seen or edge.target_id not in self.units:
                        continue
                    seen.add(edge.target_id)
                    next_frontier.append(edge.target_id)
                    out.append(self.units[edge.target_id])
            frontier = next_frontier
        return out

    def conflicts(self, unit_id: str) -> set[str]:
        return {
            edge.target_id
            for edge in self._adjacency.get(unit_id, [])
            if edge.relation == "conflicts-with"
        }

    def build_from_repo(self, repo_path: str | Path, max_file_chars: int = 12000) -> None:
        populate_from_repo(self, repo_path, max_file_chars=max_file_chars)
        self.apply_conflict_detection()

    def add_task_memory(self, task: Task, link_gold_context: bool = False) -> ContextUnit | None:
        unit = ExperienceCurator().curate_task_memory(task)
        if unit is None:
            return None
        self.add_unit(unit)
        if link_gold_context:
            for gold_id in task.gold_context_ids:
                if gold_id in self.units:
                    self.add_edge(ContextEdge(unit.id, gold_id, "derived-from", 0.7))
        return unit

    def add_run_memory(self, run: AgentRun, task: Task) -> ContextUnit:
        unit = ExperienceCurator().curate_run(run, task)
        self.add_unit(unit)
        for context_id in run.selected_context_ids:
            if context_id in self.units:
                self.add_edge(ContextEdge(unit.id, context_id, "used", 1.0))
        return unit

    def apply_conflict_detection(self) -> None:
        ConflictDetector().apply(self)

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        write_jsonl(output / "context_units.jsonl", (u.to_dict() for u in self.units.values()))
        write_jsonl(output / "context_edges.jsonl", (e.to_dict() for e in self.edges))

    @classmethod
    def load(cls, input_dir: str | Path, instance_id: str) -> "InstanceContextGraph":
        graph = cls(instance_id)
        for row in read_jsonl(Path(input_dir) / "context_units.jsonl"):
            graph.add_unit(ContextUnit.from_dict(row))
        for row in read_jsonl(Path(input_dir) / "context_edges.jsonl"):
            graph.add_edge(ContextEdge.from_dict(row))
        return graph
