"""Conflict and staleness detection for context graphs.

This is a deterministic ActMem-style layer: it marks stale or risky context and
adds conflict edges when stale evidence overlaps with fresher evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cicl_agent.core.schema import ContextEdge, ContextUnit
from cicl_agent.core.text import tokenize


DEFAULT_STALE_TERMS = {
    "deprecated",
    "legacy",
    "obsolete",
    "stale",
    "superseded",
    "old behavior",
    "old path",
    "do not use",
    "no longer use",
    "remove this",
}


@dataclass(slots=True)
class ConflictDetector:
    stale_terms: set[str] = field(default_factory=lambda: set(DEFAULT_STALE_TERMS))
    min_overlap: float = 0.18
    max_edges_per_stale_unit: int = 8

    def apply(self, graph: object) -> None:
        """Mark risky units and add conflicts-with edges in-place."""

        units: list[ContextUnit] = list(getattr(graph, "units").values())
        for unit in units:
            if self.is_stale(unit):
                unit.metadata["stale"] = True
                unit.metadata["conflict_risk"] = True
                unit.confidence = min(unit.confidence, 0.45)

        stale_units = [unit for unit in units if unit.metadata.get("stale")]
        fresh_units = [unit for unit in units if not unit.metadata.get("stale")]
        existing = {
            (edge.source_id, edge.target_id, edge.relation)
            for edge in getattr(graph, "edges")
        }
        for stale_unit in stale_units:
            ranked: list[tuple[float, ContextUnit]] = []
            stale_tokens = self._signal_tokens(stale_unit)
            if not stale_tokens:
                continue
            for fresh_unit in fresh_units:
                if fresh_unit.instance_id != stale_unit.instance_id:
                    continue
                overlap = self._overlap(stale_tokens, self._signal_tokens(fresh_unit))
                if overlap >= self.min_overlap:
                    ranked.append((overlap, fresh_unit))
            ranked.sort(key=lambda row: row[0], reverse=True)
            for overlap, fresh_unit in ranked[: self.max_edges_per_stale_unit]:
                key = (stale_unit.id, fresh_unit.id, "conflicts-with")
                reverse_key = (fresh_unit.id, stale_unit.id, "conflicts-with")
                if key in existing or reverse_key in existing:
                    continue
                getattr(graph, "add_edge")(
                    ContextEdge(
                        stale_unit.id,
                        fresh_unit.id,
                        "conflicts-with",
                        weight=round(min(1.0, overlap + 0.35), 6),
                        metadata={"detector": "lexical_staleness", "overlap": round(overlap, 6)},
                    )
                )
                existing.add(key)

    def is_stale(self, unit: ContextUnit) -> bool:
        haystack = f"{unit.content}\n{unit.source}\n{unit.metadata}".lower()
        return any(term in haystack for term in self.stale_terms)

    @staticmethod
    def _signal_tokens(unit: ContextUnit) -> set[str]:
        tokens = set(tokenize(f"{unit.content} {unit.source}"))
        return {
            token
            for token in tokens
            if len(token) > 2
            and token
            not in {
                "the",
                "and",
                "for",
                "with",
                "from",
                "this",
                "that",
                "task",
                "context",
                "use",
                "uses",
                "used",
            }
        }

    @staticmethod
    def _overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / min(len(left), len(right))
