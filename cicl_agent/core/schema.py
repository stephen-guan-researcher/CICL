"""Shared data schemas for CICL experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_tokens(text: str) -> int:
    """A deterministic token proxy used for budget accounting."""
    if not text:
        return 0
    return max(1, len(text.split()))


@dataclass(slots=True)
class ContextUnit:
    id: str
    instance_id: str
    type: str
    content: str
    source: str
    created_at: str = field(default_factory=utc_now)
    token_cost: int = 0
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_cost <= 0:
            self.token_cost = estimate_tokens(self.content)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextUnit":
        return cls(**data)


@dataclass(slots=True)
class ContextEdge:
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextEdge":
        return cls(**data)


@dataclass(slots=True)
class CausalScore:
    context_id: str
    task_id: str
    action_delta: float
    outcome_uplift: float
    necessity_score: float
    cost_penalty: float
    final_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CausalScore":
        return cls(**data)


@dataclass(slots=True)
class AgentRun:
    task_id: str
    method: str
    selected_context_ids: list[str]
    actions: list[str]
    observations: list[str]
    success: bool
    tokens: int
    runtime_seconds: float
    tool_calls: int
    candidate_context_ids: list[str] = field(default_factory=list)
    instance_id: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentRun":
        return cls(**data)


@dataclass(slots=True)
class Task:
    id: str
    instance_id: str
    instruction: str
    split: str = "eval"
    gold_context_ids: list[str] = field(default_factory=list)
    expected_actions: list[str] = field(default_factory=list)
    memory: str = ""
    difficulty: str = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(**data)

