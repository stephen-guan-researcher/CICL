"""Core data contracts, JSONL I/O, structural protocols, and text helpers."""

from .io import append_jsonl, read_jsonl, write_jsonl
from .protocols import AgentAdapter
from .schema import AgentRun, CausalScore, ContextEdge, ContextUnit, Task
from .text import jaccard, tokenize

__all__ = [
    "AgentAdapter",
    "AgentRun",
    "CausalScore",
    "ContextEdge",
    "ContextUnit",
    "Task",
    "append_jsonl",
    "jaccard",
    "read_jsonl",
    "tokenize",
    "write_jsonl",
]
