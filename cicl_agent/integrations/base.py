"""Provider-agnostic adapters for external LLM clients."""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Minimal contract any LLM provider client must satisfy."""

    def complete(self, prompt: str) -> str:
        ...


__all__ = ["LLMClient"]
