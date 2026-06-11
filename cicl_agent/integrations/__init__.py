"""External model/provider adapters.

Submodules import on demand to avoid pulling the (causal -> llm_judge)
dependency graph through the integrations namespace.
"""

from .base import LLMClient

__all__ = ["LLMClient"]
