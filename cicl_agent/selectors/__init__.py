"""Context selectors: baselines, CICL, learned, LLM-CICL, and the registry.

The package is the public surface — agents and runners depend only on this
namespace, not on the individual modules.
"""

from .base import ContextSelector, fit_budget
from .baselines import (
    AutoContextKGSelector,
    CICLRandomContextSelector,
    FullContextSelector,
    GraphMemorySelector,
    NoContextSelector,
    OracleGoldContextSelector,
    SelfGeneratedExamplesSelector,
    SummaryMemorySelector,
    VanillaRAGSelector,
)
from .cicl import CICLNoInstanceGraphSelector, CICLSelector
from .learned import LearnedCICLSelector
from .llm_cicl import LLMCICLSelector
from .registry import default_selectors

__all__ = [
    "AutoContextKGSelector",
    "CICLNoInstanceGraphSelector",
    "CICLRandomContextSelector",
    "CICLSelector",
    "ContextSelector",
    "FullContextSelector",
    "GraphMemorySelector",
    "LLMCICLSelector",
    "LearnedCICLSelector",
    "NoContextSelector",
    "OracleGoldContextSelector",
    "SelfGeneratedExamplesSelector",
    "SummaryMemorySelector",
    "VanillaRAGSelector",
    "default_selectors",
    "fit_budget",
]
