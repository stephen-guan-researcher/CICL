"""Default selector registry: one place to compose the comparison set."""

from __future__ import annotations

from cicl_agent.causal.llm_judge import LLMCausalContextJudge
from cicl_agent.causal.scoring import ScoringWeights
from cicl_agent.learning.ranker import PairwiseContextRanker
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.selectors.base import ContextSelector
from cicl_agent.selectors.baselines import (
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
from cicl_agent.selectors.cicl import CICLNoInstanceGraphSelector, CICLSelector
from cicl_agent.selectors.learned import LearnedCICLSelector
from cicl_agent.selectors.llm_cicl import LLMCICLSelector


def default_selectors(
    graph: InstanceContextGraph,
    include_ablations: bool = False,
    learned_ranker: PairwiseContextRanker | None = None,
    include_llm_judge: bool = False,
    llm_judge: LLMCausalContextJudge | None = None,
) -> list[ContextSelector]:
    selectors: list[ContextSelector] = [
        NoContextSelector(graph),
        FullContextSelector(graph),
        VanillaRAGSelector(graph),
        SummaryMemorySelector(graph),
        GraphMemorySelector(graph),
        AutoContextKGSelector(graph),
        SelfGeneratedExamplesSelector(graph),
        CICLSelector(graph),
        OracleGoldContextSelector(graph),
    ]
    if include_llm_judge:
        selectors.insert(-1, LLMCICLSelector(graph, judge=llm_judge or LLMCausalContextJudge()))
    if learned_ranker is not None:
        learned_name = "CICL_Distilled" if learned_ranker.metadata.get("label_source") == "llm" else "LearnedCICL"
        selectors.insert(-1, LearnedCICLSelector(graph, learned_ranker, name=learned_name))
    if include_ablations:
        selectors.extend(
            [
                CICLSelector(
                    graph,
                    name="CICL_w/o_causal_scoring",
                    weights=ScoringWeights(action_delta=0.0, outcome_uplift=0.0, necessity=1.0, cost=0.0),
                ),
                CICLNoInstanceGraphSelector(graph),
                CICLSelector(
                    graph,
                    name="CICL_w/o_action_intervention",
                    weights=ScoringWeights(action_delta=0.0, outcome_uplift=0.55, necessity=0.35, cost=0.10),
                ),
                CICLSelector(
                    graph,
                    name="CICL_w/o_outcome_uplift",
                    weights=ScoringWeights(action_delta=0.55, outcome_uplift=0.0, necessity=0.35, cost=0.10),
                ),
                CICLSelector(graph, name="CICL_w/o_conflict_filtering", conflict_filtering=False),
                CICLRandomContextSelector(graph),
            ]
        )
    return selectors


__all__ = ["default_selectors"]
