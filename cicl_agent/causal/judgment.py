"""Counterfactual judgment data + score/graph adapters.

The data dataclass and the score/graph projections live here so that downstream
modules (learning, selectors, runners) can consume judgments without depending
on the LLM client glue in [cicl_agent.causal.llm_judge][].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cicl_agent.core.schema import CausalScore, ContextEdge, ContextUnit, Task


@dataclass(slots=True)
class CounterfactualContextJudgment:
    task_id: str
    context_id: str
    no_context_action: str
    with_context_action: str
    action_shift: float
    necessity: float
    expected_outcome_uplift: float
    negative_transfer_risk: float
    reason: str
    confidence: float = 0.7

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict) -> "CounterfactualContextJudgment":
        return cls(
            task_id=row["task_id"],
            context_id=row["context_id"],
            no_context_action=row.get("no_context_action", ""),
            with_context_action=row.get("with_context_action", ""),
            action_shift=float(row.get("action_shift", 0.0)),
            necessity=float(row.get("necessity", 0.0)),
            expected_outcome_uplift=float(row.get("expected_outcome_uplift", 0.0)),
            negative_transfer_risk=float(row.get("negative_transfer_risk", 0.0)),
            reason=row.get("reason", ""),
            confidence=float(row.get("confidence", 0.7)),
        )


def self_confident(action_shift: float, uplift: float, negative_transfer: float) -> bool:
    return max(action_shift, uplift, negative_transfer) >= 0.55


def judgment_to_score(judgment: CounterfactualContextJudgment, unit: ContextUnit) -> CausalScore:
    cost_penalty = min(1.0, unit.token_cost / 1000 + 0.2 * judgment.negative_transfer_risk)
    final = (
        0.34 * judgment.action_shift
        + 0.26 * judgment.necessity
        + 0.28 * judgment.expected_outcome_uplift
        - 0.22 * judgment.negative_transfer_risk
        - 0.08 * cost_penalty
    )
    return CausalScore(
        context_id=judgment.context_id,
        task_id=judgment.task_id,
        action_delta=round(judgment.action_shift, 6),
        outcome_uplift=round(judgment.expected_outcome_uplift, 6),
        necessity_score=round(judgment.necessity, 6),
        cost_penalty=round(cost_penalty, 6),
        final_score=round(final, 6),
        metadata={
            "selector": "LLM-CICL",
            "negative_transfer_risk": judgment.negative_transfer_risk,
            "llm_confidence": judgment.confidence,
            "reason": judgment.reason,
            "no_context_action": judgment.no_context_action,
            "with_context_action": judgment.with_context_action,
        },
    )


def judgment_to_graph_artifact(
    task: Task,
    unit: ContextUnit,
    judgment: CounterfactualContextJudgment,
) -> tuple[ContextUnit, ContextEdge]:
    relation = "weakly-affects"
    if judgment.negative_transfer_risk >= 0.5:
        relation = "causes-negative-transfer"
    elif judgment.expected_outcome_uplift >= 0.5 or judgment.action_shift >= 0.5:
        relation = "enables-action"
    judgment_id = f"policy_judgment:{task.id}:{unit.id}"
    content = "\n".join(
        [
            f"Task: {task.instruction}",
            f"Context: {unit.id}",
            f"No-context action: {judgment.no_context_action}",
            f"With-context action: {judgment.with_context_action}",
            f"Action shift: {judgment.action_shift}",
            f"Expected uplift: {judgment.expected_outcome_uplift}",
            f"Negative-transfer risk: {judgment.negative_transfer_risk}",
            f"Reason: {judgment.reason}",
        ]
    )
    artifact = ContextUnit(
        id=judgment_id,
        instance_id=unit.instance_id,
        type="policy_judgment",
        content=content,
        source=f"llm_judge:{task.id}:{unit.id}",
        confidence=judgment.confidence,
        metadata=judgment.to_dict(),
    )
    edge = ContextEdge(
        unit.id,
        judgment_id,
        relation,
        weight=round(max(judgment.action_shift, judgment.expected_outcome_uplift, judgment.negative_transfer_risk), 6),
        metadata={"task_id": task.id, "judge": "llm_counterfactual"},
    )
    return artifact, edge
