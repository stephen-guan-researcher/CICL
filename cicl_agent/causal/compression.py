"""Causal memory-card compression for agent context.

This module turns raw context units into short action-oriented memory cards.
The compressor can call the same LLM client used by LLM-CICL; without a client
it uses the deterministic counterfactual judge so local experiments remain
reproducible.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from cicl_agent.causal.judgment import CounterfactualContextJudgment, judgment_to_score
from cicl_agent.causal.llm_judge import LLMCausalContextJudge
from cicl_agent.core.schema import ContextUnit, Task, estimate_tokens
from cicl_agent.core.text import tokenize


@dataclass(slots=True)
class CausalMemoryCard:
    task_id: str
    context_id: str
    trigger: str
    evidence: str
    action_hint: str
    failure_if_ignored: str
    scope: str
    causal_score: float
    confidence: float
    original_token_cost: int
    compressed_token_cost: int = 0

    def __post_init__(self) -> None:
        if self.compressed_token_cost <= 0:
            self.compressed_token_cost = estimate_tokens(self.to_content())

    def to_content(self) -> str:
        return "\n".join(
            [
                f"Trigger: {self.trigger}",
                f"Evidence: {self.evidence}",
                f"Action hint: {self.action_hint}",
                f"Failure if ignored: {self.failure_if_ignored}",
                f"Scope: {self.scope}",
                f"Causal score: {self.causal_score:.3f}",
            ]
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict) -> "CausalMemoryCard":
        return cls(
            task_id=str(row["task_id"]),
            context_id=str(row["context_id"]),
            trigger=str(row.get("trigger", "")),
            evidence=str(row.get("evidence", "")),
            action_hint=str(row.get("action_hint", "")),
            failure_if_ignored=str(row.get("failure_if_ignored", "")),
            scope=str(row.get("scope", "")),
            causal_score=float(row.get("causal_score", 0.0)),
            confidence=float(row.get("confidence", 0.7)),
            original_token_cost=int(row.get("original_token_cost", 0)),
            compressed_token_cost=int(row.get("compressed_token_cost", 0)),
        )


class CausalMemoryCompressor:
    """Compress raw context into action-useful memory cards."""

    def __init__(self, judge: LLMCausalContextJudge | None = None, max_field_words: int = 28) -> None:
        self.judge = judge or LLMCausalContextJudge()
        self.max_field_words = max_field_words

    def compress(
        self,
        task: Task,
        unit: ContextUnit,
        judgment: CounterfactualContextJudgment | None = None,
    ) -> tuple[ContextUnit, CausalMemoryCard, CounterfactualContextJudgment]:
        judgment = judgment or self.judge.judge(task, unit)
        if self.judge.client is None:
            card = self._simulate_card(task, unit, judgment)
        else:
            card = self._llm_card(task, unit, judgment)
        compressed = self.to_context_unit(unit, card)
        return compressed, card, judgment

    def compress_many(
        self,
        task: Task,
        units: list[ContextUnit],
        judgments: dict[str, CounterfactualContextJudgment] | None = None,
    ) -> list[tuple[ContextUnit, CausalMemoryCard, CounterfactualContextJudgment]]:
        judgments = judgments or {}
        return [self.compress(task, unit, judgments.get(unit.id)) for unit in units]

    @staticmethod
    def to_context_unit(unit: ContextUnit, card: CausalMemoryCard) -> ContextUnit:
        content = card.to_content()
        compressed_tokens = estimate_tokens(content)
        if compressed_tokens >= unit.token_cost:
            content = "\n".join(
                [
                    f"Action hint: {card.action_hint}",
                    f"Evidence: {card.evidence}",
                    f"Scope: {card.scope}",
                ]
            )
            compressed_tokens = estimate_tokens(content)
        if compressed_tokens >= unit.token_cost:
            content = "\n".join(
                [
                    f"Action hint: {card.action_hint}",
                    f"Scope: {card.scope}",
                ]
            )
            compressed_tokens = estimate_tokens(content)
        card.compressed_token_cost = compressed_tokens
        return ContextUnit(
            id=unit.id,
            instance_id=unit.instance_id,
            type="memory_card",
            content=content,
            source=f"compressed:{unit.source}",
            token_cost=compressed_tokens,
            confidence=card.confidence,
            metadata={
                "compressor": "causal_memory_card",
                "original_context_id": unit.id,
                "original_type": unit.type,
                "original_source": unit.source,
                "original_token_cost": unit.token_cost,
                "compressed_token_cost": compressed_tokens,
                "compression_ratio": round(compressed_tokens / max(1, unit.token_cost), 6),
                "causal_score": card.causal_score,
            },
        )

    def _llm_card(
        self,
        task: Task,
        unit: ContextUnit,
        judgment: CounterfactualContextJudgment,
    ) -> CausalMemoryCard:
        prompt = self._render_prompt(task, unit, judgment)
        response = self.judge.client.complete(prompt)  # type: ignore[union-attr]
        parsed = self._parse_json(response)
        return self._card_from_fields(task, unit, judgment, parsed)

    def _simulate_card(
        self,
        task: Task,
        unit: ContextUnit,
        judgment: CounterfactualContextJudgment,
    ) -> CausalMemoryCard:
        score = judgment_to_score(judgment, unit).final_score
        relevant_line = self._most_relevant_line(unit.content, task.instruction + " " + judgment.with_context_action)
        trigger = self._trim(task.instruction)
        evidence = self._trim(f"{unit.source}: {relevant_line or judgment.reason}")
        action_hint = self._trim(judgment.with_context_action or f"inspect {unit.source}")
        if judgment.negative_transfer_risk >= 0.5:
            failure = "This context may be stale; following it can cause negative transfer."
        elif judgment.action_shift >= 0.45 or judgment.expected_outcome_uplift >= 0.45:
            failure = "Without this hint, the agent may inspect broad or wrong context first."
        else:
            failure = "If ignored, the agent loses a weak but potentially useful local clue."
        return CausalMemoryCard(
            task_id=task.id,
            context_id=unit.id,
            trigger=trigger,
            evidence=evidence,
            action_hint=action_hint,
            failure_if_ignored=self._trim(failure),
            scope=self._trim(unit.source or unit.id, words=18),
            causal_score=round(score, 6),
            confidence=judgment.confidence,
            original_token_cost=unit.token_cost,
        )

    def _card_from_fields(
        self,
        task: Task,
        unit: ContextUnit,
        judgment: CounterfactualContextJudgment,
        fields: dict,
    ) -> CausalMemoryCard:
        score = judgment_to_score(judgment, unit).final_score
        return CausalMemoryCard(
            task_id=task.id,
            context_id=unit.id,
            trigger=self._trim(str(fields.get("trigger") or task.instruction)),
            evidence=self._trim(str(fields.get("evidence") or judgment.reason)),
            action_hint=self._trim(str(fields.get("action_hint") or judgment.with_context_action)),
            failure_if_ignored=self._trim(str(fields.get("failure_if_ignored") or "Agent may choose a weaker next action.")),
            scope=self._trim(str(fields.get("scope") or unit.source or unit.id), words=18),
            causal_score=round(float(fields.get("causal_score", score)), 6),
            confidence=round(float(fields.get("confidence", judgment.confidence)), 6),
            original_token_cost=unit.token_cost,
        )

    def _render_prompt(
        self,
        task: Task,
        unit: ContextUnit,
        judgment: CounterfactualContextJudgment,
    ) -> str:
        return f"""You compress context for an LLM coding/tool agent.

Goal: keep only information that can change the agent's next action or avoid a
known failure. Do not solve the task. Do not include long code. Return JSON only.

Task:
{task.instruction}

Candidate context id: {unit.id}
Candidate source: {unit.source}
Candidate context:
{unit.content}

Counterfactual judgment:
- no_context_action: {judgment.no_context_action}
- with_context_action: {judgment.with_context_action}
- action_shift: {judgment.action_shift}
- necessity: {judgment.necessity}
- expected_outcome_uplift: {judgment.expected_outcome_uplift}
- negative_transfer_risk: {judgment.negative_transfer_risk}
- reason: {judgment.reason}

Return JSON with exactly these keys:
{{
  "trigger": "when this memory card should be used",
  "evidence": "short evidence from the context",
  "action_hint": "specific next action the agent should prefer",
  "failure_if_ignored": "what likely goes wrong without it",
  "scope": "file/function/API/environment scope",
  "causal_score": 0.0,
  "confidence": 0.0
}}
"""

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"LLM compressor did not return JSON: {text[:200]}")
        return json.loads(match.group(0))

    def _trim(self, text: str, words: int | None = None) -> str:
        words = words or self.max_field_words
        tokens = text.replace("\n", " ").split()
        if len(tokens) <= words:
            return " ".join(tokens)
        return " ".join(tokens[:words]).rstrip(" ,.;:") + "..."

    @staticmethod
    def _most_relevant_line(text: str, query: str) -> str:
        query_terms = set(tokenize(query))
        best_line = ""
        best_score = -1
        for line in text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            terms = set(tokenize(clean))
            score = len(query_terms & terms)
            if score > best_score:
                best_line = clean
                best_score = score
        return best_line


class ExtractiveContextCompressor:
    """Non-causal compression baseline using relevant source lines only."""

    def __init__(self, max_lines: int = 3, max_words: int = 42) -> None:
        self.max_lines = max_lines
        self.max_words = max_words

    def compress(self, task: Task, unit: ContextUnit) -> ContextUnit:
        query = task.instruction + " " + " ".join(task.expected_actions)
        lines = self._top_lines(unit.content, query)
        content = "\n".join(
            [
                f"Summary: {self._trim(' '.join(lines))}",
                f"Scope: {unit.source or unit.id}",
            ]
        )
        compressed_tokens = estimate_tokens(content)
        if compressed_tokens >= unit.token_cost:
            content = "\n".join(
                [
                    f"Summary: {self._trim(' '.join(lines), words=max(12, self.max_words // 2))}",
                    f"Scope: {unit.source or unit.id}",
                ]
            )
            compressed_tokens = estimate_tokens(content)
        return ContextUnit(
            id=unit.id,
            instance_id=unit.instance_id,
            type="summary_card",
            content=content,
            source=f"summary:{unit.source}",
            token_cost=compressed_tokens,
            confidence=unit.confidence,
            metadata={
                "compressor": "extractive_summary",
                "original_context_id": unit.id,
                "original_type": unit.type,
                "original_source": unit.source,
                "original_token_cost": unit.token_cost,
                "compressed_token_cost": compressed_tokens,
                "compression_ratio": round(compressed_tokens / max(1, unit.token_cost), 6),
            },
        )

    def compress_many(self, task: Task, units: list[ContextUnit]) -> list[ContextUnit]:
        return [self.compress(task, unit) for unit in units]

    def _top_lines(self, text: str, query: str) -> list[str]:
        query_terms = set(tokenize(query))
        scored: list[tuple[int, int, str]] = []
        for idx, line in enumerate(text.splitlines()):
            clean = line.strip()
            if not clean:
                continue
            terms = set(tokenize(clean))
            score = len(query_terms & terms)
            scored.append((score, -idx, clean))
        scored.sort(reverse=True)
        lines = [line for _, _, line in scored[: self.max_lines]]
        return lines or [text.strip().splitlines()[0] if text.strip() else ""]

    def _trim(self, text: str, words: int | None = None) -> str:
        limit = words or self.max_words
        tokens = text.replace("\n", " ").split()
        if len(tokens) <= limit:
            return " ".join(tokens)
        return " ".join(tokens[:limit]).rstrip(" ,.;:") + "..."


__all__ = ["CausalMemoryCard", "CausalMemoryCompressor", "ExtractiveContextCompressor"]
