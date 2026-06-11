"""LLM-guided counterfactual context judging.

The judge is a thin orchestrator: prompt templating, LLM call, JSON parsing,
and a deterministic simulator fallback. Pure data shapes and score adapters
live in [cicl_agent.causal.judgment][].
"""

from __future__ import annotations

import json
import re

from cicl_agent.causal.judgment import (
    CounterfactualContextJudgment,
    judgment_to_graph_artifact,
    judgment_to_score,
    self_confident,
)
from cicl_agent.core.schema import CausalScore, ContextUnit, Task
from cicl_agent.core.text import jaccard, tokenize
from cicl_agent.integrations.base import LLMClient

__all__ = [
    "CounterfactualContextJudgment",
    "CounterfactualPromptTemplate",
    "CodeRetrievalPromptTemplate",
    "LLMCausalContextJudge",
    "LLMClient",
    "judgment_to_graph_artifact",
    "judgment_to_score",
]


class CounterfactualPromptTemplate:
    def render(self, task: Task, unit: ContextUnit) -> str:
        expected = "\n".join(f"- {action}" for action in task.expected_actions) or "- unknown"
        return f"""You are a causal critic for an LLM tool agent.

Estimate whether the candidate context would causally change the agent's next
actions and final outcome.

Task:
{task.instruction}

Expected successful actions:
{expected}

Candidate context id:
{unit.id}

Candidate context source:
{unit.source}

Candidate context:
{unit.content}

Return only JSON with these keys:
{{
  "no_context_action": "likely first action without this context",
  "with_context_action": "likely first action with this context",
  "action_shift": 0.0,
  "necessity": 0.0,
  "expected_outcome_uplift": 0.0,
  "negative_transfer_risk": 0.0,
  "reason": "short explanation",
  "confidence": 0.0
}}

Keep every string field concise (20 words or fewer). Do not wrap the JSON in
markdown fences or add prose outside the JSON object.
"""


class CodeRetrievalPromptTemplate(CounterfactualPromptTemplate):
    """Prompt calibrated for repository retrieval reranking.

    SWE-bench-RR candidates are short code snippets rather than reusable memory
    cards. The calibration below keeps the same JSON schema, but tells the model
    to reward concrete edit/test evidence and lexical anchors instead of broad
    semantic relatedness.
    """

    def __init__(self, max_context_chars: int = 6000) -> None:
        self.max_context_chars = max_context_chars

    def render(self, task: Task, unit: ContextUnit) -> str:
        expected = "\n".join(f"- {action}" for action in task.expected_actions) or "- infer from the issue"
        context = self._clip(unit.content, self.max_context_chars)
        return f"""You are ranking repository context for a coding agent.

Question: if the agent sees ONLY this candidate snippet in addition to the issue,
will it make a better next inspection/edit/test decision?

Scoring rules:
- Reward exact file paths, functions, classes, arguments, tests, stack traces,
  API contracts, invariants, or nearby implementation that would change where
  the agent looks or edits.
- Give low scores to generic docs, duplicated issue text, unrelated files, or
  snippets that merely share broad project words.
- Penalize stale, conflicting, misleading, or legacy-only context.
- Be conservative: lexical anchors and executable code details matter in real
  bug-fix retrieval.

Calibrate numeric fields from 0.0 to 1.0:
- action_shift: 0.0 means no concrete next-step change; 1.0 means a clear new
  file/function/test/edit target.
- necessity: 0.0 means optional background; 1.0 means the fix is unlikely
  without this snippet.
- expected_outcome_uplift: 0.0 means no expected repair gain; 1.0 means it
  strongly increases the chance of solving the issue.
- negative_transfer_risk: 0.0 means safe; 1.0 means likely to mislead.
- confidence: confidence in your judgment, not in the candidate itself.

Issue:
{task.instruction}

Expected successful actions:
{expected}

Candidate context id:
{unit.id}

Candidate source/path:
{unit.source}

Candidate snippet:
{context}

Return only JSON with these keys:
{{
  "no_context_action": "likely first action without this snippet",
  "with_context_action": "likely first action with this snippet",
  "action_shift": 0.0,
  "necessity": 0.0,
  "expected_outcome_uplift": 0.0,
  "negative_transfer_risk": 0.0,
  "reason": "short explanation naming the strongest concrete evidence",
  "confidence": 0.0
}}

Keep every string field concise (20 words or fewer). Do not wrap the JSON in
markdown fences or add prose outside the JSON object.
"""

    @staticmethod
    def _clip(text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        head = max_chars // 2
        tail = max_chars - head
        return text[:head] + "\n...[truncated]...\n" + text[-tail:]


class LLMCausalContextJudge:
    """Counterfactual LLM judge with deterministic fallback.

    If a real `LLMClient` is passed, the judge parses its JSON response. If no
    client is available, a deterministic simulator produces the same schema so
    experiments stay reproducible.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        prompt_template: CounterfactualPromptTemplate | None = None,
    ) -> None:
        self.client = client
        self.prompt_template = prompt_template or CounterfactualPromptTemplate()

    def judge(self, task: Task, unit: ContextUnit) -> CounterfactualContextJudgment:
        if self.client is None:
            return self._simulate(task, unit)
        prompt = self.prompt_template.render(task, unit)
        response = self.client.complete(prompt)
        parsed = self._parse_json(response)
        parsed["task_id"] = task.id
        parsed["context_id"] = unit.id
        return CounterfactualContextJudgment.from_dict(parsed)

    def judge_many(self, task: Task, units: list[ContextUnit]) -> list[CounterfactualContextJudgment]:
        return [self.judge(task, unit) for unit in units]

    def score(self, task: Task, unit: ContextUnit) -> CausalScore:
        return judgment_to_score(self.judge(task, unit), unit)

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"LLM judge did not return JSON: {text[:200]}")
        return json.loads(match.group(0))

    @staticmethod
    def _simulate(task: Task, unit: ContextUnit) -> CounterfactualContextJudgment:
        context_text = f"{unit.content} {unit.source}"
        expected_text = " ".join(task.expected_actions)
        expected_terms = set(tokenize(expected_text))
        context_terms = set(tokenize(context_text))
        instruction_terms = set(tokenize(task.instruction))
        expected_overlap = len(expected_terms & context_terms) / max(1, len(expected_terms))
        instruction_overlap = len(instruction_terms & context_terms) / max(1, len(instruction_terms | context_terms))
        memory_bonus = 0.12 if unit.type in {"rule", "strategy", "trajectory"} else 0.0
        source_bonus = min(0.15, jaccard(unit.source, expected_text) + jaccard(unit.source, task.instruction))
        stale = bool(unit.metadata.get("stale") or unit.metadata.get("conflict_risk"))
        stale_text = any(term in context_text.lower() for term in ["legacy", "deprecated", "stale", "do not use"])
        negative_transfer = min(1.0, 0.65 * float(stale or stale_text) + 0.2 * float(unit.type == "failure"))
        action_shift = min(1.0, 0.62 * expected_overlap + 0.18 * instruction_overlap + memory_bonus + source_bonus)
        necessity = min(1.0, 0.72 * expected_overlap + 0.2 * instruction_overlap + memory_bonus)
        uplift = max(0.0, min(1.0, 0.68 * necessity + 0.24 * action_shift - 0.45 * negative_transfer))

        if task.expected_actions and expected_overlap >= 0.45:
            with_action = task.expected_actions[0]
        elif unit.source:
            with_action = f"inspect {unit.source}"
        else:
            with_action = "use candidate context to choose next tool"
        no_action = "broadly inspect the repository without a specific causal hint"
        if negative_transfer >= 0.5:
            reason = "The context appears stale or conflicting and may steer the agent toward a harmful action."
        elif uplift >= 0.5:
            reason = "The context points to action terms or files that can shift the agent toward a successful path."
        else:
            reason = "The context is only weakly policy-shaping beyond semantic relevance."
        return CounterfactualContextJudgment(
            task_id=task.id,
            context_id=unit.id,
            no_context_action=no_action,
            with_context_action=with_action,
            action_shift=round(action_shift, 6),
            necessity=round(necessity, 6),
            expected_outcome_uplift=round(uplift, 6),
            negative_transfer_risk=round(negative_transfer, 6),
            reason=reason,
            confidence=0.72 if self_confident(action_shift, uplift, negative_transfer) else 0.58,
        )
