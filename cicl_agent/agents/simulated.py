"""Agent adapters.

The default simulated agent gives deterministic end-to-end tests. Replace it
with a real ReAct/coding agent adapter for paper experiments.
"""

from __future__ import annotations

import time

from cicl_agent.core.schema import AgentRun, ContextUnit, Task
from cicl_agent.core.text import tokenize


class SimulatedReActAgent:
    def run(self, task: Task, method: str, selected_units: list[ContextUnit]) -> AgentRun:
        start = time.perf_counter()
        selected_ids = [unit.id for unit in selected_units]
        selected_text = "\n".join(unit.content for unit in selected_units)
        gold_hit = bool(set(selected_ids) & set(task.gold_context_ids))
        action_hit = self._has_expected_action(selected_text, task.expected_actions)
        easy_without_context = task.metadata.get("requires_context") is False
        success = bool(gold_hit or action_hit or easy_without_context)
        actions = self._actions(task, selected_units, success)
        observations = [
            "selected_gold_context" if gold_hit else "no_gold_context_selected",
            "action_terms_supported" if action_hit else "action_terms_missing",
        ]
        runtime = time.perf_counter() - start
        return AgentRun(
            task_id=task.id,
            method=method,
            selected_context_ids=selected_ids,
            candidate_context_ids=selected_ids,
            actions=actions,
            observations=observations,
            success=success,
            tokens=sum(unit.token_cost for unit in selected_units),
            runtime_seconds=runtime,
            tool_calls=max(1, min(6, len(selected_units))),
            instance_id=task.instance_id,
        )

    @staticmethod
    def _has_expected_action(selected_text: str, expected_actions: list[str]) -> bool:
        if not expected_actions:
            return False
        selected_terms = set(tokenize(selected_text))
        for action in expected_actions:
            action_terms = set(tokenize(action))
            if action_terms and len(action_terms & selected_terms) / len(action_terms) >= 0.5:
                return True
        return False

    @staticmethod
    def _actions(task: Task, selected_units: list[ContextUnit], success: bool) -> list[str]:
        if success and task.expected_actions:
            return task.expected_actions
        if selected_units:
            return [f"inspect {selected_units[0].source}", "attempt fix"]
        return ["inspect task", "attempt from prompt only"]

