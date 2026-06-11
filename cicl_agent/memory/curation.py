"""Experience curation utilities inspired by ACE-style playbooks.

The curator converts raw task memories and trajectories into compact,
structured context units. This gives CICL a reusable playbook layer instead of
only storing raw histories.
"""

from __future__ import annotations

from dataclasses import dataclass

from cicl_agent.core.schema import AgentRun, ContextUnit, Task


@dataclass(slots=True)
class ExperienceCurator:
    """Build compact reusable context units from task and run evidence."""

    confidence_success: float = 0.82
    confidence_failure: float = 0.52
    confidence_task_memory: float = 0.9

    def curate_task_memory(self, task: Task) -> ContextUnit | None:
        if not task.memory:
            return None
        expected = "; ".join(task.expected_actions) if task.expected_actions else "infer from task"
        domain = task.metadata.get("domain") or task.metadata.get("repo") or task.instance_id
        content = "\n".join(
            [
                f"Task pattern: {task.instruction}",
                f"Reusable rule: {task.memory}",
                f"Expected action hints: {expected}",
                f"Domain: {domain}",
            ]
        )
        return ContextUnit(
            id=f"memory:{task.id}",
            instance_id=task.instance_id,
            type="rule",
            content=content,
            source=f"task:{task.id}",
            confidence=self.confidence_task_memory,
            metadata={
                "task_id": task.id,
                "split": task.split,
                "curator": "ace_style",
                "domain": domain,
                "expected_actions": list(task.expected_actions),
            },
        )

    def curate_run(self, run: AgentRun, task: Task) -> ContextUnit:
        outcome = "success" if run.success else "failure"
        useful = ", ".join(run.selected_context_ids) if run.selected_context_ids else "none"
        actions = " | ".join(run.actions) if run.actions else "none"
        observations = " | ".join(run.observations) if run.observations else "none"
        lesson = self._lesson_from_run(run, task)
        unit_type = "trajectory" if run.success else "failure"
        return ContextUnit(
            id=f"trajectory:{run.method}:{task.id}",
            instance_id=task.instance_id,
            type=unit_type,
            content="\n".join(
                [
                    f"Task pattern: {task.instruction}",
                    f"Outcome: {outcome}",
                    f"Selected context: {useful}",
                    f"Actions: {actions}",
                    f"Observations: {observations}",
                    f"Reusable lesson: {lesson}",
                ]
            ),
            source=f"run:{run.method}:{task.id}",
            confidence=self.confidence_success if run.success else self.confidence_failure,
            metadata={
                "task_id": task.id,
                "method": run.method,
                "success": run.success,
                "curator": "ace_style",
                "selected_context_ids": list(run.selected_context_ids),
            },
        )

    @staticmethod
    def _lesson_from_run(run: AgentRun, task: Task) -> str:
        if run.success:
            if run.selected_context_ids:
                return (
                    "For similar tasks, prioritize these context units before broad exploration: "
                    + ", ".join(run.selected_context_ids)
                )
            return "Task appears solvable from the instruction without extra instance context."
        if run.selected_context_ids:
            return (
                "Previous attempt failed with these context units; treat them as weak evidence "
                "unless a stronger rule or file context supports them: "
                + ", ".join(run.selected_context_ids)
            )
        return "Prompt-only attempt failed; retrieve instance rules or file-level context first."
