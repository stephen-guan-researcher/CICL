import unittest

from cicl_agent.core.schema import AgentRun, ContextUnit, Task
from cicl_agent.evaluation.replay import evaluate_run_replay


class ReplayEvaluationTest(unittest.TestCase):
    def test_gold_context_selected_but_not_actioned_is_counted_separately(self):
        task = Task(
            id="task",
            instance_id="repo",
            instruction="fix parser timestamp handling",
            gold_context_ids=["file:parser.py"],
            metadata={"requires_context": True},
        )
        selected_units = [
            ContextUnit(
                id="file:parser.py",
                instance_id="repo",
                type="file",
                source="parser.py",
                content="def parse_timestamp(value): ...",
            ),
            ContextUnit(
                id="file:client.py",
                instance_id="repo",
                type="file",
                source="client.py",
                content="class ApiClient: ...",
            ),
        ]
        run = AgentRun(
            task_id="task",
            method="ReplayTest",
            selected_context_ids=["file:parser.py", "file:client.py"],
            actions=["inspect client.py", "attempt fix"],
            observations=[],
            success=True,
            tokens=10,
            runtime_seconds=0.0,
            tool_calls=1,
        )

        row = evaluate_run_replay(run, task, selected_units, budget=40)

        self.assertEqual(row["selected_gold"], 1)
        self.assertEqual(row["gold_action_supported"], 0)
        self.assertEqual(row["selected_but_unused_gold"], 1)
        self.assertEqual(row["replay_success"], 0)

    def test_memory_card_actionability_is_measured(self):
        task = Task(
            id="task",
            instance_id="repo",
            instruction="fix parser timestamp handling",
            gold_context_ids=["file:parser.py"],
            metadata={"requires_context": True},
        )
        selected_units = [
            ContextUnit(
                id="file:parser.py",
                instance_id="repo",
                type="memory_card",
                source="compressed:parser.py",
                content="Evidence: parser.py handles Z\nAction hint: inspect parser.py\nScope: parser.py",
            )
        ]
        run = AgentRun(
            task_id="task",
            method="CausalCompressedCICL",
            selected_context_ids=["file:parser.py"],
            actions=["inspect compressed:parser.py", "attempt fix"],
            observations=[],
            success=True,
            tokens=10,
            runtime_seconds=0.0,
            tool_calls=1,
        )

        row = evaluate_run_replay(run, task, selected_units, budget=40)

        self.assertEqual(row["gold_action_supported"], 1)
        self.assertEqual(row["replay_success"], 1)
        self.assertEqual(row["card_actionability"], 1.0)
        self.assertEqual(row["evidence_marker_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

