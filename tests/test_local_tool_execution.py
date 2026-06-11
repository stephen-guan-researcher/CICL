import tempfile
import unittest
from pathlib import Path

from cicl_agent.agents.local_tool import LocalContextInspectionAgent
from cicl_agent.core.schema import ContextUnit, Task
from cicl_agent.evaluation.tool_execution import evaluate_tool_run


class LocalToolExecutionTest(unittest.TestCase):
    def test_local_agent_reads_selected_context_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "parser.py").write_text("def parse_timestamp(value):\n    return value\n", encoding="utf-8")
            task = Task(
                id="task",
                instance_id="repo",
                instruction="fix parser timestamp handling",
                gold_context_ids=["file:parser.py"],
                metadata={"requires_context": True},
            )
            unit = ContextUnit(
                id="file:parser.py",
                instance_id="repo",
                type="file",
                source="parser.py",
                content="def parse_timestamp(value): return value",
            )

            run = LocalContextInspectionAgent(repo).run(task, "LocalToolTest", [unit])
            row = evaluate_tool_run(run, task, [unit], budget=40)

            self.assertTrue(run.success)
            self.assertEqual(row["gold_read"], 1)
            self.assertEqual(row["first_read_gold"], 1)
            self.assertEqual(row["target_prefix_read"], 0)

    def test_local_agent_reports_missing_path_without_gold_credit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            task = Task(
                id="task",
                instance_id="repo",
                instruction="fix parser timestamp handling",
                gold_context_ids=["file:parser.py"],
                metadata={"requires_context": True},
            )
            unit = ContextUnit(
                id="file:parser.py",
                instance_id="repo",
                type="file",
                source="parser.py",
                content="def parse_timestamp(value): return value",
            )

            run = LocalContextInspectionAgent(repo).run(task, "LocalToolTest", [unit])
            row = evaluate_tool_run(run, task, [unit], budget=40)

            self.assertFalse(run.success)
            self.assertEqual(row["gold_read"], 0)
            self.assertEqual(row["missing_count"], 1)


if __name__ == "__main__":
    unittest.main()

