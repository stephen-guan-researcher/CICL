import tempfile
import unittest
from pathlib import Path

from cicl_agent.core.schema import ContextUnit, Task
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.runners.harmful_context_stress import (
    harmful_metric_row,
    inject_harmful_context,
    run_harmful_context_stress,
)
from cicl_agent.agents.simulated import SimulatedReActAgent


class HarmfulContextStressTest(unittest.TestCase):
    def test_injected_harmful_context_is_stale(self):
        graph = InstanceContextGraph("toy")
        task = Task(
            id="eval_parse_z_timestamp",
            instance_id="toy",
            instruction="Fix timestamp parsing.",
            expected_actions=["inspect parser.py"],
        )

        injected = inject_harmful_context(graph, [task])

        self.assertEqual(1, len(injected))
        self.assertTrue(injected[0].metadata["harmful_context"])
        self.assertTrue(injected[0].metadata["stale"])
        self.assertIn("obsolete", injected[0].content.lower())

    def test_harmful_metric_marks_first_harmful(self):
        task = Task(
            id="eval_parse_z_timestamp",
            instance_id="toy",
            instruction="Fix timestamp parsing.",
            gold_context_ids=["file:parser.py"],
            expected_actions=["inspect parser.py"],
        )
        harmful = ContextUnit(
            id="harmful:eval_parse_z_timestamp",
            instance_id="toy",
            type="failure",
            content="deprecated parser timestamp note",
            source="legacy.md",
            metadata={"harmful_context": True, "stale": True},
        )
        good = ContextUnit(
            id="file:parser.py",
            instance_id="toy",
            type="file",
            content="inspect parser.py",
            source="parser.py",
        )
        run = SimulatedReActAgent().run(task, "VanillaRAG", [harmful, good])

        row = harmful_metric_row(task, run, [harmful, good])

        self.assertEqual(1, row["harmful_selected"])
        self.assertEqual(1, row["first_harmful"])
        self.assertEqual(0, row["stress_success"])

    def test_runner_writes_outputs(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_harmful_context_stress(
                repo=root / "experiments" / "examples" / "toy_repo",
                tasks_path=root / "experiments" / "examples" / "toy_tasks.jsonl",
                output=Path(tmpdir) / "stress",
                budget=80,
            )

            methods = {row["method"] for row in result["summary"]}
            self.assertIn("CICL", methods)
            self.assertIn("VanillaRAG", methods)
            self.assertTrue((Path(tmpdir) / "stress" / "harmful_context_summary.csv").exists())
            self.assertTrue((Path(tmpdir) / "stress" / "harmful_context_metrics.csv").exists())
            self.assertTrue((Path(tmpdir) / "stress" / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
