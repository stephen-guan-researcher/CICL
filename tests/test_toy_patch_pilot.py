import tempfile
import unittest
from pathlib import Path

from cicl_agent.agents.toy_patch import ToyPatchAgent
from cicl_agent.core.schema import ContextUnit, Task
from cicl_agent.runners.toy_patch_pilot import run_toy_patch_pilot


class ToyPatchPilotTest(unittest.TestCase):
    def test_toy_patch_agent_requires_relevant_context(self):
        root = Path(__file__).resolve().parents[1]
        task = Task(
            id="eval_parse_z_timestamp",
            instance_id="toy_repo",
            instruction="Fix timestamp strings ending with Z.",
            gold_context_ids=["file:parser.py"],
            metadata={"requires_context": True},
        )
        agent = ToyPatchAgent(root / "experiments" / "examples" / "toy_repo")

        no_context = agent.run(task, "NoContext", [])
        self.assertFalse(no_context.success)
        self.assertFalse(no_context.metadata["patch_applied"])

        parser_context = ContextUnit(
            id="file:parser.py",
            instance_id="toy_repo",
            type="file",
            source="parser.py",
            content="parse_iso8601_timestamp handles trailing Z with timezone.utc",
        )
        with_context = agent.run(task, "OracleGoldContext", [parser_context])
        self.assertTrue(with_context.success)
        self.assertTrue(with_context.metadata["patch_applied"])

    def test_toy_patch_pilot_writes_summary_and_report(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_toy_patch_pilot(
                repo=root / "experiments" / "examples" / "toy_repo",
                tasks_path=root / "experiments" / "examples" / "toy_tasks.jsonl",
                output=Path(tmpdir) / "toy_patch",
                budget=80,
            )

            methods = {row["method"] for row in result["summary"]}
            self.assertIn("CICL", methods)
            self.assertIn("OracleGoldContext", methods)
            self.assertTrue((Path(tmpdir) / "toy_patch" / "toy_patch_summary.csv").exists())
            self.assertTrue((Path(tmpdir) / "toy_patch" / "toy_patch_metrics.csv").exists())
            self.assertTrue((Path(tmpdir) / "toy_patch" / "toy_patch_runs.jsonl").exists())
            self.assertTrue((Path(tmpdir) / "toy_patch" / "report.md").exists())
            report = (Path(tmpdir) / "toy_patch" / "report.md").read_text(encoding="utf-8")
            self.assertIn("mutate parser.py \\| inspect parser.py", report)


if __name__ == "__main__":
    unittest.main()
