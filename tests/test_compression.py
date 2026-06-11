import tempfile
import unittest
from pathlib import Path

from cicl_agent.causal.compression import CausalMemoryCompressor, ExtractiveContextCompressor
from cicl_agent.core.schema import ContextUnit, Task
from cicl_agent.runners.compression_experiment import run_compression_experiment, run_compression_suite


class CompressionTest(unittest.TestCase):
    def test_compressor_creates_short_action_memory_card(self):
        task = Task(
            id="task",
            instance_id="toy",
            instruction="fix timestamp strings ending with Z",
            expected_actions=["inspect parser.py", "update parse_iso8601_timestamp"],
        )
        unit = ContextUnit(
            id="file:parser.py",
            instance_id="toy",
            type="file",
            source="parser.py",
            content=(
                "def parse_iso8601_timestamp(value):\n"
                "    # Local project rule: timestamps ending with Z must be interpreted as UTC.\n"
                "    if value.endswith('Z'):\n"
                "        value = value[:-1] + '+00:00'\n"
                "    return datetime.fromisoformat(value)\n"
            )
            * 8,
        )
        compressed, card, _ = CausalMemoryCompressor().compress(task, unit)

        self.assertEqual(compressed.id, unit.id)
        self.assertEqual(compressed.type, "memory_card")
        self.assertLess(compressed.token_cost, unit.token_cost)
        self.assertIn("Action hint:", compressed.content)
        self.assertEqual(card.context_id, unit.id)

    def test_extractive_summary_compressor_is_non_causal_baseline(self):
        task = Task(id="task", instance_id="toy", instruction="fix endpoint slash handling")
        unit = ContextUnit(
            id="file:client.py",
            instance_id="toy",
            type="file",
            source="client.py",
            content=(
                "def build_url(path):\n"
                "    if not path.startswith('/'):\n"
                "        path = '/' + path\n"
                "    return base_url + path\n"
            )
            * 8,
        )
        compressed = ExtractiveContextCompressor().compress(task, unit)

        self.assertEqual(compressed.id, unit.id)
        self.assertEqual(compressed.type, "summary_card")
        self.assertIn("Summary:", compressed.content)
        self.assertNotIn("Action hint:", compressed.content)

    def test_compression_experiment_writes_raw_and_compressed_rows(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_compression_experiment(
                repo=root / "experiments" / "examples" / "toy_repo",
                tasks_path=root / "experiments" / "examples" / "toy_tasks.jsonl",
                output=Path(tmpdir) / "compression",
                budget=60,
                top_k=8,
            )

            methods = {row["method"] for row in result["summary"]}
            self.assertEqual(
                methods,
                {
                    "RawCICLSelection",
                    "SelectedThenSummaryCompressedCICL",
                    "SelectedThenCompressedCICL",
                    "SummaryCompressedCICL",
                    "CausalCompressedCICL",
                },
            )
            by_method = {row["method"]: row for row in result["summary"]}
            self.assertLessEqual(
                by_method["SelectedThenCompressedCICL"]["avg_compression_ratio"],
                by_method["RawCICLSelection"]["avg_compression_ratio"],
            )
            self.assertTrue((Path(tmpdir) / "compression" / "causal_memory_cards.jsonl").exists())

    def test_compression_suite_writes_budget_sweep_report(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_compression_suite(
                repo=root / "experiments" / "examples" / "toy_repo",
                tasks_path=root / "experiments" / "examples" / "toy_tasks.jsonl",
                output=Path(tmpdir) / "suite",
                budgets=[40, 60],
                top_k=8,
            )

            self.assertEqual(len(result["summary"]), 10)
            self.assertIn("case_studies", result)
            self.assertIn("significance", result)
            self.assertIn("replay", result)
            self.assertIn("local_tool", result)
            self.assertTrue(result["significance"])
            self.assertTrue(result["replay"])
            self.assertTrue(result["local_tool"])
            self.assertTrue((Path(tmpdir) / "suite" / "compression_budget_sweep.csv").exists())
            self.assertTrue((Path(tmpdir) / "suite" / "case_studies.csv").exists())
            self.assertTrue((Path(tmpdir) / "suite" / "paired_significance.csv").exists())
            self.assertTrue((Path(tmpdir) / "suite" / "replay_task_metrics.csv").exists())
            self.assertTrue((Path(tmpdir) / "suite" / "replay_summary.csv").exists())
            self.assertTrue((Path(tmpdir) / "suite" / "tool_task_metrics.csv").exists())
            self.assertTrue((Path(tmpdir) / "suite" / "tool_summary.csv").exists())
            self.assertTrue((Path(tmpdir) / "suite" / "tool_agent_runs.jsonl").exists())
            self.assertTrue((Path(tmpdir) / "suite" / "report" / "report.md").exists())
            self.assertTrue((Path(tmpdir) / "suite" / "report" / "case_studies.md").exists())
            self.assertTrue((Path(tmpdir) / "suite" / "report" / "paired_significance.md").exists())
            self.assertTrue((Path(tmpdir) / "suite" / "report" / "replay_summary.md").exists())
            self.assertTrue((Path(tmpdir) / "suite" / "report" / "tool_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
