import tempfile
import unittest
from pathlib import Path

from cicl_agent.runners.runner import run_experiment
from cicl_agent.evaluation.reporting import generate_report


class RunnerTest(unittest.TestCase):
    def test_toy_runner_writes_metrics(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = run_experiment(
                repo=root / "experiments" / "examples" / "toy_repo",
                tasks_path=root / "experiments" / "examples" / "toy_tasks.jsonl",
                output=tmpdir,
                budget=120,
                include_ablations=True,
            )
            methods = {row["method"] for row in rows}
            self.assertIn("CICL", methods)
            self.assertIn("CICL_w/o_causal_scoring", methods)
            self.assertTrue((Path(tmpdir) / "metrics.csv").exists())
            self.assertTrue((Path(tmpdir) / "causal_scores.jsonl").exists())

            report_dir = generate_report(tmpdir)
            self.assertTrue((report_dir / "report.md").exists())
            self.assertTrue((report_dir / "tables" / "main_results.md").exists())
            self.assertTrue((report_dir / "figures" / "context_f1_by_method.svg").exists())


if __name__ == "__main__":
    unittest.main()
