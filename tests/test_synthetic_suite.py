import tempfile
import unittest
from pathlib import Path

from cicl_agent.runners.experiment_suite import budget_sweep, causal_removal, learning_curve
from cicl_agent.benchmarks.synthetic import generate_synthetic_benchmark


class SyntheticSuiteTest(unittest.TestCase):
    def test_synthetic_suite_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo, tasks = generate_synthetic_benchmark(root / "bench", n_rules=3, n_eval_per_rule=1, seed=3)
            budget_rows = budget_sweep(repo, tasks, root / "budget", budgets=[80])
            learning_rows = learning_curve(repo, tasks, root / "learning", base_counts=[0, 1], budget=80)
            removal_rows = causal_removal(repo, tasks, root / "removal", budget=80)
            self.assertTrue(budget_rows)
            self.assertTrue(learning_rows)
            self.assertTrue(removal_rows)
            self.assertTrue((root / "budget" / "budget_sweep_metrics.csv").exists())
            self.assertTrue((root / "learning" / "learning_curve_metrics.csv").exists())
            self.assertTrue((root / "removal" / "causal_removal.csv").exists())


if __name__ == "__main__":
    unittest.main()

