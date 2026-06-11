import json
import tempfile
import unittest
from pathlib import Path

from cicl_agent.core.io import write_jsonl
from cicl_agent.evaluation.llm_agreement import evaluate_llm_agreement, spearman


class LLMAgreementEvaluationTest(unittest.TestCase):
    def test_spearman_handles_ties(self):
        self.assertAlmostEqual(spearman([1.0, 2.0, 2.0, 4.0], [1.0, 3.0, 3.0, 4.0]), 1.0)

    def test_simulator_smoke_writes_summary_and_judgments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "foo.py").write_text(
                "def normalize_value(value):\n    return str(value).strip().lower()\n",
                encoding="utf-8",
            )
            tasks = root / "tasks.jsonl"
            write_jsonl(
                tasks,
                [
                    {
                        "id": "task-1",
                        "instance_id": "toy/repo",
                        "instruction": "Use normalize_value from foo.py.",
                        "split": "eval",
                        "gold_context_ids": ["file:foo.py"],
                        "expected_actions": ["inspect foo.py"],
                    }
                ],
            )
            teacher = root / "teacher.jsonl"
            write_jsonl(
                teacher,
                [
                    {
                        "task_id": "task-1",
                        "context_id": "file:foo.py",
                        "action_delta": 0.7,
                        "outcome_delta": 0.8,
                        "negative_transfer": 0.0,
                        "metadata": {
                            "llm_action_shift": 0.7,
                            "llm_necessity": 0.8,
                            "llm_expected_uplift": 0.8,
                            "llm_negative_transfer": 0.0,
                            "llm_confidence": 0.9,
                        },
                    }
                ],
            )
            output = root / "agreement.json"
            judgments = root / "judgments.jsonl"

            summary = evaluate_llm_agreement(
                repo=repo,
                tasks_path=tasks,
                teacher_examples=teacher,
                output=output,
                judgments_output=judgments,
                llm_provider="simulator",
                top_k=1,
            )

            self.assertEqual(summary["n_tasks"], 1)
            self.assertEqual(summary["n_attempted"], 1)
            self.assertEqual(summary["n_parsed"], 1)
            self.assertEqual(summary["parse_rate"], 1.0)
            self.assertEqual(summary["avg_top_k_jaccard"], 1.0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["n_tasks"], 1)
            rows = [json.loads(line) for line in judgments.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["context_id"], "file:foo.py")
            self.assertTrue(rows[0]["ok"])


if __name__ == "__main__":
    unittest.main()
