import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cicl_agent.evaluation.real_agent_rollout_preflight import (
    check_real_agent_rollout_preflight,
    write_json_report,
    write_markdown_report,
)


class RealAgentRolloutPreflightTest(unittest.TestCase):
    def test_reports_blockers_without_credentials_or_rollout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root, include_rollout=False)

            with patch.dict("os.environ", {}, clear=True), patch(
                "cicl_agent.evaluation.real_agent_rollout_preflight.shutil.which",
                side_effect=lambda cmd: None,
            ), patch(
                "cicl_agent.evaluation.real_agent_rollout_preflight.importlib.util.find_spec",
                return_value=None,
            ):
                report = check_real_agent_rollout_preflight(root=root, min_free_mb=1)

            self.assertEqual(report["status"], "FAIL")
            blockers = {row["name"] for row in report["blocking_checks"]}
            self.assertIn("model_credentials", blockers)
            self.assertIn("docker_available", blockers)
            self.assertIn("pytest_available", blockers)
            self.assertIn("formal_rollout_artifact", blockers)
            self.assertNotIn("patch_smoke_artifact", blockers)

    def test_passes_when_prerequisites_and_formal_rollout_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root, include_rollout=True)

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}, clear=True), patch(
                "cicl_agent.evaluation.real_agent_rollout_preflight.shutil.which",
                side_effect=lambda cmd: f"/usr/bin/{cmd}",
            ):
                report = check_real_agent_rollout_preflight(root=root, min_free_mb=1)

            self.assertEqual(report["status"], "PASS")
            self.assertEqual([], report["blocking_checks"])

    def test_writes_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root, include_rollout=False)

            with patch.dict("os.environ", {}, clear=True), patch(
                "cicl_agent.evaluation.real_agent_rollout_preflight.shutil.which",
                side_effect=lambda cmd: None,
            ):
                report = check_real_agent_rollout_preflight(root=root, min_free_mb=1)

            md = write_markdown_report(report, root / "reports/real_agent_rollout_preflight.md")
            js = write_json_report(report, root / "reports/real_agent_rollout_preflight.json")

            self.assertIn("Real-Agent Rollout Preflight", md.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(js.read_text(encoding="utf-8"))["status"], "FAIL")

    def _fixture(self, root: Path, include_rollout: bool) -> None:
        tasks = root / "experiments/data/raw/swebench_verified_rr/corpus_sample_50.jsonl"
        tasks.parent.mkdir(parents=True)
        tasks.write_text('{"instance_id":"x","problem_statement":"bug"}\n', encoding="utf-8")

        retrieval = root / "artifacts/outputs/latest/swebench_verified_rr_retrieval/retrieval_summary.csv"
        retrieval.parent.mkdir(parents=True)
        retrieval.write_text(
            "method,n,recall@10\nBM25,1,1.0\nCausalRerank,1,0.5\n",
            encoding="utf-8",
        )

        smoke = root / "artifacts/outputs/latest/swebench_opus_patch_smoke_n3"
        smoke.mkdir(parents=True)
        (smoke / "summary.json").write_text(
            json.dumps({"n_tasks": 3, "n_success": 3, "official_swebench_harness": "not run"}),
            encoding="utf-8",
        )
        (smoke / "executable_validation.json").write_text(
            json.dumps({"n_passed": 3, "official_swebench_harness": "not run"}),
            encoding="utf-8",
        )

        if include_rollout:
            rollout = root / "artifacts/outputs/latest/real_agent_rollout/summary.json"
            rollout.parent.mkdir(parents=True)
            rollout.write_text(
                json.dumps(
                    {
                        "benchmark": "SWE-bench Verified",
                        "task_source": "experiments/data/raw/swebench_verified_rr/corpus_sample_50.jsonl",
                        "task_ids": ["x"],
                        "n_tasks": 3,
                        "methods": ["NoContext", "BM25", "CICL"],
                        "provider": "anthropic-compatible",
                        "model": "claude-opus-4-6",
                        "same_budget_across_methods": True,
                        "context_budget": 120,
                        "official_or_equivalent_regression_tests": True,
                        "test_harness": "official SWE-bench regression tests in Docker",
                        "gold_patch_visible_to_agent": False,
                        "claim_boundary": (
                            "Evidence is limited to a non-toy SWE-bench subset "
                            "with official or equivalent regression tests, not "
                            "toy tasks or a patch-generation smoke."
                        ),
                        "metrics_by_method": {
                            "NoContext": {"n_attempted": 3, "n_applied": 1, "n_executable": 1, "n_passed": 0},
                            "BM25": {"n_attempted": 3, "n_applied": 2, "n_executable": 2, "n_passed": 1},
                            "CICL": {"n_attempted": 3, "patch_apply_rate": 1.0, "executable_test_rate": 1.0, "pass_rate": 0.67},
                        },
                    }
                ),
                encoding="utf-8",
            )


if __name__ == "__main__":
    unittest.main()
