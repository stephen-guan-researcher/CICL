import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cicl_agent.core.io import write_jsonl
from cicl_agent.evaluation.llm_agreement_preflight import (
    check_llm_agreement_preflight,
    write_markdown_report,
)


class LLMAgreementPreflightTest(unittest.TestCase):
    def test_qwen_preflight_fails_without_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, tasks, teacher = self._fixture(Path(tmp))

            with patch.dict("os.environ", {}, clear=True):
                report = check_llm_agreement_preflight(
                    repo=repo,
                    tasks=tasks,
                    teacher_examples=teacher,
                    llm_provider="qwen",
                    min_free_mb=0,
                )

            self.assertEqual(report["status"], "FAIL")
            self.assertIn("qwen_api_key", {row["name"] for row in report["blocking_checks"]})

    def test_qwen_preflight_passes_with_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, tasks, teacher = self._fixture(Path(tmp))

            with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
                report = check_llm_agreement_preflight(
                    repo=repo,
                    tasks=tasks,
                    teacher_examples=teacher,
                    llm_provider="qwen",
                    min_free_mb=0,
                )

            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["task_count"], 1)
            self.assertEqual(report["teacher_example_count"], 1)

    def test_qwen_local_checks_base_and_adapter_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, tasks, teacher = self._fixture(root)
            base = root / "base"
            adapter = root / "adapter"
            base.mkdir()
            adapter.mkdir()

            report = check_llm_agreement_preflight(
                repo=repo,
                tasks=tasks,
                teacher_examples=teacher,
                llm_provider="qwen_local",
                llm_base_url=str(base),
                llm_model=str(adapter),
                min_free_mb=0,
            )

            self.assertEqual(report["status"], "PASS")

    def test_qwen_local_defaults_accept_huggingface_repo_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, tasks, teacher = self._fixture(Path(tmp))

            with patch.dict("os.environ", {}, clear=True):
                report = check_llm_agreement_preflight(
                    repo=repo,
                    tasks=tasks,
                    teacher_examples=teacher,
                    llm_provider="qwen_local",
                    min_free_mb=0,
                )

            self.assertEqual(report["status"], "PASS")
            details = {row["name"]: row["detail"] for row in report["checks"]}
            self.assertEqual(details["qwen_local_base"], "Hugging Face repo id: Qwen/Qwen3.5-9B")
            self.assertEqual(details["qwen_local_adapter"], "Hugging Face repo id: XinyuGuan/CICL")

    def test_markdown_report_lists_claim_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, tasks, teacher = self._fixture(Path(tmp))
            with patch.dict("os.environ", {}, clear=True):
                report = check_llm_agreement_preflight(
                    repo=repo,
                    tasks=tasks,
                    teacher_examples=teacher,
                    llm_provider="qwen",
                    min_free_mb=0,
                )
            output = Path(tmp) / "preflight.md"
            write_markdown_report(report, output)

            text = output.read_text(encoding="utf-8")
            self.assertIn("Real-Code Qwen Agreement Preflight", text)
            self.assertIn("not itself agreement evidence", text)

    def _fixture(self, root: Path):
        repo = root / "repo"
        repo.mkdir()
        (repo / "foo.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        tasks = root / "tasks.jsonl"
        write_jsonl(
            tasks,
            [
                {
                    "id": "task-1",
                    "instance_id": "toy/repo",
                    "instruction": "Inspect foo.py.",
                    "split": "eval",
                    "gold_context_ids": ["file:foo.py"],
                    "expected_actions": ["inspect foo.py"],
                }
            ],
        )
        teacher = root / "teacher.jsonl"
        write_jsonl(teacher, [{"task_id": "task-1", "context_id": "file:foo.py"}])
        return repo, tasks, teacher


if __name__ == "__main__":
    unittest.main()
