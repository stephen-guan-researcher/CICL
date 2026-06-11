import csv
import json
import tempfile
import unittest
from pathlib import Path

from cicl_agent.core.io import write_jsonl
from cicl_agent.runners.swebench_rr_retrieval import run_swebench_rr_retrieval


class SwebenchRrRetrievalTest(unittest.TestCase):
    def test_causal_rerank_writes_judgment_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tasks = tmp / "tasks.jsonl"
            corpus = tmp / "corpus.jsonl"
            output = tmp / "out"
            checkpoint = tmp / "checkpoint.jsonl"
            judgments = tmp / "judgments.jsonl"
            write_jsonl(
                tasks,
                [
                    {
                        "id": "task-1",
                        "instance_id": "swebench_verified_rr",
                        "instruction": "fix normalize_value stripping in parser.py",
                        "gold_context_ids": ["doc:good"],
                        "expected_actions": ["inspect parser.py normalize_value"],
                    }
                ],
            )
            write_jsonl(
                corpus,
                [
                    {
                        "id": "good",
                        "title": "",
                        "text": "parser.py/normalize_value\ndef normalize_value(x): return x.strip().lower()",
                    },
                    {"id": "bad", "title": "", "text": "docs/conf.py\nrelease notes for docs"},
                ],
            )

            result = run_swebench_rr_retrieval(
                tasks_path=tasks,
                corpus_path=corpus,
                output=output,
                methods=["BM25", "CausalRerank", "CausalHybridRerank"],
                ks=[1],
                pool_k=2,
                llm_provider="simulator",
                llm_checkpoint_path=checkpoint,
                llm_judgments_output=judgments,
                llm_concurrency=2,
            )

            methods = {row["method"] for row in result["summary"]}
            self.assertEqual(methods, {"BM25", "CausalRerank", "CausalHybridRerank"})
            judgment_rows = [json.loads(line) for line in judgments.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["context_id"] for row in judgment_rows}, {"doc:good", "doc:bad"})
            self.assertTrue(all(row["prompt_template"] == "code_retrieval_v1" for row in judgment_rows))
            with (output / "metadata.csv").open(encoding="utf-8") as handle:
                metadata = list(csv.DictReader(handle))[0]
            self.assertEqual(metadata["llm_concurrency"], "2")
            self.assertEqual(metadata["llm_prompt_template"], "code_retrieval_v1")


if __name__ == "__main__":
    unittest.main()
