import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cicl_agent.selectors import LLMCICLSelector
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.learning.dataset import LLMJudgmentDatasetBuilder
from cicl_agent.integrations.llm_clients import (
    AnthropicMessagesClient,
    CodexCliClient,
    DEFAULT_QWEN_LOCAL_ADAPTER,
    DEFAULT_QWEN_LOCAL_BASE,
    QwenLocalAdapterClient,
    QwenOpenAICompatibleClient,
    build_llm_judge,
    resolve_llm_model,
)
from cicl_agent.causal.llm_judge import CodeRetrievalPromptTemplate, LLMCausalContextJudge, judgment_to_score
from cicl_agent.runners.runner import run_experiment
from cicl_agent.core.schema import ContextUnit, Task
from cicl_agent.learning.train_policy import train_policy


class LLMJudgeTest(unittest.TestCase):
    def test_qwen_client_posts_openai_compatible_chat_request(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"{\\"action_shift\\":0.8,'
                    b'\\"necessity\\":0.7,\\"expected_outcome_uplift\\":0.6,'
                    b'\\"negative_transfer_risk\\":0.1,\\"reason\\":\\"useful\\",'
                    b'\\"confidence\\":0.9}"}}]}'
                )

        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["headers"] = dict(request.header_items())
            captured["payload"] = request.data.decode("utf-8")
            return FakeResponse()

        client = QwenOpenAICompatibleClient(
            api_key="test-key",
            model="qwen3.6-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout_seconds=12,
        )
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            content = client.complete("Return JSON.")

        self.assertIn('"action_shift"', content)
        self.assertEqual(captured["url"], "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        self.assertEqual(captured["timeout"], 12)
        self.assertIn("Bearer test-key", captured["headers"]["Authorization"])
        self.assertIn('"model": "qwen3.6-plus"', captured["payload"])
        self.assertIn('"response_format": {"type": "json_object"}', captured["payload"])
        self.assertIn('"enable_thinking": false', captured["payload"])

    def test_build_qwen_judge_reads_key_from_env(self):
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}, clear=False):
            judge = build_llm_judge(provider="qwen", model="qwen3.6-plus")
        self.assertIsInstance(judge, LLMCausalContextJudge)

    def test_anthropic_client_posts_messages_request(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"content":[{"type":"text","text":"{\\"action_shift\\":0.8,'
                    b'\\"necessity\\":0.7,\\"expected_outcome_uplift\\":0.6,'
                    b'\\"negative_transfer_risk\\":0.1,\\"reason\\":\\"useful\\",'
                    b'\\"confidence\\":0.9}"}]}'
                )

        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["headers"] = dict(request.header_items())
            captured["payload"] = request.data.decode("utf-8")
            return FakeResponse()

        client = AnthropicMessagesClient(
            api_key="test-key",
            model="claude-opus-4-7",
            base_url="https://api.anthropic.com",
            timeout_seconds=12,
        )
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            content = client.complete("Return JSON.")

        self.assertIn('"action_shift"', content)
        self.assertEqual(captured["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(captured["headers"]["X-api-key"], "test-key")
        self.assertEqual(captured["headers"]["Anthropic-version"], "2023-06-01")
        self.assertIn('"model": "claude-opus-4-7"', captured["payload"])

    def test_build_anthropic_judge_reads_key_from_env(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            judge = build_llm_judge(provider="anthropic")
        self.assertIsInstance(judge, LLMCausalContextJudge)

    def test_opus_alias_uses_opus_default_model(self):
        self.assertEqual(resolve_llm_model("opus"), "claude-opus-4-7")
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            judge = build_llm_judge(provider="opus")
        self.assertIsInstance(judge, LLMCausalContextJudge)

    def test_codex_alias_builds_cli_judge(self):
        self.assertEqual(resolve_llm_model("gpt5.5"), "gpt-5.5")
        judge = build_llm_judge(provider="codex", model="gpt-5.5")
        self.assertIsInstance(judge.client, CodexCliClient)

    def test_qwen_local_defaults_to_huggingface_adapter(self):
        self.assertEqual(DEFAULT_QWEN_LOCAL_BASE, "Qwen/Qwen3.5-9B")
        self.assertEqual(DEFAULT_QWEN_LOCAL_ADAPTER, "XinyuGuan/CICL")
        self.assertEqual(resolve_llm_model("qwen_local"), "XinyuGuan/CICL")
        with patch.dict("os.environ", {}, clear=True):
            client = QwenLocalAdapterClient.from_env()
        self.assertEqual(client.base_path, "Qwen/Qwen3.5-9B")
        self.assertEqual(client.adapter_path, "XinyuGuan/CICL")

    def test_code_retrieval_prompt_emphasizes_concrete_code_evidence(self):
        task = Task(
            id="task",
            instance_id="repo",
            instruction="hist no longer respects range when density=True",
        )
        unit = ContextUnit(
            id="doc:hist",
            instance_id="repo",
            type="doc",
            content=(
                "matplotlib/axes/_axes.py/Axes/hist\n"
                "if density: bins = np.histogram_bin_edges(x)\n"
                "range must be forwarded to histogram bin edge selection"
            ),
            source="matplotlib/axes/_axes.py/Axes/hist",
        )
        prompt = CodeRetrievalPromptTemplate(max_context_chars=80).render(task, unit)
        self.assertIn("exact file paths", prompt)
        self.assertIn("Candidate source/path", prompt)
        self.assertIn("...[truncated]...", prompt)

    def test_llm_judge_scores_policy_shaping_context(self):
        task = Task(
            id="task",
            instance_id="toy",
            instruction="fix timestamp parsing",
            expected_actions=["inspect parser.py"],
        )
        useful = ContextUnit(
            id="memory:parser",
            instance_id="toy",
            type="rule",
            content="inspect parser.py before changing timestamp parsing",
            source="memory",
        )
        stale = ContextUnit(
            id="file:legacy_parser.py",
            instance_id="toy",
            type="file",
            content="deprecated legacy parser path should not guide new fixes",
            source="legacy_parser.py",
            metadata={"stale": True},
        )
        judge = LLMCausalContextJudge()
        useful_judgment = judge.judge(task, useful)
        stale_judgment = judge.judge(task, stale)
        self.assertGreater(useful_judgment.action_shift, stale_judgment.action_shift)
        self.assertGreater(stale_judgment.negative_transfer_risk, useful_judgment.negative_transfer_risk)
        self.assertGreater(judgment_to_score(useful_judgment, useful).final_score, 0.0)

    def test_llm_cicl_selector_records_policy_edges(self):
        graph = InstanceContextGraph("toy")
        graph.add_unit(
            ContextUnit(
                id="memory:parser",
                instance_id="toy",
                type="rule",
                content="inspect parser.py before changing timestamp parsing",
                source="memory",
            )
        )
        graph.add_unit(
            ContextUnit(
                id="noise",
                instance_id="toy",
                type="file",
                content="unrelated retry notes",
                source="notes.md",
            )
        )
        task = Task(
            id="task",
            instance_id="toy",
            instruction="fix timestamp parsing",
            expected_actions=["inspect parser.py"],
        )
        selected = LLMCICLSelector(graph, record_policy_edges=True).select(task, budget=80, history=[])
        self.assertEqual(selected[0].id, "memory:parser")
        self.assertTrue(any(unit.type == "policy_judgment" for unit in graph.units.values()))
        self.assertTrue(any(edge.relation == "enables-action" for edge in graph.edges))

    def test_llm_label_dataset_and_distilled_policy(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary = train_policy(
                repo=root / "experiments" / "examples" / "toy_repo",
                tasks_path=root / "experiments" / "examples" / "toy_tasks.jsonl",
                output=tmp / "llm_policy.json",
                examples_output=tmp / "llm_examples.jsonl",
                epochs=20,
                label_source="llm",
            )
            self.assertEqual(summary["label_source"], "llm")
            rows = run_experiment(
                repo=root / "experiments" / "examples" / "toy_repo",
                tasks_path=root / "experiments" / "examples" / "toy_tasks.jsonl",
                output=tmp / "run",
                budget=120,
                learned_policy_path=tmp / "llm_policy.json",
                include_llm_judge=True,
            )
            methods = {row["method"] for row in rows}
            self.assertIn("LLM-CICL", methods)
            self.assertIn("CICL_Distilled", methods)

    def test_llm_judgment_dataset_builder(self):
        graph = InstanceContextGraph("toy")
        graph.add_unit(
            ContextUnit(
                id="memory:parser",
                instance_id="toy",
                type="rule",
                content="inspect parser.py before changing timestamp parsing",
                source="memory",
            )
        )
        task = Task(
            id="task",
            instance_id="toy",
            instruction="fix timestamp parsing",
            expected_actions=["inspect parser.py"],
        )
        examples = LLMJudgmentDatasetBuilder(graph).build([task])
        self.assertTrue(examples)
        self.assertEqual(examples[0].metadata["label_source"], "llm_counterfactual")


if __name__ == "__main__":
    unittest.main()
