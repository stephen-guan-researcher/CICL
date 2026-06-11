import tempfile
import unittest
from pathlib import Path

from cicl_agent.selectors import LearnedCICLSelector
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.learning import (
    CausalUtilityDatasetBuilder,
    CausalUtilityExample,
    ContextFeatureExtractor,
    PairwiseContextRanker,
)
from cicl_agent.runners.runner import run_experiment
from cicl_agent.core.schema import ContextUnit, Task
from cicl_agent.learning.train_policy import train_policy


class LearningTest(unittest.TestCase):
    def test_intervention_dataset_builder_labels_useful_context(self):
        graph = InstanceContextGraph("toy")
        useful = ContextUnit(
            id="memory:parser",
            instance_id="toy",
            type="rule",
            content="inspect parser.py before changing timestamp parsing",
            source="memory",
        )
        noise = ContextUnit(
            id="file:notes.md",
            instance_id="toy",
            type="file",
            content="timestamp notes about unrelated retry policy",
            source="notes.md",
        )
        graph.add_unit(useful)
        graph.add_unit(noise)
        task = Task(
            id="task",
            instance_id="toy",
            instruction="fix timestamp parsing",
            expected_actions=["inspect parser.py"],
        )
        examples = CausalUtilityDatasetBuilder(graph, top_k=5).build([task])
        by_id = {example.context_id: example for example in examples}
        self.assertIn("memory:parser", by_id)
        self.assertGreater(by_id["memory:parser"].reward, 0.0)

    def test_pairwise_ranker_learns_useful_over_noise(self):
        graph = InstanceContextGraph("toy")
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
        noise = ContextUnit(
            id="noise",
            instance_id="toy",
            type="file",
            content="timestamp retry notes",
            source="notes.md",
        )
        graph.add_unit(useful)
        graph.add_unit(noise)
        extractor = ContextFeatureExtractor(graph)
        examples = [
            CausalUtilityExample("task", useful.id, extractor.features(task, useful, 1.0), 1.0, 1.0, 0.0, 1.0),
            CausalUtilityExample("task", noise.id, extractor.features(task, noise, 0.2), 0.0, 0.0, 0.0, 0.0),
        ]
        ranker = PairwiseContextRanker()
        report = ranker.fit(examples, epochs=30)
        self.assertGreater(report.final_pairwise_accuracy, 0.99)
        self.assertGreater(ranker.score(examples[0].features), ranker.score(examples[1].features))

    def test_learned_selector_selects_trained_context(self):
        graph = InstanceContextGraph("toy")
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
        noise = ContextUnit(
            id="noise",
            instance_id="toy",
            type="file",
            content="timestamp retry notes",
            source="notes.md",
        )
        graph.add_unit(useful)
        graph.add_unit(noise)
        extractor = ContextFeatureExtractor(graph)
        ranker = PairwiseContextRanker()
        ranker.fit(
            [
                CausalUtilityExample("task", useful.id, extractor.features(task, useful, 1.0), 1.0, 1.0, 0.0, 1.0),
                CausalUtilityExample("task", noise.id, extractor.features(task, noise, 0.2), 0.0, 0.0, 0.0, 0.0),
            ],
            epochs=30,
        )
        selected = LearnedCICLSelector(graph, ranker).select(task, budget=80, history=[])
        self.assertEqual(selected[0].id, useful.id)

    def test_train_policy_and_runner_include_learned_cicl(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            policy_path = tmp / "policy.json"
            summary = train_policy(
                repo=root / "experiments" / "examples" / "toy_repo",
                tasks_path=root / "experiments" / "examples" / "toy_tasks.jsonl",
                output=policy_path,
                examples_output=tmp / "examples.jsonl",
                epochs=20,
            )
            self.assertTrue(policy_path.exists())
            self.assertGreater(summary["examples"], 0)
            rows = run_experiment(
                repo=root / "experiments" / "examples" / "toy_repo",
                tasks_path=root / "experiments" / "examples" / "toy_tasks.jsonl",
                output=tmp / "run",
                budget=120,
                learned_policy_path=policy_path,
            )
            self.assertIn("LearnedCICL", {row["method"] for row in rows})


if __name__ == "__main__":
    unittest.main()
