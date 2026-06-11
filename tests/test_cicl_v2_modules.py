import unittest

from cicl_agent.retrieval.assembly import BudgetAwareContextAssembler, BudgetPolicy
from cicl_agent.memory.conflict import ConflictDetector
from cicl_agent.memory.curation import ExperienceCurator
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.causal.intervention import InterventionScorer
from cicl_agent.core.schema import AgentRun, ContextUnit, Task


class CICLV2ModulesTest(unittest.TestCase):
    def test_curator_turns_task_memory_into_rule(self):
        task = Task(
            id="base1",
            instance_id="toy",
            instruction="fix UTC timestamp parsing",
            split="base",
            memory="timestamps ending with Z must be parsed as UTC",
            expected_actions=["inspect parser.py"],
        )
        unit = ExperienceCurator().curate_task_memory(task)
        self.assertIsNotNone(unit)
        self.assertEqual(unit.type, "rule")
        self.assertEqual(unit.metadata["curator"], "ace_style")
        self.assertIn("Reusable rule", unit.content)

    def test_curator_preserves_failure_lessons(self):
        task = Task(id="eval1", instance_id="toy", instruction="fix parser")
        run = AgentRun(
            task_id="eval1",
            method="CICL",
            selected_context_ids=["file:legacy_parser.py"],
            actions=["inspect legacy_parser.py"],
            observations=["tests still fail"],
            success=False,
            tokens=10,
            runtime_seconds=0.1,
            tool_calls=1,
        )
        unit = ExperienceCurator().curate_run(run, task)
        self.assertEqual(unit.type, "failure")
        self.assertIn("Previous attempt failed", unit.content)

    def test_conflict_detector_marks_stale_and_adds_edges(self):
        graph = InstanceContextGraph("toy")
        graph.add_unit(
            ContextUnit(
                id="file:legacy_parser.py",
                instance_id="toy",
                type="file",
                content="legacy old behavior parser timestamp parse should not guide new fixes",
                source="legacy_parser.py",
            )
        )
        graph.add_unit(
            ContextUnit(
                id="file:parser.py",
                instance_id="toy",
                type="file",
                content="current parser timestamp parse implementation",
                source="parser.py",
            )
        )
        ConflictDetector().apply(graph)
        self.assertTrue(graph.units["file:legacy_parser.py"].metadata["stale"])
        self.assertIn("file:parser.py", graph.conflicts("file:legacy_parser.py"))

    def test_budget_policy_rejects_low_score_context(self):
        graph = InstanceContextGraph("toy")
        graph.add_unit(
            ContextUnit(
                id="noise",
                instance_id="toy",
                type="file",
                content="unrelated retry status code",
                source="client.py",
            )
        )
        task = Task(id="task", instance_id="toy", instruction="fix timestamp parsing")
        selected, scores = BudgetAwareContextAssembler(
            graph,
            budget_policy=BudgetPolicy(min_final_score=0.95, min_score_after_first=0.95),
        ).assemble(task, token_budget=100)
        self.assertEqual(selected, [])
        self.assertEqual(scores, [])

    def test_intervention_scorer_detects_action_and_success_delta(self):
        task = Task(
            id="task",
            instance_id="toy",
            instruction="fix timestamp parsing",
            expected_actions=["inspect parser.py"],
        )
        unit = ContextUnit(
            id="memory:parser",
            instance_id="toy",
            type="rule",
            content="inspect parser.py before changing parse_iso8601",
            source="memory",
        )
        result = InterventionScorer().score_candidates(task, [unit])[0]
        self.assertFalse(result.no_context_success)
        self.assertTrue(result.with_context_success)
        self.assertGreater(result.action_delta, 0.0)
        self.assertEqual(result.success_delta, 1.0)


if __name__ == "__main__":
    unittest.main()
