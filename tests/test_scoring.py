import unittest

from cicl_agent.core.schema import ContextUnit, Task
from cicl_agent.causal.scoring import CausalUtilityScorer


class ScoringTest(unittest.TestCase):
    def test_default_scorer_does_not_use_gold_labels(self):
        unit = ContextUnit(
            id="ctx1",
            instance_id="toy",
            type="strategy",
            content="inspect parser.py and update parse_iso8601_timestamp",
            source="memory",
        )
        task = Task(
            id="task1",
            instance_id="toy",
            instruction="fix timestamp parsing",
            gold_context_ids=["ctx1"],
            expected_actions=["inspect parser.py"],
        )
        score = CausalUtilityScorer().score(unit, task)
        self.assertLess(score.necessity_score, 1.0)
        self.assertGreater(score.final_score, 0.2)

    def test_debug_scorer_can_use_gold_labels(self):
        unit = ContextUnit(
            id="ctx1",
            instance_id="toy",
            type="strategy",
            content="inspect parser.py and update parse_iso8601_timestamp",
            source="memory",
        )
        task = Task(
            id="task1",
            instance_id="toy",
            instruction="fix timestamp parsing",
            gold_context_ids=["ctx1"],
            expected_actions=["inspect parser.py"],
        )
        score = CausalUtilityScorer(allow_gold_labels=True).score(unit, task)
        self.assertEqual(score.necessity_score, 1.0)


if __name__ == "__main__":
    unittest.main()
