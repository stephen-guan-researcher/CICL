import unittest

from cicl_agent.retrieval.assembly import BudgetAwareContextAssembler
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.core.schema import ContextUnit, Task


class AssemblyTest(unittest.TestCase):
    def test_assembler_selects_gold_context(self):
        graph = InstanceContextGraph("toy")
        graph.add_unit(
            ContextUnit(
                id="memory:rule",
                instance_id="toy",
                type="strategy",
                content="timestamps ending with Z must be parsed as UTC in parser.py",
                source="memory",
            )
        )
        graph.add_unit(
            ContextUnit(
                id="noise",
                instance_id="toy",
                type="file",
                content="unrelated retry status code",
                source="client.py",
            )
        )
        task = Task(
            id="task",
            instance_id="toy",
            instruction="fix Z timestamp parsing",
            gold_context_ids=["memory:rule"],
            expected_actions=["inspect parser.py"],
        )
        selected, scores = BudgetAwareContextAssembler(graph).assemble(task, token_budget=50)
        self.assertEqual(selected[0].id, "memory:rule")
        self.assertEqual(scores[0].context_id, "memory:rule")


if __name__ == "__main__":
    unittest.main()

