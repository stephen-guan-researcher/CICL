import tempfile
import unittest
from pathlib import Path

from cicl_agent.core.io import write_jsonl
from cicl_agent.core.schema import ContextEdge, ContextUnit, Task
from cicl_agent.evaluation.leakage_audit import audit_leakage
from cicl_agent.memory.graph import InstanceContextGraph


class LeakageAuditTest(unittest.TestCase):
    def test_task_memory_does_not_link_gold_by_default(self):
        graph = InstanceContextGraph("toy")
        graph.add_unit(ContextUnit(id="file:target.py", instance_id="toy", type="file", content="x", source="target.py"))
        graph.add_task_memory(
            Task(
                id="base_task",
                instance_id="toy",
                instruction="Remember this rule.",
                split="base",
                memory="Use target.py.",
                gold_context_ids=["file:target.py"],
            )
        )

        self.assertFalse(any(edge.relation == "derived-from" for edge in graph.edges))

    def test_task_memory_can_link_gold_when_explicitly_requested(self):
        graph = InstanceContextGraph("toy")
        graph.add_unit(ContextUnit(id="file:target.py", instance_id="toy", type="file", content="x", source="target.py"))
        graph.add_task_memory(
            Task(
                id="base_task",
                instance_id="toy",
                instruction="Remember this rule.",
                split="base",
                memory="Use target.py.",
                gold_context_ids=["file:target.py"],
            ),
            link_gold_context=True,
        )

        self.assertTrue(any(edge.relation == "derived-from" for edge in graph.edges))

    def test_audit_fails_on_derived_from_gold_edge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out = root / "out"
            out.mkdir()
            tasks = root / "tasks.jsonl"
            write_jsonl(
                tasks,
                [
                    Task(
                        id="eval_task",
                        instance_id="toy",
                        instruction="Fix it.",
                        gold_context_ids=["file:target.py"],
                    ).to_dict()
                ],
            )
            write_jsonl(
                out / "context_units.jsonl",
                [
                    ContextUnit(
                        id="memory:base",
                        instance_id="toy",
                        type="rule",
                        content="Use target.py.",
                        source="task:base",
                    ).to_dict()
                ],
            )
            write_jsonl(
                out / "context_edges.jsonl",
                [ContextEdge("memory:base", "file:target.py", "derived-from").to_dict()],
            )

            items = audit_leakage(root=root, tasks_path="tasks.jsonl", artifact_dirs=["out"])

            self.assertTrue(any(item.status == "FAIL" and item.category == "context-edges" for item in items))

    def test_audit_passes_clean_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out = root / "out"
            out.mkdir()
            tasks = root / "tasks.jsonl"
            write_jsonl(tasks, [Task(id="eval_task", instance_id="toy", instruction="Fix it.").to_dict()])
            write_jsonl(
                out / "context_units.jsonl",
                [ContextUnit(id="file:target.py", instance_id="toy", type="file", content="x", source="target.py").to_dict()],
            )
            write_jsonl(
                out / "context_edges.jsonl",
                [ContextEdge("file:target.py", "symbol:target.py:f:1", "contains").to_dict()],
            )

            items = audit_leakage(root=root, tasks_path="tasks.jsonl", artifact_dirs=["out"])

            self.assertFalse([item for item in items if item.status == "FAIL"])

    def test_audit_scans_current_source_package(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "cicl_agent/causal/scoring.py"
            source.parent.mkdir(parents=True)
            source.write_text("allow_gold_labels = False\n", encoding="utf-8")

            items = audit_leakage(root=root)

            self.assertTrue(
                any(
                    item.category == "source-gold-usage"
                    and item.target == "cicl_agent/causal/scoring.py"
                    and item.status == "WARN"
                    for item in items
                )
            )


if __name__ == "__main__":
    unittest.main()
