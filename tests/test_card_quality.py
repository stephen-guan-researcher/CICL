import tempfile
import unittest
from pathlib import Path

from cicl_agent.evaluation.card_quality import audit_card_file, write_markdown_report


class CardQualityTest(unittest.TestCase):
    def test_audit_passes_complete_actionable_compressed_cards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cards = Path(tmpdir) / "cards.jsonl"
            cards.write_text(
                "\n".join(
                    [
                        '{"trigger":"timestamp task","evidence":"parser.py: Z means UTC","action_hint":"inspect parser.py and update timestamp parsing","failure_if_ignored":"timezone tests fail","scope":"parser.py","original_token_cost":100,"compressed_token_cost":35}',
                        '{"trigger":"url task","evidence":"client.py: path needs slash","action_hint":"read client.py before fixing URL builder","failure_if_ignored":"endpoint tests fail","scope":"client.py","original_token_cost":90,"compressed_token_cost":30}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            row = audit_card_file(cards)

            self.assertEqual(row.status, "PASS")
            self.assertEqual(row.cards, 2)
            self.assertEqual(row.required_field_rate, 1.0)
            self.assertEqual(row.actionable_hint_rate, 1.0)
            self.assertEqual(row.compression_success_rate, 1.0)

    def test_audit_fails_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cards = Path(tmpdir) / "cards.jsonl"
            cards.write_text(
                '{"trigger":"task","evidence":"","action_hint":"inspect parser.py","failure_if_ignored":"tests fail","scope":"parser.py","original_token_cost":100,"compressed_token_cost":30}\n',
                encoding="utf-8",
            )

            row = audit_card_file(cards)

            self.assertEqual(row.status, "FAIL")
            self.assertEqual(row.required_field_rate, 0.0)

    def test_markdown_report_mentions_structural_scope(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cards = Path(tmpdir) / "cards.jsonl"
            cards.write_text(
                '{"trigger":"task","evidence":"file.py: fact","action_hint":"inspect file.py","failure_if_ignored":"tests fail","scope":"file.py","original_token_cost":100,"compressed_token_cost":40}\n',
                encoding="utf-8",
            )
            report = Path(tmpdir) / "report.md"
            row = audit_card_file(cards)
            write_markdown_report([row], report)

            text = report.read_text(encoding="utf-8")
            self.assertIn("Causal Memory Card 质量审计", text)
            self.assertIn("结构性审计", text)


if __name__ == "__main__":
    unittest.main()
