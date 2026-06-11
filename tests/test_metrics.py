import unittest

from cicl_agent.evaluation.metrics import context_prf, mrr


class MetricsTest(unittest.TestCase):
    def test_context_prf(self):
        p, r, f = context_prf(["a", "b"], ["b", "c"])
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(r, 0.5)
        self.assertAlmostEqual(f, 0.5)

    def test_mrr(self):
        self.assertEqual(mrr(["x", "b"], ["b"]), 0.5)
        self.assertEqual(mrr(["x"], ["b"]), 0.0)


if __name__ == "__main__":
    unittest.main()

