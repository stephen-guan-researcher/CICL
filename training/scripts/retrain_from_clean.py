"""Retrain a PairwiseContextRanker from migrated llm_examples.clean.jsonl.

Skips the LLMJudgmentDatasetBuilder (no Opus calls, no graph rebuild) — just
loads existing examples and runs ranker.fit. Used after migrate_examples.py
strips the llm_* leakage and re-derives retrieval_score with the bugfix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cicl_agent.learning.dataset import CausalUtilityExample
from cicl_agent.learning.ranker import PairwiseContextRanker


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--examples", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--label-source", default="llm")
    p.add_argument("--llm-provider", default="anthropic")
    p.add_argument("--llm-model", default="claude-opus-4-7")
    args = p.parse_args()

    examples = []
    with open(args.examples) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(CausalUtilityExample.from_dict(json.loads(line)))
    print(f"[retrain] loaded {len(examples)} examples from {args.examples}")

    metadata = {
        "label_source": args.label_source,
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "post_bugfix": True,
    }
    ranker = PairwiseContextRanker(metadata=metadata)
    report = ranker.fit(examples, epochs=args.epochs, lr=args.lr)
    ranker.save(args.output, report=report)

    summary = {
        "policy_path": args.output,
        "examples_path": args.examples,
        "examples": len(examples),
        "label_source": args.label_source,
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        **report.to_dict(),
    }
    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
