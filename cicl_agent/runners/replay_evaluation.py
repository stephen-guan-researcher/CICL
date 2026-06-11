"""CLI for replay-style action evaluation over compression suite outputs."""

from __future__ import annotations

import argparse

from cicl_agent.evaluation.replay import evaluate_compression_suite_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay logged agent actions against selected/gold context.")
    parser.add_argument("--suite", required=True, help="Compression suite directory containing budget_* outputs.")
    parser.add_argument("--tasks", required=True, help="Task JSONL used by the suite.")
    args = parser.parse_args()

    result = evaluate_compression_suite_replay(args.suite, args.tasks)
    for row in result["summary"]:
        print(
            f"budget={row['budget']} method={row['method']} "
            f"replay_success={row['replay_success_rate']} "
            f"selected_gold={row['selected_gold_rate']} "
            f"gold_action={row['gold_action_rate']}"
        )


if __name__ == "__main__":
    main()

