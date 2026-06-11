"""CLI for local tool-execution evaluation over compression suite outputs."""

from __future__ import annotations

import argparse

from cicl_agent.evaluation.tool_execution import evaluate_compression_suite_local_tools


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local file-inspection tools over selected context.")
    parser.add_argument("--repo", required=True, help="Repository/context corpus directory to inspect.")
    parser.add_argument("--suite", required=True, help="Compression suite directory containing budget_* outputs.")
    parser.add_argument("--tasks", required=True, help="Task JSONL used by the suite.")
    parser.add_argument("--max-inspections", type=int, default=3, help="Maximum selected context paths to read per run.")
    args = parser.parse_args()

    result = evaluate_compression_suite_local_tools(
        repo=args.repo,
        suite_dir=args.suite,
        tasks_path=args.tasks,
        max_inspections=args.max_inspections,
    )
    for row in result["summary"]:
        print(
            f"budget={row['budget']} method={row['method']} "
            f"tool_success={row['tool_success_rate']} "
            f"selected_gold={row['selected_gold_rate']} "
            f"gold_read={row['gold_read_rate']}"
        )


if __name__ == "__main__":
    main()

