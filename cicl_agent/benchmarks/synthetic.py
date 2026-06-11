"""Synthetic benchmark generator for CICL sanity experiments.

The generator creates repo-local rules, matching base memories, related eval
tasks, and stale distractor files. It is not a replacement for SWE-ContextBench
or ContextBench; it is a controllable smoke test for context-learning claims.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from cicl_agent.core.io import write_jsonl


def generate_synthetic_benchmark(
    output_dir: str | Path,
    n_rules: int = 12,
    n_eval_per_rule: int = 2,
    seed: int = 7,
) -> tuple[Path, Path]:
    rng = random.Random(seed)
    root = Path(output_dir)
    repo = root / "repo"
    tasks_path = root / "tasks.jsonl"
    repo.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    domains = [
        "timestamp",
        "pagination",
        "cache",
        "retry",
        "auth",
        "locale",
        "serialization",
        "validation",
        "rate_limit",
        "webhook",
        "session",
        "schema",
    ]
    verbs = ["normalize", "validate", "refresh", "coerce", "resolve", "guard"]

    readme_lines = ["# Synthetic CICL Repo", "", "This repo contains local project rules."]
    for idx in range(n_rules):
        domain = domains[idx % len(domains)]
        verb = verbs[idx % len(verbs)]
        function = f"{verb}_{domain}_rule_{idx:02d}"
        module = f"{domain}_rule_{idx:02d}.py"
        stale_module = f"{domain}_rule_{idx:02d}_legacy.py"
        keyword = f"{domain}_policy_{idx:02d}"
        action = f"update {function}"

        correct_rule = (
            f"Local rule {idx:02d}: when handling {domain} requests, use {function} "
            f"because {keyword} is the only current project policy."
        )
        stale_rule = (
            f"Deprecated note {idx:02d}: old {domain} handlers mention {keyword}, "
            f"but this legacy rule is stale and should not guide new fixes."
        )
        readme_lines.append(f"- {correct_rule}")
        (repo / module).write_text(
            "\n".join(
                [
                    f"def {function}(value):",
                    f'    """{correct_rule}"""',
                    "    return value",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (repo / stale_module).write_text(
            "\n".join(
                [
                    f"def legacy_{function}(value):",
                    f'    """{stale_rule}"""',
                    "    return value",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        base_id = f"base_{domain}_{idx:02d}"
        symbol_id = f"symbol:{module}:{function}:1"
        file_id = f"file:{module}"
        tasks.append(
            {
                "id": base_id,
                "instance_id": "synthetic_cicl",
                "split": "base",
                "instruction": f"Prior issue taught the current {domain} policy for {keyword}.",
                "memory": f"{correct_rule} Inspect {module} and {action} for related tasks.",
                "gold_context_ids": [symbol_id, file_id],
                "expected_actions": [f"inspect {module}", action],
                "difficulty": "base",
                "metadata": {"requires_context": True, "domain": domain, "rule_index": idx},
            }
        )

        for eval_idx in range(n_eval_per_rule):
            distractor_domain = domains[(idx + eval_idx + 3) % len(domains)]
            task_id = f"eval_{domain}_{idx:02d}_{eval_idx:02d}"
            instruction_templates = [
                f"A related bug reappeared in {domain}: the agent must apply {keyword} without using stale legacy notes.",
                f"Fix a new {domain} failure. The symptom resembles old {distractor_domain} code, but the needed policy is {keyword}.",
                f"The {domain} path is failing again; recover the local rule and patch the current handler.",
            ]
            tasks.append(
                {
                    "id": task_id,
                    "instance_id": "synthetic_cicl",
                    "split": "eval",
                    "instruction": rng.choice(instruction_templates),
                    "gold_context_ids": [f"memory:{base_id}", symbol_id, file_id],
                    "expected_actions": [f"inspect {module}", action],
                    "difficulty": "hard" if eval_idx % 2 else "medium",
                    "metadata": {"requires_context": True, "domain": domain, "rule_index": idx},
                }
            )

    (repo / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    write_jsonl(tasks_path, tasks)
    return repo, tasks_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic CICL benchmark.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rules", type=int, default=12)
    parser.add_argument("--eval-per-rule", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    repo, tasks = generate_synthetic_benchmark(args.output, args.rules, args.eval_per_rule, args.seed)
    print(repo)
    print(tasks)


if __name__ == "__main__":
    main()

