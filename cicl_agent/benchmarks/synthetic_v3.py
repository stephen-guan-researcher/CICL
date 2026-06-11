"""Synthetic benchmark v3 with stronger semantic distractors.

v2 alt distractors were BM25-competitive but still scored lower than the
target, so under a tight token budget the heuristic CICL still picked the
target naturally. v3 changes two things to break that:

  1. The target docstring is **minimal** (just function + keyword), removing
     the prose that gave it BM25 length-normalisation advantage.
  2. Each rule gets **two** semantic alt distractors with different hidden
     disqualifiers (staging-only, feature-gated). Both alts pack the task
     template vocabulary so their BM25 scores are at least as high as the
     target's.

The intended effect under budget=120:

  * heuristic CICL is forced to choose between target and alts that look
    equally relevant to lexical retrieval;
  * Opus labels can rank alts as high-negative-transfer based on the
     disqualifier in the docstring;
  * student distilled on those labels should keep target-preference and
     outperform heuristic CICL.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from cicl_agent.core.io import write_jsonl


_ALT_TEMPLATES = [
    (
        "staging-only",
        "Apply {keyword} for {domain} requests: patches the current handler "
        "and recovers the local {domain} rule when the {domain} path is "
        "failing. {function} here runs only in the staging shard "
        "handler_origin; canonical production traffic for {keyword} stays in "
        "{module}.",
    ),
    (
        "feature-gated",
        "Apply {keyword} for {domain} requests: patches the current handler "
        "and recovers the local rule when the {domain} path is failing. This "
        "{function} branch is gated behind feature_flag_{idx:02d}; without "
        "the flag, the project's policy still routes {keyword} through the "
        "canonical handler in {module}.",
    ),
]


def generate_synthetic_benchmark_v3(
    output_dir: str | Path,
    n_rules: int = 50,
    n_eval_per_rule: int = 5,
    seed: int = 7,
) -> tuple[Path, Path]:
    rng = random.Random(seed)
    root = Path(output_dir)
    repo = root / "repo"
    tasks_path = root / "tasks.jsonl"
    repo.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    domains = [
        "timestamp", "pagination", "cache", "retry", "auth", "locale",
        "serialization", "validation", "rate_limit", "webhook", "session", "schema",
    ]
    verbs = ["normalize", "validate", "refresh", "coerce", "resolve", "guard"]

    readme_lines = ["# Synthetic CICL Repo v3 (semantic distractors, minimal target)", ""]
    for idx in range(n_rules):
        domain = domains[idx % len(domains)]
        verb = verbs[idx % len(verbs)]
        function = f"{verb}_{domain}_rule_{idx:02d}"
        module = f"{domain}_rule_{idx:02d}.py"
        stale_module = f"{domain}_rule_{idx:02d}_legacy.py"
        keyword = f"{domain}_policy_{idx:02d}"
        action = f"update {function}"

        # Minimal target docstring: just enough to be retrievable, no prose
        # that would amplify BM25 length advantage over the alts.
        target_doc = f"{function}: applies {keyword}."
        stale_doc = (
            f"Deprecated note {idx:02d}: old {domain} handlers mention {keyword}, "
            f"but this legacy rule is stale and should not guide new fixes."
        )

        readme_lines.append(f"- {function} -> {keyword} (current policy)")

        (repo / module).write_text(
            "\n".join([
                f"def {function}(value):",
                f'    """{target_doc}"""',
                "    return value",
                "",
            ]),
            encoding="utf-8",
        )
        (repo / stale_module).write_text(
            "\n".join([
                f"def legacy_{function}(value):",
                f'    """{stale_doc}"""',
                "    return value",
                "",
            ]),
            encoding="utf-8",
        )

        for alt_idx, (suffix, alt_template) in enumerate(_ALT_TEMPLATES):
            alt_module = f"{domain}_rule_{idx:02d}_{suffix.replace('-', '_')}.py"
            alt_fn = f"{function}_{suffix.replace('-', '_')}"
            alt_doc = alt_template.format(
                idx=idx, domain=domain, keyword=keyword,
                function=function, module=module,
            )
            (repo / alt_module).write_text(
                "\n".join([
                    f"def {alt_fn}(value):",
                    f'    """{alt_doc}"""',
                    "    return value",
                    "",
                ]),
                encoding="utf-8",
            )

        base_id = f"base_{domain}_{idx:02d}"
        symbol_id = f"symbol:{module}:{function}:1"
        file_id = f"file:{module}"
        tasks.append({
            "id": base_id,
            "instance_id": "synthetic_cicl_v3",
            "split": "base",
            "instruction": f"Prior issue taught the current {domain} policy for {keyword}.",
            "memory": f"Local rule {idx:02d}: use {function} for {keyword}. Inspect {module}.",
            "gold_context_ids": [symbol_id, file_id],
            "expected_actions": [f"inspect {module}", action],
            "difficulty": "base",
            "metadata": {"requires_context": True, "domain": domain, "rule_index": idx},
        })

        for eval_idx in range(n_eval_per_rule):
            distractor_domain = domains[(idx + eval_idx + 3) % len(domains)]
            task_id = f"eval_{domain}_{idx:02d}_{eval_idx:02d}"
            instruction_templates = [
                f"A related bug reappeared in {domain}: the agent must apply {keyword} without using stale legacy notes.",
                f"Fix a new {domain} failure. The symptom resembles old {distractor_domain} code, but the needed policy is {keyword}.",
                f"The {domain} path is failing again; recover the local rule and patch the current handler.",
            ]
            tasks.append({
                "id": task_id,
                "instance_id": "synthetic_cicl_v3",
                "split": "eval",
                "instruction": rng.choice(instruction_templates),
                "gold_context_ids": [f"memory:{base_id}", symbol_id, file_id],
                "expected_actions": [f"inspect {module}", action],
                "difficulty": "hard" if eval_idx % 2 else "medium",
                "metadata": {"requires_context": True, "domain": domain, "rule_index": idx},
            })

    (repo / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    write_jsonl(tasks_path, tasks)
    return repo, tasks_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the v3 synthetic CICL benchmark.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rules", type=int, default=50)
    parser.add_argument("--eval-per-rule", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    repo, tasks = generate_synthetic_benchmark_v3(args.output, args.rules, args.eval_per_rule, args.seed)
    print(repo)
    print(tasks)


if __name__ == "__main__":
    main()
