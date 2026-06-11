"""Synthetic benchmark v2 with semantic distractors.

v1 distractors were obvious (`_legacy.py` + "Deprecated"/"stale" in docstring),
so heuristic CICL with path/name/Jaccard features already separated targets
from distractors. v2 keeps the v1 target and the v1 legacy distractor, and
adds a **semantic** distractor that:

  * shares the same rule index (path/name match cannot tell it from target);
  * mentions the task keywords with equal or higher token overlap
    (Jaccard cannot tell it from target);
  * contains a disqualifying condition that requires reading the docstring
    semantically (only LLM judges should catch it).

The intended effect: heuristic CICL F1 should drop on v2; an Opus-distilled
student should — if the distillation captures the LLM's semantic separation —
retain a higher F1 than the heuristic.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from cicl_agent.core.io import write_jsonl


_ALT_TEMPLATES = [
    # Each template repeats the task-template vocabulary densely so the alt's
    # BM25/Jaccard score is at least as high as the target's, and hides a
    # disqualifier that only semantic reading can catch. Disqualifiers avoid
    # ConflictDetector keywords (deprecated/legacy/stale/obsolete/...) so the
    # heuristic stale-detection layer cannot filter them out for free.
    (
        "staging-only",
        "Apply {keyword} when handling {domain} requests: this rule patches "
        "the current handler and recovers the local {domain} rule when the "
        "{domain} path is failing. {function} runs only inside the staging "
        "shard handler_origin; production traffic for {keyword} must call the "
        "canonical handler in {module} via {function}.",
    ),
    (
        "shadow-traffic",
        "Apply {keyword} for {domain} requests: patches the current handler "
        "and recovers the local rule when the {domain} path is failing. This "
        "{function} branch is wired only into the shadow-traffic mirror; the "
        "needed policy for production {domain} fixes is still {function} in "
        "{module}.",
    ),
    (
        "feature-gate",
        "Apply {keyword} to handle {domain} requests: recovers the local "
        "{domain} rule and patches the current handler when the {domain} path "
        "is failing. {function} here is gated behind the experimental "
        "feature_flag_{idx:02d}; without that flag, the needed policy is the "
        "canonical {function} defined in {module}.",
    ),
    (
        "mirror-fork",
        "Apply {keyword} for {domain} requests: patches the current handler "
        "and recovers the local rule when the {domain} path is failing. This "
        "{function} is a mirror-fork kept in tree for an internal consumer; "
        "the project's current policy still routes {keyword} fixes through "
        "the canonical {function} in {module}.",
    ),
]


def generate_synthetic_benchmark_v2(
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

    readme_lines = ["# Synthetic CICL Repo v2 (semantic distractors)", "",
                    "Each rule has a target, a legacy distractor (v1-compatible) and a semantic alt distractor."]
    for idx in range(n_rules):
        domain = domains[idx % len(domains)]
        verb = verbs[idx % len(verbs)]
        function = f"{verb}_{domain}_rule_{idx:02d}"
        module = f"{domain}_rule_{idx:02d}.py"
        stale_module = f"{domain}_rule_{idx:02d}_legacy.py"
        alt_module = f"{domain}_rule_{idx:02d}_alt.py"
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

        _, alt_template = _ALT_TEMPLATES[idx % len(_ALT_TEMPLATES)]
        alt_doc = alt_template.format(
            idx=idx, domain=domain, keyword=keyword,
            function=function, module=module,
        )

        readme_lines.append(f"- {correct_rule}")

        (repo / module).write_text(
            "\n".join([
                f"def {function}(value):",
                f'    """{correct_rule}"""',
                "    return value",
                "",
            ]),
            encoding="utf-8",
        )
        (repo / stale_module).write_text(
            "\n".join([
                f"def legacy_{function}(value):",
                f'    """{stale_rule}"""',
                "    return value",
                "",
            ]),
            encoding="utf-8",
        )
        # Semantic distractor: same path/name index, same keyword density, but
        # a disqualifying condition only readable via the docstring.
        (repo / alt_module).write_text(
            "\n".join([
                f"def {function}_alt(value):",
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
            "instance_id": "synthetic_cicl_v2",
            "split": "base",
            "instruction": f"Prior issue taught the current {domain} policy for {keyword}.",
            "memory": f"{correct_rule} Inspect {module} and {action} for related tasks.",
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
                "instance_id": "synthetic_cicl_v2",
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
    parser = argparse.ArgumentParser(description="Generate the v2 synthetic CICL benchmark with semantic distractors.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rules", type=int, default=50)
    parser.add_argument("--eval-per-rule", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    repo, tasks = generate_synthetic_benchmark_v2(args.output, args.rules, args.eval_per_rule, args.seed)
    print(repo)
    print(tasks)


if __name__ == "__main__":
    main()
