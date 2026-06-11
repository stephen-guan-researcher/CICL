"""Train a learned CICL context policy from intervention labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cicl_agent.core.io import write_jsonl
from cicl_agent.integrations.llm_clients import build_llm_judge, normalize_provider, resolve_llm_model
from cicl_agent.learning.dataset import CausalUtilityDatasetBuilder, LLMJudgmentDatasetBuilder
from cicl_agent.learning.ranker import PairwiseContextRanker
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.runners.runner import load_tasks


def train_policy(
    repo: str | Path,
    tasks_path: str | Path,
    output: str | Path,
    examples_output: str | Path | None = None,
    train_split: str = "base",
    use_all_tasks: bool = False,
    max_train_tasks: int | None = None,
    top_k: int = 20,
    neighbor_depth: int = 1,
    epochs: int = 80,
    lr: float = 0.08,
    label_source: str = "intervention",
    llm_provider: str = "simulator",
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key_env: str = "DASHSCOPE_API_KEY",
    llm_timeout_seconds: float = 60.0,
    llm_max_tokens: int | None = None,
    concurrency: int = 1,
    checkpoint_path: str | Path | None = None,
    max_retries: int = 3,
) -> dict:
    tasks = load_tasks(tasks_path)
    if not tasks:
        raise ValueError(f"No tasks found in {tasks_path}")
    instance_id = tasks[0].instance_id
    graph = InstanceContextGraph(instance_id)
    graph.build_from_repo(repo)
    for task in tasks:
        if task.split == "base":
            graph.add_task_memory(task)
    graph.apply_conflict_detection()

    train_tasks = tasks if use_all_tasks else [task for task in tasks if task.split == train_split]
    if not train_tasks:
        train_tasks = [task for task in tasks if task.split == "base"] or tasks
    if max_train_tasks is not None:
        train_tasks = train_tasks[: max(0, max_train_tasks)]

    judge = None
    llm_provider = normalize_provider(llm_provider)
    resolved_llm_model = resolve_llm_model(llm_provider, llm_model)
    if label_source == "llm":
        judge = build_llm_judge(
            provider=llm_provider,
            model=resolved_llm_model,
            base_url=llm_base_url,
            api_key_env=llm_api_key_env,
            timeout_seconds=llm_timeout_seconds,
            max_tokens=llm_max_tokens,
        )
        examples = LLMJudgmentDatasetBuilder(
            graph,
            judge=judge,
            top_k=top_k,
            neighbor_depth=neighbor_depth,
            concurrency=concurrency,
            checkpoint_path=checkpoint_path,
            max_retries=max_retries,
        ).build(train_tasks)
    elif label_source == "intervention":
        examples = CausalUtilityDatasetBuilder(graph, top_k=top_k, neighbor_depth=neighbor_depth).build(train_tasks)
    else:
        raise ValueError(f"Unsupported label_source: {label_source}")
    ranker_metadata = {"label_source": label_source}
    if label_source == "llm":
        ranker_metadata.update(
            {
                "llm_provider": llm_provider,
                "llm_model": resolved_llm_model if llm_provider != "simulator" else "simulator",
                "llm_base_url": llm_base_url or "",
            }
        )
    ranker = PairwiseContextRanker(metadata=ranker_metadata)
    report = ranker.fit(examples, epochs=epochs, lr=lr)
    ranker.save(output, report=report)
    if examples_output:
        write_jsonl(examples_output, (example.to_dict() for example in examples))

    summary = {
        "policy_path": str(output),
        "examples_path": str(examples_output) if examples_output else "",
        "train_tasks": len(train_tasks),
        "label_source": label_source,
        "llm_provider": llm_provider if label_source == "llm" else "",
        "llm_model": resolved_llm_model if label_source == "llm" else "",
        **report.to_dict(),
    }
    summary_path = Path(output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CICL learned context policy.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--examples-output")
    parser.add_argument("--train-split", default="base")
    parser.add_argument("--use-all-tasks", action="store_true")
    parser.add_argument("--max-train-tasks", type=int, help="Limit train tasks for pilot LLM-label subsets.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--neighbor-depth", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--label-source", choices=["intervention", "llm"], default="intervention")
    parser.add_argument("--llm-provider", choices=["simulator", "qwen", "qwen_local", "anthropic", "opus", "claude"], default="simulator")
    parser.add_argument("--llm-model", help="Teacher model name. Defaults to the provider-specific model.")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--llm-max-tokens", type=int)
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel Opus/Qwen calls.")
    parser.add_argument("--checkpoint-path", help="Resumable jsonl checkpoint for LLM labels.")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()
    summary = train_policy(
        repo=args.repo,
        tasks_path=args.tasks,
        output=args.output,
        examples_output=args.examples_output,
        train_split=args.train_split,
        use_all_tasks=args.use_all_tasks,
        max_train_tasks=args.max_train_tasks,
        top_k=args.top_k,
        neighbor_depth=args.neighbor_depth,
        epochs=args.epochs,
        lr=args.lr,
        label_source=args.label_source,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key_env=args.llm_api_key_env,
        llm_timeout_seconds=args.llm_timeout_seconds,
        llm_max_tokens=args.llm_max_tokens,
        concurrency=args.concurrency,
        checkpoint_path=args.checkpoint_path,
        max_retries=args.max_retries,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
