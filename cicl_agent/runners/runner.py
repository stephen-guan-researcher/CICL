"""Command-line experiment runner for CICL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cicl_agent.agents.simulated import SimulatedReActAgent
from cicl_agent.core.io import append_jsonl, read_jsonl
from cicl_agent.core.schema import AgentRun, Task
from cicl_agent.evaluation.logger import TrajectoryLogger
from cicl_agent.evaluation.metrics import summarize_runs, write_metrics_csv
from cicl_agent.integrations.llm_clients import build_llm_judge, normalize_provider, resolve_llm_model
from cicl_agent.learning.ranker import PairwiseContextRanker
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.selectors import default_selectors


def load_tasks(path: str | Path) -> list[Task]:
    return [Task.from_dict(row) for row in read_jsonl(path)]


def run_experiment(
    repo: str | Path,
    tasks_path: str | Path,
    output: str | Path,
    budget: int,
    include_ablations: bool = False,
    eval_offset: int = 0,
    max_eval_tasks: int | None = None,
    learned_policy_path: str | Path | None = None,
    include_llm_judge: bool = False,
    llm_provider: str = "simulator",
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key_env: str = "DASHSCOPE_API_KEY",
    llm_timeout_seconds: float = 60.0,
) -> list[dict]:
    tasks = load_tasks(tasks_path)
    if not tasks:
        raise ValueError(f"No tasks found in {tasks_path}")

    instance_id = tasks[0].instance_id
    graph = InstanceContextGraph(instance_id)
    graph.build_from_repo(repo)

    base_tasks = [task for task in tasks if task.split == "base"]
    eval_tasks = [task for task in tasks if task.split != "base"]
    if eval_offset:
        eval_tasks = eval_tasks[max(0, eval_offset) :]
    if max_eval_tasks is not None:
        eval_tasks = eval_tasks[: max(0, max_eval_tasks)]
    for task in base_tasks:
        graph.add_task_memory(task)
    graph.apply_conflict_detection()

    output = Path(output)
    graph.save(output)

    agent = SimulatedReActAgent()
    logger = TrajectoryLogger(output)
    learned_ranker = _load_ranker(learned_policy_path) if learned_policy_path else None
    llm_provider = normalize_provider(llm_provider)
    resolved_llm_model = resolve_llm_model(llm_provider, llm_model)
    llm_judge = (
        build_llm_judge(
            provider=llm_provider,
            model=resolved_llm_model,
            base_url=llm_base_url,
            api_key_env=llm_api_key_env,
            timeout_seconds=llm_timeout_seconds,
        )
        if include_llm_judge
        else None
    )
    selectors = default_selectors(
        graph,
        include_ablations=include_ablations,
        learned_ranker=learned_ranker,
        include_llm_judge=include_llm_judge,
        llm_judge=llm_judge,
    )
    all_runs: list[AgentRun] = []

    for task in eval_tasks:
        history_for_task = list(all_runs)
        for selector in selectors:
            selected_units = selector.select(task, budget=budget, history=history_for_task)
            run = agent.run(task, method=selector.name, selected_units=selected_units)
            all_runs.append(run)
            logger.log_run(run, task, selected_units)
            update = getattr(selector, "update", None)
            if callable(update):
                update(task, selected_units, run)
            for score in getattr(selector, "last_scores", []):
                append_jsonl(output / "causal_scores.jsonl", score.to_dict())
            if selector.name == "CICL":
                graph.add_run_memory(run, task)
                graph.apply_conflict_detection()

    task_map = {task.id: task for task in tasks}
    rows = summarize_runs(all_runs, task_map)
    write_metrics_csv(output / "metrics.csv", rows)
    graph.save(output)
    return rows


def _load_ranker(path: str | Path) -> object:
    """Load learned policies while keeping torch optional for non-MLP runs."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model_type = payload.get("model_type", "pairwise_linear_context_ranker")
    if model_type == "pairwise_mlp_context_ranker":
        try:
            from cicl_agent.learning.mlp_ranker import MLPContextRanker
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise RuntimeError(
                    "Loading an MLP learned policy requires torch. Install torch or use a linear policy checkpoint."
                ) from exc
            raise
        return MLPContextRanker.load(path)
    return PairwiseContextRanker.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CICL context-learning experiment.")
    parser.add_argument("--repo", required=True, help="Repository or environment instance to index.")
    parser.add_argument("--tasks", required=True, help="JSONL task file.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--budget", type=int, default=4000, help="Context token budget.")
    parser.add_argument("--include-ablations", action="store_true", help="Run CICL ablation selectors.")
    parser.add_argument("--eval-offset", type=int, default=0, help="Skip this many eval tasks before running.")
    parser.add_argument("--max-eval-tasks", type=int, help="Limit eval tasks for pilot runs.")
    parser.add_argument("--learned-policy", help="Optional path to a trained pairwise context policy.")
    parser.add_argument("--include-llm-judge", action="store_true", help="Run LLM-CICL counterfactual judge reranker.")
    parser.add_argument("--llm-provider", choices=["simulator", "qwen", "qwen_local", "anthropic", "opus", "claude"], default="simulator")
    parser.add_argument("--llm-model", help="Teacher model name. Defaults to the provider-specific model.")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    rows = run_experiment(
        args.repo,
        args.tasks,
        args.output,
        args.budget,
        args.include_ablations,
        eval_offset=args.eval_offset,
        max_eval_tasks=args.max_eval_tasks,
        learned_policy_path=args.learned_policy,
        include_llm_judge=args.include_llm_judge,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key_env=args.llm_api_key_env,
        llm_timeout_seconds=args.llm_timeout_seconds,
    )
    for row in rows:
        print(
            f"{row['method']}: success={row['success_rate']} "
            f"f1={row['context_f1']} tokens={row['avg_tokens']}"
        )


if __name__ == "__main__":
    main()
