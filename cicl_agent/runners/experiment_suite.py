"""Higher-level experiment suite for budget sweeps and causal diagnostics."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from cicl_agent.agents.simulated import SimulatedReActAgent
from cicl_agent.retrieval.assembly import BudgetAwareContextAssembler
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.core.io import read_jsonl, write_jsonl
from cicl_agent.evaluation.metrics import context_prf, summarize_runs, write_metrics_csv
from cicl_agent.evaluation.reporting import generate_report
from cicl_agent.runners.runner import load_tasks, run_experiment
from cicl_agent.core.schema import AgentRun, Task


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def budget_sweep(
    repo: str | Path,
    tasks: str | Path,
    output: str | Path,
    budgets: list[int],
    learned_policy_path: str | Path | None = None,
    include_llm_judge: bool = False,
    llm_provider: str = "simulator",
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key_env: str = "DASHSCOPE_API_KEY",
    llm_timeout_seconds: float = 60.0,
) -> list[dict]:
    output = Path(output)
    rows: list[dict] = []
    for budget in budgets:
        run_dir = output / f"budget_{budget}"
        metrics = run_experiment(
            repo,
            tasks,
            run_dir,
            budget=budget,
            include_ablations=True,
            learned_policy_path=learned_policy_path,
            include_llm_judge=include_llm_judge,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key_env=llm_api_key_env,
            llm_timeout_seconds=llm_timeout_seconds,
        )
        generate_report(run_dir)
        for row in metrics:
            rows.append({"budget": budget, **row})
    write_csv(output / "budget_sweep_metrics.csv", rows)
    return rows


def learning_curve(
    repo: str | Path,
    tasks: str | Path,
    output: str | Path,
    base_counts: list[int],
    budget: int,
    learned_policy_path: str | Path | None = None,
    include_llm_judge: bool = False,
    llm_provider: str = "simulator",
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key_env: str = "DASHSCOPE_API_KEY",
    llm_timeout_seconds: float = 60.0,
) -> list[dict]:
    output = Path(output)
    all_tasks = load_tasks(tasks)
    base = [task for task in all_tasks if task.split == "base"]
    eval_tasks = [task for task in all_tasks if task.split != "base"]
    rows: list[dict] = []
    for count in base_counts:
        subset = base[:count] + eval_tasks
        task_file = output / f"learning_{count}_base_tasks.jsonl"
        write_jsonl(task_file, (task.to_dict() for task in subset))
        run_dir = output / f"learning_base_{count}"
        metrics = run_experiment(
            repo,
            task_file,
            run_dir,
            budget=budget,
            include_ablations=False,
            learned_policy_path=learned_policy_path,
            include_llm_judge=include_llm_judge,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key_env=llm_api_key_env,
            llm_timeout_seconds=llm_timeout_seconds,
        )
        for row in metrics:
            if row["method"] in {
                "CICL",
                "LLM-CICL",
                "CICL_Distilled",
                "LearnedCICL",
                "VanillaRAG",
                "NoContext",
                "FullContext",
                "GraphMemory",
            }:
                rows.append({"base_tasks": count, **row})
    write_csv(output / "learning_curve_metrics.csv", rows)
    return rows


def causal_removal(repo: str | Path, tasks: str | Path, output: str | Path, budget: int, seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    output = Path(output)
    all_tasks = load_tasks(tasks)
    instance_id = all_tasks[0].instance_id
    graph = InstanceContextGraph(instance_id)
    graph.build_from_repo(repo)
    for task in all_tasks:
        if task.split == "base":
            graph.add_task_memory(task)
    graph.apply_conflict_detection()
    eval_tasks = [task for task in all_tasks if task.split != "base"]
    agent = SimulatedReActAgent()
    history: list[AgentRun] = []
    rows: list[dict] = []
    runs: list[AgentRun] = []

    for task in eval_tasks:
        selected, scores = BudgetAwareContextAssembler(graph, history=history).assemble(task, budget)
        full_run = agent.run(task, "CICL_full", selected)
        runs.append(full_run)
        if not selected:
            continue
        top_id = max(scores, key=lambda score: score.final_score).context_id if scores else selected[0].id
        without_top = [unit for unit in selected if unit.id != top_id]
        top_removed_run = agent.run(task, "CICL_remove_top", without_top)
        random_unit = rng.choice(selected)
        without_random = [unit for unit in selected if unit.id != random_unit.id]
        random_removed_run = agent.run(task, "CICL_remove_random", without_random)
        runs.extend([top_removed_run, random_removed_run])
        for label, run in [
            ("full", full_run),
            ("remove_top", top_removed_run),
            ("remove_random", random_removed_run),
        ]:
            p, r, f1 = context_prf(run.selected_context_ids, task.gold_context_ids)
            rows.append(
                {
                    "task_id": task.id,
                    "condition": label,
                    "success": int(run.success),
                    "context_precision": round(p, 6),
                    "context_recall": round(r, 6),
                    "context_f1": round(f1, 6),
                    "tokens": run.tokens,
                    "removed_top_context": top_id if label == "remove_top" else "",
                    "removed_random_context": random_unit.id if label == "remove_random" else "",
                }
            )
        history.append(full_run)

    write_csv(output / "causal_removal.csv", rows)
    write_metrics_csv(output / "causal_removal_summary.csv", summarize_runs(runs, {task.id: task for task in all_tasks}))
    return rows


def run_full_suite(
    repo: str | Path,
    tasks: str | Path,
    output: str | Path,
    budgets: list[int],
    base_counts: list[int],
    learned_policy_path: str | Path | None = None,
    include_llm_judge: bool = False,
    llm_provider: str = "simulator",
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key_env: str = "DASHSCOPE_API_KEY",
    llm_timeout_seconds: float = 60.0,
) -> None:
    output = Path(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    budget_sweep(
        repo,
        tasks,
        output / "budget_sweep",
        budgets,
        learned_policy_path=learned_policy_path,
        include_llm_judge=include_llm_judge,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key_env=llm_api_key_env,
        llm_timeout_seconds=llm_timeout_seconds,
    )
    learning_curve(
        repo,
        tasks,
        output / "learning_curve",
        base_counts,
        budget=max(budgets),
        learned_policy_path=learned_policy_path,
        include_llm_judge=include_llm_judge,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key_env=llm_api_key_env,
        llm_timeout_seconds=llm_timeout_seconds,
    )
    causal_removal(repo, tasks, output / "causal_removal", budget=max(budgets))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run expanded CICL experiment suite.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", default="80,120,200,400")
    parser.add_argument("--base-counts", default="0,1,3,5,10")
    parser.add_argument("--learned-policy")
    parser.add_argument("--include-llm-judge", action="store_true")
    parser.add_argument("--llm-provider", choices=["simulator", "qwen", "qwen_local", "anthropic", "opus", "claude"], default="simulator")
    parser.add_argument("--llm-model", help="Teacher model name. Defaults to the provider-specific model.")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()
    budgets = [int(value) for value in args.budgets.split(",") if value]
    base_counts = [int(value) for value in args.base_counts.split(",") if value]
    run_full_suite(
        args.repo,
        args.tasks,
        args.output,
        budgets,
        base_counts,
        learned_policy_path=args.learned_policy,
        include_llm_judge=args.include_llm_judge,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key_env=args.llm_api_key_env,
        llm_timeout_seconds=args.llm_timeout_seconds,
    )
    print(args.output)


if __name__ == "__main__":
    main()
