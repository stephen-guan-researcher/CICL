"""Evaluate selection-level agreement between an LLM judge and teacher labels."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import OrderedDict
from pathlib import Path

from cicl_agent.causal.judgment import CounterfactualContextJudgment, judgment_to_score
from cicl_agent.core.io import append_jsonl, read_jsonl, write_jsonl
from cicl_agent.integrations.llm_clients import build_llm_judge, normalize_provider, resolve_llm_model
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.runners.runner import load_tasks


NUMERIC_FIELDS = (
    "action_shift",
    "necessity",
    "expected_outcome_uplift",
    "negative_transfer_risk",
    "confidence",
)


def evaluate_llm_agreement(
    repo: str | Path,
    tasks_path: str | Path,
    teacher_examples: str | Path,
    output: str | Path,
    judgments_output: str | Path | None = None,
    llm_provider: str = "simulator",
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key_env: str = "DASHSCOPE_API_KEY",
    llm_timeout_seconds: float = 60.0,
    llm_max_tokens: int | None = None,
    max_tasks: int | None = None,
    max_candidates_per_task: int | None = None,
    top_k: int = 5,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Compare a candidate judge to teacher labels on the same contexts.

    The function is intentionally provider-agnostic. Passing
    ``llm_provider="qwen"`` runs the OpenAI-compatible DashScope/Qwen client,
    ``qwen_local`` runs a local Qwen LoRA adapter, and ``simulator`` provides a
    deterministic smoke path for CI and machines without model credentials.
    """

    tasks = load_tasks(tasks_path)
    if not tasks:
        raise ValueError(f"No tasks found in {tasks_path}")
    task_map = {task.id: task for task in tasks}
    graph = InstanceContextGraph(tasks[0].instance_id)
    graph.build_from_repo(repo)
    for task in tasks:
        if task.split == "base":
            graph.add_task_memory(task)
    graph.apply_conflict_detection()

    rows_by_task = _group_teacher_examples(teacher_examples)
    sampled_task_ids = [task_id for task_id in rows_by_task if task_id in task_map]
    if max_tasks is not None:
        sampled_task_ids = sampled_task_ids[: max(0, max_tasks)]
    if not sampled_task_ids:
        raise ValueError("No teacher-example task ids matched the task file.")

    provider = normalize_provider(llm_provider)
    resolved_model = resolve_llm_model(provider, llm_model)
    judge = build_llm_judge(
        provider=provider,
        model=resolved_model,
        base_url=llm_base_url,
        api_key_env=llm_api_key_env,
        timeout_seconds=llm_timeout_seconds,
        max_tokens=llm_max_tokens,
    )
    cached = _load_checkpoint(checkpoint_path)

    per_candidate_rows: list[dict] = []
    per_task: list[dict] = []
    n_attempted = 0
    n_parsed = 0
    n_errors = 0
    field_errors: dict[str, list[float]] = {field: [] for field in NUMERIC_FIELDS}
    t0 = time.time()

    for task_id in sampled_task_ids:
        task = task_map[task_id]
        teacher_rows = rows_by_task[task_id]
        if max_candidates_per_task is not None:
            teacher_rows = teacher_rows[: max(0, max_candidates_per_task)]
        teacher_scores: list[tuple[str, float]] = []
        target_scores: list[tuple[str, float]] = []

        for example in teacher_rows:
            context_id = example["context_id"]
            unit = graph.units.get(context_id)
            if unit is None:
                per_candidate_rows.append(
                    {
                        "task_id": task_id,
                        "context_id": context_id,
                        "ok": False,
                        "error": "context_id_missing_from_graph",
                    }
                )
                n_errors += 1
                continue

            teacher_judgment = teacher_judgment_from_example(example)
            teacher_score = judgment_to_score(teacher_judgment, unit).final_score
            teacher_scores.append((context_id, teacher_score))

            cache_key = f"{task_id}\t{context_id}"
            cached_row = cached.get(cache_key)
            if cached_row is not None:
                row = cached_row
            else:
                row = _judge_candidate(judge, task, unit, teacher_judgment, teacher_score)
                if checkpoint_path is not None:
                    append_jsonl(checkpoint_path, row)
            per_candidate_rows.append(row)
            n_attempted += 1
            if not row.get("ok"):
                n_errors += 1
                continue
            n_parsed += 1
            target_score = float(row["target_score"])
            target_scores.append((context_id, target_score))
            target = row["target_judgment"]
            for field in NUMERIC_FIELDS:
                field_errors[field].append(abs(float(target[field]) - getattr(teacher_judgment, field)))

        task_report = _task_agreement(task_id, teacher_scores, target_scores, top_k)
        per_task.append(task_report)

    elapsed = time.time() - t0
    summary = {
        "teacher_examples": str(teacher_examples),
        "tasks_path": str(tasks_path),
        "repo": str(repo),
        "judge_provider": provider,
        "judge_model": resolved_model if provider != "simulator" else "simulator",
        "n_tasks": len(per_task),
        "top_k": top_k,
        "n_attempted": n_attempted,
        "n_parsed": n_parsed,
        "n_errors": n_errors,
        "parse_rate": n_parsed / n_attempted if n_attempted else None,
        "field_mae_vs_teacher_on_same_candidates": {
            field: {
                "mae": (sum(values) / len(values) if values else None),
                "n": len(values),
            }
            for field, values in field_errors.items()
        },
        "avg_top_k_jaccard": _mean(
            row["top_k_jaccard"] for row in per_task if row["top_k_jaccard"] is not None
        ),
        "avg_spearman_rho": _mean(
            row["spearman_rho"] for row in per_task if row["spearman_rho"] is not None
        ),
        "elapsed_sec": round(elapsed, 2),
        "per_task": per_task,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if judgments_output:
        write_jsonl(judgments_output, per_candidate_rows)
    return summary


def teacher_judgment_from_example(example: dict) -> CounterfactualContextJudgment:
    meta = example.get("metadata") or {}
    return CounterfactualContextJudgment(
        task_id=example["task_id"],
        context_id=example["context_id"],
        no_context_action=meta.get("no_context_action", ""),
        with_context_action=meta.get("with_context_action", ""),
        action_shift=float(meta.get("llm_action_shift", example.get("action_delta", 0.0))),
        necessity=float(meta.get("llm_necessity", 0.0)),
        expected_outcome_uplift=float(meta.get("llm_expected_uplift", example.get("outcome_delta", 0.0))),
        negative_transfer_risk=float(
            meta.get("llm_negative_transfer", example.get("negative_transfer", 0.0))
        ),
        reason=meta.get("reason", ""),
        confidence=float(meta.get("llm_confidence", 0.7)),
    )


def _judge_candidate(
    judge,
    task,
    unit,
    teacher_judgment: CounterfactualContextJudgment,
    teacher_score: float,
) -> dict:
    started = time.time()
    row = {
        "task_id": task.id,
        "context_id": unit.id,
        "teacher_judgment": teacher_judgment.to_dict(),
        "teacher_score": teacher_score,
    }
    try:
        target_judgment = judge.judge(task, unit)
        target_score = judgment_to_score(target_judgment, unit).final_score
    except Exception as exc:  # noqa: BLE001
        row.update(
            {
                "ok": False,
                "error": str(exc)[:500],
                "latency_sec": round(time.time() - started, 3),
            }
        )
        return row
    row.update(
        {
            "ok": True,
            "target_judgment": target_judgment.to_dict(),
            "target_score": target_score,
            "latency_sec": round(time.time() - started, 3),
        }
    )
    return row


def _group_teacher_examples(path: str | Path) -> OrderedDict[str, list[dict]]:
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for row in read_jsonl(path):
        grouped.setdefault(row["task_id"], []).append(row)
    return grouped


def _load_checkpoint(path: str | Path | None) -> dict[str, dict]:
    if path is None or not Path(path).exists():
        return {}
    cached: dict[str, dict] = {}
    for row in read_jsonl(path):
        task_id = row.get("task_id")
        context_id = row.get("context_id")
        if task_id and context_id:
            cached[f"{task_id}\t{context_id}"] = row
    return cached


def _task_agreement(
    task_id: str,
    teacher_scores: list[tuple[str, float]],
    target_scores: list[tuple[str, float]],
    top_k: int,
) -> dict:
    teacher_top = {
        cid for cid, _ in sorted(teacher_scores, key=lambda item: (-item[1], item[0]))[:top_k]
    }
    target_top = {
        cid for cid, _ in sorted(target_scores, key=lambda item: (-item[1], item[0]))[:top_k]
    }
    jaccard = None
    if target_top:
        jaccard = len(teacher_top & target_top) / max(1, len(teacher_top | target_top))
    teacher_by_cid = dict(teacher_scores)
    target_by_cid = dict(target_scores)
    common = [cid for cid in teacher_by_cid if cid in target_by_cid]
    rho = None
    if len(common) >= 2:
        rho = spearman([teacher_by_cid[cid] for cid in common], [target_by_cid[cid] for cid in common])
        if math.isnan(rho):
            rho = None
    return {
        "task_id": task_id,
        "n_teacher_candidates": len(teacher_scores),
        "n_target_judged": len(target_scores),
        "top_k_jaccard": jaccard,
        "spearman_rho": rho,
        "teacher_top": sorted(teacher_top),
        "target_top": sorted(target_top),
    }


def spearman(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n != len(y) or n < 2:
        return float("nan")
    rx = _ranks(x)
    ry = _ranks(y)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in rx) * sum((b - mean_y) ** 2 for b in ry)
    )
    return numerator / denominator if denominator > 0 else float("nan")


def _ranks(values: list[float]) -> list[float]:
    n = len(values)
    order = sorted(range(n), key=lambda index: values[index])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _mean(values) -> float | None:
    collected = [value for value in values if value is not None and not math.isnan(value)]
    return sum(collected) / len(collected) if collected else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM judge agreement with teacher labels.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--teacher-examples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judgments-output")
    parser.add_argument(
        "--llm-provider",
        choices=["simulator", "qwen", "qwen_local", "anthropic", "opus", "claude", "codex", "gpt5.5"],
        default="simulator",
    )
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--llm-max-tokens", type=int)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-candidates-per-task", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--checkpoint-path")
    args = parser.parse_args()
    summary = evaluate_llm_agreement(
        repo=args.repo,
        tasks_path=args.tasks,
        teacher_examples=args.teacher_examples,
        output=args.output,
        judgments_output=args.judgments_output,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key_env=args.llm_api_key_env,
        llm_timeout_seconds=args.llm_timeout_seconds,
        llm_max_tokens=args.llm_max_tokens,
        max_tasks=args.max_tasks,
        max_candidates_per_task=args.max_candidates_per_task,
        top_k=args.top_k,
        checkpoint_path=args.checkpoint_path,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
