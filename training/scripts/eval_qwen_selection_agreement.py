"""Selection-level agreement between the Qwen LoRA judge and the Opus teacher.

For each sampled task we already have all candidate (task, context) judgments
from Opus in ``llm_examples.clean.jsonl``. We re-judge the SAME candidates
with the local Qwen adapter and compare:

  - per-field MAE (action_shift, necessity, expected_outcome_uplift,
    negative_transfer_risk, confidence)
  - Spearman rank correlation between Opus's final_score and Qwen's
    final_score over the candidate set of each task
  - Jaccard overlap of top-K selected context ids (K=5)
  - JSON parse rate

This is the missing piece for the paper's Qwen-as-judge end-to-end claim:
field-level MAE was already shown on the val split, this adds *selection-level*
agreement on the same candidate pool the Opus teacher saw.

Usage:
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m \
        training.scripts.eval_qwen_selection_agreement \
        --repo experiments/data/synthetic/v1/repo \
        --tasks experiments/data/synthetic/v1/tasks.jsonl \
        --opus artifacts/outputs/latest/synthetic_opus_v1/llm_examples.clean.jsonl \
        --num-tasks 5 \
        --output artifacts/outputs/latest/qwen_local_eval/selection_agreement.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from cicl_agent.causal.judgment import (
    CounterfactualContextJudgment,
    judgment_to_score,
)
from cicl_agent.causal.llm_judge import CounterfactualPromptTemplate
from cicl_agent.core.io import read_jsonl
from cicl_agent.core.schema import Task
from cicl_agent.integrations.llm_clients import QwenLocalAdapterClient
from cicl_agent.memory.graph import InstanceContextGraph


NUMERIC_FIELDS = (
    "action_shift",
    "necessity",
    "expected_outcome_uplift",
    "negative_transfer_risk",
    "confidence",
)


def opus_judgment_from_example(example: dict) -> CounterfactualContextJudgment:
    meta = example.get("metadata") or {}
    return CounterfactualContextJudgment(
        task_id=example["task_id"],
        context_id=example["context_id"],
        no_context_action=meta.get("no_context_action", ""),
        with_context_action=meta.get("with_context_action", ""),
        action_shift=float(meta.get("llm_action_shift", example.get("action_delta", 0.0))),
        necessity=float(meta.get("llm_necessity", 0.0)),
        expected_outcome_uplift=float(
            meta.get("llm_expected_uplift", example.get("outcome_delta", 0.0))
        ),
        negative_transfer_risk=float(
            meta.get("llm_negative_transfer", example.get("negative_transfer", 0.0))
        ),
        reason=meta.get("reason", ""),
        confidence=float(meta.get("llm_confidence", 0.7)),
    )


def parse_qwen_json(text: str) -> dict | None:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    fragment = text[start : end + 1]
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        return None


def spearman(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg_rank
            i = j + 1
        return out

    rx, ry = ranks(x), ranks(y)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den = math.sqrt(
        sum((a - mean_x) ** 2 for a in rx) * sum((b - mean_y) ** 2 for b in ry)
    )
    return num / den if den > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--opus", required=True, help="Opus llm_examples.clean.jsonl")
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-input-tokens", type=int, default=896)
    parser.add_argument(
        "--adapter", default="XinyuGuan/CICL"
    )
    parser.add_argument(
        "--base", default="Qwen/Qwen3.5-9B"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Build graph and task map for prompt rendering
    tasks = [Task.from_dict(row) for row in read_jsonl(args.tasks)]
    task_map = {t.id: t for t in tasks}
    base_tasks = [t for t in tasks if t.split == "base"]
    graph = InstanceContextGraph(tasks[0].instance_id)
    graph.build_from_repo(args.repo)
    for t in base_tasks:
        graph.add_task_memory(t)
    graph.apply_conflict_detection()

    units_by_id = {uid: unit for uid, unit in graph.units.items()}

    # Group Opus examples by task
    opus_by_task: dict[str, list[dict]] = {}
    for ex in read_jsonl(args.opus):
        opus_by_task.setdefault(ex["task_id"], []).append(ex)

    sampled = list(opus_by_task.keys())[: args.num_tasks]
    print(f"[setup] {len(sampled)} tasks sampled, "
          f"avg candidates={sum(len(opus_by_task[t]) for t in sampled)/len(sampled):.1f}")

    client = QwenLocalAdapterClient(
        base_path=args.base,
        adapter_path=args.adapter,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
    )
    template = CounterfactualPromptTemplate()

    per_task = []
    n_calls = 0
    n_parsed = 0
    t0 = time.time()
    field_errors: dict[str, list[float]] = {f: [] for f in NUMERIC_FIELDS}

    for task_id in sampled:
        task = task_map.get(task_id)
        if task is None:
            print(f"[skip] task {task_id} not in task file")
            continue
        opus_rows = opus_by_task[task_id]
        opus_scores: list[tuple[str, float]] = []
        qwen_scores: list[tuple[str, float]] = []
        qwen_judgments: list[dict] = []

        for ex in opus_rows:
            cid = ex["context_id"]
            unit = units_by_id.get(cid)
            if unit is None:
                # Skip candidates whose unit isn't in the rebuilt graph
                continue
            opus_judgment = opus_judgment_from_example(ex)
            opus_score = judgment_to_score(opus_judgment, unit).final_score
            opus_scores.append((cid, opus_score))

            prompt = template.render(task, unit)
            t1 = time.time()
            try:
                raw = client.complete(prompt)
            except Exception as exc:  # noqa: BLE001
                print(f"[error] {task_id} {cid}: {exc}")
                continue
            n_calls += 1
            parsed = parse_qwen_json(raw)
            if parsed is None:
                continue
            n_parsed += 1
            parsed["task_id"] = task_id
            parsed["context_id"] = cid
            try:
                qwen_judgment = CounterfactualContextJudgment.from_dict(parsed)
            except Exception:
                continue
            qwen_score = judgment_to_score(qwen_judgment, unit).final_score
            qwen_scores.append((cid, qwen_score))
            qwen_judgments.append({
                "context_id": cid,
                "raw": raw,
                "parsed": parsed,
                "opus": opus_judgment.to_dict(),
                "qwen_score": qwen_score,
                "opus_score": opus_score,
                "latency_sec": round(time.time() - t1, 3),
            })
            for f in NUMERIC_FIELDS:
                field_errors[f].append(
                    abs(float(parsed.get(f, 0.0)) - getattr(opus_judgment, f))
                )

        # Top-K agreement
        if not qwen_scores:
            continue
        opus_top = {cid for cid, _ in sorted(opus_scores, key=lambda x: -x[1])[: args.top_k]}
        qwen_top = {cid for cid, _ in sorted(qwen_scores, key=lambda x: -x[1])[: args.top_k]}
        jaccard = (
            len(opus_top & qwen_top) / max(1, len(opus_top | qwen_top))
        )
        # Rank correlation on candidates present in both
        opus_by_cid = {cid: s for cid, s in opus_scores}
        common = [cid for cid, _ in qwen_scores if cid in opus_by_cid]
        rho = spearman(
            [opus_by_cid[c] for c in common],
            [next(s for cid, s in qwen_scores if cid == c) for c in common],
        )
        per_task.append({
            "task_id": task_id,
            "n_candidates": len(opus_scores),
            "n_qwen_judged": len(qwen_scores),
            "top_k_jaccard": jaccard,
            "spearman_rho": rho,
            "opus_top": sorted(opus_top),
            "qwen_top": sorted(qwen_top),
        })
        print(
            f"[task] {task_id} "
            f"n_qwen={len(qwen_scores)}/{len(opus_scores)} "
            f"top{args.top_k}_jaccard={jaccard:.2f} rho={rho:.2f}"
        )

    elapsed = time.time() - t0
    field_mae = {
        f: (sum(v) / len(v) if v else None, len(v)) for f, v in field_errors.items()
    }
    summary = {
        "tasks_evaluated": [p["task_id"] for p in per_task],
        "n_tasks": len(per_task),
        "top_k": args.top_k,
        "n_qwen_calls": n_calls,
        "n_parsed": n_parsed,
        "parse_rate": (n_parsed / n_calls) if n_calls else None,
        "field_mae_vs_opus_on_same_candidates": {
            f: {"mae": v[0], "n": v[1]} for f, v in field_mae.items()
        },
        "avg_top_k_jaccard": (
            sum(p["top_k_jaccard"] for p in per_task) / len(per_task)
            if per_task else None
        ),
        "avg_spearman_rho": (
            sum(p["spearman_rho"] for p in per_task if not math.isnan(p["spearman_rho"]))
            / max(1, sum(1 for p in per_task if not math.isnan(p["spearman_rho"])))
            if per_task else None
        ),
        "elapsed_sec": round(elapsed, 2),
        "per_task": per_task,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(
        f"[done] tasks={len(per_task)} calls={n_calls} parse_rate="
        f"{summary['parse_rate']} avg_top{args.top_k}_jaccard="
        f"{summary['avg_top_k_jaccard']} elapsed={elapsed:.1f}s -> {out}"
    )


if __name__ == "__main__":
    main()
