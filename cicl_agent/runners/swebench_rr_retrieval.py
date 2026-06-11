"""SWE-bench Verified-RR retrieval benchmark runner.

This runner evaluates context retrieval directly. It does not execute patches
or run a coding agent. The benchmark question is: given an issue, can a method
rank the gold context document highly among candidate repository snippets?
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from cicl_agent.benchmarks.download import fetch_hf_rows
from cicl_agent.causal.judgment import judgment_to_score
from cicl_agent.causal.llm_judge import CodeRetrievalPromptTemplate
from cicl_agent.core.io import append_jsonl, read_jsonl, write_jsonl
from cicl_agent.core.schema import ContextUnit, Task
from cicl_agent.integrations.llm_clients import build_llm_judge, normalize_provider, resolve_llm_model
from cicl_agent.retrieval.retrievers import BM25Retriever, HashedEmbeddingRetriever, HybridRetriever
from cicl_agent.runners.runner import load_tasks


DEFAULT_KS = [1, 5, 10, 20]
CAUSAL_RERANK_METHODS = {"CausalRerank", "CausalHybridRerank"}


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


def write_markdown_table(path: str | Path, rows: list[dict], title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def hydrate_candidate_corpus(
    *,
    top_ranked_path: str | Path,
    output_path: str | Path,
    candidate_limit: int,
    max_queries: int | None = None,
    allow_partial: bool = False,
    refresh: bool = False,
) -> Path:
    """Fetch top-ranked candidate texts for each sampled query.

    The HF rows endpoint is offset-based. In this dataset, the corpus rows are
    ordered by the same query groups listed in `top_ranked`, so cumulative
    lengths give a stable offset for each query's candidate block.
    """

    top_ranked_path = Path(top_ranked_path)
    output_path = Path(output_path)
    if output_path.exists() and not refresh:
        return output_path

    rows = list(read_jsonl(top_ranked_path))
    if max_queries is not None:
        rows = rows[:max_queries]
    hydrated: list[dict] = []
    offset = 0
    for row in rows:
        candidate_ids = [str(item) for item in row.get("corpus-ids", []) or []]
        if not candidate_ids:
            continue
        fetch_len = min(candidate_limit, len(candidate_ids))
        try:
            corpus_rows, _ = fetch_hf_rows("mteb/SWEbenchVerifiedRR", "corpus", "train", fetch_len, offset=offset)
        except Exception:
            if allow_partial and hydrated:
                break
            raise
        expected = set(candidate_ids[:fetch_len])
        for corpus_row in corpus_rows:
            if str(corpus_row.get("id", "")) in expected:
                hydrated.append(corpus_row)
        offset += len(candidate_ids)

    write_jsonl(output_path, hydrated)
    return output_path


def load_context_units(corpus_path: str | Path) -> list[ContextUnit]:
    units: list[ContextUnit] = []
    for row in read_jsonl(corpus_path):
        corpus_id = str(row["id"])
        title = str(row.get("title") or "")
        text = str(row.get("text") or "")
        content = text if not title else f"{title}\n{text}"
        units.append(
            ContextUnit(
                id=f"doc:{corpus_id}",
                instance_id="swebench_verified_rr",
                type="doc",
                content=content,
                source=corpus_id,
                metadata={"source_dataset": "mteb/SWEbenchVerifiedRR"},
            )
        )
    return units


def _causal_cache_key(task_id: str, context_id: str) -> str:
    return f"{task_id}\t{context_id}"


def _load_judgment_cache(path: str | Path | None) -> dict[str, dict]:
    if path is None or not Path(path).exists():
        return {}
    cached: dict[str, dict] = {}
    for row in read_jsonl(path):
        task_id = row.get("task_id")
        context_id = row.get("context_id")
        if task_id and context_id:
            cached[_causal_cache_key(str(task_id), str(context_id))] = row
    return cached


def _judge_causal_candidate(judge, task: Task, unit: ContextUnit, retrieval_score: float, pool_rank: int) -> dict:
    started = time.time()
    row = {
        "task_id": task.id,
        "context_id": unit.id,
        "unit_source": unit.source,
        "pool_rank": pool_rank,
        "retrieval_score": retrieval_score,
        "prompt_template": "code_retrieval_v1",
    }
    try:
        judgment = judge.judge(task, unit)
        causal_score = judgment_to_score(judgment, unit).final_score
    except Exception as exc:  # noqa: BLE001
        row.update(
            {
                "ok": False,
                "error": str(exc)[:500],
                "causal_score": -1.0,
                "latency_sec": round(time.time() - started, 3),
            }
        )
        return row
    row.update(
        {
            "ok": True,
            "judgment": judgment.to_dict(),
            "causal_score": causal_score,
            "latency_sec": round(time.time() - started, 3),
        }
    )
    return row


def _precompute_causal_judgments(
    *,
    tasks: list[Task],
    pools_by_task: dict[str, list[tuple[ContextUnit, float]]],
    judge,
    cache: dict[str, dict],
    concurrency: int,
    checkpoint_path: str | Path | None,
) -> None:
    pending: list[tuple[Task, ContextUnit, float, int]] = []
    for task in tasks:
        for index, (unit, retrieval_score) in enumerate(pools_by_task.get(task.id, []), start=1):
            key = _causal_cache_key(task.id, unit.id)
            if key not in cache:
                pending.append((task, unit, retrieval_score, index))

    if not pending:
        return

    def record(row: dict) -> None:
        cache[_causal_cache_key(row["task_id"], row["context_id"])] = row
        if checkpoint_path is not None:
            append_jsonl(checkpoint_path, row)

    worker_count = max(1, min(concurrency, len(pending)))
    if worker_count == 1:
        for task, unit, retrieval_score, pool_rank in pending:
            record(_judge_causal_candidate(judge, task, unit, retrieval_score, pool_rank))
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_judge_causal_candidate, judge, task, unit, retrieval_score, pool_rank)
            for task, unit, retrieval_score, pool_rank in pending
        ]
        for future in as_completed(futures):
            record(future.result())


def _judgment_rows_for_pools(
    tasks: list[Task],
    pools_by_task: dict[str, list[tuple[ContextUnit, float]]],
    cache: dict[str, dict],
) -> list[dict]:
    rows: list[dict] = []
    for task in tasks:
        for index, (unit, retrieval_score) in enumerate(pools_by_task.get(task.id, []), start=1):
            key = _causal_cache_key(task.id, unit.id)
            row = dict(cache.get(key) or {})
            row.setdefault("task_id", task.id)
            row.setdefault("context_id", unit.id)
            row.setdefault("unit_source", unit.source)
            row["pool_rank"] = index
            row["retrieval_score"] = retrieval_score
            rows.append(row)
    return rows


def _normalise_retrieval_scores(pool: list[tuple[ContextUnit, float]]) -> dict[str, float]:
    if not pool:
        return {}
    values = [score for _, score in pool]
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return {unit.id: 1.0 for unit, _ in pool}
    return {unit.id: (score - lo) / (hi - lo) for unit, score in pool}


def rank_units(
    *,
    method: str,
    task: Task,
    units: list[ContextUnit],
    max_k: int,
    pool_k: int,
    judge,
    causal_cache: dict[str, dict] | None = None,
    prefetched_pool: list[tuple[ContextUnit, float]] | None = None,
    causal_blend_weight: float = 0.35,
) -> list[str]:
    if method == "BM25":
        return [unit.id for unit, _ in BM25Retriever(units).search(task.instruction, top_k=max_k)]
    if method == "HashedEmbedding":
        return [unit.id for unit, _ in HashedEmbeddingRetriever(units).search(task.instruction, top_k=max_k)]
    if method == "HybridRAG":
        return [unit.id for unit, _ in HybridRetriever(units).search(task, top_k=max_k)]
    if method in CAUSAL_RERANK_METHODS:
        pool = prefetched_pool if prefetched_pool is not None else HybridRetriever(units).search(task, top_k=pool_k)
        retrieval_norm = _normalise_retrieval_scores(pool)
        scored = []
        for index, (unit, retrieval_score) in enumerate(pool):
            cached = (causal_cache or {}).get(_causal_cache_key(task.id, unit.id))
            if cached is None:
                cached = _judge_causal_candidate(judge, task, unit, retrieval_score, index + 1)
                if causal_cache is not None:
                    causal_cache[_causal_cache_key(task.id, unit.id)] = cached
            causal_score = float(cached.get("causal_score", -1.0)) if cached.get("ok") else -1.0
            if method == "CausalHybridRerank":
                retrieval_score_norm = retrieval_norm.get(unit.id, 0.0)
                rank_score = causal_blend_weight * causal_score + (1.0 - causal_blend_weight) * retrieval_score_norm
            else:
                rank_score = causal_score
            scored.append((unit.id, rank_score, causal_score, retrieval_score, -index))
        scored.sort(key=lambda row: (row[1], row[2], row[3], row[4]), reverse=True)
        return [unit_id for unit_id, _, _, _, _ in scored[:max_k]]
    if method == "Random":
        ids = [unit.id for unit in units]
        rng = random.Random(f"swebench-rr:{task.id}")
        rng.shuffle(ids)
        return ids[:max_k]
    if method == "OracleGoldContext":
        gold = [unit_id for unit_id in task.gold_context_ids if any(unit.id == unit_id for unit in units)]
        rest = [unit.id for unit in units if unit.id not in set(gold)]
        return (gold + rest)[:max_k]
    raise ValueError(f"Unknown retrieval method: {method}")


def reciprocal_rank(ranked_ids: list[str], gold_ids: set[str], max_k: int) -> float:
    for index, unit_id in enumerate(ranked_ids[:max_k], start=1):
        if unit_id in gold_ids:
            return 1.0 / index
    return 0.0


def task_metric_row(task: Task, method: str, ranked_ids: list[str], ks: list[int]) -> dict:
    gold = set(task.gold_context_ids)
    rank = 0
    for index, unit_id in enumerate(ranked_ids, start=1):
        if unit_id in gold:
            rank = index
            break
    row = {
        "task_id": task.id,
        "method": method,
        "gold_context_ids": "|".join(task.gold_context_ids),
        "gold_rank": rank,
        "top_context_ids": "|".join(ranked_ids[: max(ks)]),
    }
    for k in ks:
        hit = int(any(unit_id in gold for unit_id in ranked_ids[:k]))
        row[f"hit@{k}"] = hit
        row[f"recall@{k}"] = round(hit / max(1, len(gold)), 6)
        row[f"precision@{k}"] = round(hit / k, 6)
        row[f"mrr@{k}"] = round(reciprocal_rank(ranked_ids, gold, k), 6)
    return row


def summarize(rows: list[dict], ks: list[int]) -> list[dict]:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
    summary_rows: list[dict] = []
    for method in sorted(by_method):
        method_rows = by_method[method]
        summary = {
            "method": method,
            "n": len(method_rows),
            "coverage": round(
                sum(1 for row in method_rows if int(row["gold_rank"]) > 0) / max(1, len(method_rows)),
                6,
            ),
            "mean_gold_rank": round(
                sum(int(row["gold_rank"]) for row in method_rows if int(row["gold_rank"]) > 0)
                / max(1, sum(1 for row in method_rows if int(row["gold_rank"]) > 0)),
                3,
            ),
        }
        for k in ks:
            for metric in [f"hit@{k}", f"recall@{k}", f"precision@{k}", f"mrr@{k}"]:
                summary[metric] = round(
                    sum(float(row[metric]) for row in method_rows) / max(1, len(method_rows)),
                    6,
                )
        summary_rows.append(summary)
    return summary_rows


def run_swebench_rr_retrieval(
    *,
    tasks_path: str | Path,
    corpus_path: str | Path,
    output: str | Path,
    top_ranked_path: str | Path | None = None,
    hydrate: bool = False,
    refresh_hydration: bool = False,
    candidate_limit: int = 50,
    max_hydrate_queries: int | None = None,
    allow_partial_hydration: bool = False,
    methods: Iterable[str] = ("BM25", "HashedEmbedding", "HybridRAG", "CausalRerank", "Random", "OracleGoldContext"),
    ks: list[int] | None = None,
    pool_k: int = 50,
    llm_provider: str = "simulator",
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key_env: str = "DASHSCOPE_API_KEY",
    llm_timeout_seconds: float = 60.0,
    llm_max_tokens: int | None = None,
    llm_concurrency: int = 1,
    llm_context_chars: int = 6000,
    llm_checkpoint_path: str | Path | None = None,
    llm_judgments_output: str | Path | None = None,
    causal_blend_weight: float = 0.35,
) -> dict[str, list[dict]]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ks = ks or DEFAULT_KS
    max_k = max(ks)

    corpus_path = Path(corpus_path)
    if hydrate:
        if top_ranked_path is None:
            raise ValueError("--top-ranked is required when --hydrate is set")
        corpus_path = hydrate_candidate_corpus(
            top_ranked_path=top_ranked_path,
            output_path=corpus_path,
            candidate_limit=candidate_limit,
            max_queries=max_hydrate_queries,
            allow_partial=allow_partial_hydration,
            refresh=refresh_hydration,
        )

    tasks = load_tasks(tasks_path)
    units = load_context_units(corpus_path)
    unit_ids = {unit.id for unit in units}
    evaluable_tasks = [task for task in tasks if any(gold_id in unit_ids for gold_id in task.gold_context_ids)]

    llm_provider = normalize_provider(llm_provider)
    judge = build_llm_judge(
        provider=llm_provider,
        model=resolve_llm_model(llm_provider, llm_model),
        base_url=llm_base_url,
        api_key_env=llm_api_key_env,
        timeout_seconds=llm_timeout_seconds,
        max_tokens=llm_max_tokens,
    )
    judge.prompt_template = CodeRetrievalPromptTemplate(max_context_chars=llm_context_chars)

    methods = list(methods)
    causal_methods = [method for method in methods if method in CAUSAL_RERANK_METHODS]
    pools_by_task: dict[str, list[tuple[ContextUnit, float]]] = {}
    judgment_cache = _load_judgment_cache(llm_checkpoint_path)
    if causal_methods:
        hybrid = HybridRetriever(units)
        pools_by_task = {task.id: hybrid.search(task, top_k=pool_k) for task in evaluable_tasks}
        _precompute_causal_judgments(
            tasks=evaluable_tasks,
            pools_by_task=pools_by_task,
            judge=judge,
            cache=judgment_cache,
            concurrency=llm_concurrency,
            checkpoint_path=llm_checkpoint_path,
        )

    metric_rows: list[dict] = []
    for task in evaluable_tasks:
        for method in methods:
            ranked_ids = rank_units(
                method=method,
                task=task,
                units=units,
                max_k=max_k,
                pool_k=pool_k,
                judge=judge,
                causal_cache=judgment_cache,
                prefetched_pool=pools_by_task.get(task.id),
                causal_blend_weight=causal_blend_weight,
            )
            metric_rows.append(task_metric_row(task, method, ranked_ids, ks))

    summary_rows = summarize(metric_rows, ks)
    metadata_rows = [
        {
            "tasks_total": len(tasks),
            "tasks_evaluable": len(evaluable_tasks),
            "corpus_units": len(units),
            "candidate_limit": candidate_limit,
            "max_hydrate_queries": max_hydrate_queries if max_hydrate_queries is not None else "",
            "allow_partial_hydration": int(allow_partial_hydration),
            "methods": ",".join(methods),
            "ks": ",".join(str(k) for k in ks),
            "llm_provider": llm_provider,
            "llm_model": resolve_llm_model(llm_provider, llm_model) if llm_provider != "simulator" else "simulator",
            "llm_concurrency": llm_concurrency,
            "llm_prompt_template": "code_retrieval_v1" if causal_methods else "",
            "llm_context_chars": llm_context_chars if causal_methods else "",
            "llm_checkpoint_path": str(llm_checkpoint_path) if llm_checkpoint_path else "",
            "llm_judgments_output": str(llm_judgments_output) if llm_judgments_output else "",
            "causal_blend_weight": causal_blend_weight,
            "corpus_path": str(corpus_path),
        }
    ]
    write_csv(output / "retrieval_task_metrics.csv", metric_rows)
    write_csv(output / "retrieval_summary.csv", summary_rows)
    write_csv(output / "metadata.csv", metadata_rows)
    write_markdown_table(output / "report" / "retrieval_summary.md", summary_rows, "SWE-bench Verified-RR Retrieval Summary")
    write_markdown_table(output / "report" / "retrieval_task_metrics_sample.md", metric_rows[:120], "SWE-bench Verified-RR Task Metrics Sample")
    write_jsonl(output / "corpus_units.jsonl", (unit.to_dict() for unit in units))
    if causal_methods:
        judgment_path = Path(llm_judgments_output) if llm_judgments_output else output / "causal_judgments.jsonl"
        write_jsonl(judgment_path, _judgment_rows_for_pools(evaluable_tasks, pools_by_task, judgment_cache))
    return {"summary": summary_rows, "metrics": metric_rows, "metadata": metadata_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SWE-bench Verified-RR retrieval comparisons.")
    parser.add_argument("--tasks", default="data/processed/swebench_verified_rr_tasks.jsonl")
    parser.add_argument("--corpus", default="data/processed/swebench_verified_rr_candidate_corpus_top50.jsonl")
    parser.add_argument("--top-ranked", default="data/raw/swebench_verified_rr/top_ranked_sample_50.jsonl")
    parser.add_argument("--output", default="outputs/latest/swebench_verified_rr_retrieval")
    parser.add_argument("--hydrate", action="store_true", help="Fetch candidate corpus text from Hugging Face rows.")
    parser.add_argument("--refresh-hydration", action="store_true")
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--max-hydrate-queries", type=int)
    parser.add_argument("--allow-partial-hydration", action="store_true")
    parser.add_argument("--methods", default="BM25,HashedEmbedding,HybridRAG,CausalRerank,Random,OracleGoldContext")
    parser.add_argument("--ks", default="1,5,10,20")
    parser.add_argument("--pool-k", type=int, default=50)
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
    parser.add_argument("--llm-concurrency", type=int, default=1)
    parser.add_argument("--llm-context-chars", type=int, default=6000)
    parser.add_argument("--llm-checkpoint-path")
    parser.add_argument("--llm-judgments-output")
    parser.add_argument("--causal-blend-weight", type=float, default=0.35)
    args = parser.parse_args()

    result = run_swebench_rr_retrieval(
        tasks_path=args.tasks,
        corpus_path=args.corpus,
        output=args.output,
        top_ranked_path=args.top_ranked,
        hydrate=args.hydrate,
        refresh_hydration=args.refresh_hydration,
        candidate_limit=args.candidate_limit,
        max_hydrate_queries=args.max_hydrate_queries,
        allow_partial_hydration=args.allow_partial_hydration,
        methods=[item for item in args.methods.split(",") if item],
        ks=[int(item) for item in args.ks.split(",") if item],
        pool_k=args.pool_k,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key_env=args.llm_api_key_env,
        llm_timeout_seconds=args.llm_timeout_seconds,
        llm_max_tokens=args.llm_max_tokens,
        llm_concurrency=args.llm_concurrency,
        llm_context_chars=args.llm_context_chars,
        llm_checkpoint_path=args.llm_checkpoint_path,
        llm_judgments_output=args.llm_judgments_output,
        causal_blend_weight=args.causal_blend_weight,
    )
    for row in result["summary"]:
        print(
            f"{row['method']}: n={row['n']} hit@1={row.get('hit@1')} "
            f"hit@5={row.get('hit@5')} mrr@10={row.get('mrr@10')}"
        )


if __name__ == "__main__":
    main()
