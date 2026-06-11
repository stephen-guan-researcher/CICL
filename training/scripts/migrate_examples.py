"""Migrate Opus-labeled llm_examples.jsonl to the post-bugfix schema.

The old `_make_example` merged llm_* judgment fields into `features` (label
leakage) and stored `retrieval_score` via a hard clamp `min(1.0, score)` that
saturated every BM25-driven score to exactly 1.0 (zero variance).

This script re-derives `features` from scratch using the fixed extractor,
moves the llm_* judgment fields back into `metadata`, and writes a cleaned
JSONL. It does NOT re-call Opus — the judgments themselves are kept verbatim
from the cached file. Graph + retrieval are rebuilt locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cicl_agent.learning.features import ContextFeatureExtractor
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.retrieval.retrievers import HybridRetriever
from cicl_agent.runners.runner import load_tasks


LLM_FIELDS = (
    "llm_action_shift",
    "llm_necessity",
    "llm_expected_uplift",
    "llm_negative_transfer",
    "llm_confidence",
)


def migrate(examples_in: Path, repo: Path, tasks_path: Path, examples_out: Path, top_k: int = 30) -> dict:
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

    tasks_by_id = {t.id: t for t in tasks}
    extractor = ContextFeatureExtractor(graph)

    retriever = HybridRetriever(list(graph.units.values()))
    retrieval_cache: dict[str, dict[str, float]] = {}

    def retrieval_for(task_id: str) -> dict[str, float]:
        if task_id not in retrieval_cache:
            task = tasks_by_id.get(task_id)
            if task is None:
                retrieval_cache[task_id] = {}
            else:
                retrieval_cache[task_id] = {
                    unit.id: score for unit, score in retriever.search(task, top_k=top_k * 3)
                }
        return retrieval_cache[task_id]

    written = 0
    skipped = 0
    retrieval_hits = 0
    retrieval_misses = 0
    examples_out.parent.mkdir(parents=True, exist_ok=True)

    with examples_in.open() as fin, examples_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task_id = row["task_id"]
            context_id = row["context_id"]
            task = tasks_by_id.get(task_id)
            unit = graph.units.get(context_id)
            if task is None or unit is None:
                skipped += 1
                continue

            scores = retrieval_for(task_id)
            if context_id in scores:
                retrieval_score = scores[context_id]
                retrieval_hits += 1
            else:
                # Neighbor candidates that BM25 didn't return — fall back to 0
                # like the LearnedCICLSelector would for an unretrieved unit.
                retrieval_score = 0.0
                retrieval_misses += 1

            old_features = row.get("features", {}) or {}
            old_metadata = row.get("metadata", {}) or {}

            new_features = extractor.features(task, unit, retrieval_score=retrieval_score)

            # Lift llm_* judgment fields out of (old) features into metadata.
            new_metadata = dict(old_metadata)
            for key in LLM_FIELDS:
                if key in new_metadata:
                    continue
                if key in old_features:
                    new_metadata[key] = old_features[key]
                elif key in old_metadata:
                    pass

            row["features"] = new_features
            row["metadata"] = new_metadata
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    return {
        "examples_in": str(examples_in),
        "examples_out": str(examples_out),
        "written": written,
        "skipped": skipped,
        "retrieval_hits": retrieval_hits,
        "retrieval_misses": retrieval_misses,
        "tasks_loaded": len(tasks),
        "graph_units": len(graph.units),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-in", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--examples-out", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()
    summary = migrate(
        examples_in=Path(args.examples_in),
        repo=Path(args.repo),
        tasks_path=Path(args.tasks),
        examples_out=Path(args.examples_out),
        top_k=args.top_k,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
