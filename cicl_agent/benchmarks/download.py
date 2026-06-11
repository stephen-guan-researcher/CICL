"""Download small open benchmark samples for CICL experiments.

The downloader is intentionally dependency-free. It uses `curl` for network
fetching because this workspace can reach Hugging Face through curl even when
Python's default SSL certificate lookup is unavailable.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from cicl_agent.benchmarks.adapters import (
    repobench_row_to_task,
    swebench_row_to_task,
    swebench_rr_query_to_task,
)
from cicl_agent.core.io import write_jsonl


# datasets-server.huggingface.co is blocked from this DSW; fall back to the
# `datasets` library via the configured HF_ENDPOINT (default hf-mirror.com,
# which is reachable). The mirror serves direct file blobs and the standard
# /api endpoints, so the parquet shards loaded by `datasets` work.
HF_ENDPOINT = os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
REPOBENCH_R_PYTHON_CFF_URL = (
    f"{HF_ENDPOINT}/datasets/tianyang/repobench-r/resolve/main/data/python_cff.gz"
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def curl_bytes(url: str) -> bytes:
    result = subprocess.run(
        ["curl", "-LfsS", "--retry", "3", "--retry-delay", "2", "--retry-all-errors", url],
        check=True,
        capture_output=True,
    )
    return result.stdout


def curl_file(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-LfsS", "--retry", "3", "--retry-delay", "2", "--retry-all-errors", url, "-o", str(output)],
        check=True,
    )


def fetch_hf_rows(
    dataset: str,
    config: str,
    split: str,
    length: int,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Stream the first ``offset+length`` rows via the `datasets` library.

    Falls back to this path because datasets-server.huggingface.co is not
    reachable from this DSW. Uses streaming so we don't pull entire shards.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split, streaming=True)
    sliced = itertools.islice(ds, offset, offset + length)
    rows = [dict(row) for row in sliced]
    # `streaming=True` doesn't expose the total row count cheaply; report
    # what we actually pulled so the manifest stays honest.
    return rows, len(rows)


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def save_swebench_sample(
    *,
    dataset_name: str,
    dataset_id: str,
    split: str,
    limit: int,
    output_root: Path,
    processed_name: str,
) -> dict[str, Any]:
    rows, total = fetch_hf_rows(dataset_id, "default", split, limit)
    for row in rows:
        row["source_dataset"] = dataset_id
    raw_path = output_root / "raw" / dataset_name / f"{split}_sample_{len(rows)}.jsonl"
    task_path = output_root / "processed" / f"{dataset_name}_{split}_tasks.jsonl"
    write_jsonl(raw_path, rows)
    tasks = [swebench_row_to_task(row).to_dict() for row in rows]
    write_jsonl(task_path, tasks)
    return {
        "name": processed_name,
        "dataset_id": dataset_id,
        "source_url": f"https://huggingface.co/datasets/{dataset_id}",
        "raw_path": raw_path.as_posix(),
        "processed_task_path": task_path.as_posix(),
        "downloaded_rows": len(rows),
        "total_rows": total,
        "notes": "Processed tasks exclude solution patch/test patch; patch is used only to derive gold file ids.",
    }


def save_swebench_verified_rr_sample(limit: int, corpus_limit: int, output_root: Path) -> dict[str, Any]:
    dataset_id = "mteb/SWEbenchVerifiedRR"
    queries, total_queries = fetch_hf_rows(dataset_id, "queries", "train", limit)
    try:
        qrels, total_qrels = fetch_hf_rows(dataset_id, "qrels", "train", min(100, max(corpus_limit, limit * 2)))
    except subprocess.CalledProcessError:
        qrels, total_qrels = [], 0
    try:
        corpus, total_corpus = fetch_hf_rows(dataset_id, "corpus", "train", corpus_limit)
    except subprocess.CalledProcessError:
        try:
            corpus, total_corpus = fetch_hf_rows(dataset_id, "corpus", "train", min(50, corpus_limit))
        except subprocess.CalledProcessError:
            corpus, total_corpus = [], 0
    top_ranked, total_top_ranked = fetch_hf_rows(dataset_id, "top_ranked", "train", limit)

    raw_dir = output_root / "raw" / "swebench_verified_rr"
    write_jsonl(raw_dir / f"queries_sample_{len(queries)}.jsonl", queries)
    write_jsonl(raw_dir / f"qrels_sample_{len(qrels)}.jsonl", qrels)
    write_jsonl(raw_dir / f"corpus_sample_{len(corpus)}.jsonl", corpus)
    write_jsonl(raw_dir / f"top_ranked_sample_{len(top_ranked)}.jsonl", top_ranked)

    positives_by_query: dict[str, list[str]] = {}
    for row in qrels:
        if int(row.get("score", 0)) <= 0:
            continue
        positives_by_query.setdefault(str(row.get("query-id")), []).append(str(row.get("corpus-id")))
    for row in top_ranked:
        query_id = str(row.get("query-id"))
        for corpus_id in row.get("corpus-ids", []) or []:
            corpus_id = str(corpus_id)
            if "positive" in corpus_id:
                positives_by_query.setdefault(query_id, []).append(corpus_id)
    tasks = [
        swebench_rr_query_to_task(query, sorted(set(positives_by_query.get(str(query["id"]), [])))).to_dict()
        for query in queries
    ]
    task_path = output_root / "processed" / "swebench_verified_rr_tasks.jsonl"
    write_jsonl(task_path, tasks)

    return {
        "name": "SWEbenchVerifiedRR",
        "dataset_id": dataset_id,
        "source_url": f"https://huggingface.co/datasets/{dataset_id}",
        "raw_paths": {
            "queries": (raw_dir / f"queries_sample_{len(queries)}.jsonl").as_posix(),
            "qrels": (raw_dir / f"qrels_sample_{len(qrels)}.jsonl").as_posix(),
            "corpus": (raw_dir / f"corpus_sample_{len(corpus)}.jsonl").as_posix(),
            "top_ranked": (raw_dir / f"top_ranked_sample_{len(top_ranked)}.jsonl").as_posix(),
        },
        "processed_task_path": task_path.as_posix(),
        "downloaded_rows": {
            "queries": len(queries),
            "qrels": len(qrels),
            "corpus": len(corpus),
            "top_ranked": len(top_ranked),
        },
        "total_rows": {
            "queries": total_queries,
            "qrels": total_qrels,
            "corpus": total_corpus,
            "top_ranked": total_top_ranked,
        },
        "notes": "Retrieval benchmark for issue-to-context ranking; positives are qrels with score > 0.",
    }


def save_repobench_r_sample(limit: int, output_root: Path) -> dict[str, Any]:
    raw_dir = output_root / "raw" / "repobench_r"
    processed_dir = output_root / "processed"
    raw_path = raw_dir / f"python_cff_test_easy_sample_{limit}.jsonl"
    task_path = processed_dir / "repobench_r_python_cff_tasks.jsonl"
    repo_dir = processed_dir / "repobench_r_python_cff_repo"

    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "python_cff.gz"
        curl_file(REPOBENCH_R_PYTHON_CFF_URL, archive_path)
        with gzip.open(archive_path, "rb") as handle:
            payload = pickle.load(handle)

    records = list(payload["test"]["easy"][:limit])
    write_jsonl(raw_path, records)
    tasks = [repobench_row_to_task(row, index=i).to_dict() for i, row in enumerate(records)]
    write_jsonl(task_path, tasks)

    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    (repo_dir / "contexts").mkdir(parents=True, exist_ok=True)
    (repo_dir / "targets").mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(records):
        for j, snippet in enumerate(row.get("context", []) or []):
            context_file = repo_dir / "contexts" / f"{i:05d}_{j:02d}.py"
            context_file.write_text(str(snippet), encoding="utf-8")
        target_file = repo_dir / "targets" / f"{i:05d}.py"
        target_file.write_text(str(row.get("code", "")), encoding="utf-8")

    return {
        "name": "RepoBench-R Python CFF",
        "dataset_id": "tianyang/repobench-r",
        "source_url": "https://huggingface.co/datasets/tianyang/repobench-r",
        "raw_path": raw_path.as_posix(),
        "processed_task_path": task_path.as_posix(),
        "processed_repo_path": repo_dir.as_posix(),
        "downloaded_rows": len(records),
        "total_rows": {
            "test.easy": len(payload["test"]["easy"]),
            "test.hard": len(payload["test"]["hard"]),
            "train": len(payload["train"]),
        },
        "notes": "Creates a tiny local repo from context snippets so the existing CICL graph builder can run.",
    }


def download_starter(output_root: str | Path, limit: int, rr_corpus_limit: int, repobench_limit: int) -> list[dict[str, Any]]:
    output_root = Path(output_root)
    manifest: list[dict[str, Any]] = []
    manifest.append(
        save_swebench_sample(
            dataset_name="swebench_lite",
            dataset_id="SWE-bench/SWE-bench_Lite",
            split="test",
            limit=limit,
            output_root=output_root,
            processed_name="SWE-bench Lite",
        )
    )
    manifest.append(
        save_swebench_sample(
            dataset_name="swebench_verified",
            dataset_id="SWE-bench/SWE-bench_Verified",
            split="test",
            limit=limit,
            output_root=output_root,
            processed_name="SWE-bench Verified",
        )
    )
    manifest.append(save_swebench_verified_rr_sample(limit=limit, corpus_limit=rr_corpus_limit, output_root=output_root))
    manifest.append(save_repobench_r_sample(limit=repobench_limit, output_root=output_root))
    for row in manifest:
        row["downloaded_at"] = utc_timestamp()
    write_manifest(output_root / "manifests" / "datasets_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download starter open datasets for CICL.")
    parser.add_argument("--output-root", default="data", help="Root containing raw/, processed/, and manifests/.")
    parser.add_argument("--limit", type=int, default=50, help="Rows for SWE-bench and SWEbenchVerifiedRR queries.")
    parser.add_argument("--rr-corpus-limit", type=int, default=500, help="Rows for SWEbenchVerifiedRR corpus/qrels samples.")
    parser.add_argument("--repobench-limit", type=int, default=100, help="Rows for RepoBench-R Python CFF sample.")
    args = parser.parse_args()
    manifest = download_starter(
        output_root=args.output_root,
        limit=args.limit,
        rr_corpus_limit=args.rr_corpus_limit,
        repobench_limit=args.repobench_limit,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
