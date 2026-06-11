"""CSV / Markdown / JSONL helpers for CICL reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: str | Path) -> list[dict]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []
    rows: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: str | Path, rows: list[dict]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: str | Path, rows: list[dict], title: str) -> None:
    md_path = Path(path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        md_path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "read_csv_dicts",
    "read_jsonl",
    "write_csv",
    "write_markdown_table",
]
