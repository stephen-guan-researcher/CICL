"""Repo-walking builders for InstanceContextGraph.

The graph itself only manages units and edges. Anything that materialises
context from external sources (filesystem, AST, downstream tools) lives here so
the graph stays storage-only.
"""

from __future__ import annotations

import ast
from pathlib import Path

from cicl_agent.core.schema import ContextEdge, ContextUnit


TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
}

SKIP_PARTS = {".git", "__pycache__", ".venv", "node_modules"}


def populate_from_repo(graph, repo_path: str | Path, max_file_chars: int = 12000) -> None:
    """Walk a repo and add file/symbol units onto the given graph."""

    repo = Path(repo_path).resolve()
    for path in sorted(repo.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        rel = path.relative_to(repo).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        file_unit = ContextUnit(
            id=f"file:{rel}",
            instance_id=graph.instance_id,
            type="file",
            content=text[:max_file_chars],
            source=rel,
            metadata={"path": rel, "truncated": len(text) > max_file_chars},
        )
        graph.add_unit(file_unit)
        if path.suffix == ".py":
            _add_python_symbols(graph, rel, text, file_unit.id)


def _add_python_symbols(graph, rel: str, text: str, file_unit_id: str) -> None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    lines = text.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start)
        snippet = "\n".join(lines[start - 1 : end])
        kind = "class" if isinstance(node, ast.ClassDef) else "symbol"
        unit_id = f"symbol:{rel}:{node.name}:{start}"
        graph.add_unit(
            ContextUnit(
                id=unit_id,
                instance_id=graph.instance_id,
                type=kind,
                content=snippet,
                source=f"{rel}:{start}",
                metadata={"path": rel, "name": node.name, "line": start},
            )
        )
        graph.add_edge(ContextEdge(file_unit_id, unit_id, "contains", 1.0))
