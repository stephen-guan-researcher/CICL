"""Shared text tokenization and lexical similarity helpers."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def jaccard(a: str, b: str) -> float:
    aa = set(tokenize(a))
    bb = set(tokenize(b))
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)
