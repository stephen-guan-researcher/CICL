"""Retrieval backends for context units."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict

from cicl_agent.core.schema import ContextUnit, Task
from cicl_agent.core.text import tokenize


class BM25Retriever:
    def __init__(self, units: list[ContextUnit], k1: float = 1.5, b: float = 0.75) -> None:
        self.units = units
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(unit.content + " " + unit.source) for unit in units]
        self.avgdl = sum(len(doc) for doc in self.docs) / max(1, len(self.docs))
        df: dict[str, int] = defaultdict(int)
        for doc in self.docs:
            for token in set(doc):
                df[token] += 1
        self.idf = {
            token: math.log(1 + (len(self.docs) - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def search(self, query: str, top_k: int = 20) -> list[tuple[ContextUnit, float]]:
        query_tokens = tokenize(query)
        scored: list[tuple[ContextUnit, float]] = []
        for unit, doc in zip(self.units, self.docs):
            if not doc:
                continue
            counts = Counter(doc)
            score = 0.0
            for token in query_tokens:
                if token not in counts:
                    continue
                freq = counts[token]
                denom = freq + self.k1 * (1 - self.b + self.b * len(doc) / max(1e-9, self.avgdl))
                score += self.idf.get(token, 0.0) * freq * (self.k1 + 1) / denom
            if score > 0:
                scored.append((unit, score))
        scored.sort(key=lambda row: row[1], reverse=True)
        return scored[:top_k]


class HashedEmbeddingRetriever:
    """A dependency-free hashed bag-of-words cosine retriever.

    This is not a substitute for a production embedding model. It exists so the
    framework can run anywhere and keep the same adapter slot for real embeddings.
    """

    def __init__(self, units: list[ContextUnit], dims: int = 256) -> None:
        self.units = units
        self.dims = dims
        self.vectors = [self._vectorize(unit.content + " " + unit.source) for unit in units]

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.dims

    def _vectorize(self, text: str) -> dict[int, float]:
        counts: dict[int, float] = defaultdict(float)
        for token in tokenize(text):
            counts[self._bucket(token)] += 1.0
        norm = math.sqrt(sum(value * value for value in counts.values()))
        if norm == 0:
            return {}
        return {key: value / norm for key, value in counts.items()}

    @staticmethod
    def _cosine(a: dict[int, float], b: dict[int, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(value * b.get(key, 0.0) for key, value in a.items())

    def search(self, query: str, top_k: int = 20) -> list[tuple[ContextUnit, float]]:
        query_vector = self._vectorize(query)
        scored = [
            (unit, self._cosine(query_vector, vector))
            for unit, vector in zip(self.units, self.vectors)
        ]
        scored = [row for row in scored if row[1] > 0]
        scored.sort(key=lambda row: row[1], reverse=True)
        return scored[:top_k]


class HybridRetriever:
    def __init__(self, units: list[ContextUnit], bm25_weight: float = 0.65) -> None:
        self.units = units
        self.bm25 = BM25Retriever(units)
        self.embedding = HashedEmbeddingRetriever(units)
        self.bm25_weight = bm25_weight

    def search(self, task: Task, top_k: int = 30) -> list[tuple[ContextUnit, float]]:
        query = task.instruction + " " + " ".join(task.expected_actions)
        scores: dict[str, float] = defaultdict(float)
        unit_by_id = {unit.id: unit for unit in self.units}
        for unit, score in self.bm25.search(query, top_k=top_k * 2):
            scores[unit.id] += self.bm25_weight * score
        for unit, score in self.embedding.search(query, top_k=top_k * 2):
            scores[unit.id] += (1 - self.bm25_weight) * score
        ranked = [(unit_by_id[unit_id], score) for unit_id, score in scores.items()]
        ranked.sort(key=lambda row: row[1], reverse=True)
        return ranked[:top_k]

