"""Pairwise context ranker — small linear model trained from utility examples."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from cicl_agent.learning.dataset import CausalUtilityExample


@dataclass(slots=True)
class PairwiseTrainingReport:
    examples: int
    pairs: int
    epochs: int
    final_pairwise_accuracy: float

    def to_dict(self) -> dict:
        return asdict(self)


class PairwiseContextRanker:
    """A small trainable ranker for context utility."""

    def __init__(self, weights: dict[str, float] | None = None, metadata: dict | None = None) -> None:
        self.weights = weights or {}
        self.metadata = metadata or {}

    def score_raw(self, features: dict[str, float]) -> float:
        return sum(self.weights.get(name, 0.0) * value for name, value in features.items())

    def score(self, features: dict[str, float]) -> float:
        raw = max(-40.0, min(40.0, self.score_raw(features)))
        return 1.0 / (1.0 + math.exp(-raw))

    def fit(
        self,
        examples: list[CausalUtilityExample],
        epochs: int = 80,
        lr: float = 0.08,
        l2: float = 0.0005,
        margin: float = 0.05,
        seed: int = 17,
    ) -> PairwiseTrainingReport:
        pairs = self._pairs(examples, margin=margin)
        rng = random.Random(seed)
        if not pairs:
            return PairwiseTrainingReport(len(examples), 0, epochs, 0.0)
        for _ in range(epochs):
            rng.shuffle(pairs)
            for pos, neg in pairs:
                diff_features = self._diff(pos.features, neg.features)
                diff = self.score_raw(diff_features)
                grad_scale = 1.0 / (1.0 + math.exp(max(-40.0, min(40.0, diff))))
                for name, value in diff_features.items():
                    old = self.weights.get(name, 0.0)
                    self.weights[name] = old + lr * grad_scale * value - lr * l2 * old
        return PairwiseTrainingReport(
            examples=len(examples),
            pairs=len(pairs),
            epochs=epochs,
            final_pairwise_accuracy=self.pairwise_accuracy(examples, margin=margin),
        )

    def pairwise_accuracy(self, examples: list[CausalUtilityExample], margin: float = 0.05) -> float:
        pairs = self._pairs(examples, margin=margin)
        if not pairs:
            return 0.0
        correct = 0
        for pos, neg in pairs:
            if self.score_raw(pos.features) > self.score_raw(neg.features):
                correct += 1
        return correct / len(pairs)

    def save(self, path: str | Path, report: PairwiseTrainingReport | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "pairwise_linear_context_ranker",
            "weights": dict(sorted(self.weights.items())),
            "metadata": dict(sorted(self.metadata.items())),
            "report": report.to_dict() if report else None,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PairwiseContextRanker":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            weights={key: float(value) for key, value in payload.get("weights", {}).items()},
            metadata=payload.get("metadata", {}),
        )

    @staticmethod
    def _pairs(
        examples: list[CausalUtilityExample], margin: float
    ) -> list[tuple[CausalUtilityExample, CausalUtilityExample]]:
        by_task: dict[str, list[CausalUtilityExample]] = defaultdict(list)
        for example in examples:
            by_task[example.task_id].append(example)
        pairs: list[tuple[CausalUtilityExample, CausalUtilityExample]] = []
        for rows in by_task.values():
            for pos in rows:
                for neg in rows:
                    if pos.reward >= neg.reward + margin:
                        pairs.append((pos, neg))
        return pairs

    @staticmethod
    def _diff(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
        keys = set(left) | set(right)
        return {key: left.get(key, 0.0) - right.get(key, 0.0) for key in keys}


__all__ = ["PairwiseContextRanker", "PairwiseTrainingReport"]
