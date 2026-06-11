"""Small MLP pairwise context ranker — non-linear distilled student."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from cicl_agent.learning.dataset import CausalUtilityExample
from cicl_agent.learning.ranker import PairwiseContextRanker


@dataclass(slots=True)
class MLPTrainingReport:
    examples: int
    pairs: int
    epochs: int
    final_pairwise_accuracy: float
    final_loss: float
    hidden_sizes: list[int]

    def to_dict(self) -> dict:
        return asdict(self)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPContextRanker:
    """Drop-in replacement for PairwiseContextRanker backed by a small MLP."""

    def __init__(
        self,
        feature_names: list[str] | None = None,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.0,
        state_dict: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.feature_names: list[str] = list(feature_names or [])
        self.hidden_sizes: list[int] = list(hidden_sizes or [64, 32])
        self.dropout = dropout
        self.metadata = metadata or {}
        self._model: _MLP | None = None
        if self.feature_names:
            self._build_model()
            if state_dict is not None:
                self._model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()})
        # weights kept for API compatibility (some legacy code peeks at it)
        self.weights: dict[str, float] = {}

    def _build_model(self) -> None:
        torch.manual_seed(17)
        self._model = _MLP(len(self.feature_names), self.hidden_sizes, self.dropout)
        self._model.eval()

    def _vec(self, features: dict[str, float]) -> torch.Tensor:
        return torch.tensor(
            [float(features.get(name, 0.0)) for name in self.feature_names],
            dtype=torch.float32,
        )

    def score_raw(self, features: dict[str, float]) -> float:
        assert self._model is not None, "Model not initialized"
        with torch.no_grad():
            return float(self._model(self._vec(features).unsqueeze(0)).item())

    def score(self, features: dict[str, float]) -> float:
        raw = self.score_raw(features)
        if raw > 40:
            return 1.0
        if raw < -40:
            return 0.0
        return 1.0 / (1.0 + math.exp(-raw))

    def fit(
        self,
        examples: list[CausalUtilityExample],
        epochs: int = 120,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        margin: float = 0.05,
        batch_size: int = 256,
        seed: int = 17,
    ) -> MLPTrainingReport:
        if not self.feature_names:
            # Infer from first example with non-empty features
            for ex in examples:
                if ex.features:
                    self.feature_names = sorted(ex.features.keys())
                    break
            self._build_model()
        assert self._model is not None

        pairs = self._pairs(examples, margin=margin)
        if not pairs:
            return MLPTrainingReport(len(examples), 0, epochs, 0.0, 0.0, self.hidden_sizes)
        rng = random.Random(seed)

        # Precompute feature vectors for all examples used in any pair
        feat_cache: dict[int, torch.Tensor] = {}
        for pos, neg in pairs:
            for ex in (pos, neg):
                key = id(ex)
                if key not in feat_cache:
                    feat_cache[key] = self._vec(ex.features)
        pos_idx = list(range(len(pairs)))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        opt = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=weight_decay)

        final_loss = 0.0
        for epoch in range(epochs):
            self._model.train()
            rng.shuffle(pos_idx)
            total = 0.0
            steps = 0
            for start in range(0, len(pos_idx), batch_size):
                batch = pos_idx[start : start + batch_size]
                pos_vecs = torch.stack([feat_cache[id(pairs[i][0])] for i in batch]).to(device)
                neg_vecs = torch.stack([feat_cache[id(pairs[i][1])] for i in batch]).to(device)
                pos_scores = self._model(pos_vecs)
                neg_scores = self._model(neg_vecs)
                # BPR pairwise loss: -log sigmoid(pos - neg)
                loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item())
                steps += 1
            final_loss = total / max(steps, 1)
        self._model.to("cpu")
        self._model.eval()
        return MLPTrainingReport(
            examples=len(examples),
            pairs=len(pairs),
            epochs=epochs,
            final_pairwise_accuracy=self.pairwise_accuracy(examples, margin=margin),
            final_loss=final_loss,
            hidden_sizes=self.hidden_sizes,
        )

    def pairwise_accuracy(self, examples: list[CausalUtilityExample], margin: float = 0.05) -> float:
        pairs = self._pairs(examples, margin=margin)
        if not pairs:
            return 0.0
        correct = 0
        with torch.no_grad():
            for pos, neg in pairs:
                if self.score_raw(pos.features) > self.score_raw(neg.features):
                    correct += 1
        return correct / len(pairs)

    def save(self, path: str | Path, report: MLPTrainingReport | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        assert self._model is not None
        state_dict = {k: v.cpu().tolist() for k, v in self._model.state_dict().items()}
        payload = {
            "model_type": "pairwise_mlp_context_ranker",
            "feature_names": self.feature_names,
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "state_dict": state_dict,
            "metadata": dict(sorted(self.metadata.items())),
            "report": report.to_dict() if report else None,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MLPContextRanker":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_type") != "pairwise_mlp_context_ranker":
            raise ValueError(f"Not an MLP ranker checkpoint: {payload.get('model_type')}")
        return cls(
            feature_names=payload["feature_names"],
            hidden_sizes=payload["hidden_sizes"],
            dropout=payload.get("dropout", 0.0),
            state_dict=payload["state_dict"],
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


def load_ranker(path: str | Path) -> object:
    """Factory: load whichever ranker class matches the JSON's model_type."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model_type = payload.get("model_type", "pairwise_linear_context_ranker")
    if model_type == "pairwise_mlp_context_ranker":
        return MLPContextRanker.load(path)
    return PairwiseContextRanker.load(path)


__all__ = ["MLPContextRanker", "MLPTrainingReport", "load_ranker"]
