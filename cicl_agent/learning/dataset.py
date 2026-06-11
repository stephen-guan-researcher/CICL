"""Training-set builders: intervention probes and LLM counterfactual labels."""

from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from cicl_agent.agents.simulated import SimulatedReActAgent
from cicl_agent.causal.intervention import InterventionScorer
from cicl_agent.causal.llm_judge import LLMCausalContextJudge
from cicl_agent.core.protocols import AgentAdapter
from cicl_agent.core.schema import AgentRun, ContextUnit, Task
from cicl_agent.learning.features import ContextFeatureExtractor
from cicl_agent.memory.graph import InstanceContextGraph
from cicl_agent.retrieval.retrievers import HybridRetriever


@dataclass(slots=True)
class CausalUtilityExample:
    task_id: str
    context_id: str
    features: dict[str, float]
    action_delta: float
    outcome_delta: float
    negative_transfer: float
    reward: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict) -> "CausalUtilityExample":
        return cls(**row)


class CausalUtilityDatasetBuilder:
    """Generate trainable labels from intervention probes."""

    def __init__(
        self,
        graph: InstanceContextGraph,
        agent: AgentAdapter | None = None,
        history: list[AgentRun] | None = None,
        top_k: int = 20,
        neighbor_depth: int = 1,
    ) -> None:
        self.graph = graph
        self.agent = agent or SimulatedReActAgent()
        self.history = history or []
        self.top_k = top_k
        self.neighbor_depth = neighbor_depth

    def build(self, tasks: Iterable[Task]) -> list[CausalUtilityExample]:
        examples: list[CausalUtilityExample] = []
        for task in tasks:
            candidates = self._candidates(task)
            if not candidates:
                continue
            extractor = ContextFeatureExtractor(self.graph, history=self.history)
            interventions = {
                row.context_id: row
                for row in InterventionScorer(self.agent).score_candidates(
                    task, [unit for unit, _ in candidates]
                )
            }
            for unit, retrieval_score in candidates:
                row = interventions[unit.id]
                negative_transfer = float(row.success_delta < 0 or unit.metadata.get("stale", False))
                reward = self.reward(row.action_delta, row.success_delta, unit, negative_transfer)
                examples.append(
                    CausalUtilityExample(
                        task_id=task.id,
                        context_id=unit.id,
                        features=extractor.features(task, unit, retrieval_score=retrieval_score),
                        action_delta=row.action_delta,
                        outcome_delta=row.success_delta,
                        negative_transfer=negative_transfer,
                        reward=reward,
                        metadata={
                            "unit_type": unit.type,
                            "source": unit.source,
                            "no_context_success": row.no_context_success,
                            "with_context_success": row.with_context_success,
                        },
                    )
                )
        return examples

    def _candidates(self, task: Task) -> list[tuple[ContextUnit, float]]:
        retrieved = HybridRetriever(list(self.graph.units.values())).search(task, top_k=self.top_k)
        candidates: dict[str, tuple[ContextUnit, float]] = {unit.id: (unit, score) for unit, score in retrieved}
        for unit, score in retrieved[: max(1, self.top_k // 3)]:
            for neighbor in self.graph.neighbors(unit.id, depth=self.neighbor_depth):
                candidates.setdefault(neighbor.id, (neighbor, score * 0.6))
        return list(candidates.values())

    @staticmethod
    def reward(action_delta: float, success_delta: float, unit: ContextUnit, negative_transfer: float) -> float:
        return (
            1.0 * max(0.0, success_delta)
            + 0.45 * action_delta
            - 0.08 * min(1.0, unit.token_cost / 1000)
            - 0.35 * negative_transfer
        )


class LLMJudgmentDatasetBuilder(CausalUtilityDatasetBuilder):
    """Generate distillation labels from LLM counterfactual judgments.

    Supports concurrent labeling with a ThreadPoolExecutor and resumable
    checkpointing so a long Opus run can be interrupted and continued.
    """

    def __init__(
        self,
        graph: InstanceContextGraph,
        judge: LLMCausalContextJudge | None = None,
        history: list[AgentRun] | None = None,
        top_k: int = 20,
        neighbor_depth: int = 1,
        concurrency: int = 1,
        checkpoint_path: str | Path | None = None,
        max_retries: int = 3,
        progress_every: int = 25,
    ) -> None:
        super().__init__(graph, agent=None, history=history, top_k=top_k, neighbor_depth=neighbor_depth)
        self.judge = judge or LLMCausalContextJudge()
        self.concurrency = max(1, int(concurrency))
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.max_retries = max(1, int(max_retries))
        self.progress_every = max(1, int(progress_every))

    def build(self, tasks: Iterable[Task]) -> list[CausalUtilityExample]:
        tasks = list(tasks)
        seen: set[tuple[str, str]] = set()
        examples: list[CausalUtilityExample] = []
        if self.checkpoint_path and self.checkpoint_path.exists():
            for line in self.checkpoint_path.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                example = CausalUtilityExample.from_dict(row)
                seen.add((example.task_id, example.context_id))
                examples.append(example)
            print(f"[LLMJudgmentDatasetBuilder] resumed {len(examples)} examples from checkpoint", flush=True)

        extractor = ContextFeatureExtractor(self.graph, history=self.history)

        jobs: list[tuple[Task, ContextUnit, float]] = []
        for task in tasks:
            candidates = self._candidates(task)
            if not candidates:
                continue
            for unit, retrieval_score in candidates:
                if (task.id, unit.id) in seen:
                    continue
                jobs.append((task, unit, retrieval_score))

        if not jobs:
            return examples

        print(
            f"[LLMJudgmentDatasetBuilder] {len(jobs)} judgments to run "
            f"(concurrency={self.concurrency}, checkpoint={self.checkpoint_path})",
            flush=True,
        )

        checkpoint_lock = threading.Lock()
        checkpoint_fp = None
        if self.checkpoint_path:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_fp = self.checkpoint_path.open("a", encoding="utf-8")

        def write_checkpoint(example: CausalUtilityExample) -> None:
            if checkpoint_fp is None:
                return
            with checkpoint_lock:
                checkpoint_fp.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
                checkpoint_fp.flush()

        def work(job: tuple[Task, ContextUnit, float]) -> CausalUtilityExample | None:
            task, unit, retrieval_score = job
            last_err: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    judgment = self.judge.judge(task, unit)
                    last_err = None
                    break
                except Exception as exc:  # network/JSON errors
                    last_err = exc
                    time.sleep(min(8.0, 0.8 * (2 ** attempt)) + random.random() * 0.4)
            if last_err is not None:
                print(f"[LLMJudgmentDatasetBuilder] giving up on ({task.id}, {unit.id}): {last_err}", flush=True)
                return None
            return self._make_example(task, unit, retrieval_score, judgment, extractor)

        completed = 0
        start = time.time()
        try:
            if self.concurrency <= 1:
                for job in jobs:
                    example = work(job)
                    if example is None:
                        continue
                    examples.append(example)
                    write_checkpoint(example)
                    completed += 1
                    if completed % self.progress_every == 0:
                        elapsed = time.time() - start
                        rate = completed / elapsed if elapsed > 0 else 0
                        print(
                            f"[LLMJudgmentDatasetBuilder] {completed}/{len(jobs)} "
                            f"({rate:.2f} req/s, {elapsed:.0f}s elapsed)",
                            flush=True,
                        )
            else:
                with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                    futures = [pool.submit(work, job) for job in jobs]
                    for future in as_completed(futures):
                        example = future.result()
                        if example is None:
                            continue
                        examples.append(example)
                        write_checkpoint(example)
                        completed += 1
                        if completed % self.progress_every == 0:
                            elapsed = time.time() - start
                            rate = completed / elapsed if elapsed > 0 else 0
                            eta = (len(jobs) - completed) / rate if rate > 0 else 0
                            print(
                                f"[LLMJudgmentDatasetBuilder] {completed}/{len(jobs)} "
                                f"({rate:.2f} req/s, {elapsed:.0f}s elapsed, ~{eta:.0f}s ETA)",
                                flush=True,
                            )
        finally:
            if checkpoint_fp is not None:
                checkpoint_fp.close()

        print(
            f"[LLMJudgmentDatasetBuilder] done: {completed} new judgments, "
            f"{len(examples)} total examples (incl. resumed)",
            flush=True,
        )
        return examples

    def _make_example(
        self,
        task: Task,
        unit: ContextUnit,
        retrieval_score: float,
        judgment,
        extractor: ContextFeatureExtractor,
    ) -> CausalUtilityExample:
        negative_transfer = judgment.negative_transfer_risk
        reward = self.reward(
            judgment.action_shift,
            judgment.expected_outcome_uplift,
            unit,
            negative_transfer,
        )
        # NOTE: do NOT merge llm_* judgment fields into `features`. The ranker
        # treats the features dict as input X; including the teacher's
        # judgment there leaks the label and produces an artifact ranker that
        # only learns to copy llm_action_shift. Keep them on metadata for
        # downstream analysis instead.
        features = extractor.features(task, unit, retrieval_score=retrieval_score)
        return CausalUtilityExample(
            task_id=task.id,
            context_id=unit.id,
            features=features,
            action_delta=judgment.action_shift,
            outcome_delta=judgment.expected_outcome_uplift,
            negative_transfer=negative_transfer,
            reward=reward,
            metadata={
                "label_source": "llm_counterfactual",
                "unit_type": unit.type,
                "source": unit.source,
                "reason": judgment.reason,
                "no_context_action": judgment.no_context_action,
                "with_context_action": judgment.with_context_action,
                "llm_action_shift": judgment.action_shift,
                "llm_necessity": judgment.necessity,
                "llm_expected_uplift": judgment.expected_outcome_uplift,
                "llm_negative_transfer": judgment.negative_transfer_risk,
                "llm_confidence": judgment.confidence,
            },
        )


__all__ = [
    "CausalUtilityDatasetBuilder",
    "CausalUtilityExample",
    "LLMJudgmentDatasetBuilder",
]
