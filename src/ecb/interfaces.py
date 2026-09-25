"""Implement these adapters around existing, frozen recommendation components."""
from typing import Iterable, Protocol
from .types import (Candidate, Context, RankedList, RetrievalResult,
                    RewardExample, ScoredList, State, TrainingRecord)


class BaseRecommender(Protocol):
    def rank(self, state: State, pool: tuple[Candidate, ...], limit: int) -> RankedList:
        """Return unique candidates in descending base-score order."""


class EvidenceRanker(Protocol):
    def estimate_cost(self, state: State, candidates: tuple[Candidate, ...]) -> float:
        """Return a conservative upper bound in declared budget units."""

    def score(self, state: State, candidates: tuple[Candidate, ...]) -> ScoredList:
        """One scalar per candidate, same order; report actual usage."""


class Retriever(Protocol):
    def estimate_cost(self, quotas: tuple[int, int, int]) -> float:
        """Return a conservative upper bound, including permitted retries."""

    def retrieve(self, state: State, signal: tuple[float, ...],
                 quotas: tuple[int, int, int], excluded: frozenset[str]) -> RetrievalResult:
        """Query three sources; omit excluded IDs and report realized costs."""


class Integration(Protocol):
    base: BaseRecommender
    ranker: EvidenceRanker
    retriever: Retriever

    def training_records(self) -> Iterable[TrainingRecord]:
        """Fresh finite iterable for each epoch; training split only."""

    def reward_examples(self) -> Iterable[RewardExample]:
        """Scored, sampled training trajectories for reward-model warmup."""

    def evaluation_contexts(self) -> Iterable[Context]:
        """Target-free contexts in fixed order."""

    def evaluation_targets(self) -> Iterable[str]:
        """Read only after all inference trajectories have finished."""
