"""Pointwise scoring of organized, observed evidence through an external provider."""
from dataclasses import dataclass, replace
import math
from typing import Protocol

from .types import Candidate, Cost, ScoredList, ServiceFailure, State


@dataclass(frozen=True)
class CandidateEvidence:
    """Target-free evidence for one candidate; embeddings stay in the core state."""

    history_text: tuple[str, ...]
    request_text: str
    retrieval_signal: tuple[float, ...]
    candidate_id: str
    candidate_text: str
    construction_evidence: tuple[str, ...]


class CandidateEvidenceProvider(Protocol):
    """Implement service access, caching, and retries behind this boundary.

    Report physical requests, retries, cache hits, tokens, latency, billing, and
    declared budget units in Cost. The adapter owns logical evaluation counts.
    Raise ServiceFailure with all usage incurred by a failed candidate call.
    """

    def estimate_cost(self, record: CandidateEvidence) -> float:
        """Return a finite, nonnegative upper bound including allowed retries."""

    def score(self, record: CandidateEvidence) -> tuple[float, Cost]:
        """Return one finite score and actual usage, including cache or retries."""


class OrganizedEvidenceRanker:
    """Adapt a pointwise provider to the EvidenceRanker interface.

    Each candidate is exposed once in input order. A cache hit still counts as
    one logical evaluation. Provider-reported logical counts are replaced, so
    retries cannot inflate the number of logical candidate evaluations.
    """

    def __init__(self, provider: CandidateEvidenceProvider):
        self.provider = provider

    @staticmethod
    def _record(state: State, candidate: Candidate) -> CandidateEvidence:
        return CandidateEvidence(
            history_text=tuple(state.context.history_text),
            request_text=state.context.text,
            retrieval_signal=tuple(state.signal),
            candidate_id=candidate.item_id,
            candidate_text=candidate.text,
            construction_evidence=tuple(candidate.evidence),
        )

    def estimate_cost(self, state: State, candidates: tuple[Candidate, ...]) -> float:
        """Sum validated provider bounds without making scoring requests."""
        total = 0.0
        for candidate in candidates:
            estimate = self.provider.estimate_cost(self._record(state, candidate))
            try:
                valid = math.isfinite(estimate) and estimate >= 0
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise ValueError("provider cost estimates must be finite and nonnegative")
            total += float(estimate)
            if not math.isfinite(total):
                raise ValueError("total cost estimate must be finite and nonnegative")
        return total

    def score(self, state: State, candidates: tuple[Candidate, ...]) -> ScoredList:
        """Score in order and retain all known usage if any candidate fails.

        Unexpected provider exceptions preserve prior usage and the attempted
        logical call, but cannot supply unreported physical usage. Providers
        must use ServiceFailure to account for failed requests and retries.
        """
        cost = Cost()
        scores = []
        for candidate in candidates:
            record = self._record(state, candidate)
            cost += Cost(logical_evaluations=1)
            try:
                score, usage = self.provider.score(record)
            except ServiceFailure as error:
                cost += replace(error.cost, logical_evaluations=0)
                raise ServiceFailure(str(error), cost) from error
            except Exception as error:
                raise ServiceFailure("evidence provider failed", cost) from error
            if not isinstance(usage, Cost):
                raise ServiceFailure("evidence provider must return Cost usage", cost)
            cost += replace(usage, logical_evaluations=0)
            try:
                valid = math.isfinite(score)
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise ServiceFailure("evidence provider must return a finite numeric score", cost)
            scores.append(float(score))
        return ScoredList(tuple(candidate.item_id for candidate in candidates), tuple(scores), cost)
