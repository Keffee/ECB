"""Records shared with external adapters. Inference records have no target field."""
from dataclasses import asdict, dataclass, field
import math
from typing import Any
import torch


@dataclass(frozen=True)
class Candidate:
    item_id: str
    embedding: tuple[float, ...]
    text: str = ""
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class Context:
    request_id: str
    history: tuple[tuple[float, ...], ...]
    embedding: tuple[float, ...]
    text: str = ""
    history_text: tuple[str, ...] = ()


@dataclass(frozen=True)
class State:
    context: Context
    signal: tuple[float, ...] = ()


@dataclass(frozen=True)
class Cost:
    # units are explicitly chosen by the adapter: e.g. request/token cost proxies.
    units: float = 0.0
    logical_evaluations: int = 0
    physical_requests: int = 0
    retries: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    billed_amount: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid cost field: {name}")
            if name not in ("units", "latency_seconds", "billed_amount") and int(value) != value:
                raise ValueError(f"{name} must be an integer")

    def __add__(self, other):
        return Cost(**{key: value + asdict(other)[key] for key, value in asdict(self).items()})


@dataclass(frozen=True)
class RankedList:
    candidates: tuple[Candidate, ...]
    logits: tuple[float, ...]


@dataclass(frozen=True)
class ScoredList:
    item_ids: tuple[str, ...]
    scores: tuple[float, ...]
    cost: Cost


@dataclass(frozen=True)
class RetrievalResult:
    candidates: tuple[Candidate, ...]
    cost: Cost
    source_counts: tuple[int, int, int]


class ServiceFailure(RuntimeError):
    """Adapters raise this with costs already incurred, including failed retries."""
    def __init__(self, message: str, cost: Cost = Cost()):
        super().__init__(message)
        self.cost = cost


@dataclass(frozen=True)
class Action:
    lambda_: float
    rho: tuple[float, float, float]
    omega: tuple[float, float, float]


@dataclass
class Step:
    pre_features: torch.Tensor
    before_state: State
    before_ranked: RankedList
    before_pool: tuple[Candidate, ...]
    cost_before: Cost
    action: Action | None = None
    quotas: tuple[int, int, int] | None = None
    evidence_distribution: tuple[float, ...] = ()
    cost: Cost = field(default_factory=Cost)
    log_prob: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
    critic_features: torch.Tensor | None = None
    reason: str = "complete"
    ranker_cost: Cost = field(default_factory=Cost)
    retrieval_cost: Cost = field(default_factory=Cost)
    source_counts: tuple[int, int, int] = (0, 0, 0)
    added_ids: tuple[str, ...] = ()


@dataclass
class Trajectory:
    state: State
    ranked: RankedList
    pool: tuple[Candidate, ...]
    steps: list[Step]
    cost: Cost
    stop_reason: str
    decisions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def item_ids(self):
        return tuple(candidate.item_id for candidate in self.ranked.candidates)

    def prefix(self, index):
        step = self.steps[index]
        return Trajectory(step.before_state, step.before_ranked, step.before_pool,
                          self.steps[:index], step.cost_before, "training_stop_reference", self.decisions[:index])

    def public_trace(self) -> dict[str, Any]:
        return {
            "request_id": self.state.context.request_id,
            "item_ids": self.item_ids,
            "stop_reason": self.stop_reason,
            "cost": asdict(self.cost),
            "decisions": self.decisions,
            "steps": [{
                "candidates": [c.item_id for c in step.before_ranked.candidates],
                "action": asdict(step.action) if step.action else None,
                "quotas": step.quotas,
                "evidence_distribution": step.evidence_distribution,
                "cost": asdict(step.cost),
                "reason": step.reason,
                "ranker_cost": asdict(step.ranker_cost),
                "retrieval_cost": asdict(step.retrieval_cost),
                "source_counts": list(step.source_counts),
                "added_ids": list(step.added_ids),
            } for step in self.steps],
        }


@dataclass(frozen=True)
class TrainingRecord:
    context: Context
    target_embedding: tuple[float, ...]
    split: str = "train"

    def __post_init__(self):
        if self.split != "train":
            raise ValueError("controller fitting accepts training records only")


@dataclass(frozen=True)
class RewardExample:
    trajectory: Trajectory
    target_embedding: tuple[float, ...]
    label: float
    split: str = "train"

    def __post_init__(self):
        if self.split != "train":
            raise ValueError("reward fitting accepts training records only")
        if not math.isfinite(self.label) or not 0 <= self.label <= 1:
            raise ValueError("reward labels must be finite in [0, 1]")
