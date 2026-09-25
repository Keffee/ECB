from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Config:
    embedding_dim: int = 128
    hidden_dim: int = 64
    max_rounds: int = 3
    history_size: int = 32
    shortlist_size: int = 20
    pool_size: int = 100
    retrieval_slots: int = 6
    budget: float = 100.0
    seed: int = 42
    epochs: int = 10
    warmup_epochs: int = 1
    batch_size: int = 8
    gate_passes: int = 5
    learning_rate: float = 0.001
    entropy_weight: float = 0.001
    gradient_clip: float = 5.0
    gate_threshold: float = 0.0

    def __post_init__(self):
        for name in ("embedding_dim", "hidden_dim", "max_rounds", "history_size",
                     "shortlist_size", "pool_size", "retrieval_slots", "epochs",
                     "batch_size", "gate_passes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.pool_size < self.shortlist_size:
            raise ValueError("pool_size must be >= shortlist_size")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must be in [0, epochs)")
        for name in ("budget", "learning_rate", "gradient_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.entropy_weight) or self.entropy_weight < 0:
            raise ValueError("entropy_weight must be finite and nonnegative")
        if not math.isfinite(self.gate_threshold):
            raise ValueError("gate_threshold must be finite")

    @property
    def state_dim(self):
        return 5 * self.embedding_dim + 8

    @property
    def critic_dim(self):
        return self.state_dim + self.embedding_dim + 2

    def as_dict(self):
        return asdict(self)
