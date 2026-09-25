"""Declared mapping from the continuous action to a bounded retrieval signal."""
import math
import torch
import torch.nn.functional as F
from .types import Action, State


def allocate_quotas(weights, slots):
    if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("expected three finite nonnegative source weights")
    if not isinstance(slots, int) or slots < 0 or sum(weights) <= 0:
        raise ValueError("invalid retrieval budget or weights")
    raw = [slots * weight / sum(weights) for weight in weights]
    quotas = [math.floor(value) for value in raw]
    order = sorted(range(3), key=lambda i: (-(raw[i] - quotas[i]), i))
    for i in order[:slots - sum(quotas)]:
        quotas[i] += 1
    return tuple(quotas)


def retrieval_signal(state: State, candidates, distribution, action: Action):
    dimension = len(state.context.embedding)
    if state.context.history:
        history = torch.tensor(state.context.history, dtype=torch.float32)
        positions = torch.linspace(-1., 0., len(history))
        # lambda smoothly changes uniform history pooling to recent-interest pooling.
        recency = (10. * action.lambda_ * positions).softmax(0)
        interest = (history * recency[:, None]).sum(0)
    else:
        interest = torch.zeros(dimension)
    evidence = (torch.tensor([c.embedding for c in candidates], dtype=torch.float32) *
                torch.tensor(distribution, dtype=torch.float32)[:, None]).sum(0)
    parts = (interest, torch.tensor(state.context.embedding, dtype=torch.float32), evidence)
    mixed = sum(weight * F.normalize(part, dim=0) for weight, part in zip(action.rho, parts))
    return tuple(float(x) for x in F.normalize(mixed, dim=0))
