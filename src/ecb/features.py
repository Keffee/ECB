import math
import torch
from .types import Candidate, Context, RankedList


def vector(value, dimension):
    if len(value) != dimension or not all(math.isfinite(float(x)) for x in value):
        raise ValueError("expected finite vector of configured embedding_dim")
    return torch.tensor(value, dtype=torch.float32)


def validate_context(context: Context, dimension):
    if not isinstance(context, Context) or not isinstance(context.request_id, str) or not context.request_id:
        raise ValueError("expected a target-free Context with request_id")
    vector(context.embedding, dimension)
    if context.history_text and len(context.history_text) != len(context.history):
        raise ValueError("history_text must align with history embeddings")
    for item in context.history:
        vector(item, dimension)


def validate_candidates(candidates, dimension, limit):
    if len(candidates) > limit:
        raise ValueError("candidate limit exceeded")
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, Candidate) or not candidate.item_id or candidate.item_id in seen:
            raise ValueError("candidate identities must be nonempty and unique")
        seen.add(candidate.item_id)
        vector(candidate.embedding, dimension)


def validate_ranking(ranked: RankedList, config):
    validate_candidates(ranked.candidates, config.embedding_dim, config.shortlist_size)
    if len(ranked.logits) != len(ranked.candidates) or not all(math.isfinite(x) for x in ranked.logits):
        raise ValueError("base logits must align with candidates and be finite")
    if any(a < b for a, b in zip(ranked.logits, ranked.logits[1:])):
        raise ValueError("base output must be sorted in descending logit order")


def pre_score_features(state, ranked, pool, cost, round_index, config):
    d = config.embedding_dim
    history = torch.tensor(state.context.history).float().mean(0) if state.context.history else torch.zeros(d)
    context = vector(state.context.embedding, d)
    signal = vector(state.signal, d) if state.signal else torch.zeros(d)
    p = torch.tensor(ranked.logits).float().softmax(0)
    candidate_vectors = torch.tensor([c.embedding for c in ranked.candidates]).float()
    base = (p[:, None] * candidate_vectors).sum(0)
    pool_mean = torch.tensor([c.embedding for c in pool]).float().mean(0)
    entropy = float(-(p * p.clamp_min(1e-12).log()).sum()) / max(1., math.log(len(p)))
    gap = float(p[0] - p[1]) if len(p) > 1 else 1.
    scalars = torch.tensor([
        max(0., config.budget - cost.units) / config.budget,
        (config.max_rounds - round_index) / config.max_rounds,
        len(p) / config.shortlist_size, len(pool) / config.pool_size,
        entropy, gap, cost.units / config.budget, round_index / config.max_rounds,
    ])
    return torch.cat((history, context, signal, base, pool_mean, scalars)).unsqueeze(0)
