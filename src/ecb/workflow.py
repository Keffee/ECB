"""Gate-first execution. The held-out target is never an argument."""
from dataclasses import replace
import math
import torch

from .features import (pre_score_features, validate_candidates, validate_context,
                       validate_ranking)
from .policy import critic_features, masked_distribution
from .retrieval import allocate_quotas, retrieval_signal
from .types import (Action, Cost, RankedList, ServiceFailure, State, Step, Trajectory)


def merge_pool(previous, additions, limit):
    by_id = {item.item_id: item for item in previous}
    for item in additions:
        by_id.setdefault(item.item_id, item)
    return tuple(by_id.values())[:limit]


def estimate(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("adapter cost bound must be finite and nonnegative")
    return float(value)


class Workflow:
    def __init__(self, config, base, ranker, retriever, gate, actor):
        self.config, self.base = config, base
        self.ranker, self.retriever = ranker, retriever
        self.gate, self.actor = gate, actor

    def run(self, context, *, training=False, deterministic=False,
            force_first=False, force_all=False):
        cfg = self.config
        validate_context(context, cfg.embedding_dim)
        context = replace(context, history=tuple(tuple(x) for x in context.history[-cfg.history_size:]),
                          embedding=tuple(context.embedding),
                          history_text=tuple(context.history_text[-cfg.history_size:]))
        state, cost, steps, pool = State(context), Cost(), [], ()
        try:
            with torch.no_grad():
                ranked = self.base.rank(state, pool, cfg.shortlist_size)
            validate_ranking(ranked, cfg)
        except (ServiceFailure, ValueError):
            return Trajectory(state, RankedList((), ()), (), [], cost, "initial_base_failure")
        pool = merge_pool((), ranked.candidates, cfg.pool_size)
        if not ranked.candidates:
            return Trajectory(state, ranked, pool, steps, cost, "empty_candidates")
        reason = "round_limit"
        decisions = []
        device = next(self.actor.parameters()).device

        for t in range(cfg.max_rounds):
            if cost.units >= cfg.budget:
                reason = "budget_limit"
                break
            if len(pool) >= cfg.pool_size:
                reason = "pool_limit"
                break
            features = pre_score_features(state, ranked, pool, cost, t, cfg).to(device)
            with torch.no_grad():
                gate_value = float(self.gate(features).item())
                forced = force_all or (force_first and t == 0)
                upgrade = forced or gate_value > cfg.gate_threshold
            decision = {
                "round": t, "decision": "upgrade" if upgrade else "stop",
                "reason": "forced_training" if forced else "gate",
                "gate_value": gate_value, "gate_threshold": cfg.gate_threshold,
                "remaining_budget": max(0., cfg.budget - cost.units),
                "remaining_rounds": cfg.max_rounds - t,
                "candidate_ids": [c.item_id for c in ranked.candidates],
                "base_logits": list(ranked.logits),
                "pre_score_features": features[0].detach().cpu().tolist(),
                "ranker_called": False,
            }
            decisions.append(decision)
            if not upgrade:
                reason = "gate_stop"
                break
            slots = min(cfg.retrieval_slots, cfg.pool_size - len(pool))
            rank_bound = estimate(self.ranker.estimate_cost(state, ranked.candidates))
            if cost.units + rank_bound > cfg.budget:
                reason = "budget_limit"
                break
            step = Step(features.detach(), state, ranked, pool, cost)
            steps.append(step)
            decision["ranker_called"] = True
            try:
                with torch.no_grad():
                    scored = self.ranker.score(state, ranked.candidates)
                step.ranker_cost = scored.cost
                step.cost = step.cost + scored.cost
                cost = cost + scored.cost
            except ServiceFailure as exc:
                step.ranker_cost = exc.cost
                step.cost = step.cost + exc.cost
                cost = cost + exc.cost
                step.reason = reason = "ranker_failure"
                break
            if scored.cost.units > rank_bound + 1e-8 or cost.units > cfg.budget:
                step.reason = reason = "ranker_cost_overrun"
                break
            if scored.item_ids != tuple(c.item_id for c in ranked.candidates) or len(scored.scores) != len(ranked.candidates):
                step.reason = reason = "invalid_ranker_output"
                break
            candidate_tensor = torch.tensor([[c.embedding for c in ranked.candidates]],
                                            dtype=torch.float32, device=device)
            scores = torch.tensor([scored.scores], dtype=torch.float32, device=device)
            mask = torch.ones_like(scores, dtype=torch.bool)
            try:
                distribution = masked_distribution(scores, mask)
            except ValueError:
                step.reason = reason = "invalid_ranker_output"
                break
            post_features = pre_score_features(state, ranked, pool, cost, t, cfg).to(device)
            with torch.set_grad_enabled(training):
                sample = self.actor.sample(post_features, candidate_tensor, scores, mask,
                                           deterministic=deterministic)
            action = Action(float(sample.lambda_[0]), tuple(sample.rho[0].tolist()),
                            tuple(sample.omega[0].tolist()))
            quotas = allocate_quotas(action.omega, slots)
            step.action, step.quotas = action, quotas
            step.evidence_distribution = tuple(distribution[0].tolist())
            step.log_prob, step.entropy = sample.log_prob.squeeze(0), sample.entropy.squeeze(0)
            step.critic_features = critic_features(post_features, candidate_tensor, scores, mask)
            retrieval_bound = estimate(self.retriever.estimate_cost(quotas))
            if cost.units + retrieval_bound > cfg.budget:
                step.reason = reason = "retrieval_budget_limit"
                break
            signal = retrieval_signal(state, ranked.candidates, step.evidence_distribution, action)
            proposed_state = State(context, signal)
            try:
                with torch.no_grad():
                    retrieved = self.retriever.retrieve(
                        state, signal, quotas, frozenset(c.item_id for c in pool))
                step.retrieval_cost = retrieved.cost
                step.cost = step.cost + retrieved.cost
                cost = cost + retrieved.cost
            except ServiceFailure as exc:
                step.retrieval_cost = exc.cost
                step.cost = step.cost + exc.cost
                cost = cost + exc.cost
                step.reason = reason = "retrieval_failure"
                break
            if retrieved.cost.units > retrieval_bound + 1e-8 or cost.units > cfg.budget:
                step.reason = reason = "retrieval_cost_overrun"
                break
            try:
                validate_candidates(retrieved.candidates, cfg.embedding_dim, slots)
                if len(retrieved.source_counts) != 3 or any(
                        not isinstance(n, int) or not 0 <= n <= q
                        for n, q in zip(retrieved.source_counts, quotas)):
                    raise ValueError("source counts exceed requested quotas")
                if len(retrieved.candidates) > sum(retrieved.source_counts):
                    raise ValueError("returned more unique candidates than source hits")
                if set(c.item_id for c in retrieved.candidates) & set(c.item_id for c in pool):
                    raise ValueError("retriever returned excluded candidate IDs")
                step.source_counts = retrieved.source_counts
                step.added_ids = tuple(c.item_id for c in retrieved.candidates)
                proposed_pool = merge_pool(pool, retrieved.candidates, cfg.pool_size)
                with torch.no_grad():
                    new_ranked = self.base.rank(proposed_state, proposed_pool, cfg.shortlist_size)
                validate_ranking(new_ranked, cfg)
                if not new_ranked.candidates or not set(c.item_id for c in new_ranked.candidates).issubset(
                        c.item_id for c in proposed_pool):
                    raise ValueError("base reranking must use the exposed pool")
            except (ServiceFailure, ValueError):
                step.reason = reason = "retrieval_or_rerank_failure"
                break
            state, pool, ranked = proposed_state, proposed_pool, new_ranked
        if reason != "gate_stop":
            decisions.append({
                "round": len(steps), "decision": "stop", "reason": reason,
                "gate_value": None, "gate_threshold": cfg.gate_threshold,
                "remaining_budget": max(0., cfg.budget - cost.units),
                "remaining_rounds": max(0, cfg.max_rounds - len(steps)),
                "candidate_ids": [c.item_id for c in ranked.candidates],
                "base_logits": list(ranked.logits), "ranker_called": False,
            })
        return Trajectory(state, ranked, pool, steps, cost, reason, decisions)
