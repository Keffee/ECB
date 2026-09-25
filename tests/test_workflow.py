from dataclasses import fields

import pytest
import torch

from ecb.types import Candidate, Context, Cost, RankedList, ScoredList, RetrievalResult, ServiceFailure
from ecb.config import Config
from ecb.models import Gate
from ecb.policy import JointPolicy
from ecb.workflow import Workflow


def context():
    return Context("request-1", ((1., 0.), (0., 1.)), (.5, .5), "observed context")


class Base:
    def __init__(self):
        self.calls = 0

    def rank(self, state, pool, limit):
        self.calls += 1
        candidates = {c.item_id: c for c in pool}
        candidates.setdefault("a", Candidate("a", (1., 0.), "item a"))
        ordered = sorted(candidates.values(), key=lambda c: c.item_id, reverse=True)[:limit]
        return RankedList(tuple(ordered), tuple(float(len(ordered) - i) for i in range(len(ordered))))


class Ranker:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def estimate_cost(self, state, candidates):
        return .1 * len(candidates)

    def score(self, state, candidates):
        self.calls += 1
        cost = Cost(units=.1 * len(candidates), physical_requests=1, logical_evaluations=len(candidates))
        if self.fail:
            raise ServiceFailure("ranker unavailable", cost)
        return ScoredList(tuple(c.item_id for c in candidates), tuple(float(i) for i in range(len(candidates))), cost)


class Retriever:
    def __init__(self):
        self.calls = 0
        self.signals = []

    def estimate_cost(self, quotas):
        return .1 * sum(quotas)

    def retrieve(self, state, signal, quotas, excluded):
        self.calls += 1
        self.signals.append(signal)
        c = Candidate(chr(ord("a") + self.calls), (0., 1.), "retrieved item")
        return RetrievalResult((c,), Cost(units=.1 * sum(quotas)), quotas)


def setup_workflow(upgrade=True, budget=10., rounds=2, ranker=None):
    config = Config(embedding_dim=2, hidden_dim=12, max_rounds=rounds, shortlist_size=5, pool_size=8, retrieval_slots=3, budget=budget)
    gate = Gate(config.state_dim, 12)
    with torch.no_grad():
        for p in gate.parameters():
            p.zero_()
        gate.network[-1].bias.fill_(1 if upgrade else -1)
    base, ranker, retriever = Base(), ranker or Ranker(), Retriever()
    actor = JointPolicy(config.state_dim, 2, 12)
    return Workflow(config, base, ranker, retriever, gate, actor), base, ranker, retriever


def test_stop_returns_base_without_scoring_or_retrieval():
    workflow, base, ranker, retriever = setup_workflow(False)
    result = workflow.run(context())
    assert result.item_ids == ("a",)
    assert result.stop_reason == "gate_stop"
    assert ranker.calls == retriever.calls == 0
    assert result.cost.units == 0


def test_latest_signal_replaces_but_pool_persists():
    workflow, base, ranker, retriever = setup_workflow()
    result = workflow.run(context())
    assert ranker.calls == retriever.calls == 2
    assert base.calls == 3
    assert result.item_ids == ("c", "b", "a")
    assert result.state.context.history == context().history
    assert result.state.signal == retriever.signals[-1]
    assert len(result.state.signal) == 2
    assert result.stop_reason == "round_limit"


def test_budget_prevents_ranker_call():
    workflow, _, ranker, retriever = setup_workflow(budget=.05)
    result = workflow.run(context())
    assert result.cost.units == 0
    assert ranker.calls == retriever.calls == 0
    assert result.stop_reason == "budget_limit"


def test_ranker_failure_keeps_last_complete_ranking_and_cost():
    workflow, _, ranker, retriever = setup_workflow(ranker=Ranker(fail=True))
    result = workflow.run(context())
    assert result.item_ids == ("a",)
    assert result.stop_reason == "ranker_failure"
    assert result.cost.units == pytest.approx(.1)
    assert result.cost.physical_requests == 1
    assert retriever.calls == 0


def test_target_is_not_an_inference_field():
    for cls in (Context, Candidate):
        assert all("target" not in f.name for f in fields(cls))
    workflow, _, _, _ = setup_workflow(False)
    result = workflow.run(context())
    assert "target" not in result.public_trace()


def test_misaligned_scores_rejected_and_charged():
    class Wrong(Ranker):
        def score(self, state, candidates):
            return ScoredList(("wrong",), (2.,), Cost(units=.1))
    workflow, _, _, _ = setup_workflow(ranker=Wrong())
    result = workflow.run(context())
    assert result.stop_reason == "invalid_ranker_output"
    assert result.cost.units == pytest.approx(.1)


def test_integer_context_vectors_are_supported():
    from dataclasses import replace
    workflow, _, _, _ = setup_workflow()
    value = replace(context(), history=((1, 0), (0, 1)), embedding=(1, 0))
    assert workflow.run(value).stop_reason == "round_limit"


def test_stop_decision_and_pre_score_state_are_auditable():
    workflow, _, _, _ = setup_workflow(False)
    trace = workflow.run(context()).public_trace()
    assert trace["decisions"][0]["decision"] == "stop"
    assert trace["decisions"][0]["base_logits"] == [1.]
    assert trace["decisions"][0]["gate_value"] < 0
    assert trace["decisions"][0]["ranker_called"] is False


def test_trace_keeps_service_costs_and_added_candidates():
    workflow, _, _, _ = setup_workflow(rounds=1)
    trace = workflow.run(context()).public_trace()
    assert trace["steps"][0]["ranker_cost"]["units"] == pytest.approx(.1)
    assert trace["steps"][0]["retrieval_cost"]["units"] == pytest.approx(.3)
    assert trace["steps"][0]["added_ids"] == ["b"]
    assert len(trace["decisions"]) == 2
    assert trace["decisions"][-1]["reason"] == "round_limit"



def test_actor_observes_budget_after_ranker_charge():
    workflow, _, _, _ = setup_workflow(rounds=1)
    observed = []
    original = workflow.actor.sample
    def capture(state, *args, **kwargs):
        observed.append(state.detach().clone())
        return original(state, *args, **kwargs)
    workflow.actor.sample = capture
    workflow.run(context())
    assert observed[0][0, -8].item() == pytest.approx(.99)
    assert observed[0][0, -2].item() == pytest.approx(.01)
