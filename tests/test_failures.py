import pytest

from ecb.types import Cost, RetrievalResult, ServiceFailure, RankedList
from test_workflow import setup_workflow, context, Ranker


def test_retrieval_failure_charged_and_last_base_returned():
    workflow, _, _, retriever = setup_workflow()
    def failed(*args):
        raise ServiceFailure("failed", Cost(units=.2, retries=1, physical_requests=2))
    retriever.retrieve = failed
    result = workflow.run(context())
    assert result.item_ids == ("a",)
    assert result.stop_reason == "retrieval_failure"
    assert result.cost.units == pytest.approx(.3)
    assert result.cost.retries == 1


def test_retrieval_budget_stop_keeps_scoring_cost():
    workflow, _, ranker, retriever = setup_workflow(budget=.15)
    result = workflow.run(context())
    assert ranker.calls == 1
    assert retriever.calls == 0
    assert result.cost.units == pytest.approx(.1)
    assert result.stop_reason == "retrieval_budget_limit"


def test_base_failure_after_retrieval_returns_previous_ranking():
    workflow, base, _, _ = setup_workflow()
    original = base.rank
    def failing(state, pool, limit):
        if pool:
            raise ServiceFailure("base unavailable")
        return original(state, pool, limit)
    base.rank = failing
    result = workflow.run(context())
    assert result.item_ids == ("a",)
    assert result.stop_reason == "retrieval_or_rerank_failure"
    assert result.cost.units == pytest.approx(.4)


def test_duplicate_base_ids_not_accepted():
    workflow, base, _, _ = setup_workflow()
    original = base.rank
    def duplicated(state, pool, limit):
        ranked = original(state, pool, limit)
        return RankedList(ranked.candidates * 2, (1., 0.))
    base.rank = duplicated
    assert workflow.run(context()).stop_reason == "initial_base_failure"


def test_cost_overrun_stops_and_keeps_realized_cost():
    class Overrun(Ranker):
        def estimate_cost(self, state, candidates):
            return .01
    workflow, _, _, retriever = setup_workflow(ranker=Overrun())
    result = workflow.run(context())
    assert result.stop_reason == "ranker_cost_overrun"
    assert result.cost.units == pytest.approx(.1)
    assert retriever.calls == 0



def test_retriever_must_respect_excluded_ids():
    from ecb.types import Candidate
    workflow, _, _, retriever = setup_workflow()
    def repeated(*args):
        return RetrievalResult((Candidate("a", (1., 0.)),), Cost(units=.1), (1, 0, 0))
    retriever.retrieve = repeated
    result = workflow.run(context())
    assert result.stop_reason == "retrieval_or_rerank_failure"
    assert result.item_ids == ("a",)
    assert result.cost.units == pytest.approx(.2)
