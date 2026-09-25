from dataclasses import FrozenInstanceError, fields

import pytest

from ecb.ranker import CandidateEvidence, OrganizedEvidenceRanker
from ecb.types import Candidate, Context, Cost, ServiceFailure, State


def state():
    return State(
        Context(
            "request-example",
            ((1.0, 0.0), (0.0, 1.0)),
            (0.5, 0.5),
            "a request based on observed information",
            history_text=("observed first item", "observed second item"),
        ),
        signal=(0.25, 0.75),
    )


def candidates():
    return (
        Candidate("first", (1.0, 0.0), "first item", ("catalog", "semantic match")),
        Candidate("second", (0.0, 1.0), "second item", ("recent item",)),
        Candidate("third", (0.5, 0.5), "third item"),
    )


class Provider:
    def __init__(self, outcomes=None, estimates=None):
        self.outcomes = outcomes or {}
        self.estimates = estimates or {}
        self.score_records = []
        self.estimate_records = []

    def estimate_cost(self, record):
        self.estimate_records.append(record)
        return self.estimates.get(record.candidate_id, 0.5)

    def score(self, record):
        self.score_records.append(record)
        result = self.outcomes.get(record.candidate_id, (0.5, Cost()))
        if isinstance(result, Exception):
            raise result
        return result


def test_organizes_target_free_immutable_evidence_for_each_candidate():
    provider = Provider()
    ranker = OrganizedEvidenceRanker(provider)

    ranker.score(state(), candidates()[:1])

    record = provider.score_records[0]
    assert record == CandidateEvidence(
        history_text=("observed first item", "observed second item"),
        request_text="a request based on observed information",
        retrieval_signal=(0.25, 0.75),
        candidate_id="first",
        candidate_text="first item",
        construction_evidence=("catalog", "semantic match"),
    )
    with pytest.raises(FrozenInstanceError):
        record.candidate_text = "changed"
    assert not any("target" in field.name for field in fields(record))


def test_score_preserves_order_and_counts_logical_calls_separately_from_usage():
    first_cost = Cost(units=0.1, logical_evaluations=8, cache_hits=1, latency_seconds=0.01)
    second_cost = Cost(
        units=0.3, logical_evaluations=6, physical_requests=2, retries=1,
        input_tokens=20, output_tokens=3, latency_seconds=0.25, billed_amount=0.002,
    )
    provider = Provider({"first": (0.8, first_cost), "second": (-0.2, second_cost)})

    result = OrganizedEvidenceRanker(provider).score(state(), candidates()[:2])

    assert result.item_ids == ("first", "second")
    assert result.scores == (0.8, -0.2)
    assert [record.candidate_id for record in provider.score_records] == ["first", "second"]
    assert result.cost.logical_evaluations == 2
    assert result.cost.physical_requests == 2
    assert result.cost.retries == 1
    assert result.cost.cache_hits == 1
    assert result.cost.input_tokens == 20
    assert result.cost.output_tokens == 3
    assert result.cost.units == pytest.approx(0.4)
    assert result.cost.latency_seconds == pytest.approx(0.26)
    assert result.cost.billed_amount == pytest.approx(0.002)


def test_estimate_sums_pointwise_upper_bounds_without_scoring():
    provider = Provider(estimates={"first": 0.4, "second": 0.7})
    ranker = OrganizedEvidenceRanker(provider)

    assert ranker.estimate_cost(state(), candidates()[:2]) == pytest.approx(1.1)
    assert provider.score_records == []
    assert [record.candidate_id for record in provider.estimate_records] == ["first", "second"]
    ranker.score(state(), candidates()[:2])
    assert provider.estimate_records == provider.score_records


@pytest.mark.parametrize("estimate", [-0.1, float("nan"), float("inf"), float("-inf"), "bad", None])
def test_rejects_invalid_cost_estimates_before_any_score_call(estimate):
    provider = Provider(estimates={"first": estimate})

    with pytest.raises(ValueError, match="finite and nonnegative"):
        OrganizedEvidenceRanker(provider).estimate_cost(state(), candidates())

    assert provider.score_records == []
    assert len(provider.estimate_records) == 1


def test_rejects_overflow_of_sum_of_finite_estimates():
    provider = Provider(estimates={"first": 1e308, "second": 1e308})

    with pytest.raises(ValueError, match="finite and nonnegative"):
        OrganizedEvidenceRanker(provider).estimate_cost(state(), candidates()[:2])


def test_provider_failure_retains_previous_and_failed_request_usage():
    original = ServiceFailure(
        "provider unavailable",
        Cost(units=0.4, logical_evaluations=9, physical_requests=3, retries=2, input_tokens=7),
    )
    provider = Provider({
        "first": (0.8, Cost(units=0.1, physical_requests=1, output_tokens=2)),
        "second": original,
    })

    with pytest.raises(ServiceFailure, match="provider unavailable") as captured:
        OrganizedEvidenceRanker(provider).score(state(), candidates())

    assert captured.value.__cause__ is original
    assert captured.value.cost == Cost(
        units=0.5, logical_evaluations=2, physical_requests=4, retries=2,
        input_tokens=7, output_tokens=2,
    )
    assert [record.candidate_id for record in provider.score_records] == ["first", "second"]


def test_unaccounted_provider_exception_retains_known_cost_and_attempted_call():
    original = RuntimeError("provider unavailable")
    provider = Provider({
        "first": (0.8, Cost(units=0.1, physical_requests=1)),
        "second": original,
    })

    with pytest.raises(ServiceFailure) as captured:
        OrganizedEvidenceRanker(provider).score(state(), candidates())

    assert captured.value.__cause__ is original
    assert captured.value.cost == Cost(units=0.1, logical_evaluations=2, physical_requests=1)
    assert len(provider.score_records) == 2


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), "bad", None])
def test_invalid_score_preserves_realized_cost_of_both_calls(score):
    provider = Provider({
        "first": (0.8, Cost(units=0.1, physical_requests=1)),
        "second": (score, Cost(units=0.2, physical_requests=2, retries=1)),
    })

    with pytest.raises(ServiceFailure, match="finite numeric score") as captured:
        OrganizedEvidenceRanker(provider).score(state(), candidates())

    assert captured.value.cost.units == pytest.approx(0.3)
    assert captured.value.cost.logical_evaluations == 2
    assert captured.value.cost.physical_requests == 3
    assert captured.value.cost.retries == 1
    assert len(provider.score_records) == 2


def test_empty_candidates_do_not_call_provider():
    provider = Provider()
    ranker = OrganizedEvidenceRanker(provider)

    assert ranker.estimate_cost(state(), ()) == 0.0
    result = ranker.score(state(), ())

    assert result.item_ids == result.scores == ()
    assert result.cost == Cost()
    assert provider.score_records == provider.estimate_records == []


def test_context_without_history_text_produces_empty_text_history():
    provider = Provider()
    empty_state = State(Context("request-example", (), (1.0, 0.0)))

    OrganizedEvidenceRanker(provider).score(empty_state, candidates()[:1])

    assert provider.score_records[0].history_text == ()
    assert provider.score_records[0].request_text == ""
    assert provider.score_records[0].retrieval_signal == ()
