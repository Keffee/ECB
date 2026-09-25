import torch
import pytest

from ecb.training import reward_to_go
from ecb.evaluation import ranking_metrics
from ecb.reward import TrajectoryReward
from test_workflow import setup_workflow, context


def test_returns_charge_remaining_costs_only():
    assert reward_to_go(1., [.1, .2, .3]) == pytest.approx([.4, .5, .7])


def test_reward_depends_on_terminal_workflow():
    torch.manual_seed(12)
    reward = TrajectoryReward(2, 12)
    a, _, _, _ = setup_workflow(False)
    b, _, _, _ = setup_workflow(True)
    stopped, upgraded = a.run(context()), b.run(context())
    x = reward.features(stopped, (0., 1.))
    y = reward.features(upgraded, (0., 1.))
    assert x.shape == y.shape
    assert not torch.equal(x, y)
    reward.freeze()
    assert not any(p.requires_grad for p in reward.parameters())


def test_metrics_keep_misses_and_failures_in_denominator():
    metrics = ranking_metrics([("a",), (), ("b", "a")], ["a", "b", "a"], (1, 2))
    assert metrics["count"] == 3
    assert metrics["hit@1"] == pytest.approx(1/3)
    assert metrics["hit@2"] == pytest.approx(2/3)
    with pytest.raises(ValueError):
        ranking_metrics([("a",)], [], (1,))
