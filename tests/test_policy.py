import pytest
import torch

from ecb.policy import JointPolicy, masked_distribution
from ecb.retrieval import allocate_quotas


def test_masked_softmax_ignores_padding():
    scores = torch.tensor([[1.0, 2.0, float("nan")]])
    mask = torch.tensor([[True, True, False]])
    result = masked_distribution(scores, mask)
    assert result[0, 2] == 0
    assert torch.allclose(result[0, :2], torch.softmax(scores[0, :2], -1))
    with pytest.raises(ValueError):
        masked_distribution(scores, torch.zeros_like(mask))


def test_actor_joint_density_and_all_heads_have_gradients():
    torch.manual_seed(17)
    policy = JointPolicy(6, 4, 12)
    state = torch.randn(2, 6)
    candidates = torch.randn(2, 3, 4)
    scores = torch.randn(2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    action = policy.sample(state, candidates, scores, mask)
    assert not action.lambda_.requires_grad
    assert not action.rho.requires_grad
    assert not action.omega.requires_grad
    assert torch.allclose(action.log_prob, action.head_log_probs.sum(-1))
    assert torch.allclose(action.rho.sum(-1), torch.ones(2))
    assert torch.allclose(action.omega.sum(-1), torch.ones(2))
    (-action.log_prob.mean()).backward()
    for head in (policy.lambda_head, policy.rho_head, policy.omega_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_candidate_score_pairs_are_permutation_invariant():
    torch.manual_seed(8)
    policy = JointPolicy(6, 4, 12)
    state = torch.randn(1, 6)
    candidates = torch.randn(1, 4, 4)
    scores = torch.tensor([[1., 3., 2., 0.]])
    mask = torch.tensor([[True, True, True, False]])
    expected = policy.encode(state, candidates, scores, mask)
    permutation = [2, 0, 3, 1]
    actual = policy.encode(state, candidates[:, permutation], scores[:, permutation], mask[:, permutation])
    assert torch.allclose(expected, actual, atol=1e-6)
    candidates[:, 3] = float("nan")
    scores[:, 3] = float("nan")
    assert torch.allclose(expected, policy.encode(state, candidates, scores, mask), atol=1e-6)


def test_distribution_is_candidate_conditioned():
    torch.manual_seed(8)
    policy = JointPolicy(6, 4, 12)
    state, candidates = torch.randn(1, 6), torch.randn(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    a = policy.encode(state, candidates, torch.tensor([[7., 0., 0.]]), mask)
    b = policy.encode(state, candidates, torch.tensor([[0., 7., 0.]]), mask)
    assert not torch.allclose(a, b)


def test_quotas_preserve_exact_budget():
    assert allocate_quotas((.2, .5, .3), 7) == (1, 4, 2)
    for budget in range(20):
        quotas = allocate_quotas((.001, .499, .5), budget)
        assert sum(quotas) == budget
        assert min(quotas) >= 0
    with pytest.raises(ValueError):
        allocate_quotas((0., 0., 0.), 4)



def test_lambda_and_rho_change_retrieval_signal():
    from ecb.retrieval import retrieval_signal
    from ecb.types import Action, Candidate, Context, State
    state = State(Context("r", ((1., 0.), (0., 1.)), (.7, .3)))
    candidates = (Candidate("a", (0., 1.)),)
    uniform = retrieval_signal(state, candidates, (1.,), Action(0., (1., 0., 0.), (.2, .3, .5)))
    recent = retrieval_signal(state, candidates, (1.,), Action(1., (1., 0., 0.), (.2, .3, .5)))
    request = retrieval_signal(state, candidates, (1.,), Action(1., (0., 1., 0.), (.2, .3, .5)))
    assert uniform != recent
    assert recent != request
