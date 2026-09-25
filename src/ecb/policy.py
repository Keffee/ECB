"""Conditional joint action density and masked candidate evidence encoder."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.distributions import Beta, Dirichlet


def masked_distribution(scores, mask):
    if scores.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("scores and boolean mask must have identical shape")
    if not mask.any(-1).all():
        raise ValueError("each row needs at least one valid candidate")
    if not torch.isfinite(scores[mask]).all():
        raise ValueError("valid scores must be finite")
    return scores.masked_fill(~mask, -torch.inf).softmax(-1)


def critic_features(state, candidates, scores, mask):
    distribution = masked_distribution(scores, mask)
    safe = candidates.masked_fill(~mask.unsqueeze(-1), 0.)
    pooled = (safe * distribution.unsqueeze(-1)).sum(-2)
    entropy = -(distribution * distribution.clamp_min(1e-12).log()).sum(-1, keepdim=True)
    return torch.cat((state, pooled, entropy, distribution.max(-1, keepdim=True).values), -1).detach()


@dataclass
class PolicySample:
    lambda_: torch.Tensor
    rho: torch.Tensor
    omega: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    head_log_probs: torch.Tensor


class JointPolicy(nn.Module):
    def __init__(self, state_dim, embedding_dim, hidden_dim=64):
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim), nn.Tanh())
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.lambda_head = nn.Sequential(nn.Linear(hidden_dim, 2), nn.Softplus())
        self.rho_head = nn.Sequential(nn.Linear(hidden_dim + 1, 3), nn.Softplus())
        self.omega_head = nn.Sequential(nn.Linear(hidden_dim + 4, 3), nn.Softplus())

    def encode(self, state, candidates, scores, mask):
        distribution = masked_distribution(scores, mask)
        if candidates.shape[:-1] != scores.shape or not torch.isfinite(candidates[mask]).all():
            raise ValueError("invalid candidate embeddings")
        safe = candidates.masked_fill(~mask.unsqueeze(-1), 0.)
        encoded = self.candidate_encoder(torch.cat((safe, distribution.unsqueeze(-1)), -1))
        pooled = (encoded * distribution.unsqueeze(-1)).sum(-2)
        return self.state_encoder(torch.cat((state, pooled), -1))

    def sample(self, state, candidates, scores, mask, deterministic=False):
        hidden = self.encode(state, candidates, scores, mask)
        parameters = self.lambda_head(hidden) + 1.
        lambda_dist = Beta(parameters[..., 0], parameters[..., 1])
        lam = lambda_dist.mean.detach() if deterministic else lambda_dist.sample()
        rho_dist = Dirichlet(self.rho_head(torch.cat((hidden, lam.unsqueeze(-1)), -1)) + 1.)
        rho = rho_dist.mean.detach() if deterministic else rho_dist.sample()
        omega_dist = Dirichlet(self.omega_head(torch.cat((hidden, lam.unsqueeze(-1), rho), -1)) + 1.)
        omega = omega_dist.mean.detach() if deterministic else omega_dist.sample()
        terms = torch.stack((lambda_dist.log_prob(lam), rho_dist.log_prob(rho),
                             omega_dist.log_prob(omega)), -1)
        # Conditional entropy of later heads also depends on upstream sampled heads.
        h_rho, h_omega = rho_dist.entropy(), omega_dist.entropy()
        entropy = lambda_dist.entropy() + h_rho + h_omega
        # Zero-valued score-function terms give the upstream contribution to the
        # gradient of expected conditional entropies.
        entropy = entropy + (terms[..., 0] - terms[..., 0].detach()) * (h_rho + h_omega).detach()
        entropy = entropy + (terms[..., 1] - terms[..., 1].detach()) * h_omega.detach()
        return PolicySample(lam, rho, omega, terms.sum(-1), entropy, terms)
