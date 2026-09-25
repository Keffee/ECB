"""Target-conditioned trajectory utility, fitted only on training records."""
import math
import torch
from torch import nn
from .features import vector
from .types import RewardExample, Trajectory


class TrajectoryReward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(nn.Linear(6 * embedding_dim + 3, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, 1))
        self.fitted = False

    def features(self, trajectory: Trajectory, target_embedding):
        d = self.embedding_dim
        context = trajectory.state.context
        history = torch.tensor(context.history).float().mean(0) if context.history else torch.zeros(d)
        request = vector(context.embedding, d)
        signal = vector(trajectory.state.signal, d) if trajectory.state.signal else torch.zeros(d)
        target = vector(target_embedding, d)
        candidates = trajectory.ranked.candidates
        if candidates:
            ranks = torch.arange(1, len(candidates) + 1, dtype=torch.float32)
            weights = 1. / torch.log2(ranks + 1.)
            weights = weights / weights.sum()
            final_list = (weights[:, None] * torch.tensor([c.embedding for c in candidates])).sum(0)
        else:
            final_list = torch.zeros(d)
        # Utility sees the actual terminal list and final signal. Cost is subtracted
        # by the trainer, so budget units are deliberately not included here.
        scalars = torch.tensor([math.log1p(len(candidates)), math.log1p(len(trajectory.pool)),
                                math.log1p(len(trajectory.steps))])
        return torch.cat((history, request, signal, target, final_list, target * final_list, scalars))

    def forward(self, features):
        return self.network(features).squeeze(-1).sigmoid()

    def utility(self, trajectory, target_embedding):
        with torch.no_grad():
            features = self.features(trajectory, target_embedding).to(next(self.parameters()).device)
            return float(self(features).item())

    def freeze(self):
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def fit(self, examples, *, epochs=20, learning_rate=.001):
        examples = list(examples)
        if not examples or any(not isinstance(x, RewardExample) or x.split != "train" for x in examples):
            raise ValueError("reward fit requires nonempty train-only RewardExample records")
        if epochs <= 0 or learning_rate <= 0:
            raise ValueError("positive fit epochs and learning_rate required")
        self.train()
        for p in self.parameters():
            p.requires_grad_(True)
        device = next(self.parameters()).device
        features = torch.stack([self.features(x.trajectory, x.target_embedding) for x in examples]).to(device)
        labels = torch.tensor([x.label for x in examples], device=device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for _ in range(epochs):
            loss = nn.functional.mse_loss(self(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.fitted = True
        self.freeze()
        return float(loss.detach())

    def checkpoint(self):
        if not self.fitted:
            raise ValueError("fit reward model before saving")
        return {"format": "ecb-reward-v1", "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim, "state_dict": self.state_dict(),
                "fit_split": "train"}
