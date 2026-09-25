from torch import nn


class Gate(nn.Module):
    """Predict executed continuation utility minus stopping utility."""
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)


class ValueBaseline(nn.Module):
    """Action-independent baseline with parameters separate from the actor."""
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)
