"""Policy and value networks. Policies return a torch.distributions object."""
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical


class CartPolePolicy(nn.Module):
    def __init__(self, obs_dim: int = 4, hidden: int = 128, n_act: int = 2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_act))

    def forward(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(obs))


class PongPolicy(nn.Module):
    """6400 -> 200 -> 1; the output is the logit of P(RIGHT). Input is (B, 80, 80)."""

    def __init__(self, hidden: int = 200):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(6400, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, obs: torch.Tensor) -> Bernoulli:
        # logits instead of sigmoid probs: same distribution, stable log_prob near 0/1
        return Bernoulli(logits=self.net(obs).squeeze(-1))


class PongCNN(nn.Module):
    """Small conv alternative to PongPolicy (--arch cnn)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Unflatten(1, (1, 80)),
            nn.Conv2d(1, 16, 8, stride=4), nn.ReLU(),   # -> 16 x 19 x 19
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),  # -> 32 x 8 x 8
            nn.Flatten(), nn.Linear(32 * 8 * 8, 256), nn.ReLU(), nn.Linear(256, 1),
        )

    def forward(self, obs: torch.Tensor) -> Bernoulli:
        return Bernoulli(logits=self.net(obs).squeeze(-1))


class ValueNet(nn.Module):
    """V_phi(s); in_dim=4 for CartPole, 6400 for Pong (same preprocessed frame diff as the policy)."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
