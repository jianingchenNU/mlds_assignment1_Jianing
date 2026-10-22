"""Baselines b_t for REINFORCE. Each one has __call__(obs, G) -> b and update(obs, G) -> value_loss."""
from collections import deque

import torch
import torch.nn.functional as F

from pg.networks import ValueNet
from pg.returns import normalize


class NoBaseline:
    def __call__(self, obs: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(G)

    def update(self, obs: torch.Tensor, G: torch.Tensor) -> float:
        return float("nan")

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, d: dict) -> None:
        pass


class MeanBaseline:
    """Constant b = mean of all G_t over the last M episodes.

    Averages G_t (not episode G_0) so b is on the same scale as the targets it is subtracted from.
    The current episode is excluded, so b is independent of its actions and the gradient stays unbiased.
    """

    def __init__(self, M: int = 100):
        self.hist = deque(maxlen=M)  # (sum of G_t, number of steps) per episode

    def __call__(self, obs: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        n = sum(c for _, c in self.hist)
        b = sum(s for s, _ in self.hist) / n if n else 0.0
        return torch.full_like(G, b)

    def update(self, obs: torch.Tensor, G: torch.Tensor) -> float:
        self.hist.append((G.sum().item(), G.numel()))
        return float("nan")

    def state_dict(self) -> dict:
        return {"hist": list(self.hist)}

    def load_state_dict(self, d: dict) -> None:
        self.hist.clear()
        self.hist.extend(d["hist"])


class ValueBaseline:
    """State-dependent b_t = V_phi(s_t), fit by MSE to the raw G_t with its own Adam."""

    def __init__(self, in_dim: int, hidden: int, lr: float, device: torch.device):
        self.net = ValueNet(in_dim, hidden).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def __call__(self, obs: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        # detach: the policy loss must not push gradients into phi
        return self.net(obs).detach()

    def update(self, obs: torch.Tensor, G: torch.Tensor) -> float:
        loss = F.mse_loss(self.net(obs), G)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def state_dict(self) -> dict:
        return {"net": self.net.state_dict(), "opt": self.opt.state_dict()}

    def load_state_dict(self, d: dict) -> None:
        self.net.load_state_dict(d["net"])
        self.opt.load_state_dict(d["opt"])


def advantage(baseline, obs: torch.Tensor, G: torch.Tensor,
              norm: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (A, raw) with raw = G_t - b_t and A = raw standardized if norm.

    raw is kept because its second moment E[(G - b)^2] is what the baselines are compared on;
    after standardization it is always ~1.

    Note that standardizing removes any constant shift, so with norm=True MeanBaseline
    gives exactly the same update as NoBaseline.
    """
    raw = G - baseline(obs, G)
    return (normalize(raw) if norm else raw), raw
