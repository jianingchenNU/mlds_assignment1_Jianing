"""Discounted reward-to-go and standardization."""
import numpy as np


def discounted_returns(rew: np.ndarray, gamma: float, reset_on_nonzero: bool = False) -> np.ndarray:
    """G_t = sum_k gamma^k r_{t+k}, computed within one episode (no bootstrap).

    With reset_on_nonzero (Pong), the sum restarts at every scored point so a rally is only
    credited with its own outcome.
    """
    G = np.zeros(len(rew), dtype=np.float64)
    run = 0.0
    for t in range(len(rew) - 1, -1, -1):
        if reset_on_nonzero and rew[t] != 0:
            run = 0.0
        run = rew[t] + gamma * run
        G[t] = run
    return G


def normalize(x, eps: float = 1e-8):
    """Works for numpy arrays and torch tensors; a constant input maps to zeros."""
    return (x - x.mean()) / (x.std() + eps)
