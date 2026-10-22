"""Shuttle dispatch MDP: cost matrix (S, 2) and transition tensor (2, S, S)."""
import numpy as np


class ShuttleMDP:
    def __init__(self, K: int = 15, c_f: float = 100.0, c_h: float = 2.0, s_max: int = 200,
                 gamma: float = 0.95, arrivals: dict[int, float] | None = None):
        self.K, self.c_f, self.c_h, self.s_max, self.gamma = K, c_f, c_h, s_max, gamma
        self.arrivals = arrivals if arrivals is not None else {a: 0.2 for a in range(1, 6)}
        self.S = s_max + 1
        self.states = np.arange(self.S)
        self.cost = self._build_cost()
        self.P = self._build_P()
        assert np.allclose(self.P.sum(-1), 1.0)

    def left(self, s: np.ndarray, a: int) -> np.ndarray:
        return np.maximum(s - self.K * a, 0)

    def _build_cost(self) -> np.ndarray:
        s = self.states
        return np.stack([self.c_f * a + self.c_h * self.left(s, a) for a in (0, 1)], axis=1)

    def _build_P(self) -> np.ndarray:
        P = np.zeros((2, self.S, self.S))
        for a in (0, 1):
            rem = self.left(self.states, a)
            for n, p in self.arrivals.items():
                # np.add.at accumulates mass from different n truncated onto s_max
                np.add.at(P[a], (self.states, np.minimum(rem + n, self.s_max)), p)
        return P

    def q_values(self, V: np.ndarray) -> np.ndarray:
        """Q(s, a) = c(s, a) + gamma * E[V(s')], shape (S, 2)."""
        return self.cost + self.gamma * (self.P @ V).T


def threshold(pi: np.ndarray) -> int | None:
    """Smallest state where the policy dispatches, or None if it never does."""
    idx = np.flatnonzero(pi == 1)
    return int(idx[0]) if idx.size else None
