"""Finite-horizon backward induction with terminal value V_T = 0."""
import numpy as np

from mdp.shuttle_mdp import ShuttleMDP


def enumerate_mdp(m: ShuttleMDP, T: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Returns V_0 of shape (S,) and the time-indexed policy pi of shape (T, S)."""
    V = np.zeros(m.S)
    pi = np.zeros((T, m.S), dtype=int)
    for t in range(T - 1, -1, -1):
        Q = m.q_values(V)
        pi[t] = Q.argmin(1)  # ties -> 0 (don't dispatch)
        V = Q.min(1)
    return V, pi
