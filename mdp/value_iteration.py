"""Infinite-horizon value iteration, stopped on sup-norm residual."""
import numpy as np

from mdp.shuttle_mdp import ShuttleMDP


def value_iteration(m: ShuttleMDP, tol: float = 1e-8,
                    max_iter: int = 100_000) -> tuple[np.ndarray, np.ndarray, list[float]]:
    V = np.zeros(m.S)
    res = []
    for _ in range(max_iter):
        Q = m.q_values(V)
        V_new = Q.min(1)
        res.append(float(np.abs(V_new - V).max()))
        V = V_new
        if res[-1] < tol:
            break
    return V, m.q_values(V).argmin(1), res
