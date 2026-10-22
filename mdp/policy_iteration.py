"""Policy iteration with exact linear-solve evaluation."""
import numpy as np

from mdp.shuttle_mdp import ShuttleMDP


def evaluate(m: ShuttleMDP, pi: np.ndarray) -> np.ndarray:
    s = m.states
    P_pi = m.P[pi, s]
    c_pi = m.cost[s, pi]
    return np.linalg.solve(np.eye(m.S) - m.gamma * P_pi, c_pi)


def policy_iteration(m: ShuttleMDP, tie_tol: float = 1e-10,
                     max_iter: int = 1000) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Returns V, pi, and the number of states whose action changed in each iteration."""
    pi = np.zeros(m.S, dtype=int)
    changes = []
    for _ in range(max_iter):
        V = evaluate(m, pi)
        Q = m.q_values(V)
        # keep the incumbent action unless strictly better by tie_tol, otherwise PI can cycle on ties
        keep = Q[m.states, pi] <= Q.min(1) + tie_tol
        new_pi = np.where(keep, pi, Q.argmin(1))
        changes.append(int((new_pi != pi).sum()))
        pi = new_pi
        if changes[-1] == 0:
            break
    return V, pi, changes
