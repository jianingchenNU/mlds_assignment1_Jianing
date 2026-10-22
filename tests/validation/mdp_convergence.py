"""Convergence rates of value iteration and policy iteration.

What it tests
  - VI residual |V_{k+1} - V_k|_inf and true error |V_k - V*|_inf (V* from PI), on a log scale,
    against the contraction bound r_0 * gamma^k.
  - The empirical tail rate r_{k+1} / r_k.
  - Number of PI iterations and how many states change action in each.

Expected
  - Both curves are straight lines on a log scale lying on or below the gamma^k line; the tail rate
    is ~0.95 (it cannot exceed gamma).
  - VI needs ~430 iterations for tol = 1e-8, no more than log(tol / r_0) / log(gamma) (the bound is
    loose in the first iterations, where the residual shrinks faster than gamma).
  - PI converges in a handful of iterations (6 here), with the number of changed states shrinking
    to 0: PI is Newton's method on the Bellman equation and converges superlinearly.

Why
  The Bellman operator is a gamma-contraction in the sup norm. A rate faster than gamma would mean
  the update is not a true Bellman backup; a slower one would point to a bug in P or gamma.

    python -m tests.validation.mdp_convergence
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from common.plotting import COLORS, save, style
from mdp.policy_iteration import policy_iteration
from mdp.shuttle_mdp import ShuttleMDP
from tests.validation._common import FIG, OUT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--smoke", action="store_true", help="tol = 1e-3")
    args = p.parse_args()
    tol = 1e-3 if args.smoke else args.tol

    m = ShuttleMDP()
    V_star, _, changes = policy_iteration(m)
    V = np.zeros(m.S)
    res, err = [], [np.abs(V - V_star).max()]
    while not res or res[-1] >= tol:
        V_new = m.q_values(V).min(1)
        res.append(np.abs(V_new - V).max())
        V = V_new
        err.append(np.abs(V - V_star).max())
    res, err = np.array(res), np.array(err)
    k = np.arange(len(res))
    rate = res[1:] / res[:-1]

    print(f"VI: {len(res)} iterations to residual < {tol:g}")
    print(f"tail rate r_(k+1)/r_k (last 50): mean {rate[-50:].mean():.5f}, max {rate.max():.5f}  (gamma = {m.gamma})")
    print(f"predicted iterations log(tol/r0)/log(gamma) = {np.log(tol / res[0]) / np.log(m.gamma):.1f}")
    print(f"PI: {len(changes)} iterations, states changed per iteration = {changes}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "mdp_convergence.npz", residuals=res, errors=err, pi_changes=np.array(changes))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(k, res, color=COLORS[0], linewidth=2, label="residual |V_k+1 - V_k|")
    ax.semilogy(np.arange(len(err)), err, color=COLORS[2], linewidth=2, label="error |V_k - V*|")
    ax.semilogy(k, res[0] * m.gamma**k, "--", color=COLORS[1], linewidth=1.5, label="r_0 * gamma^k")
    style(ax, "Value iteration convergence", "iteration k", "sup norm (cost units)")
    ax.legend(frameon=False)
    save(fig, FIG / "mdp_convergence.png")


if __name__ == "__main__":
    main()
