"""Deterministic arrivals (A = 3 every period): a trace that can be checked by hand.

What it tests
  Solves the MDP with A = 3, then follows the optimal policy from s0 for 10 periods and prints
  t, s, a, c(s, a), V(s), Q(s, wait), Q(s, dispatch). It also simulates the policy for 2000
  periods and compares the realized discounted cost with V(s0).

Expected
  - The state path is 0, 3, 6, ... until s reaches s*, then a dispatch takes everyone and the cycle
    repeats with a fixed period, so the trace is periodic.
  - At every step V(s) = min(Q(s, 0), Q(s, 1)) and the chosen action is the argmin.
  - Simulated discounted cost matches V(s0) to ~1e-6 (the tail beyond 2000 periods is
    0.95^2000 * |V| ~ 0).
  - Hand check: with cycle length n (dispatch every n periods), one cycle from s = 0 costs
    c_h * 3 * (0 + 1 + ... + (n-1)) discounted + c_f at the dispatch; the printed costs should match.

Why
  Removing randomness turns the Bellman equation into a plain recursion along one path, so every
  number in the table can be reproduced with a calculator.

    python -m tests.validation.mdp_deterministic
"""
import argparse

import numpy as np
import pandas as pd

from mdp.policy_iteration import policy_iteration
from mdp.shuttle_mdp import ShuttleMDP, threshold
from tests.validation._common import OUT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arrivals", type=int, default=3)
    p.add_argument("--s0", type=int, default=0)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--smoke", action="store_true", help="200-period simulation instead of 2000")
    args = p.parse_args()

    m = ShuttleMDP(arrivals={args.arrivals: 1.0})
    V, pi, _ = policy_iteration(m)
    Q = m.q_values(V)
    print(f"A = {args.arrivals} deterministic, s* = {threshold(pi)}")

    rows, s = [], args.s0
    for t in range(args.steps):
        a = int(pi[s])
        rows.append(dict(t=t, s=s, a=a, cost=m.cost[s, a], V=V[s], Q_wait=Q[s, 0], Q_dispatch=Q[s, 1]))
        s = int(np.argmax(m.P[a, s]))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    horizon = 200 if args.smoke else 2000
    s, total = args.s0, 0.0
    for t in range(horizon):
        a = int(pi[s])
        total += m.gamma**t * m.cost[s, a]
        s = int(np.argmax(m.P[a, s]))
    print(f"simulated discounted cost over {horizon} periods = {total:.6f}")
    print(f"V(s0)                                      = {V[args.s0]:.6f}")
    print(f"difference                                 = {abs(total - V[args.s0]):.3e}")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "mdp_deterministic.csv", index=False)


if __name__ == "__main__":
    main()
