"""Cross-check the three MDP solvers against each other and against theory.

What it tests
  1. |V_VI - V_PI|_inf: two independent algorithms (fixed-point iteration vs exact linear solve)
     must reach the same V*. VI and PI policies must be identical.
  2. |V_enum,0(T) - V*|_inf for T in {10, 25, 50, 100, 200, 500}, against the bound gamma^T |V*|_inf.
  3. The finite-horizon policy pi_t near the end of the horizon.

Expected
  1. ~1e-7 (set by VI's tol=1e-8 / (1 - gamma)), policies equal, s* = 13.
  2. The gap is below the bound for every T and decays like 0.95^T; at T = 500 it is ~1e-7,
     i.e. enumeration has converged to the infinite-horizon solution.
  3. pi_{T-1} never dispatches; the threshold rises as t -> T and equals s* far from T.

Why
  The Bellman operator is a gamma-contraction, so V_enum,0(T) = B^T(0) and |B^T(0) - V*| <= gamma^T |V*|.
  With V_T = 0 the last step dispatches only if c_f < c_h * min(s, K) = 30, which never holds for
  c_f = 100, so the end-of-horizon policy must be lazier than the stationary one.

    python -m tests.validation.mdp_consistency
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.plotting import COLORS, save, style
from mdp.enumeration import enumerate_mdp
from mdp.policy_iteration import policy_iteration
from mdp.shuttle_mdp import ShuttleMDP, threshold
from mdp.value_iteration import value_iteration
from tests.validation._common import FIG, OUT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    Ts = [10, 50] if args.smoke else [10, 25, 50, 100, 200, 500]

    m = ShuttleMDP()
    V_vi, pi_vi, _ = value_iteration(m)
    V_pi, pi_pi, _ = policy_iteration(m)
    print(f"|V_VI - V_PI|_inf = {np.abs(V_vi - V_pi).max():.3e}")
    print(f"policies identical: {np.array_equal(pi_vi, pi_pi)}   s*_VI = {threshold(pi_vi)}  s*_PI = {threshold(pi_pi)}")

    vmax = np.abs(V_pi).max()
    rows = []
    for T in Ts:
        V0, pi_t = enumerate_mdp(m, T)
        rows.append(dict(T=T, gap=np.abs(V0 - V_pi).max(), bound=m.gamma**T * vmax,
                         s_star_t0=threshold(pi_t[0])))
    df = pd.DataFrame(rows)
    df["within_bound"] = df.gap <= df.bound
    print(df.to_string(index=False))

    T = Ts[-1]
    _, pi_t = enumerate_mdp(m, T)
    near_end = {f"t=T-{k}": threshold(pi_t[T - k]) for k in range(1, 8)}
    print(f"thresholds near the horizon (T={T}): {near_end}  (None = never dispatch)")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "mdp_consistency.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(df["T"], df.gap, "o-", color=COLORS[0], linewidth=2, label="|V_enum,0 - V*|_inf")
    ax.semilogy(df["T"], df.bound, "--", color=COLORS[1], linewidth=2, label="bound gamma^T |V*|_inf")
    style(ax, "Enumeration converges to the infinite-horizon value", "horizon T (periods)",
          "sup-norm gap (cost units)")
    ax.legend(frameon=False)
    save(fig, FIG / "mdp_consistency.png")


if __name__ == "__main__":
    main()
