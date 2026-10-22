"""Parameter sweeps: does the threshold s* move the way economics says it should?

What it tests (policy iteration, all other parameters at default)
  - c_f = 0:  dispatching is free.
  - c_h = 0:  waiting is free.
  - c_f in {0, 25, 50, 100, 200, 400, 800}
  - c_h in {0.5, 1, 2, 4, 8}
  - K in {5, 10, 15, 20, 30, 50}

Expected
  - c_f = 0: s* = 1 (dispatch whenever anyone waits; at s = 0 both actions cost 0 and the tie
    goes to "wait").
  - c_h = 0: never dispatch (s* = None), since dispatching costs c_f > 0 and buys nothing.
  - s* non-decreasing in c_f (dispatch is pricier, so batch more people per trip). "Never
    dispatch" counts as s* = +inf in this check.
  - c_f = 800: never dispatch. Not a bug: the station cap S_MAX = 200 drops excess arrivals for
    free, so never dispatching costs at most c_h * 200 = 400 per period and V(200) = 400 / (1 - gamma)
    = 8000 exactly. Each trip at c_f = 800 saves too little holding cost before s refills to 200.
  - s* non-increasing in c_h (waiting is pricier, so dispatch sooner).
  - K: s* grows with K at first, then saturates; a bigger shuttle is only worth filling up to the
    point where holding cost of the wait outweighs c_f.

Why
  A sanity check on the cost/transition model as a whole: an error in the order of events or the
  cost formula usually shows up as a sweep moving in the wrong direction.

    python -m tests.validation.mdp_sensitivity
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.plotting import COLORS, save, style
from mdp.policy_iteration import policy_iteration
from mdp.shuttle_mdp import ShuttleMDP, threshold
from tests.validation._common import FIG, OUT


def s_star(**kw) -> int | None:
    return threshold(policy_iteration(ShuttleMDP(**kw))[1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="three values per sweep")
    args = p.parse_args()
    sweeps = {
        "c_f": [0, 25, 50, 100, 200, 400, 800],
        "c_h": [0.5, 1, 2, 4, 8],
        "K": [5, 10, 15, 20, 30, 50],
    }
    if args.smoke:
        sweeps = {k: v[:3] for k, v in sweeps.items()}

    print(f"c_f = 0 -> s* = {s_star(c_f=0)}   (expected 1)")
    print(f"c_h = 0 -> s* = {s_star(c_h=0)}   (expected None)")

    rows = [dict(param=k, value=v, s_star=s_star(**{k: v})) for k, vals in sweeps.items() for v in vals]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    for k, sign in (("c_f", 1), ("c_h", -1)):
        s = df[df.param == k].s_star.astype(float).fillna(np.inf).to_numpy()  # None = never = +inf
        print(f"s* {'non-decreasing' if sign > 0 else 'non-increasing'} in {k}: "
              f"{bool(np.all(sign * np.diff(s) >= 0))}")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "mdp_sensitivity.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    labels = {"c_f": "dispatch cost c_f (cost units)", "c_h": "holding cost c_h (cost units / person / period)",
              "K": "shuttle capacity K (people)"}
    for ax, (k, c) in zip(axes, zip(sweeps, COLORS)):
        d = df[df.param == k]
        ax.plot(d.value, d.s_star.astype(float), "o-", color=c, linewidth=2)
        top = d.s_star.max() + 1  # "never dispatch" (s* = +inf) drawn as an x above the curve
        for v in d.value[d.s_star.isna()]:
            ax.plot([v], [top], "x", color=c, markersize=9, markeredgewidth=2)
            ax.annotate("never dispatch", (v, top), xytext=(0, -14), textcoords="offset points",
                        ha="right", color="#52514e", fontsize=8)
        style(ax, f"s* vs {k}", labels[k], "threshold s* (people)")
    save(fig, FIG / "mdp_sensitivity.png")


if __name__ == "__main__":
    main()
