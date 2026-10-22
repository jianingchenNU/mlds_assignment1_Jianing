"""Check structural properties of the optimal solution under several parameter settings.

What it tests
  For the default parameters and a few variants (c_f, c_h, K, gamma changed):
  - V* is non-decreasing in s;
  - pi* is a threshold policy (non-decreasing in s: wait below s*, dispatch from s* on);
  - whether every finite-horizon policy pi_t from enumeration is threshold-type, and where not.

Expected
  V monotone and pi* threshold-type in every config, s* = 13 for the default parameters.
  The finite-horizon policies can break the threshold structure in the last few periods for s
  close to S_MAX = 200 (e.g. at t = T-4, T-5 for the default parameters): there, arrivals beyond
  200 are truncated, i.e. turned away at no cost, so a full station "absorbs" new arrivals for free,
  while dispatching reopens room that refills within a few periods. With only a few periods left
  this makes waiting at s ~ 200 cheaper than at s ~ 195. It is an artifact of the capacity cap
  (an edge case of the model), not of the solver, and it disappears for the stationary policy.

Why
  More people waiting can never be cheaper: from a larger s you can mimic any action sequence from
  a smaller s and pay at least as much holding cost, so V is monotone. With a monotone V, the
  benefit of dispatching (removing up to K people) grows with s, which makes the optimal policy a
  threshold rule. A violation in the stationary V* or pi* would point to an indexing or truncation
  bug in P.

    python -m tests.validation.mdp_structure
"""
import argparse

import numpy as np
import pandas as pd

from mdp.enumeration import enumerate_mdp
from mdp.policy_iteration import policy_iteration
from mdp.shuttle_mdp import ShuttleMDP, threshold
from tests.validation._common import OUT

CONFIGS = {
    "default": {},
    "c_f=200": dict(c_f=200),
    "c_h=5": dict(c_h=5),
    "K=10": dict(K=10),
    "K=30": dict(K=30),
    "gamma=0.99": dict(gamma=0.99),
}


def is_threshold(pi: np.ndarray) -> bool:
    return bool(np.all(np.diff(pi) >= 0))


def violations(pi_t: np.ndarray, k: int = 3) -> str:
    """The k (T - t, s) pairs closest to the horizon where pi_t drops from dispatch back to wait."""
    T = len(pi_t)
    out = [(T - t, int(np.flatnonzero(np.diff(x) < 0)[0]) + 1) for t, x in enumerate(pi_t) if not is_threshold(x)]
    return "none" if not out else " ".join(f"(T-{a},s={b})" for a, b in out[-k:])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="enumeration with T=50")
    args = p.parse_args()
    T = 50 if args.smoke else 500

    rows = []
    for name, kw in CONFIGS.items():
        m = ShuttleMDP(**kw)
        V, pi, _ = policy_iteration(m)
        _, pi_t = enumerate_mdp(m, T)
        rows.append(dict(config=name, V_monotone=bool(np.all(np.diff(V) >= -1e-9)),
                         pi_threshold=is_threshold(pi), s_star=threshold(pi),
                         pi_t_violations=violations(pi_t),
                         V0=V[0], V200=V[-1]))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "mdp_structure.csv", index=False)


if __name__ == "__main__":
    main()
