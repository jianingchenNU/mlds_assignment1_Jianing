"""Solve the shuttle MDP with enumeration, value iteration and policy iteration.

Saves results/mdp/{enum,vi,pi}.npz.
    python -m mdp.run_all
    python -m mdp.run_all --c-f 200 --out results/mdp_cf200
"""
import argparse
import time
from pathlib import Path

import numpy as np

from mdp.enumeration import enumerate_mdp
from mdp.policy_iteration import policy_iteration
from mdp.shuttle_mdp import ShuttleMDP, threshold
from mdp.value_iteration import value_iteration


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=15)
    p.add_argument("--c-f", type=float, default=100.0)
    p.add_argument("--c-h", type=float, default=2.0)
    p.add_argument("--s-max", type=int, default=200)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--out", default="results/mdp")
    p.add_argument("--smoke", action="store_true", help="short horizon T=50")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    T = 50 if args.smoke else args.T
    m = ShuttleMDP(K=args.K, c_f=args.c_f, c_h=args.c_h, s_max=args.s_max, gamma=args.gamma)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(K=args.K, c_f=args.c_f, c_h=args.c_h, s_max=args.s_max, gamma=args.gamma)

    t0 = time.perf_counter()
    V0, pi_t = enumerate_mdp(m, T)
    t1 = time.perf_counter()
    V_vi, pi_vi, res = value_iteration(m, args.tol)
    t2 = time.perf_counter()
    V_pi, pi_pi, changes = policy_iteration(m)
    t3 = time.perf_counter()

    np.savez(out / "enum.npz", V0=V0, pi=pi_t, T=T, **cfg)
    np.savez(out / "vi.npz", V=V_vi, pi=pi_vi, residuals=np.array(res), tol=args.tol, **cfg)
    np.savez(out / "pi.npz", V=V_pi, pi=pi_pi, changes=np.array(changes), **cfg)

    print(f"enum: T={T}  s*={threshold(pi_t[0])}  {t1 - t0:.2f}s")
    print(f"vi:   {len(res)} iters  s*={threshold(pi_vi)}  {t2 - t1:.2f}s")
    print(f"pi:   {len(changes)} iters  changes={changes}  s*={threshold(pi_pi)}  {t3 - t2:.2f}s")
    print(f"|V_vi - V_pi|_inf   = {np.abs(V_vi - V_pi).max():.3e}")
    print(f"|V_enum0 - V_vi|_inf = {np.abs(V0 - V_vi).max():.3e}")
    print(f"policies agree (vi vs pi): {np.array_equal(pi_vi, pi_pi)}")


if __name__ == "__main__":
    main()
