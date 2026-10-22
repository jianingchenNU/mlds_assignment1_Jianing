"""How well does the value baseline V_phi(s_t) track the returns G_t during training?

What it tests
  Trains with --baseline value and, for each episode (before that episode's value update), records
  corr(V_phi(s_t), G_t) and the explained variance 1 - Var(G - V) / Var(G).

Expected
  Both start near 0 (random V_phi) and rise over training. CartPole: correlation well above 0.5 once
  episodes get long. They should not reach 1: G_t also depends on how far t is from the 500-step
  truncation, and time is not part of the state, so part of G_t is unpredictable from s_t alone.
  Explained variance > 0 is the condition under which the baseline actually reduces variance
  (Var(G - V) < Var(G)).

Why
  A value baseline only helps if it predicts the return. A flat or negative explained variance
  would mean the critic is not learning (wrong target, detached targets, or too small lr) and
  the "value" run in Part 2 is effectively a noisy baseline.

    python -m tests.validation.pg_value_fit --env cartpole --episodes 1000
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.plotting import COLORS, moving_average, save, style
from pg.returns import discounted_returns
from pg.train import DEFAULTS
from tests.validation._common import FIG, OUT, train_with_hook


def make_hook(env_name: str):
    gamma = DEFAULTS[env_name]["gamma"]

    def hook(obs, act, rew, policy, baseline, device) -> dict:
        G = discounted_returns(rew, gamma, reset_on_nonzero=env_name == "pong")
        with torch.no_grad():
            V = baseline.net(torch.as_tensor(obs, device=device)).cpu().numpy()
        # constant G (e.g. length-1 episode) makes both statistics undefined
        if len(G) < 2 or G.std() == 0:
            return dict(corr=np.nan, explained_var=np.nan)
        corr = np.corrcoef(V, G)[0, 1] if V.std() > 0 else np.nan
        return dict(corr=corr, explained_var=1 - np.var(G - V) / np.var(G))

    return hook


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["cartpole", "pong"], default="cartpole")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--normalize-returns", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smoke", action="store_true", help="5 episodes (1 for pong)")
    args = p.parse_args()
    E = (1 if args.env == "pong" else 5) if args.smoke else args.episodes

    df, _, _ = train_with_hook(args.env, "value", E, args.seed, norm=args.normalize_returns,
                               hook=make_hook(args.env))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"pg_value_fit_{args.env}.csv", index=False)

    k = max(1, min(50, E // 2))
    for col in ("corr", "explained_var"):
        x = df[col].to_numpy()
        print(f"{col:14s} first {k}: {np.nanmean(x[:k]):.3f}   last {k}: {np.nanmean(x[-k:]):.3f}")

    w = min(100, E)
    ep = df.episode + 1
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, c, label in (("corr", COLORS[0], "corr(V(s_t), G_t)"), ("explained_var", COLORS[2], "explained variance")):
        x = df[col].fillna(0).to_numpy()
        ax.plot(ep, moving_average(x, w), color=c, linewidth=2, label=f"{label} ({w}-ep MA)")
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set_ylim(-1, 1)
    style(ax, f"{args.env}: value baseline fit during training", "training episode", "statistic (unitless)")
    ax.legend(frameon=False)
    save(fig, FIG / f"pg_value_fit_{args.env}.png")


if __name__ == "__main__":
    main()
