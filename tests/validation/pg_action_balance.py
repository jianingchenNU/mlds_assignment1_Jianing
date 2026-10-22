"""Track the fraction of RIGHT (action 1) and the mean P(RIGHT) over training.

What it tests
  Trains with the same setup as pg.train and, for every episode, records the fraction of steps
  where the sampled action was 1 (Pong: RIGHT, CartPole: push right) and the policy's mean
  probability of that action over the episode.

Expected
  The fraction stays well inside (0, 1): it starts at ~0.5 and drifts as the policy learns which
  side is useful, but the 100-episode moving average never collapses below ~0.05 or above ~0.95.
  In Pong both directions are needed to track the ball, so a collapse means the policy has
  degenerated into a constant action. The script prints the extremes of the moving average and
  flags a collapse.

Why
  A sigmoid output can saturate early with a large lr (large logits, vanishing log-prob gradient);
  at lr = 0.01 Pong entropy fell to ~0.1 within ~110 episodes, which is why Pong uses lr = 0.001.
  Collapse would also show up as entropy -> 0 in the training log; this makes it explicit per action.

    python -m tests.validation.pg_action_balance --env pong --episodes 300
"""
import argparse

import matplotlib.pyplot as plt
import torch

from common.plotting import COLORS, moving_average, save, style
from tests.validation._common import FIG, OUT, train_with_hook


def hook(obs, act, rew, policy, baseline, device) -> dict:
    with torch.no_grad():
        d = policy(torch.as_tensor(obs, device=device))
        p1 = d.probs[:, 1] if d.probs.dim() == 2 else d.probs
    return dict(frac_right=float(act.mean()), mean_p_right=p1.mean().item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["cartpole", "pong"], default="pong")
    p.add_argument("--baseline", choices=["none", "mean", "value"], default="none")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="2 episodes")
    args = p.parse_args()
    E = 2 if args.smoke else args.episodes

    df, _, _ = train_with_hook(args.env, args.baseline, E, args.seed, hook=hook)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"pg_action_balance_{args.env}.csv", index=False)

    w = min(100, E)
    ma = moving_average(df.frac_right.to_numpy(), w)
    print(f"fraction of action 1: overall {df.frac_right.mean():.3f}, "
          f"{w}-ep moving average range [{ma.min():.3f}, {ma.max():.3f}]")
    print(f"collapse (MA < 0.05 or > 0.95): {bool((ma < 0.05).any() or (ma > 0.95).any())}")

    ep = df.episode + 1
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, df.frac_right, color=COLORS[0], alpha=0.25, linewidth=0.8)
    ax.plot(ep, ma, color=COLORS[0], linewidth=2, label="fraction of action 1 (MA)")
    ax.plot(ep, moving_average(df.mean_p_right.to_numpy(), w), color=COLORS[1], linewidth=2,
            label="mean P(action 1) (MA)")
    ax.axhline(0.5, color="#52514e", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    label = "RIGHT" if args.env == "pong" else "push right"
    style(ax, f"{args.env}: action balance during training", "training episode", f"share of {label}")
    ax.legend(frameon=False)
    save(fig, FIG / f"pg_action_balance_{args.env}.png")


if __name__ == "__main__":
    main()
