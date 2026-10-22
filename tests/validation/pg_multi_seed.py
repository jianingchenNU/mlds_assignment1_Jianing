"""Train one configuration under several seeds and summarise mean +- std learning curves.

What it tests
  Runs pg.train for each seed (unknown arguments are passed through, e.g. --no-normalize-returns),
  then plots the across-seed mean of the 100-episode moving average with a +-1 std band and
  prints the final-100-episode mean per seed.

Expected
  CartPole, 1000 episodes, no baseline: every seed ends well above random (~22); seeds differ
  substantially in how fast they get there (REINFORCE is high-variance), which shows as a wide
  band in the middle of training. With the value baseline the band should be narrower.

Why
  One lucky seed proves nothing. Consistent improvement across seeds shows the algorithm works,
  and the spread is a direct measure of the run-to-run variance a baseline is meant to reduce.

    python -m tests.validation.pg_multi_seed --env cartpole --baseline none --seeds 0 1 2 3 4
    python -m tests.validation.pg_multi_seed --env cartpole --baseline value --no-normalize-returns
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.plotting import COLORS, moving_average, save, style
from pg import train
from pg.plot_pg import TITLES
from tests.validation._common import FIG, OUT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["cartpole", "pong"], default="cartpole")
    p.add_argument("--baseline", choices=["none", "mean", "value"], default="none")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int)
    p.add_argument("--plot-only", action="store_true", help="reuse existing runs")
    p.add_argument("--smoke", action="store_true", help="2 seeds x 5 episodes")
    args, passthrough = p.parse_known_args()
    seeds = args.seeds[:2] if args.smoke else args.seeds
    episodes = 5 if args.smoke else args.episodes
    out = OUT / "multi_seed"
    tag = f"{args.env}_{args.baseline}"

    curves, finals = [], []
    for s in seeds:
        run = f"{tag}_s{s}"
        if not args.plot_only:
            argv = ["--env", args.env, "--baseline", args.baseline, "--seed", str(s),
                    "--out", str(out), "--run-name", run] + passthrough
            if episodes:
                argv += ["--episodes", str(episodes)]
            train.main(argv)
        rew = pd.read_csv(out / run / "train_log.csv")["reward"].to_numpy()
        curves.append(moving_average(rew, 100))
        finals.append(rew[-100:].mean())

    n = min(map(len, curves))
    C = np.stack([c[:n] for c in curves])
    summary = pd.DataFrame({"seed": seeds, "final100_mean": finals})
    print(summary.to_string(index=False))
    print(f"across seeds: {np.mean(finals):.2f} +- {np.std(finals):.2f}")
    summary.to_csv(out / f"{tag}_summary.csv", index=False)

    ep = np.arange(1, n + 1)
    mu, sd = C.mean(0), C.std(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, mu, color=COLORS[0], linewidth=2, label=f"mean over {len(seeds)} seeds")
    ax.fill_between(ep, mu - sd, mu + sd, color=COLORS[0], alpha=0.2, linewidth=0, label="+-1 std")
    style(ax, f"{TITLES[args.env]}: baseline = {args.baseline}, multiple seeds",
          "training episode", "100-episode moving average reward")
    ax.legend(frameon=False)
    save(fig, FIG / f"pg_multi_seed_{tag}.png")


if __name__ == "__main__":
    main()
