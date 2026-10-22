"""Training curves, eval histograms and baseline comparisons from results/ into figures/.

Looks for runs named <prefix><env>_<baseline>_s<seed>; missing runs are skipped.
    python -m pg.plot_pg --env cartpole
    python -m pg.plot_pg --env pong --seed 1
    python -m pg.plot_pg --env cartpole --smoke     # uses smoke_* runs, writes figures/smoke/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.plotting import (COLORS, moving_average, plot_curves, plot_histogram, plot_training_curve,
                             save, style)

TITLES = {"cartpole": "CartPole-v1", "pong": "Pong-v5"}
BASELINES = ["none", "mean", "value"]


def plot_adv_var(logs: dict[str, pd.DataFrame], path: Path, title: str, w: int = 100) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for (name, df), c in zip(logs.items(), COLORS):
        ax.plot(df["episode"] + 1, moving_average(df["adv_var"].to_numpy(), w), color=c, linewidth=2, label=name)
    # variances differ by orders of magnitude between baselines
    ax.set_yscale("log")
    style(ax, title, "training episode", f"E[(G_t - b_t)^2], {w}-episode moving average")
    ax.legend(frameon=False)
    save(fig, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["cartpole", "pong"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", default="results")
    p.add_argument("--figures", default="figures")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    prefix = "smoke_" if args.smoke else ""
    figs = Path(args.figures) / ("smoke" if args.smoke else "")
    game = TITLES[args.env]

    logs = {}
    for b in BASELINES:
        run = Path(args.results) / f"{prefix}{args.env}_{b}_s{args.seed}"
        if not (run / "train_log.csv").exists():
            print(f"skip {run}: no train_log.csv")
            continue
        df = pd.read_csv(run / "train_log.csv")
        logs[b] = df
        plot_training_curve(df["reward"].to_numpy(), figs / f"{args.env}_{b}_train.png",
                            f"{game} training, baseline = {b}", args.window)
        ev = run / "eval_rewards.npy"
        if ev.exists():
            plot_histogram(np.load(ev), figs / f"{args.env}_{b}_hist.png",
                           f"{game} evaluation rewards, baseline = {b}")

    if len(logs) > 1:
        series = {b: df["reward"].to_numpy() for b, df in logs.items()}
        plot_curves(series, figs / f"{args.env}_compare.png",
                    f"{game}: baselines, {args.window}-episode moving average", "episode reward", args.window)
        plot_adv_var(logs, figs / f"{args.env}_compare_adv_var.png",
                     f"{game}: advantage variance by baseline", args.window)


if __name__ == "__main__":
    main()
