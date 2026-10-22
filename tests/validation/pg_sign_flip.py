"""Train CartPole on -G_t instead of G_t: the learner should get worse than random.

What it tests
  Two runs with identical seed and settings, one normal and one with --flip-sign (gradient ascent
  on minus the return). Compares the last-100-episode mean reward with a random policy.

Expected
  normal  : climbs well above random (towards 200-500 after ~1000 episodes).
  flipped : drops below random (~22) towards ~9-10, the shortest episode CartPole allows, because
            the agent learns to tip the pole over as fast as possible.

Why
  If the gradient sign, the log-prob or the return computation were wrong, the two runs would
  not separate in opposite directions from the random level. This is a direct test that the
  update moves probability mass toward high-return actions.

    python -m tests.validation.pg_sign_flip --episodes 1000
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.plotting import COLORS, moving_average, save, style
from pg import train
from tests.validation._common import FIG, OUT
from tests.validation.pg_random_baseline import random_rewards


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="5 episodes each")
    args = p.parse_args()
    E = 5 if args.smoke else args.episodes
    out = OUT / "sign_flip"

    common = ["--env", "cartpole", "--episodes", str(E), "--seed", str(args.seed), "--out", str(out)]
    train.main(common + ["--run-name", "normal"])
    train.main(common + ["--run-name", "flipped", "--flip-sign"])
    rand = random_rewards("cartpole", 10 if args.smoke else 200, args.seed).mean()

    w = min(100, E)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, c in (("normal", COLORS[0]), ("flipped", COLORS[1])):
        rew = pd.read_csv(out / name / "train_log.csv")["reward"].to_numpy()
        print(f"{name:8s} last-{w} mean reward {rew[-w:].mean():.2f}")
        ax.plot(np.arange(1, E + 1), moving_average(rew, w), color=c, linewidth=2, label=name)
    print(f"random   mean reward {rand:.2f}")
    ax.axhline(rand, color="#52514e", linestyle="--", linewidth=1, label="random policy")
    ax.set_yscale("log")
    style(ax, "CartPole: normal vs sign-flipped returns", "training episode",
          f"episode reward, {w}-episode moving average")
    ax.legend(frameon=False)
    save(fig, FIG / "pg_sign_flip.png")


if __name__ == "__main__":
    main()
