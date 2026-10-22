"""Visualise Pong preprocessing: raw frame, two preprocessed frames, and their difference.

What it tests
  Steps Pong with random RIGHT/LEFT actions until the ball is in play, then plots
  raw RGB | preprocess(previous) | preprocess(current) | difference, and prints pixel statistics.

Expected
  - Preprocessed frames are 80x80 binary: the two paddles (left column ~8, right column ~70-71)
    and the ball are 1, background is 0; the score digits are cropped away.
  - The background colours (red channel 144 and 109) are present in the raw crop and absent after
    preprocessing.
  - In the difference only moving objects remain: the ball shows as a +1/-1 pair, a paddle that did
    not move cancels out. That is what lets an MLP infer direction of motion from one input.

Why
  If the crop, downsampling or background values were wrong, the network would see the score,
  noise, or nothing at all, and Pong training would fail silently.

    python -m tests.validation.pg_preprocess_viz
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from common.plotting import save
from common.seed import set_seed
from pg.pong_utils import preprocess, to_env_action
from pg.train import make_env
from tests.validation._common import FIG


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=60, help="random steps before the snapshot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="same as default (a few dozen steps)")
    args = p.parse_args()

    env = make_env("pong")
    set_seed(args.seed, env)
    raw, _ = env.reset()
    prev_raw = raw
    for _ in range(args.warmup):
        prev_raw = raw
        raw, *_ = env.step(to_env_action(np.random.randint(2)))
    env.close()

    prev, cur = preprocess(prev_raw.copy()), preprocess(raw.copy())
    diff = cur - prev
    crop_red = raw[35:195:2, ::2, 0]
    print(f"raw red-channel values in crop: {sorted(np.unique(crop_red).tolist())}")
    print(f"background 144/109 left after preprocess: {bool(np.isin(crop_red[cur == 1], [144, 109]).any())}")
    print(f"preprocessed values {np.unique(cur).tolist()}, nonzero pixels {int(cur.sum())} of 6400")
    print(f"columns with nonzero pixels: {np.flatnonzero(cur.any(0)).tolist()}")
    print(f"difference values {np.unique(diff).tolist()}, +1 pixels {int((diff > 0).sum())}, "
          f"-1 pixels {int((diff < 0).sum())}")

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    axes[0].imshow(raw)
    axes[0].set_title("raw frame (210x160x3)")
    axes[1].imshow(prev, cmap="gray_r", vmin=0, vmax=1)
    axes[1].set_title("preprocessed t-1 (80x80)")
    axes[2].imshow(cur, cmap="gray_r", vmin=0, vmax=1)
    axes[2].set_title("preprocessed t (80x80)")
    im = axes[3].imshow(diff, cmap="bwr", vmin=-1, vmax=1)
    axes[3].set_title("difference: network input")
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, FIG / "pg_preprocess_viz.png")


if __name__ == "__main__":
    main()
