"""Plot MDP results from results/mdp into figures/.

    python -m mdp.plot_mdp
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.plotting import COLORS, save, style
from mdp.shuttle_mdp import threshold

V_LABEL = "expected discounted cost (cost units)"
S_LABEL = "customers waiting, s (people)"


def plot_value(V: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(V)), V, color=COLORS[0], linewidth=2)
    style(ax, title, S_LABEL, V_LABEL)
    save(fig, path)


def plot_policy(pi: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(np.arange(len(pi)), pi, where="post", color=COLORS[0], linewidth=2)
    s_star = threshold(pi)
    if s_star is not None:
        ax.axvline(s_star, color="#52514e", linestyle="--", linewidth=1)
        ax.annotate(f"s* = {s_star}", (s_star, 0.5), xytext=(8, 0), textcoords="offset points")
    ax.set_yticks([0, 1], ["0: wait", "1: dispatch"])
    ax.set_ylim(-0.1, 1.1)
    style(ax, title, S_LABEL, "optimal action")
    save(fig, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/mdp")
    p.add_argument("--figures", default="figures")
    p.add_argument("--smoke", action="store_true", help="no-op; plotting is already fast")
    args = p.parse_args()
    r, f = Path(args.results), Path(args.figures)

    enum = np.load(r / "enum.npz")
    vi = np.load(r / "vi.npz")
    pi = np.load(r / "pi.npz")
    plot_value(enum["V0"], f / "mdp_enum_V0.png",
               f"Enumeration: optimal value at t=0 (T={int(enum['T'])})")
    plot_value(vi["V"], f / "mdp_vi_V.png", "Value iteration: optimal value function V*")
    plot_policy(pi["pi"], f / "mdp_pi_policy.png", "Policy iteration: optimal policy")


if __name__ == "__main__":
    main()
