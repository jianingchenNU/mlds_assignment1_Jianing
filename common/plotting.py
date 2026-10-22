"""Shared matplotlib helpers: training curves with moving average, reward histograms."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# fixed categorical order so a series keeps its color across figures
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]


def moving_average(x: np.ndarray, w: int = 100) -> np.ndarray:
    """Trailing mean over the last min(t+1, w) points, same length as x."""
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - w, 0)
    return (c[idx] - c[lo]) / (idx - lo)


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curve(rew: np.ndarray, path: str | Path, title: str, w: int = 100) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ep = np.arange(1, len(rew) + 1)
    ax.plot(ep, rew, color=COLORS[0], alpha=0.25, linewidth=0.8, label="episode reward")
    ax.plot(ep, moving_average(rew, w), color=COLORS[0], linewidth=2, label=f"{w}-episode moving average")
    style(ax, title, "training episode", "episode reward")
    ax.legend(frameon=False)
    save(fig, path)


def plot_histogram(rew: np.ndarray, path: str | Path, title: str, bins: int = 30) -> None:
    mu, sd = float(np.mean(rew)), float(np.std(rew))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rew, bins=bins, color=COLORS[0], edgecolor="white", linewidth=0.5)
    ax.axvline(mu, color="#0b0b0b", linewidth=1.5, linestyle="--")
    ax.text(0.02, 0.95, f"mean = {mu:.2f}\nstd = {sd:.2f}\nn = {len(rew)}",
            transform=ax.transAxes, va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    style(ax, title, "episode reward", "count")
    save(fig, path)


def plot_curves(series: dict[str, np.ndarray], path: str | Path, title: str, ylabel: str,
                w: int = 100) -> None:
    """Overlay moving averages of several runs (e.g. baseline comparison)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for (name, y), c in zip(series.items(), COLORS):
        ax.plot(np.arange(1, len(y) + 1), moving_average(y, w), color=c, linewidth=2, label=name)
    style(ax, title, "training episode", ylabel)
    ax.legend(frameon=False)
    save(fig, path)
