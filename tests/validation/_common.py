"""Shared paths and a training loop with a per-episode hook for the PG validation scripts."""
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from tqdm import trange

from common.device import get_device
from common.seed import set_seed
from pg.reinforce import rollout, update
from pg.train import DEFAULTS, act_mapper, encoder, make_baseline, make_env, make_policy

OUT = Path("results/validation")
FIG = Path("figures/validation")


def train_with_hook(env_name: str, baseline: str, episodes: int, seed: int = 0, norm: bool = True,
                    device: str | None = None, hook: Callable | None = None, sign: float = 1.0):
    """Same setup as pg.train; hook(obs, act, rew, policy, baseline, device) runs before each
    update and its dict is merged into that episode's row. Returns (DataFrame, policy, baseline)."""
    dev = get_device(device, env_name)
    env = make_env(env_name)
    set_seed(seed, env)
    policy = make_policy(env_name).to(dev)
    lr = DEFAULTS[env_name]["lr"]  # same lr as pg.train so validation matches the real runs
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    b = make_baseline(SimpleNamespace(baseline=baseline, env=env_name, mean_window=100, value_lr=lr), dev)
    encode, act_map = encoder(env_name), act_mapper(env_name)
    gamma = DEFAULTS[env_name]["gamma"]

    rows = []
    for ep in trange(episodes, desc=f"{env_name}/{baseline}/s{seed}"):
        obs, act, rew = rollout(env, policy, dev, encode, act_map)
        extra = hook(obs, act, rew, policy, b, dev) if hook else {}
        st = update(policy, opt, b, [(obs, act, rew)], gamma, dev, norm=norm,
                    reset_on_nonzero=env_name == "pong", sign=sign)[0]
        rows.append(dict(episode=ep, reward=rew.sum(), length=len(rew), **st, **extra))
    env.close()
    return pd.DataFrame(rows), policy, b
