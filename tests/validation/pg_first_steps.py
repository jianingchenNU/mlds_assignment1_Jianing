"""Print the first steps of one episode: state, pi(a|s), action, reward, G_t, advantage.

What it tests
  Plays one episode with an untrained policy (or a checkpoint via --run) and prints the first
  --steps steps. For Pong the state is summarised as the number of nonzero pixels in the frame
  difference. The advantage uses no baseline; A_t is G_t standardized over the episode.

Expected
  CartPole (gamma = 0.95, reward 1 per step, episode length L):
    G_t = (1 - 0.95^(L - t)) / 0.05, strictly decreasing, at most 20. Early steps have positive
    standardized advantage, the last steps before the pole falls negative. Untrained
    P(a=1) ~ 0.5. A trained policy shows P(a=1) reacting to the pole angle (state[2]).
  Pong (gamma = 0.99, reward only when a point is scored):
    r_t = 0 until the first point; G_t = r_T * 0.99^(T - t) for every step of that rally, with the
    sign of the point (mostly -1 for an untrained agent). The t = 0 input is the first preprocessed
    frame itself (previous frame is zeros); later inputs are 0 wherever nothing moved, so
    nnz_diff = 0 on frames where neither paddle nor ball changed.

Why
  Every column can be checked by hand from the reward sequence and gamma, which isolates the
  return/advantage computation from the learning dynamics.

    python -m tests.validation.pg_first_steps --env cartpole
    python -m tests.validation.pg_first_steps --env pong --run pong_none_s0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.device import get_device
from common.seed import set_seed
from pg.baselines import NoBaseline, advantage
from pg.reinforce import rollout
from pg.returns import discounted_returns
from pg.train import DEFAULTS, act_mapper, encoder, make_env, make_policy
from tests.validation._common import OUT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["cartpole", "pong"], default="cartpole")
    p.add_argument("--run", help="load results/<run>/ckpt.pt instead of an untrained policy")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="same as default (one episode)")
    args = p.parse_args()

    name = args.env
    arch = "mlp"
    if args.run:
        cfg = json.loads((Path("results") / args.run / "config.json").read_text())
        name, arch = cfg["env"], cfg.get("arch", "mlp")
    device = get_device(None, name)
    env = make_env(name)
    set_seed(args.seed, env)
    policy = make_policy(name, arch).to(device)
    if args.run:
        ckpt = torch.load(Path("results") / args.run / "ckpt.pt", map_location="cpu", weights_only=False)
        policy.load_state_dict(ckpt["policy"])

    gamma = DEFAULTS[name]["gamma"]
    obs, act, rew = rollout(env, policy, device, encoder(name), act_mapper(name))
    G = discounted_returns(rew, gamma, reset_on_nonzero=name == "pong")
    obs_t = torch.as_tensor(obs, device=device)
    with torch.no_grad():
        dist = policy(obs_t)
        p1 = (dist.probs[:, 1] if name == "cartpole" else dist.probs).cpu().numpy()
    A, _ = advantage(NoBaseline(), obs_t, torch.as_tensor(G, dtype=torch.float32), norm=True)

    n = min(args.steps, len(rew))
    if name == "cartpole":
        state = {f"s{i}": obs[:n, i] for i in range(4)}
    else:
        state = {"nnz_diff": (obs[:n] != 0).reshape(n, -1).sum(1)}
    df = pd.DataFrame({"t": np.arange(n), **state, "P(a=1)": p1[:n], "a": act[:n], "r": rew[:n],
                       "G_t": G[:n], "A_t": A[:n].cpu().numpy()})
    print(f"{name}, episode length {len(rew)}, total reward {rew.sum():.0f}, gamma {gamma}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if name == "cartpole":
        L = len(rew)
        exp = (1 - gamma ** (L - np.arange(n))) / (1 - gamma)
        print(f"max |G_t - closed form| = {np.abs(G[:n] - exp).max():.2e}")
    else:
        first = int(np.flatnonzero(rew)[0]) if rew.any() else None
        print(f"first point scored at step {first}, reward {rew[first] if first is not None else None}")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"pg_first_steps_{name}.csv", index=False)


if __name__ == "__main__":
    main()
