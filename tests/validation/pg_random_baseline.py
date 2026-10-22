"""Uniform-random policy on both games: the reference level a trained agent has to beat.

What it tests
  Plays N episodes per game choosing uniformly between the two allowed actions
  (CartPole: left/right; Pong: RIGHT/LEFT only, env actions 2/3) and reports mean and std of the
  episode reward.

Expected
  CartPole-v1: mean ~22, std ~10 (the pole falls after a couple dozen random pushes).
  Pong-v5: mean ~-20.5 to -21; a random paddle almost never returns the ball.

Why
  Learning curves only mean something relative to chance. These numbers are also the reference for
  pg_sign_flip.py (a sign-flipped learner should do worse than random) and are a check that the
  environment and the action mapping are wired up as intended.

    python -m tests.validation.pg_random_baseline --episodes 100
"""
import argparse

import numpy as np
from tqdm import trange

from common.seed import set_seed
from pg.train import act_mapper, make_env
from tests.validation._common import OUT


def random_rewards(name: str, n: int, seed: int) -> np.ndarray:
    env = make_env(name)
    set_seed(seed, env)
    act_map = act_mapper(name)
    out = []
    for _ in trange(n, desc=f"random {name}"):
        env.reset()
        total, done = 0.0, False
        while not done:
            _, r, term, trunc, _ = env.step(act_map(np.random.randint(2)))
            total += r
            done = term or trunc
        out.append(total)
    env.close()
    return np.array(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--pong-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="3 CartPole and 1 Pong episode")
    args = p.parse_args()
    n = {"cartpole": 3, "pong": 1} if args.smoke else {"cartpole": args.episodes, "pong": args.pong_episodes}

    OUT.mkdir(parents=True, exist_ok=True)
    for name, k in n.items():
        rew = random_rewards(name, k, args.seed)
        np.save(OUT / f"random_{name}.npy", rew)
        print(f"{name}: n={k}  mean {rew.mean():.2f}  std {rew.std():.2f}  min {rew.min():.0f}  max {rew.max():.0f}")


if __name__ == "__main__":
    main()
