"""Roll out a trained policy; saves results/<run>/eval_rewards.npy (eval_rewards_greedy.npy with --greedy).

    python -m pg.evaluate --run cartpole_none_s0
    python -m pg.evaluate --run pong_value_s0 --num-envs 8 --device cuda
    python -m pg.evaluate --run smoke_cartpole_none_s0 --episodes 3
"""
import argparse
import json
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from common.device import get_device
from common.seed import set_seed
from pg.reinforce import rollout, rollout_vec
from pg.train import act_mapper, encoder, make_env, make_policy


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run directory name under --results")
    p.add_argument("--results", default="results")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--greedy", action="store_true", help="argmax action instead of sampling")
    p.add_argument("--num-envs", type=int, default=1)
    # different from the training seeds so evaluation episodes are not replays of training ones
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--device")
    p.add_argument("--smoke", action="store_true", help="3 episodes")
    args = p.parse_args(argv)
    if args.smoke:
        args.episodes = 3
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    run = Path(args.results) / args.run
    cfg = json.loads((run / "config.json").read_text())
    name = cfg["env"]
    device = get_device(args.device, name)

    policy = make_policy(name, cfg.get("arch", "mlp")).to(device)
    ckpt = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    vec = args.num_envs > 1
    env = gym.vector.AsyncVectorEnv([partial(make_env, name)] * args.num_envs) if vec else make_env(name)
    set_seed(args.seed, env)
    encode, act_map = encoder(name), act_mapper(name)

    rewards, lengths = [], []
    bar = tqdm(total=args.episodes, desc=f"eval {args.run}")
    while len(rewards) < args.episodes:
        if vec:
            eps = rollout_vec(env, policy, device, encode, act_map, args.greedy)
        else:
            eps = [rollout(env, policy, device, encode, act_map, args.greedy)]
        for _, _, rew in eps[:args.episodes - len(rewards)]:
            rewards.append(rew.sum())
            lengths.append(len(rew))
            bar.update(1)
    bar.close()
    env.close()

    suffix = "_greedy" if args.greedy else ""
    np.save(run / f"eval_rewards{suffix}.npy", np.array(rewards))
    np.save(run / f"eval_lengths{suffix}.npy", np.array(lengths))
    print(f"{args.run} ({'greedy' if args.greedy else 'sampled'}, n={len(rewards)}, "
          f"ckpt episode {ckpt['episode']}): mean {np.mean(rewards):.2f}  std {np.std(rewards):.2f}")


if __name__ == "__main__":
    main()
