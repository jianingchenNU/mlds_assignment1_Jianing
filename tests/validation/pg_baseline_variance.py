"""Compare gradient and advantage variance of the none / mean / value baselines.

What it tests
  A. Controlled (same policy, same episodes, only b differs):
     - get a fixed policy: --run checkpoint, or --warmup episodes of plain REINFORCE;
     - sample K "fit" episodes: b_mean = mean of their G_t, V_phi fit by MSE on them;
     - sample K fresh episodes and, for each baseline, compute the per-episode policy gradient
       g_i = grad of -sum_t log pi(a_t|s_t) (G_t - b_t) (no standardization).
     Reports the total gradient variance tr Cov(g) = sum_j Var_i(g_ij), Var(|g_i|), and the
     advantage second moment E[(G_t - b_t)^2] pooled over all steps.
  B. During training: with the same seed, trains each baseline for E episodes with
     --no-normalize-returns and reports the mean adv_var and the variance of grad_norm over training.

Expected
  value < mean < none for every statistic in A.
  - none vs mean: E[(G - c)^2] = Var(G) + (E[G] - c)^2 is minimised by c = E[G]; with
    CartPole's all-positive returns (E[G] ~ 10-20) this removes a large constant term.
  - mean vs value: V(s_t) also explains state-dependent variation of G_t, leaving only the noise.
  B should show the same ordering, but less cleanly: the three runs drift to different policies,
  so their statistics are not measured on the same distribution.

Why
  Variance reduction is the whole point of a baseline. A is the clean test of that claim (the
  gradient expectation is unchanged, only its spread differs); B shows whether it holds in the
  actual Part 2 runs. Standardization is off because it rescales every episode and would hide
  the effect (and cancels the mean baseline exactly, see
  tests/unit/test_networks_baselines.py::test_mean_baseline_cancelled_by_normalization).

    python -m tests.validation.pg_baseline_variance --env cartpole
    python -m tests.validation.pg_baseline_variance --env cartpole --run cartpole_none_s0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.device import get_device
from common.seed import set_seed
from pg.baselines import ValueBaseline
from pg.reinforce import rollout
from pg.returns import discounted_returns
from pg.train import DEFAULTS, act_mapper, encoder, make_env, make_policy
from tests.validation._common import OUT, train_with_hook

VALUE_DIMS = {"cartpole": (4, 128), "pong": (6400, 200)}


def sample(env, policy, device, name: str, k: int):
    gamma = DEFAULTS[name]["gamma"]
    eps = []
    for _ in range(k):
        obs, act, rew = rollout(env, policy, device, encoder(name), act_mapper(name))
        G = discounted_returns(rew, gamma, reset_on_nonzero=name == "pong")
        eps.append((torch.as_tensor(obs, device=device), torch.as_tensor(act, device=device),
                    torch.as_tensor(G, device=device, dtype=torch.float32)))
    return eps


def episode_grad(policy, obs, act, adv) -> torch.Tensor:
    dist = policy(obs)
    a = act.long() if isinstance(dist, torch.distributions.Categorical) else act.float()
    loss = -(dist.log_prob(a) * adv).sum()
    grads = torch.autograd.grad(loss, list(policy.parameters()))
    return torch.cat([g.flatten() for g in grads])


def controlled(name: str, policy, device, k: int, fit_steps: int, seed: int) -> pd.DataFrame:
    env = make_env(name)
    set_seed(seed, env)
    fit, test = sample(env, policy, device, name, k), sample(env, policy, device, name, k)
    env.close()

    G_fit = torch.cat([G for _, _, G in fit])
    b_mean = G_fit.mean()
    vb = ValueBaseline(*VALUE_DIMS[name], 0.01, device)
    obs_fit = torch.cat([o for o, _, _ in fit])
    for _ in range(fit_steps):
        vb.update(obs_fit, G_fit)

    baselines = {"none": lambda o, G: torch.zeros_like(G),
                 "mean": lambda o, G: torch.full_like(G, b_mean.item()),
                 "value": lambda o, G: vb(o, G)}
    rows = []
    for bname, b in baselines.items():
        # running sums keep memory O(#params) even for the 1.3M-parameter Pong MLP
        s1 = s2 = None
        norms, sq = [], []
        for obs, act, G in test:
            adv = G - b(obs, G)
            g = episode_grad(policy, obs, act, adv)
            s1 = g if s1 is None else s1 + g
            s2 = g * g if s2 is None else s2 + g * g
            norms.append(g.norm().item())
            sq.append(adv.pow(2).cpu())
        tr_cov = (s2 / k - (s1 / k) ** 2).sum().item()
        rows.append(dict(baseline=bname, trace_cov_grad=tr_cov, var_grad_norm=np.var(norms),
                         adv_second_moment=torch.cat(sq).mean().item()))
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["cartpole", "pong"], default="cartpole")
    p.add_argument("--run", help="fixed policy from results/<run>/ckpt.pt (else --warmup)")
    p.add_argument("--warmup", type=int, default=100, help="REINFORCE episodes to get a non-trivial policy")
    p.add_argument("--k", type=int, default=100, help="episodes in each of the fit and test sets")
    p.add_argument("--fit-steps", type=int, default=500, help="Adam steps to fit V_phi")
    p.add_argument("--train-episodes", type=int, default=500, help="episodes per baseline in part B")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.warmup, args.k, args.fit_steps, args.train_episodes = 2, 3, 5, 3
    name = args.env
    device = get_device(None, name)

    if args.run:
        cfg = json.loads((Path("results") / args.run / "config.json").read_text())
        name = cfg["env"]
        device = get_device(None, name)
        policy = make_policy(name, cfg.get("arch", "mlp")).to(device)
        ckpt = torch.load(Path("results") / args.run / "ckpt.pt", map_location="cpu", weights_only=False)
        policy.load_state_dict(ckpt["policy"])
    else:
        _, policy, _ = train_with_hook(name, "none", args.warmup, args.seed)

    OUT.mkdir(parents=True, exist_ok=True)
    a = controlled(name, policy, device, args.k, args.fit_steps, args.seed + 1)
    print("A. fixed policy, same test episodes:")
    print(a.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    a.to_csv(OUT / f"pg_baseline_variance_controlled_{name}.csv", index=False)

    rows = []
    for b in ("none", "mean", "value"):
        df, _, _ = train_with_hook(name, b, args.train_episodes, args.seed, norm=False)
        df.to_csv(OUT / f"pg_baseline_variance_train_{name}_{b}.csv", index=False)
        rows.append(dict(baseline=b, mean_adv_var=df.adv_var.mean(), var_grad_norm=df.grad_norm.var(),
                         final100_reward=df.reward.tail(100).mean()))
    bdf = pd.DataFrame(rows)
    print("B. during training (same seed, no return normalization):")
    print(bdf.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    bdf.to_csv(OUT / f"pg_baseline_variance_train_{name}.csv", index=False)


if __name__ == "__main__":
    main()
