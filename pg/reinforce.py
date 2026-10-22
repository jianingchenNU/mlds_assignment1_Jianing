"""REINFORCE: episode rollouts (single or vectorized) and the policy update."""
from collections.abc import Callable

import numpy as np
import torch

from pg.baselines import advantage
from pg.returns import discounted_returns

# encode(raw_obs, state) -> (network input, new state); state carries e.g. Pong's previous frame
Encoder = Callable[[np.ndarray, object], tuple[np.ndarray, object]]
Episode = tuple[np.ndarray, np.ndarray, np.ndarray]  # (obs, act, rew)


def rollout(env, policy, device: torch.device, encode: Encoder,
            act_map: Callable[[int], int] = int, greedy: bool = False) -> Episode:
    """Play one episode with the current policy (greedy: take the most likely action)."""
    raw, _ = env.reset()
    state = None
    xs, acts, rews = [], [], []
    done = False
    while not done:
        x, state = encode(raw, state)
        with torch.no_grad():
            dist = policy(torch.as_tensor(x, device=device)[None])
            a = int((dist.mode if greedy else dist.sample()).item())
        raw, r, term, trunc, _ = env.step(act_map(a))
        xs.append(x)
        acts.append(a)
        rews.append(r)
        # truncation (CartPole's 500-step cap) ends the episode too; no bootstrap
        done = term or trunc
    return np.stack(xs), np.array(acts), np.array(rews, dtype=np.float64)


def rollout_vec(envs, policy, device: torch.device, encode: Encoder,
                act_map: Callable[[int], int] = int, greedy: bool = False) -> list[Episode]:
    """Play one episode in each sub-env of a VectorEnv, all with the same policy.

    A sub-env that finishes early keeps stepping (autoreset) until the slowest one is done;
    those extra steps are discarded so each sub-env contributes exactly one episode.
    """
    raw, _ = envs.reset()
    n = envs.num_envs
    states = [None] * n
    bufs = [([], [], []) for _ in range(n)]
    active = np.ones(n, dtype=bool)
    while active.any():
        xs = []
        for i in range(n):
            x, states[i] = encode(raw[i], states[i])
            xs.append(x)
        with torch.no_grad():
            dist = policy(torch.as_tensor(np.stack(xs), device=device))
            a = (dist.mode if greedy else dist.sample()).long().cpu().numpy()
        raw, r, term, trunc, _ = envs.step(np.array([act_map(int(ai)) for ai in a]))
        for i in np.flatnonzero(active):
            bufs[i][0].append(xs[i])
            bufs[i][1].append(int(a[i]))
            bufs[i][2].append(r[i])
            active[i] = not (term[i] or trunc[i])
    return [(np.stack(x), np.array(a), np.array(r, dtype=np.float64)) for x, a, r in bufs]


def update(policy, opt, baseline, episodes: list[Episode], gamma: float, device: torch.device,
           norm: bool = True, reset_on_nonzero: bool = False, sign: float = 1.0) -> list[dict]:
    """One gradient step on the mean over episodes of L = -sum_t log pi(a_t|s_t) * A_t.

    Returns, advantages and standardization are per episode. With one episode this is plain
    REINFORCE; sign=-1 flips G for the sign-flip validation.
    """
    batch = []
    # baseline values are taken before any baseline update, so b never sees the current batch
    for obs, act, rew in episodes:
        obs_t = torch.as_tensor(obs, device=device)
        act_t = torch.as_tensor(act, device=device, dtype=torch.float32)
        G = torch.as_tensor(sign * discounted_returns(rew, gamma, reset_on_nonzero),
                            device=device, dtype=torch.float32)
        adv, raw = advantage(baseline, obs_t, G, norm)
        batch.append((obs_t, act_t, G, adv, raw))

    opt.zero_grad()
    stats = []
    for obs_t, act_t, G, adv, raw in batch:
        dist = policy(obs_t)
        # Categorical.log_prob needs integer actions, Bernoulli needs floats
        logp = dist.log_prob(act_t.long() if isinstance(dist, torch.distributions.Categorical) else act_t)
        loss = -(logp * adv).sum()
        # backward per episode so only one Pong episode's graph is in memory at a time
        (loss / len(batch)).backward()
        stats.append(dict(loss=loss.item(), entropy=dist.entropy().mean().item(),
                          # second moment, not Var: Var(G - c) = Var(G) is blind to a constant baseline
                          adv_var=raw.pow(2).mean().item()))
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf")).item()
    opt.step()

    for s, (obs_t, _, G, _, _) in zip(stats, batch):
        s["grad_norm"] = grad_norm
        s["value_loss"] = baseline.update(obs_t, G)
    return stats
