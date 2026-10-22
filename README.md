# IEMS 469 Assignment 1

Shuttle-dispatch MDP (enumeration, value iteration, policy iteration) and REINFORCE on
CartPole-v1 and Pong-v5, with no / mean / value baselines.

The report with all results, plots and validation tests is [docs/report.md](docs/report.md).

## Setup

The reported results were produced on Windows 11 with Python 3.13 and PyTorch with CUDA:
CartPole was trained on the CPU and Pong on an NVIDIA RTX 5070 Ti GPU. Dependencies are listed in
`requirements.txt`; install the PyTorch build that matches your own machine. ale-py ships the Pong ROM.

Run everything from the repository root; scripts are modules (`python -m ...`).

Unit tests (a few seconds):

```
pytest
```

## Part A: shuttle dispatch MDP

```
python -m mdp.run_all          # solves with all three methods -> results/mdp/{enum,vi,pi}.npz
python -m mdp.plot_mdp         # -> figures/mdp_enum_V0.png, mdp_vi_V.png, mdp_pi_policy.png
```

`run_all` prints iteration counts, the threshold s*, and the sup-norm gaps between methods.
All parameters are flags (`--K --c-f --c-h --s-max --gamma --T --tol --out`).

## Part B: policy gradient

Every training run writes `results/<env>_<baseline>_s<seed>/` with `config.json` (all settings,
including learning rates), `train_log.csv` (one row per episode: reward, length, loss, entropy,
grad_norm, adv_var, value_loss) and `ckpt.pt`.

### Part 1: no baseline

```
python -m pg.train --env cartpole --baseline none                 # 1000 episodes, gamma 0.95, cpu
python -m pg.train --env pong --baseline none                     # 5000 episodes, gamma 0.99, cuda

python -m pg.evaluate --run cartpole_none_s0                      # 500 episodes -> eval_rewards.npy
python -m pg.evaluate --run pong_none_s0 --num-envs 8

python -m pg.plot_pg --env cartpole                               # -> figures/cartpole_none_{train,hist}.png
python -m pg.plot_pg --env pong
```

Pong is slow (about 1-2 s per episode early on, more once rallies get longer). If a run stops,
rerun the same command with `--resume`; it continues from the last checkpoint (every
`--ckpt-every 100` episodes) and the episode numbering carries on.

### Part 2: value baseline

Same settings as Part 1, with a learned state-value baseline `V(s)`:

```
python -m pg.train --env cartpole --baseline value
python -m pg.evaluate --run cartpole_value_s0
python -m pg.train --env pong --baseline value                    # --resume after an interruption
python -m pg.evaluate --run pong_value_s0 --num-envs 8

python -m pg.plot_pg --env cartpole                               # -> figures/cartpole_value_{train,hist}.png
python -m pg.plot_pg --env pong
```

When runs for more than one baseline exist, `plot_pg` also writes `{env}_compare.png` (100-episode
moving averages) and `{env}_compare_adv_var.png` (mean squared advantage, log scale).

A constant mean baseline is also implemented (`--baseline mean`). With return standardization on
(the default) it gives exactly the same update as no baseline, so it only has an effect with
`--no-normalize-returns`; it is used in the `pg_baseline_variance` validation.

### Useful flags

| flag | meaning |
|---|---|
| `--lr`, `--value-lr` | Adam learning rates for the policy and the value network. Defaults: CartPole 0.01, Pong 0.001 (see report 3.1.1); `--value-lr` defaults to `--lr` |
| `--normalize-returns` / `--no-normalize-returns` | standardize advantages per episode (default on) |
| `--mean-window 100` | number of past episodes in the mean baseline |
| `--arch cnn` | small CNN instead of the 6400-200-1 MLP for Pong |
| `--num-envs N` | N parallel envs; N episodes are averaged into one update (default 1 = one update per episode) |
| `--seed`, `--device`, `--episodes`, `--gamma`, `--run-name`, `--out` | as named |
| `--greedy` (evaluate) | argmax actions instead of sampling; saves `eval_rewards_greedy.npy` |
| `--smoke` | a few episodes into `results/smoke_*`, to check that a script runs |

## Validation experiments

Each script's docstring states what it tests, the expected result and why. Numbers are printed,
data goes to `results/validation/`, figures to `figures/validation/`. Every script accepts `--smoke`.

### MDP (seconds)

| command | what it checks | expected |
|---|---|---|
| `python -m tests.validation.mdp_consistency` | VI vs PI vs enumeration; enumeration gap for T = 10...500 against gamma^T max\|V\| | \|V_VI - V_PI\| ~ 1e-7, gap always under the bound, ~1e-7 at T = 500; pi_{T-1} never dispatches |
| `python -m tests.validation.mdp_structure` | V non-decreasing in s, policy is a threshold rule, under 6 parameter settings | always true for the stationary solution; finite-horizon policies break it only in the last few periods near s = 200 (free truncation of arrivals) |
| `python -m tests.validation.mdp_sensitivity` | s* under sweeps of c_f, c_h, K; c_f = 0 and c_h = 0 | c_f = 0: s* = 1; c_h = 0: never dispatch; s* up with c_f (c_f = 800: never dispatch, because the cap S_MAX = 200 bounds the holding cost), down with c_h |
| `python -m tests.validation.mdp_deterministic` | arrivals fixed at 3, 10-step trace of s, a, cost, V, Q; simulated cost vs V(s0) | periodic trace (dispatch at s = 15), simulated cost = V(s0) to ~1e-6 |
| `python -m tests.validation.mdp_convergence` | VI residual and error on a log scale vs gamma^k; PI iterations | rate 0.95, ~430 VI iterations, 6 PI iterations |

### Policy gradient (these train; minutes for CartPole, hours for Pong)

| command | what it checks | expected |
|---|---|---|
| `python -m tests.validation.pg_random_baseline` | random policy on both games | CartPole ~22, Pong ~-20.5 to -21 |
| `python -m tests.validation.pg_sign_flip` | CartPole trained on -G_t vs G_t | flipped run falls below random (~9-10), normal run rises |
| `python -m tests.validation.pg_first_steps --env cartpole` (or `--env pong`, `--run <run>`) | first 10 steps: state, P(a=1), a, r, G_t, A_t | CartPole G_t matches the closed form; Pong G_t has the sign of the next point |
| `python -m tests.validation.pg_multi_seed --env cartpole --baseline none` | 5 seeds, mean +- std learning curve (extra flags are passed to `pg.train`); also run with `--baseline value` | large spread between seeds; without a baseline 2 of 5 seeds collapse to ~9 steps, with the value baseline none do |
| `python -m tests.validation.pg_preprocess_viz` | raw frame, preprocessed frames, frame difference | binary 80x80, only paddles and ball, background gone, static objects cancel in the difference |
| `python -m tests.validation.pg_action_balance --env pong` | share of RIGHT and mean P(RIGHT) during training | stays inside (0.05, 0.95), no collapse to one action |
| `python -m tests.validation.pg_baseline_variance --env cartpole` | gradient variance and E[(G-b)^2] for the 3 baselines: (A) same fixed policy and episodes, (B) during training | value < mean < none |
| `python -m tests.validation.pg_value_fit --env cartpole` | corr(V(s_t), G_t) and explained variance during training | both start below 0 (untrained network) and rise to ~0.99 |

## Files

```
common/seed.py           set_seed for python, numpy, torch and the env
common/device.py         get_device: --device, else cpu for CartPole, cuda otherwise
common/logger.py         append one episode row to a CSV
common/plotting.py       moving average, training curve, histogram, overlay plot
mdp/shuttle_mdp.py       ShuttleMDP: cost (S, 2), P (2, S, S), Q-values, threshold()
mdp/enumeration.py       finite-horizon backward induction, V_T = 0
mdp/value_iteration.py   value iteration with residual history
mdp/policy_iteration.py  policy iteration, exact evaluation, ties keep the current action
mdp/run_all.py           runs the three solvers, saves results/mdp/
mdp/plot_mdp.py          MDP figures
pg/networks.py           CartPolePolicy, PongPolicy, PongCNN, ValueNet
pg/returns.py            discounted returns (with Pong reset on score), standardization
pg/baselines.py          NoBaseline, MeanBaseline, ValueBaseline, advantage()
pg/pong_utils.py         assignment preprocess(), frame difference, action mapping
pg/reinforce.py          rollout, vectorized rollout, REINFORCE update
pg/train.py              training entry point
pg/evaluate.py           rollout of a checkpoint, saves eval rewards
pg/plot_pg.py            training curves, histograms, baseline comparison
tests/unit/              pytest unit tests
tests/validation/        validation experiments above
```
