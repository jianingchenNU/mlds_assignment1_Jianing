# IEMS 469 — Assignment 1 Report

**Name:** Jianing Chen  
**GitHub username:** jianingchenNU  
**Repository:** mlds_assignment1_Jianing  
**Link:** <https://github.com/jianingchenNU/mlds_assignment1_Jianing>

---

## 1. How to run

**Hardware used for the reported results:** Windows 11, Python 3.13, PyTorch with CUDA; CartPole
was trained on the CPU and Pong on an NVIDIA RTX 5070 Ti GPU. Dependencies are listed in
`requirements.txt`. Run every command from the repository root; all randomness is seeded
(`--seed`, default 0).

**Run** (unit tests take seconds, Part A seconds, CartPole ~30 s, Pong ~3–4 h on a GPU):

```
python -m pytest                                   # unit tests
python -m mdp.run_all && python -m mdp.plot_mdp     # Part A: solve, Figures 1-3
python -m pg.train --env <cartpole|pong> --baseline <none|value>    # Part 1: none, Part 2: value
python -m pg.evaluate --run <env>_<baseline>_s0    # 500-episode evaluation, prints mean and std
python -m pg.plot_pg --env <cartpole|pong>         # training curves, histograms, comparisons
```

Defaults follow the assignment (Adam, γ = 0.95 CartPole / 0.99 Pong, lr 0.01), except the Pong
learning rate (0.001, see 3.1.1). A stopped Pong run continues with `--resume`. The command for each
validation experiment is given in its section; the README lists every command.

---

## 2. Value and Policy Iteration (shuttle dispatch)

### 2.1 MDP formulation

| Element | Definition |
|---|---|
| State | `s ∈ {0, 1, …, 200}` = customers waiting at the start of a period (201 states) |
| Action | `a ∈ {0, 1}`: 0 = wait, 1 = dispatch one shuttle |
| Order of events | observe `s` → choose `a` → shuttle takes `min(s, K)` → holding cost on those left → `A_t` new customers arrive |
| One-period cost | `c(s, a) = c_f · a + c_h · max(s − K a, 0)` |
| Transition | `s' = min(max(s − K a, 0) + A, 200)`, `A ~ Unif{1,…,5}` |
| Objective | minimize `E[Σ_t γ^t c(s_t, a_t)]`, `γ = 0.95` |

Parameters: `K = 15, c_f = 100, c_h = 2, S_MAX = 200, γ = 0.95`.

When several arrival values are truncated to 200, their probabilities are added (e.g. from
`s = 198, a = 0`: `P(s' = 199) = 0.2`, `P(s' = 200) = 0.8`). Bellman equation (costs, so min):

`V(s) = min_a { c(s, a) + γ Σ_{s'} P(s' | s, a) V(s') }`

**Assumptions.**

1. The holding cost is charged on the customers left after dispatch; customers arriving during period
   `t` are first charged in period `t + 1`.
2. Customers who arrive when the station already holds 200 people leave at no cost (truncation).
3. Dispatching at `s = 0` is a legal action that costs `c_f`.
4. Ties are broken toward "wait" (enumeration, value iteration); policy iteration keeps the current
   action unless the other one is better by more than 10⁻¹⁰.

### 2.2 Methods

**(a) Enumeration (backward induction), T = 500.** `V_T ≡ 0`; for `t = T−1, …, 0`:
`Q_t(s, a) = c(s, a) + γ P_a V_{t+1}`, `V_t = min_a Q_t`, `π_t = argmin_a Q_t`. This gives `V_0`
and the time-indexed policy `π_t` (500 × 201).

**(b) Value iteration, T = ∞.** Start from `V_0 ≡ 0` and apply the Bellman operator until
`‖V_{k+1} − V_k‖_∞ < 10⁻⁸`. Converged in **429 iterations**.

**(c) Policy iteration.** Start from "never dispatch". Evaluation solves `(I − γ P_π) V = c_π`
exactly; improvement takes the greedy action (ties keep the current action). Converged in
**6 iterations**; number of states whose action changed per iteration: `[196, 28, 22, 7, 1, 0]`.

### 2.3 Results

| Method | s* (dispatch iff s ≥ s*) | Iterations | V(0) | V(200) |
|---|---|---|---|---|
| Enumeration, `V_0` (T = 500) | 13 | 500 backward steps | 576.67 | 3821.59 |
| Value iteration | 13 | 429 | 576.67 | 3821.59 |
| Policy iteration | 13 | 6 | 576.67 | 3821.59 |

`‖V_VI − V_PI‖_∞ = 1.9·10⁻⁷`, `‖V_enum,0 − V_PI‖_∞ = 4.9·10⁻⁹`, and the VI and PI policies are
identical in all 201 states.

<figure>
<img src="../figures/mdp_enum_V0.png" width="520">
<figcaption>Figure 1. (a) Enumeration, T = 500: optimal expected discounted cost at t = 0 vs. number of customers waiting.</figcaption>
</figure>

<figure>
<img src="../figures/mdp_vi_V.png" width="520">
<figcaption>Figure 2. (b) Value iteration: optimal value function V* vs. number of customers waiting.</figcaption>
</figure>

<figure>
<img src="../figures/mdp_pi_policy.png" width="520">
<figcaption>Figure 3. (c) Policy iteration: optimal action vs. number of customers waiting; threshold s* = 13.</figcaption>
</figure>

**Interpretation.** The optimal policy is a threshold rule: wait while fewer than 13 people are
waiting and dispatch once 13 or more are. The threshold is below the capacity 15 because holding 13
people for one more period costs 26 and makes it likelier that some cannot board later. `V` increases
with `s`, and faster once `s` exceeds `K`, because the extra people need more than one trip.

### 2.4 Edge cases and how they are handled

| Edge case | Handling | Evidence |
|---|---|---|
| Dispatch with an empty station (`s = 0`) | Allowed, cost exactly `c_f = 100`, next state `A` | unit test; never chosen by the optimal policy |
| Dispatch with fewer than K people | `max(s − K, 0) = 0`, everyone boards, no negative states | unit tests on cost and transition |
| More than K people when dispatching | `s − K` remain and pay holding cost; one shuttle per period | unit test (`c(20, 1) = 100 + 2·5 = 110`; dispatch from 200 → 186…190) |
| Station cap at 200 | arrival values that overflow are mapped to 200 and their probabilities **added** | unit test; every row of `P` sums to 1 (checked on construction) |
| Cap makes holding cost bounded | Never dispatching costs at most `2·200 = 400` per period, `V_never(200) = 8000` | with `c_f = 800` the optimal policy never dispatches (2.5.5) |
| Ties between actions | VI/enumeration → wait; PI keeps the current action within 10⁻¹⁰ so it cannot cycle | `c_f = 0`: at `s = 0` both actions cost 0, s* = 1 as expected |
| `c_h = 0` / `c_f = 0` | Degenerate cases | `c_h = 0` → never dispatch; `c_f = 0` → dispatch whenever `s ≥ 1` |
| End of a finite horizon | Enumeration's `π_t` is not stationary near `t = T` | last 3 periods never dispatch, as derived by hand (2.5.3) |
| Numerical convergence | VI stops at residual `10⁻⁸`; theory bounds the error by `tol·γ/(1−γ) = 1.9·10⁻⁷` | observed `‖V_VI − V_PI‖_∞ = 1.885·10⁻⁷`, inside the bound |

### 2.5 Validation tests

#### 2.5.1 The three methods agree; enumeration converges at rate γ^T

`python -m tests.validation.mdp_consistency`

Theory: `‖V_{enum,0}^{(T)} − V*‖_∞ ≤ γ^T ‖V*‖_∞`. For T = 10, 25, 50, 100, 200, 500 the measured
gap is always under the bound and falls with the predicted slope, from 917 at T = 10 to 4.9·10⁻⁹ at
T = 500 (bound 2.8·10⁻⁸); s* at t = 0 is 13 for every T (Figure 4).

<figure>
<img src="../figures/validation/mdp_consistency.png" width="520">
<figcaption>Figure 4. Gap between the finite-horizon value V_enum,0 and V* as a function of the horizon T, against the bound γ^T‖V*‖.</figcaption>
</figure>

#### 2.5.2 Convergence rate of VI and PI

`python -m tests.validation.mdp_convergence`

VI is a γ-contraction, so the residual should shrink by a factor 0.95 per iteration. Measured ratio
`r_{k+1}/r_k` over the last 50 iterations: mean **0.95000**, max 0.95003; on a log scale the
residual is parallel to `r_0 γ^k` (Figure 5). VI took 429 iterations (the bound
`log(tol/r_0)/log γ` predicts at most 476). PI took 6 iterations, as expected from its superlinear
convergence.

<figure>
<img src="../figures/validation/mdp_convergence.png" width="520">
<figcaption>Figure 5. Value iteration residual ‖V_{k+1} − V_k‖_∞ on a log scale with the theoretical slope γ^k.</figcaption>
</figure>

#### 2.5.3 Hand-checkable cases

`python -m tests.validation.mdp_deterministic`

**Deterministic arrivals.** With `A ≡ 3`, s* is still 13; the reachable states are multiples of 3,
so the shuttle leaves at `s = 15` with exactly 15 people and the system cycles
`3 → 6 → 9 → 12 → 15 → (dispatch) → 3`:

```
 t  s  a    cost       V
 0  0  0   0.000 569.753
 1  3  0   6.000 599.740
 2  6  0  12.000 624.989
 3  9  0  18.000 645.252
 4 12  0  24.000 660.265
 5 15  1 100.000 669.753
 6  3  0   6.000 599.740
```

By hand: `V(3) = (6 + 12γ + 18γ² + 24γ³ + 100γ⁴) / (1 − γ⁵) = 135.673 / 0.22622 = 599.74` and
`V(0) = γ V(3) = 569.75`, matching the solver. Simulating this policy for 2000 periods gives a
discounted cost that differs from `V(0)` by 1.5·10⁻¹².

**End of horizon in enumeration.** In the last period a dispatch saves at most `2·15 = 30` of holding
cost but costs 100, so it is never worth it. The saving over the last k periods is at most
`30(1 + γ + … + γ^{k−1})`: 30, 58.5, 85.6 for k = 1, 2, 3 (all < 100) and 111.3 for k = 4. So
`π_{T−1}, π_{T−2}, π_{T−3}` should never dispatch, and dispatching should first appear at `t = T−4`.
The solver gives exactly this (thresholds from the end: none, none, none, 14, 12, 13, 13, …).

#### 2.5.4 Structural properties

`python -m tests.validation.mdp_structure`

Theory predicts `V` non-decreasing in `s` and a threshold policy. Both hold under all six settings
checked: default (s* = 13), c_f = 200 (14), c_h = 5 (10), K = 10 (9), K = 30 (17) and γ = 0.99 (13).
Only the finite-horizon policies of the last few periods break the threshold shape, and only near
`s = 200`, where extra arrivals are dropped for free (assumption 2).

#### 2.5.5 Parameter sensitivity

`python -m tests.validation.mdp_sensitivity`

| sweep | s* | expectation |
|---|---|---|
| c_f = 0, 25, 50, 100, 200, 400, 800 | 1, 8, 11, 13, 14, 15, never | non-decreasing: costlier trips → larger batches ✓ |
| c_h = 0.5, 1, 2, 4, 8 | 15, 14, 13, 11, 8 | non-increasing: costlier waiting → dispatch sooner ✓ |
| K = 5, 10, 15, 20, 30, 50 | 5, 9, 13, 16, 17, 17 | rises then saturates: a big shuttle is not worth filling beyond the point where holding cost exceeds the saved trip ✓ |
| c_f = 0 | 1 | dispatch whenever anyone waits ✓ |
| c_h = 0 | never | waiting is free ✓ |

`c_f = 800` gives "never dispatch" because of the cap (assumption 2): ignoring the queue costs at
most 400 per period (`V(200) = 8000`), so a trip at 800 never pays for itself. Evaluating all 200
threshold policies exactly confirms that none is better in any state.

<figure>
<img src="../figures/validation/mdp_sensitivity.png" width="560">
<figcaption>Figure 6. Optimal threshold s* under sweeps of c_f, c_h and K ("x" = never dispatch).</figcaption>
</figure>

### 2.6 With 3 more months (Part A)

- **Simulation check.** Simulate the optimal policy from every start state under random arrivals and
  compare the average discounted cost with `V*(s)` using confidence intervals.
- **Remove the cap.** Increase `S_MAX` or replace truncation with a lost-customer penalty, to confirm
  the cap does not drive the answer.
- **Richer models.** Time-varying (rush-hour) arrivals, several shuttles with return times, and the
  average-cost criterion; check that model-free Q-learning recovers s* = 13.

---

## 3. Policy Gradient, Part 1 (REINFORCE, no baseline)

### 3.1 Algorithm

REINFORCE with one parameter update per episode:

1. Roll out one episode with the current stochastic policy `π_θ`.
2. Discounted reward-to-go within the episode: `G_t = Σ_{k≥0} γ^k r_{t+k}` (no bootstrapping; an
   episode ends at `terminated` **or** `truncated`).
3. Standardize over the episode: `A_t = (G_t − mean G) / (std G + 10⁻⁸)`.
4. Loss `L(θ) = −Σ_t log π_θ(a_t | s_t) · A_t`; one Adam step.

Standardization is the only addition to plain REINFORCE in Part 1; no baseline is used.

| | CartPole-v1 | Pong-v5 |
|---|---|---|
| γ | 0.95 | 0.99 |
| Input | 4-d state | `preprocess(frame_t) − preprocess(frame_{t−1})`, 80×80 = 6400 (zeros for the previous frame at t = 0) |
| Network | 4 → 128 (ReLU) → 2 logits, Categorical | 6400 → 200 (ReLU) → 1 logit = log-odds of RIGHT, Bernoulli |
| Actions | 0 = push left, 1 = push right | 1 → env action 2 (RIGHT), 0 → env action 3 (LEFT) |
| Returns | standard | reset to 0 whenever a point is scored (`r ≠ 0`), so each rally is credited only with its own outcome |
| Episodes | 1000 | 5000 |
| Optimizer | Adam, lr 0.01 | Adam, **lr 0.001** |

`preprocess` is the assignment's function, used verbatim; Pong-v5 keeps its defaults (frameskip 4,
sticky actions p = 0.25). Evaluation samples actions from the final policy, as in training.

#### 3.1.1 Why Pong uses lr = 0.001 instead of 0.01

The assignment allows a different learning rate if training is more stable. In two short Pong runs
with the same seed, at lr = 0.01 the mean policy entropy (maximum ln 2 = 0.693) fell from 0.635 to
**0.104 within ~110 episodes** while the reward was still ≈ −20: the policy became nearly
deterministic before it had learned anything. At lr = 0.001 entropy was still **0.641 after 137
episodes**, so the full run uses 0.001. CartPole learns well at the default 0.01 and keeps it.

### 3.2 Results

#### 3.2.1 CartPole-v1

**Evaluation over 500 episodes: mean = 110.78, std = 4.84** (min 99, max 123).

<figure>
<img src="../figures/cartpole_none_train.png" width="520">
<figcaption>Figure 7. CartPole-v1, no baseline: episode reward vs. training episode, with the 100-episode moving average.</figcaption>
</figure>

<figure>
<img src="../figures/cartpole_none_hist.png" width="480">
<figcaption>Figure 8. CartPole-v1, no baseline: histogram of 500 evaluation episodes of the final policy (mean 110.78, std 4.84).</figcaption>
</figure>

100-episode mean training reward:

| episodes | 0–99 | 100–199 | 200–299 | 300–399 | 400–499 | 500–599 | 600–699 | 700–799 | 800–899 | 900–999 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean | 100.2 | 169.4 | 470.3 | **485.7** | 225.2 | 111.5 | 244.5 | 131.6 | 118.1 | 121.4 |
| episodes hitting 500 | 0 | 7 | 86 | 93 | 25 | 0 | 0 | 0 | 0 | 0 |

The agent solves the task around episode 300 (93 of 100 episodes reach the 500-step cap), then
collapses and settles near 120. The evaluation uses the final policy, as the assignment asks, so it
reports the collapsed policy, not the peak. The 5-seed run (3.4.5) shows this is not a bad seed.
Two causes:

1. **Truncation looks like failure after standardization.** CartPole-v1 stops at 500 steps even when
   the pole is balanced. With γ = 0.95 such an episode has `G_0 ≈ 20` but `G_499 = 1`, so after
   standardization the **last 63 steps get negative advantage (down to −7.0)**: a perfect episode
   pushes down the probability of correct actions.
2. **Short effective horizon.** γ = 0.95 looks about 20 steps ahead. In 49 of 50 evaluation episodes
   the **cart leaves the track (|x| > 2.4)** rather than the pole falling: the slow drift costs
   nothing within 20 steps, so the policy never learns to correct it. This also explains the small
   evaluation std.

#### 3.2.2 Pong-v5

**Evaluation over 500 episodes: mean = 13.20, std = 4.37** (min −12, median 14, max 21); the agent
won 98.4% of the games. This matches the last 100 training episodes (+13.94).

<figure>
<img src="../figures/pong_none_train.png" width="520">
<figcaption>Figure 9. Pong-v5, no baseline: episode reward vs. training episode, with the 100-episode moving average.</figcaption>
</figure>

<figure>
<img src="../figures/pong_none_hist.png" width="480">
<figcaption>Figure 10. Pong-v5, no baseline: histogram of 500 evaluation episodes of the final policy (mean 13.20, std 4.37).</figcaption>
</figure>

- 5000 episodes at lr = 0.001 (interrupted once and continued with `--resume`). For ~500 episodes
  the agent loses almost every point; the 100-episode average then passes **0 at episode 623** and
  +10 at episode 1768, and ends at **+13.94**.
- Entropy decreases gradually (0.54 → 0.07 per 500 episodes), unlike the early collapse at lr = 0.01.
- Episodes get ~4× longer (1072 → ~4000 steps) once the agent returns the ball. The per-episode
  reward stays noisy because actions are sampled and v5 has sticky actions.

### 3.3 Edge cases and how they are handled

| Edge case | Handling | How it is checked |
|---|---|---|
| Episode ends by truncation (CartPole 500-step cap) | `terminated or truncated` ends the episode; returns are computed inside the episode only, no bootstrap | 3.2.1 analyses the effect of truncation on standardized advantages |
| Standardizing a constant return vector (e.g. a 1-step episode) | `std + 10⁻⁸`; constant input maps to zeros, no NaN | unit test |
| Pong rewards are sparse and per point | returns reset at every nonzero reward, so a point lost later does not blame actions from an earlier rally that was won | unit test (hand-computed example); 3.4.3: `G_0 = −0.99^63` for a point lost at step 63 |
| No previous frame at the start of a Pong episode | the previous frame is all zeros | unit test; 3.4.3 (t = 0 input is the frame itself) |
| `preprocess` modifies its input in place (slice assignment) | called on a copy so the environment's observation is untouched | unit test |
| Only RIGHT/LEFT allowed in Pong | policy is binary; mapping 1→2, 0→3 | unit test |
| Saturated sigmoid gives `log 0` | network outputs logits; `Bernoulli(logits=…)` computes log-prob stably | no NaN in training logs |
| Long Pong runs can be interrupted | checkpoint every 100 episodes (policy, optimizer, RNG states); `--resume` continues the episode count | the Pong run was interrupted and resumed with a continuous log |
| Reproducibility | one seed fixes python, numpy, torch (CPU and CUDA) and the environment | CartPole seed 0 retrained: all 1000 episode rewards identical |
| Policy collapses to one action (π → 1) | not prevented (plain REINFORCE has no entropy bonus); detected through the logged entropy and gradient norm | 2 of 5 CartPole seeds collapse to ~9 steps (3.4.5); Pong lr chosen to avoid early collapse (3.1.1) |

### 3.4 Validation tests

#### 3.4.1 Random policy reference

`python -m tests.validation.pg_random_baseline`

| game | episodes | mean | std | min | max |
|---|---|---|---|---|---|
| CartPole | 100 | 23.16 | 11.90 | 8 | 63 |
| Pong (RIGHT/LEFT only) | 20 | −20.55 | 0.74 | −21 | −19 |

Trained CartPole ends ~5× above random (~21× at its peak); trained Pong scores +13 against −20.6.

#### 3.4.2 Sign flip

`python -m tests.validation.pg_sign_flip`

Same seed and settings, but trained on `−G_t`. If the gradient is correct, the flipped agent should
learn the *opposite* task (drop the pole as fast as possible) and do worse than random.

| run | last-100 mean reward |
|---|---|
| normal | 121.37 |
| random policy | 22.87 |
| **flipped** | **9.27** |

The flipped agent reaches ~9, the shortest possible episode, while the normal run rises: the two
leave the random level in opposite directions (Figure 11), so the update moves probability toward
high-return actions.

<figure>
<img src="../figures/validation/pg_sign_flip.png" width="520">
<figcaption>Figure 11. CartPole trained on G_t (normal) and on −G_t (flipped), 100-episode moving averages (log scale), with the random-policy level.</figcaption>
</figure>

#### 3.4.3 First 10 steps by hand

```
python -m tests.validation.pg_first_steps --env cartpole
python -m tests.validation.pg_first_steps --env cartpole --run cartpole_none_s0
python -m tests.validation.pg_first_steps --env pong
```

- **CartPole, untrained**: episode of length L = 12. Every reward is 1, so
  `G_t = (1 − 0.95^{L−t}) / 0.05`; max deviation from this closed form is 1.8·10⁻¹⁵. `P(a=1)` is
  0.37–0.48. Advantages are positive for early steps and negative for the last ones, i.e. the steps
  right before the pole falls are blamed.
- **CartPole, trained**: `P(push right)` follows the pole's angular velocity `s3`: 0.88–0.99 when the
  pole falls right (`s3 ≈ 0.6`), 0.00–0.19 when `s3 ≈ 0.02–0.35`, a physically sensible rule.
- **Pong, untrained**: `P(RIGHT) ≈ 0.48–0.49`. The first point is lost at step 63 (`r = −1`), so every
  step of that rally has `G_t = −0.99^{63−t}`: `G_0 = −0.5309` ✓.

#### 3.4.4 Preprocessing check

`python -m tests.validation.pg_preprocess_viz`

After preprocessing the frame is binary 80×80 with only the two paddles and the ball (32 of 6400
pixels on); background and score are gone. In the difference image static objects cancel and moving
objects remain as +1/−1 pairs, which lets an MLP see the direction of motion (Figure 12).

<figure>
<img src="../figures/validation/pg_preprocess_viz.png" width="620">
<figcaption>Figure 12. Raw Pong frame, two consecutive preprocessed frames, and their difference (the network input).</figcaption>
</figure>

#### 3.4.5 Five seeds

`python -m tests.validation.pg_multi_seed --env cartpole --baseline none --seeds 0 1 2 3 4`

Final 100-episode mean per seed: 121.37, 89.63, 480.59, 9.59, 9.33 (**142.10 ± 174.90**). In
Figure 13 the ±1 std band dips below 0 only because the 5 values are skewed; no episode scores below 8.

<figure>
<img src="../figures/validation/pg_multi_seed_cartpole_none.png" width="520">
<figcaption>Figure 13. CartPole, no baseline, 5 seeds: mean ± 1 std of the 100-episode moving average.</figcaption>
</figure>

- **The algorithm learns**: every seed rises well above random, and seeds 0–3 reach ~450–500 at
  some point.
- **It is unstable**: seeds 0–2 keep oscillating between ~100 and 500.
- **Seeds 3 and 4 collapse** to a deterministic single-action policy (final entropy 0.06 and 0.001,
  ~9 steps per episode). Each collapse starts with one large update (grad norm ~30 vs a median of
  ~6); entropy then falls to ≈ 0 within a few episodes, and once π ≈ 1, `∇ log π ≈ 0`, so the policy
  cannot recover. With the same code the sign-flip test and seeds 0–2 behave as expected, so this is
  a property of plain REINFORCE with these settings, not a bug.

### 3.5 With 3 more months (Part 1)

- Handle truncation correctly (bootstrap or add the time step to the state) and measure how much of
  the CartPole collapse disappears.
- Learning-rate and γ sweeps over many seeds with confidence intervals, plus keeping the best
  checkpoint.
- Batch several episodes per update and add an entropy bonus; for Pong, try a CNN policy and
  evaluate without sticky actions.

---

## 4. Policy Gradient, Part 2 (REINFORCE with a value baseline)

### 4.1 Chosen baseline and implementation

We use a **state-dependent (non-constant) baseline `b_t = V_φ(s_t)`**, a learned value network.
Returns in both games depend strongly on the state, so `V(s)` removes more variance than a constant
(4.4.2: 99× vs 9× lower gradient variance). It depends only on `s_t`, not on `a_t`, so the gradient
stays unbiased. Unlike a constant baseline, it is not cancelled by the per-episode standardization,
so all Part 1 settings stay the same and Part 1 serves as the no-baseline comparison.

| Item | Choice |
|---|---|
| Value network | CartPole: 4 → 128 (ReLU) → 1. Pong: 6400 → 200 (ReLU) → 1, same frame-difference input as the policy |
| Target | raw discounted return `G_t` (same `G_t` as the policy, including the Pong reset at each point) |
| Loss | `MSE(V_φ(s_t), G_t)` over the episode, one Adam step per episode |
| Optimizer | separate Adam, lr equal to the policy lr (CartPole 0.01, Pong 0.001) |
| Advantage | `A_t = standardize(G_t − V_φ(s_t).detach())`: the baseline is subtracted **before** standardization |
| Order per episode | compute `b_t` with the current `V_φ` → policy step → value step, so `b_t` never uses the current episode's targets |

`detach()` keeps the policy loss from training `φ` (checked by a unit test). All other settings are
identical to Part 1, so differences come from the baseline alone.

### 4.2 Results

#### 4.2.1 CartPole-v1

**Evaluation over 500 episodes: mean = 124.46, std = 5.53** (Part 1 without baseline:
110.78 ± 4.84).

<figure>
<img src="../figures/cartpole_value_train.png" width="520">
<figcaption>Figure 14. CartPole-v1, value baseline: episode reward vs. training episode, with the 100-episode moving average.</figcaption>
</figure>

<figure>
<img src="../figures/cartpole_value_hist.png" width="480">
<figcaption>Figure 15. CartPole-v1, value baseline: histogram of 500 evaluation episodes of the final policy (mean 124.46, std 5.53).</figcaption>
</figure>

<figure>
<img src="../figures/cartpole_compare.png" width="520">
<figcaption>Figure 16. CartPole-v1: 100-episode moving average with no baseline (Part 1) and with the value baseline.</figcaption>
</figure>

<figure>
<img src="../figures/cartpole_compare_adv_var.png" width="520">
<figcaption>Figure 17. CartPole-v1: mean squared advantage E[(G_t − b_t)²] per episode (before standardization), log scale.</figcaption>
</figure>

- **Faster early learning**: the value baseline is ahead for the first 300 episodes (100-episode
  means 149 / 253 / 496 vs 100 / 169 / 470); 227 vs 211 episodes reach 500 in total.
- **Much lower advantage variance**: `E[(G_t − b_t)²]` stays around 300 without a baseline and falls
  to **0.22** with the value baseline (3 orders of magnitude, Figure 17).
- **But it collapses after episode ~400 like Part 1.** The cause is truncation (3.2.1): time is not
  in the CartPole state, so V predicts ≈ 20 for any balanced state while the true `G_t` at the end of
  a truncated episode falls to 1, and the last ~60 steps of a perfect episode still get large
  negative advantages. Terminated episodes, in contrast, are fitted almost perfectly
  (corr(V, G) = 1.00).

#### 4.2.2 Pong-v5

5000 episodes, seed 0, lr 0.001 for both networks, all other settings as in Part 1.

**Evaluation over 500 episodes: mean = 12.57, std = 4.36** (Part 1 without baseline: 13.20 ± 4.37).

<figure>
<img src="../figures/pong_value_train.png" width="520">
<figcaption>Figure 18. Pong-v5, value baseline: episode reward vs. training episode, with the 100-episode moving average.</figcaption>
</figure>

<figure>
<img src="../figures/pong_value_hist.png" width="480">
<figcaption>Figure 19. Pong-v5, value baseline: histogram of 500 evaluation episodes of the final policy (mean 12.57, std 4.36).</figcaption>
</figure>

| evaluation, 500 episodes | none (Part 1) | value |
|---|---|---|
| mean ± std | **13.20** ± 4.37 | 12.57 ± 4.36 |
| median | 14 | 13 |
| games won | 98.4% | **99.0%** |
| worst game | −12 | **−4** |

The value policy scores 0.63 points less on average (Welch t-test p = 0.02) but loses fewer games
and has no very bad game. With one training seed per method this compares two trained policies, not
the two methods in general.

<figure>
<img src="../figures/pong_compare.png" width="520">
<figcaption>Figure 20. Pong-v5: 100-episode moving average with no baseline (Part 1) and with the value baseline.</figcaption>
</figure>

<figure>
<img src="../figures/pong_compare_adv_var.png" width="520">
<figcaption>Figure 21. Pong-v5: mean squared advantage E[(G_t − b_t)²] per episode (before standardization), log scale.</figcaption>
</figure>

| training | none (Part 1) | value |
|---|---|---|
| 100-ep moving average first ≥ −10 | episode 551 | **479** |
| 100-ep moving average first ≥ 0 | **623** | 765 |
| 100-ep moving average first ≥ +10 | **1768** | 2086 |
| final 100-episode mean | 13.94 | 12.83 |
| `E[(G_t − b_t)²]`, episodes 4000–4999 | 0.328 | **0.108** |

- **Advantage variance is about 3× lower** for the whole run (0.15 vs 0.46 in the first 1000
  episodes, 0.108 vs 0.328 in the last 1000).
- **Earlier start, smoother curve**: the value run leaves the −21 floor sooner and has none of the
  drops of the no-baseline run (around episodes 2300 and 4100). The no-baseline run crossed 0 and +10
  earlier because of a sharp jump near episode 620.
- **Same final level**: both end near +13 with almost every game won.

### 4.3 Edge cases and how they are handled

| Edge case | Handling | How it is checked |
|---|---|---|
| Policy gradient leaking into the value network | `V_φ(s).detach()` in the advantage | unit test: policy loss gives no gradient to φ |
| Baseline using the current episode's targets (bias) | `b_t` computed before the value update | fixed order in the update function |
| Scale mismatch between V and standardized returns | V fits raw `G_t`; standardization applied after subtracting | value loss decreases to 0.22 (CartPole) |
| Untrained V at the start (random predictions) | advantage is standardized, so a bad early V cannot blow up the step size | explained variance starts at −0.49 and training still proceeds (4.4.1) |
| Constant baseline cancelled by standardization | reason for choosing a state-dependent baseline | unit test: mean baseline + standardization gives the same update as no baseline |
| Resume of a long Pong run | value network weights and its Adam state are saved in the checkpoint | resumed run continues with the value network restored |
| Truncation (time not in the state) | not fixable by a state-value baseline; documented | 4.2.1 analysis |

### 4.4 Validation tests

#### 4.4.1 Does the value network learn?

`python -m tests.validation.pg_value_fit --env cartpole`

Before each value update, the correlation between `V_φ(s_t)` and `G_t` and the explained variance
`1 − Var(G − V) / Var(G)` are recorded:

| | first 50 episodes | last 50 episodes |
|---|---|---|
| corr(V, G) | −0.414 | **0.997** |
| explained variance | −0.485 | **0.992** |

Explained variance > 0 is exactly the condition under which the baseline reduces variance. It starts
negative (random network) and ends at 0.99 (Figure 22). Between episodes ~200 and ~400, when most
episodes reach the 500-step cap, the fit drops back to about 0: the truncation effect of 4.2.1.

<figure>
<img src="../figures/validation/pg_value_fit_cartpole.png" width="520">
<figcaption>Figure 22. CartPole, value baseline: correlation between V(s_t) and G_t and explained variance during training.</figcaption>
</figure>

#### 4.4.2 Does the baseline reduce gradient variance?

`python -m tests.validation.pg_baseline_variance --env cartpole`

Fix a policy (100 warm-up REINFORCE episodes), fit each baseline on 100 episodes, then compute
per-episode policy gradients on the *same* 100 fresh episodes with each baseline (no
standardization, so only `b` differs). The expected gradient is the same for all three; only the
spread should differ.

| baseline | tr Cov(g) | Var(‖g‖) | E[(G − b)²] |
|---|---|---|---|
| none | 4.95 × 10⁵ | 1.19 × 10⁵ | 346.2 |
| mean (constant) | 5.47 × 10⁴ (9× lower) | 1.52 × 10⁴ | 14.17 |
| **value** | **4995 (99× lower)** | **1904** | **1.058** |

This is the ordering theory predicts, value < mean < none. The same ordering holds during training
(500 episodes per baseline, standardization off): mean `E[(G − b)²]` is 216.7 / 16.93 / 15.71 for
none / mean / value.

#### 4.4.3 Five seeds with the value baseline

`python -m tests.validation.pg_multi_seed --env cartpole --baseline value --seeds 0 1 2 3 4`

Same settings as 3.4.5 (1000 episodes, standardization on), seeds 0–4.

| seed | 0 | 1 | 2 | 3 | 4 | across seeds |
|---|---|---|---|---|---|---|
| none: final-100 mean (Part 1) | 121.37 | 89.63 | 480.59 | 9.59 | 9.33 | 142.10 ± 174.90 |
| **value: final-100 mean** | 121.45 | 466.06 | 168.35 | 136.59 | 127.53 | **204.00 ± 132.02** |
| none: mean over all 1000 episodes | 217.8 | 299.6 | 322.6 | 165.5 | 16.9 | 204.5 ± 122.3 |
| value: mean over all 1000 episodes | 214.6 | 247.4 | 185.0 | 105.8 | 209.7 | 192.5 ± 53.3 |

<figure>
<img src="../figures/validation/pg_multi_seed_cartpole_value.png" width="520">
<figcaption>Figure 23. CartPole, value baseline, 5 seeds: mean ± 1 std of the 100-episode moving average (compare Figure 13).</figcaption>
</figure>

- **No seed collapsed with the value baseline** (every seed keeps entropy ≥ 0.17 and a final-100
  mean ≥ 121), while 2 of 5 seeds collapsed without a baseline.
- **More consistent across seeds**: the spread of the whole-run mean drops from ±122 to ±53, but the
  whole-run mean itself is about the same (192.5 vs 204.5); truncation (4.2.1) limits both.
- **Caveat**: with 5 seeds, 0/5 vs 2/5 collapses is not statistically significant (Fisher exact test
  p ≈ 0.44), and the largest gradient norms are similar with and without the baseline (130–275), so
  more seeds would be needed to claim the effect and its mechanism.

### 4.5 With 3 more months (Part 2)

- Add the time step to the value network's input, or bootstrap at truncation, and test whether the
  CartPole collapse disappears.
- Actor-critic / GAE(λ): use the value network for bootstrapped advantages and study the
  bias–variance trade-off as λ varies.
- Run the Pong comparison with several seeds per baseline, so that the difference in evaluation
  scores can be attributed to the method rather than to one training run.
