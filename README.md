# Was lernen Q-Funktionen in modellfreiem Reinforcement Learning wirklich?

## Overview

Model-free RL algorithms learn Q-functions that theoretically converge
toward the optimal value function, but in practice often show instability and strong
fluctuations. This project trains three such algorithms (DQN, SAC, TD3) and investigates what their
learned Q-functions actually represent: how they evolve during training, how they
compare to policy performance, and how far they deviate from the values a converged,
noise-free policy would imply.

**Algorithms / environments:**
- DQN on `CartPole-v1`
- SAC on `Pendulum-v1`
- TD3 on `Pendulum-v1`

**Frameworks:** [Gymnasium](https://gymnasium.farama.org/), [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

## Requirements

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

```bash
python train_and_evaluate.py
```

This trains all three algorithms, writes training artifacts to `logs/<ALGO>_<ENV_ID>/`,
and produces four plots in the project root:

| File                      | Content                                                              |
|---------------------------|----------------------------------------------------------------------|
| `reward_curves.png`       | Deterministic evaluation reward, all three runs combined             |
| `reward_curves_split.png` | Same, split into a CartPole and a Pendulum subplot                   |
| `q_values.png`            | Estimated Q-value on fixed reference states, all three runs combined |
| `q_values_split.png`      | Same, split into a CartPole and a Pendulum subplot                   |

## Changes from the original notebook (`boilerplate.ipynb`)

### Deterministic-evaluation plotting instead of raw training rewards

The original plots read `monitor.csv` raw, per-episode rewards collected while the agent was still
exploring (epsilon-greedy for DQN, stochastic sampling for SAC/TD3), which is inherently
noisy. `EvalCallback` already runs periodic deterministic evaluation rollouts during
training and writes them to `evaluations.npz`; `load_eval_results()` reads that file
instead, so the plotted curves show policy performance without exploration noise mixed in.

### Q-value extraction and logging

`collect_reference_states()` samples a fixed set of 50 states once per training run via
a random rollout; `QValueLoggingCallback` then queries the model's own Q-network on those same
states every `EVAL_FREQ` steps and saves the results to `q_values.npz`. Reading Q-values differs
by action space, so the callback branches on algorithm type:
  - **DQN** (discrete actions): `model.q_net(obs)` returns Q-values for all actions
    directly; the callback logs the max (the value of the greedy action).
  - **SAC / TD3** (continuous actions, share the same code path): the critic needs an
    explicit `(state, action)` pair, so the callback first asks the current deterministic
    policy for an action, then queries `model.critic(obs, action)` and averages the twin
    critics.

## Observations (run on 2026-08-01, `TOTAL_TIMESTEPS=10_000`)

**Reward curves:**
- DQN/CartPole stayed flat at a mean evaluation reward of ca. 8.6 - 9.6 for the entire run.
  The 10,000 steps are too few for CartPole to show visible policy improvement.
- SAC and TD3 on Pendulum both converged smoothly, from roughly ‑1,100 / ‑1,400 at step
  2,000 to about ‑150 to ‑170 by step 10,000. Both curves were monotonic with no jaggedness.

**Q-value curves:**
- DQN's estimated Q-value rose from ca. 0.99 to ca. 1.10 over training, then plateaued.
- SAC and TD3 both started near 0 and grew steadily more negative as training progressed,
  settling around ‑65 to ‑70 by step 10,000.
- Discrepancy: at step 10,000, the SAC/TD3 critics estimate roughly ‑65 to ‑70,
  while the actual deterministic evaluation reward at the same point is ‑150 to ‑170. The
  critics are substantially more optimistic than what the policy actually achieves (matches the
  overestimation-bias pattern).

## Limitations / next steps

- SAC and TD3 each call `collect_reference_states("Pendulum-v1", seed=0)` independently.
  Only `env.reset(seed=...)` is seeded, not `env.action_space.seed(...)`, so the two runs'
  reference states are not guaranteed to be identical. The SAC-vs-TD3 comparison of
  the Q-value curves is only approximate.
- Q-values are currently only compared to each other over time, not to empirical
  Monte-Carlo returns. Instead, measure the actual bias (over-/underestimation relative to the true
  return).
- This script needs to be run with an appropriate amount of `TOTAL_TIMESTEPS`.
