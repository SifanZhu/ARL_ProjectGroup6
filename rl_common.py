"""
rl_common.py

Shared utilities for discovering trained runs, setting environment
state, and computing Q-values / MC-return baselines. Used by
visualization.py, temporalAnalysis.py and spatialAnalysis.py.
"""

import glob
import os
import re

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, SAC, TD3

from Qvalue_bias import estimate_return_at_state, get_q_value

# ============================================================
# Configuration
# ============================================================

MODEL_DIR = "./models"
PLOT_DIR = "./plots"

GRID_N = 21
N_MC_EPISODES = 3
MAX_STEPS = 500

ALGORITHMS = {
    "dqn": DQN,
    "sac": SAC,
    "td3": TD3,
}

RUN_RE = re.compile(r"^(dqn|sac|td3)_(.+)_steps(\d+)_seed(\d+)$")

GRID_SPECS = {
    "CartPole-v1": {
        "x_label": "Pole angle (rad)",
        "y_label": "Pole angular velocity (rad/s)",
        "x": np.linspace(-0.20, 0.20, GRID_N),
        "y": np.linspace(-2.0, 2.0, GRID_N),
    },
    "Pendulum-v1": {
        "x_label": "Theta (rad)",
        "y_label": "Theta dot (rad/s)",
        "x": np.linspace(-np.pi, np.pi, GRID_N),
        "y": np.linspace(-8.0, 8.0, GRID_N),
    },
}

# Discrete action set (DQN) or representative continuous actions (SAC/TD3).
# For continuous envs we sweep a few fixed torques rather than the full
# action space, so "one curve per action" stays readable.
ACTION_SETS = {
    "CartPole-v1": [0, 1],
    "Pendulum-v1": [np.array([-2.0]), np.array([0.0]), np.array([2.0])],
}

# Fixed set of 5 initial states used for temporal (Q-vs-training-steps) plots.
INITIAL_STATES = {
    "CartPole-v1": [
        (-0.15, -1.5), (-0.075, -0.75), (0.0, 0.0), (0.075, 0.75), (0.15, 1.5)
    ],
    "Pendulum-v1": [
        (-2.5, -6.0), (-1.25, -3.0), (0.0, 0.0), (1.25, 3.0), (2.5, 6.0)
    ],
}


# ============================================================
# Discovery / loading
# ============================================================

def discover_runs():
    """Return a list of dicts: algo, env_id, steps, seed, path."""
    runs = []

    for path in glob.glob(os.path.join(MODEL_DIR, "*")):
        name = os.path.basename(path)
        match = RUN_RE.match(name)

        if not match:
            continue

        algo, env_id, steps, seed = match.groups()

        if "PendulumDiscrete" in env_id:
            continue

        if os.path.exists(os.path.join(path, "final_model.zip")):
            runs.append({
                "algo": algo,
                "env_id": env_id,
                "steps": int(steps),
                "seed": int(seed),
                "path": path,
            })

    return sorted(runs, key=lambda r: (r["algo"], r["env_id"], r["seed"], r["steps"]))


def load_model(algo, path):
    return ALGORITHMS[algo].load(os.path.join(path, "final_model"))


def group_by(runs, keys):
    """Group run dicts by a tuple of dict keys, e.g. ('algo', 'env_id', 'seed')."""
    groups = {}
    for r in runs:
        key = tuple(r[k] for k in keys)
        groups.setdefault(key, []).append(r)
    return groups


# ============================================================
# Environment state control
# ============================================================

def set_state(env_id, env, x, y):
    env.reset()
    base = env.unwrapped

    if env_id == "CartPole-v1":
        state = np.array([0.0, 0.0, x, y], dtype=np.float32)
        base.state = state
        return state.copy()

    if env_id == "Pendulum-v1":
        base.state = np.array([x, y], dtype=np.float64)
        return base._get_obs()

    raise ValueError(f"Unsupported environment: {env_id}")


# ============================================================
# Q-value / MC-return helpers
# ============================================================

def q_value_for_action(model, obs, action):
    """Scalar Q(s, a). Collapses an ensemble result ('min') to a float."""
    q = get_q_value(model, obs, action)
    return float(q if np.isscalar(q) else q["min"])


def action_label(action):
    return f"a={action}" if np.isscalar(action) else f"a={float(np.asarray(action).item()):.1f}"


def rollout_return_forced_action(env_id, env, model, gamma, x, y, forced_action,
                                  n_episodes=N_MC_EPISODES, max_steps=MAX_STEPS):
    """
    Monte-Carlo discounted return starting at state (x, y), forcing
    `forced_action` on the first step and following the model's policy
    afterwards.

    Qvalue_bias.estimate_return_at_state only follows the policy's own
    first action, so it can't give a per-action MC baseline -- this does.
    """
    returns = []

    for _ in range(n_episodes):
        obs = set_state(env_id, env, x, y)
        done = False
        truncated = False
        step = 0
        discount = 1.0
        total = 0.0
        action = forced_action

        while not (done or truncated) and step < max_steps:
            obs, reward, done, truncated, _ = env.step(action)
            total += discount * reward
            discount *= gamma
            step += 1
            action, _ = model.predict(obs, deterministic=True)

        returns.append(total)

    return float(np.mean(returns))