"""
visualization.py

Simple, non-fancy plotting script. Lives at project root next to
Qvalue_bias.py and train.py.

Produces under plots/:
  1. heatmap_<algo>_<env>.png   -- 21x21 state-space grid, Q_pred - MC_return
  2. reward_curves.png          -- eval reward vs. timesteps, all runs together

The MC-return-vs-Q-value-over-training-time plot is dropped: it needs a
q_values.npz log that nothing in train.py currently produces.

Auto-discovers runs under ./models/ (folders named
"<algo>_<env>_steps<N>_seed<S>"), skips anything with "PendulumDiscrete"
in the name since it isn't trained on main.
"""

import glob
import os
import re

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, SAC, TD3

from Qvalue_bias import estimate_return_at_state, get_q_value

MODEL_DIR = "./models"
PLOT_DIR = "./plots"
GRID_N = 21

ALGO_CLASSES = {"dqn": DQN, "sac": SAC, "td3": TD3}

RUN_RE = re.compile(r"^(dqn|sac|td3)_(.+)_steps(\d+)_seed(\d+)$")

# 2D state-space grid per env: which two state dims to vary, rest fixed at 0.
GRID_SPECS = {
    "CartPole-v1": {
        "x_label": "Pole angle (rad)",
        "y_label": "Pole angular velocity (rad/s)",
        "x_vals": np.linspace(-0.20, 0.20, GRID_N),
        "y_vals": np.linspace(-2.0, 2.0, GRID_N),
    },
    "Pendulum-v1": {
        "x_label": "Theta (rad)",
        "y_label": "Theta dot (rad/s)",
        "x_vals": np.linspace(-np.pi, np.pi, GRID_N),
        "y_vals": np.linspace(-8.0, 8.0, GRID_N),
    },
}


def discover_runs():
    """Find (algo, env_id, run_dir) for every trained model under ./models."""
    runs = []
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "*"))):
        name = os.path.basename(path)
        m = RUN_RE.match(name)
        if not m:
            continue
        algo, env_id, _, _ = m.groups()
        if "PendulumDiscrete" in env_id:
            continue
        if not os.path.exists(os.path.join(path, "final_model.zip")):
            continue
        runs.append((algo, env_id, path))
    return runs


def set_state(env_id, env, x, y):
    """Force env into a given (x, y) state, return the obs. Not general-purpose,
    just enough for CartPole-v1 and Pendulum-v1."""
    env.reset()  # required so wrappers (OrderEnforcing/TimeLimit) accept step() after this
    base = env.unwrapped
    if env_id == "CartPole-v1":
        state = np.array([0.0, 0.0, x, y], dtype=np.float32)  # cart pos/vel fixed at 0
        base.state = state
        return state.copy()
    else:  # Pendulum-v1
        base.state = np.array([x, y], dtype=np.float64)
        return base._get_obs()


# ============================================================
# 1. Heatmap
# ============================================================

def make_heatmap(algo, env_id, run_dir):
    spec = GRID_SPECS[env_id]
    model = ALGO_CLASSES[algo].load(os.path.join(run_dir, "final_model"))
    env = gym.make(env_id)

    bias_grid = np.zeros((GRID_N, GRID_N))
    for i, y in enumerate(spec["y_vals"]):
        for j, x in enumerate(spec["x_vals"]):
            def reset_fn(env, x=x, y=y):
                return set_state(env_id, env, x, y)

            result = estimate_return_at_state(
                model, env, model.gamma, reset_fn, n_episodes=3, max_steps=500
            )
            q = get_q_value(model, result["start_obs"], result["start_action"])
            q_pred = q if np.isscalar(q) else q["min"]
            bias_grid[i, j] = q_pred - result["mean"]

    env.close()

    vmax = np.abs(bias_grid).max() or 1.0
    plt.figure(figsize=(6, 5))
    plt.imshow(
        bias_grid,
        extent=[spec["x_vals"][0], spec["x_vals"][-1], spec["y_vals"][0], spec["y_vals"][-1]],
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.colorbar(label="Q_pred - MC_return (+ = overestimation)")
    plt.xlabel(spec["x_label"])
    plt.ylabel(spec["y_label"])
    plt.title(f"Q-value bias -- {algo.upper()} ({env_id})")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/heatmap_{algo}_{env_id}.png", dpi=150)
    plt.close()


# ============================================================
# 2. Reward curves
# ============================================================

def plot_reward_curves(runs):
    plt.figure(figsize=(8, 5))
    for algo, env_id, run_dir in runs:
        eval_path = os.path.join(run_dir, "eval", "evaluations.npz")
        if not os.path.exists(eval_path):
            print(f"[skip] no eval log for {algo}_{env_id}")
            continue
        data = np.load(eval_path)
        plt.plot(data["timesteps"], data["results"].mean(axis=1), label=f"{algo.upper()} ({env_id})")

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Evaluation Reward")
    plt.title("Deterministic Evaluation Reward Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/reward_curves.png", dpi=150)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    runs = discover_runs()

    if not runs:
        print("No trained models found under ./models/ -- nothing to plot.")
        return

    print(f"Found {len(runs)} run(s): {[r[:2] for r in runs]}")

    for algo, env_id, run_dir in runs:
        print(f"Heatmap for {algo}_{env_id} ...")
        make_heatmap(algo, env_id, run_dir)

    print("Reward curves ...")
    plot_reward_curves(runs)

    print(f"Plots saved to: {PLOT_DIR}/")


if __name__ == "__main__":
    main()