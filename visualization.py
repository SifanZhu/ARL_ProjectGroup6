
"""
visualization.py

Plots trained RL models from ./models/.

For each (algorithm, environment):
    1. heatmap_<algo>_<env>.png
       - Q-value heatmap
       - Q-bias heatmap (Q - MC return)

    2. snapshot_<algo>_<env>.png
       - Q-value
       - MC-return baseline
       - state2 fixed to 0

Also:
    reward_curves_cartpole.png
    reward_curves_pendulum.png
"""

import glob
import os
import re

import gymnasium as gym
import matplotlib.pyplot as plt
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


# ============================================================
# Find trained models
# ============================================================

def discover_runs():
    runs = []

    for path in glob.glob(os.path.join(MODEL_DIR, "*")):
        name = os.path.basename(path)
        match = RUN_RE.match(name)

        if not match:
            continue

        algo, env_id, _, _ = match.groups()

        if "PendulumDiscrete" in env_id:
            continue

        if os.path.exists(os.path.join(path, "final_model.zip")):
            runs.append((algo, env_id, path))

    return sorted(runs)


# ============================================================
# Set environment to a specific state
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
# Compute Q, MC return and bias grids
# ============================================================

def make_heatmaps(model, env_id):
    spec = GRID_SPECS[env_id]
    env = gym.make(env_id)

    q_grid = np.zeros((GRID_N, GRID_N))
    mc_grid = np.zeros((GRID_N, GRID_N))

    for i, y in enumerate(spec["y"]):
        for j, x in enumerate(spec["x"]):

            def reset_fn(env, x=x, y=y):
                return set_state(env_id, env, x, y)

            result = estimate_return_at_state(
                model,
                env,
                model.gamma,
                reset_fn,
                n_episodes=N_MC_EPISODES,
                max_steps=MAX_STEPS,
            )

            q = get_q_value(
                model,
                result["start_obs"],
                result["start_action"],
            )

            q_pred = float(q if np.isscalar(q) else q["min"])

            q_grid[i, j] = q_pred
            mc_grid[i, j] = result["mean"]

    env.close()

    return q_grid, mc_grid, q_grid - mc_grid


# ============================================================
# Q-value + bias heatmaps in one PNG
# ============================================================

def plot_heatmaps(algo, env_id, q_grid, bias_grid):
    spec = GRID_SPECS[env_id]

    vmax = np.abs(bias_grid).max() or 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Q-value
    im1 = axes[0].imshow(
        q_grid,
        extent=[
            spec["x"][0], spec["x"][-1],
            spec["y"][0], spec["y"][-1],
        ],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )

    axes[0].set_title("Q-value")
    axes[0].set_xlabel(spec["x_label"])
    axes[0].set_ylabel(spec["y_label"])
    fig.colorbar(im1, ax=axes[0], label="Q-value")

    # Bias
    im2 = axes[1].imshow(
        bias_grid,
        extent=[
            spec["x"][0], spec["x"][-1],
            spec["y"][0], spec["y"][-1],
        ],
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )

    axes[1].set_title("Q-value Bias")
    axes[1].set_xlabel(spec["x_label"])
    axes[1].set_ylabel(spec["y_label"])
    fig.colorbar(im2, ax=axes[1], label="Q - MC return")

    fig.suptitle(f"{algo.upper()} ({env_id})")
    plt.tight_layout()

    filename = f"heatmap_{algo}_{env_id}.png"
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close()


# ============================================================
# Q-value vs MC-return snapshot
# ============================================================

def plot_snapshot(algo, env_id, q_grid, mc_grid):
    """Plot two 1D slices: state1=0 and state2=0."""

    spec = GRID_SPECS[env_id]

    x_zero = np.argmin(np.abs(spec["x"]))
    y_zero = np.argmin(np.abs(spec["y"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --------------------------------------------------------
    # state1 = 0 -> vary state2
    # --------------------------------------------------------

    axes[0].plot(
        spec["y"],
        q_grid[:, x_zero],
        marker="o",
        markersize=3,
        label="Q-value",
    )

    axes[0].plot(
        spec["y"],
        mc_grid[:, x_zero],
        marker="o",
        markersize=3,
        label="MC return",
    )

    axes[0].axvline(0, linestyle="--", linewidth=1)

    axes[0].set_xlabel(spec["y_label"])
    axes[0].set_ylabel("Return")
    axes[0].set_title(
        f"state1 = {spec['x'][x_zero]:.2f}"
    )
    axes[0].legend()

    # --------------------------------------------------------
    # state2 = 0 -> vary state1
    # --------------------------------------------------------

    axes[1].plot(
        spec["x"],
        q_grid[y_zero, :],
        marker="o",
        markersize=3,
        label="Q-value",
    )

    axes[1].plot(
        spec["x"],
        mc_grid[y_zero, :],
        marker="o",
        markersize=3,
        label="MC return",
    )

    axes[1].axvline(0, linestyle="--", linewidth=1)

    axes[1].set_xlabel(spec["x_label"])
    axes[1].set_ylabel("Return")
    axes[1].set_title(
        f"state2 = {spec['y'][y_zero]:.2f}"
    )
    axes[1].legend()

    fig.suptitle(f"{algo.upper()} ({env_id})")
    plt.tight_layout()

    filename = f"snapshot_{algo}_{env_id}.png"
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close()


# ============================================================
# Reward curves
# ============================================================

def plot_reward_curves(runs, env_id, filename):
    plt.figure(figsize=(8, 5))

    for algo, run_env, run_dir in runs:
        if run_env != env_id:
            continue

        path = os.path.join(run_dir, "eval", "evaluations.npz")

        if not os.path.exists(path):
            continue

        data = np.load(path)

        plt.plot(
            data["timesteps"],
            data["results"].mean(axis=1),
            label=os.path.basename(run_dir),
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Evaluation Reward")
    plt.title(f"{env_id} -- Evaluation Reward")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    runs = discover_runs()
    print(f"Found {len(runs)} run(s).")

    for algo, env_id, run_dir in runs:
        print(f"Processing {algo}_{env_id}...")

        model = ALGORITHMS[algo].load(
            os.path.join(run_dir, "final_model")
        )

        q_grid, mc_grid, bias_grid = make_heatmaps(
            model,
            env_id,
        )

        plot_heatmaps(
            algo,
            env_id,
            q_grid,
            bias_grid,
        )

        plot_snapshot(
            algo,
            env_id,
            q_grid,
            mc_grid,
        )

    plot_reward_curves(
        runs,
        "CartPole-v1",
        "reward_curves_cartpole.png",
    )

    plot_reward_curves(
        runs,
        "Pendulum-v1",
        "reward_curves_pendulum.png",
    )

    print(f"Plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()

