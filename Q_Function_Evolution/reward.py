"""
reward.py

Plot evaluation reward curves.

Two types of plots:

1. Separate:
   Uses the max-step run for each algorithm/environment.
   Output:
       plots/separate/reward/

2. Mean:
   Uses the max-step runs and averages reward curves across seeds.
   Output:
       plots/mean/reward/
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl_common import PLOT_DIR, discover_runs, group_by


# ============================================================
# Configuration
# ============================================================

ENVS = [
    "CartPole-v1",
    "Pendulum-v1",
]

SEPARATE_REWARD_DIR = os.path.join(
    PLOT_DIR,
    "separate",
    "reward",
)

MEAN_REWARD_DIR = os.path.join(
    PLOT_DIR,
    "mean",
    "reward",
)


# ============================================================
# Helpers
# ============================================================

def best_run_per_algo(runs, env_id):
    """Among runs for this env_id, keep only the max-step run per algo."""
    env_runs = [
        r for r in runs
        if r["env_id"] == env_id
    ]

    by_algo = group_by(
        env_runs,
        ("algo",),
    )

    return [
        max(group, key=lambda r: r["steps"])
        for group in by_algo.values()
    ]


def load_reward_curve(run):
    """Return (timesteps, mean_reward) or None if no reward data found."""

    eval_path = os.path.join(
        run["path"],
        "eval",
        "evaluations.npz",
    )

    if os.path.exists(eval_path):
        data = np.load(eval_path)

        return (
            data["timesteps"],
            data["results"].mean(axis=1),
        )

    monitor_path = os.path.join(
        run["path"],
        "monitor",
        "monitor.csv",
    )

    if os.path.exists(monitor_path):
        df = pd.read_csv(
            monitor_path,
            skiprows=1,
        )

        timesteps = df["l"].cumsum().to_numpy()

        # Smooth raw episode rewards.
        window = max(1, len(df) // 100)

        rewards = (
            df["r"]
            .rolling(window, min_periods=1)
            .mean()
            .to_numpy()
        )

        return timesteps, rewards

    return None


def format_steps(steps):
    """5_000_000 -> '5M', 100_000 -> '100k'."""

    if steps >= 1_000_000:
        return f"{steps / 1_000_000:g}M"

    if steps >= 1_000:
        return f"{steps / 1_000:g}k"

    return str(steps)


# ============================================================
# Separate reward plot
# ============================================================

def plot_reward_panel(ax, runs, env_id):
    """Plot reward curves for individual selected runs."""

    for run in runs:
        curve = load_reward_curve(run)

        if curve is None:
            print(
                f"  no reward data for "
                f"{run['algo']}_{env_id}_steps{run['steps']}, "
                "skipping"
            )
            continue

        timesteps, rewards = curve

        label = (
            f"{run['algo']}_{env_id}:"
            f"{format_steps(run['steps'])}"
        )

        ax.plot(
            timesteps,
            rewards,
            label=label,
        )

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Evaluation Reward")
    ax.set_title(env_id)
    ax.legend(fontsize=8)

def make_reward_plot(runs):
    """
    Create one reward figure per seed.

    Uses the max-step run for each algorithm/environment.

    Output:
        plots/separate/reward/reward_curves_seed<seed>.png
    """

    seeds = sorted(set(run["seed"] for run in runs))

    for seed in seeds:
        seed_runs = [
            run for run in runs
            if run["seed"] == seed
        ]

        fig, axes = plt.subplots(
            1,
            len(ENVS),
            figsize=(14, 5),
        )

        for ax, env_id in zip(axes, ENVS):
            best_runs = best_run_per_algo(
                seed_runs,
                env_id,
            )

            plot_reward_panel(
                ax,
                best_runs,
                env_id,
            )

        fig.suptitle(
            f"Evaluation Reward -- seed={seed}"
        )

        plt.tight_layout()

        filename = f"reward_curves_seed{seed}.png"

        output_path = os.path.join(
            SEPARATE_REWARD_DIR,
            filename,
        )

        plt.savefig(
            output_path,
            dpi=150,
        )

        plt.close()

        print(f"saved {output_path}")


# ============================================================
# Mean reward
# ============================================================

def mean_reward_curve(runs):
    """
    Calculate the mean reward curve across seeds.

    Among `runs`, only runs with the maximum training step count
    are used.

    Curves are aligned by their actual timestep before averaging.

    Returns:
        (timesteps, mean_rewards) or None.
    """

    if not runs:
        return None

    max_steps = max(
        run["steps"]
        for run in runs
    )

    selected_runs = [
        run
        for run in runs
        if run["steps"] == max_steps
    ]

    curves = []

    for run in selected_runs:
        curve = load_reward_curve(run)

        if curve is None:
            print(
                f"  no reward data for "
                f"{run['algo']}_{run['env_id']}_"
                f"steps{run['steps']}_seed{run['seed']}, "
                "skipping"
            )
            continue

        timesteps, rewards = curve

        curves.append(
            {
                "seed": run["seed"],
                "timesteps": timesteps,
                "rewards": rewards,
            }
        )

    if not curves:
        return None

    # --------------------------------------------------------
    # Find timesteps shared by all valid seeds.
    # --------------------------------------------------------

    common_timesteps = set(
        curves[0]["timesteps"]
    )

    for curve in curves[1:]:
        common_timesteps &= set(
            curve["timesteps"]
        )

    common_timesteps = sorted(
        common_timesteps
    )

    if len(common_timesteps) < 2:
        print(
            f"  fewer than two common timesteps for "
            f"{runs[0]['algo']}_{runs[0]['env_id']}"
        )
        return None

    # --------------------------------------------------------
    # Align rewards by timestep.
    # --------------------------------------------------------

    aligned_rewards = []

    for curve in curves:
        timestep_to_index = {
            timestep: i
            for i, timestep in enumerate(
                curve["timesteps"]
            )
        }

        indices = [
            timestep_to_index[timestep]
            for timestep in common_timesteps
        ]

        aligned_rewards.append(
            curve["rewards"][indices]
        )

    # Shape:
    # (num_seeds, num_timesteps)

    aligned_rewards = np.stack(
        aligned_rewards,
        axis=0,
    )

    # Average over seeds.
    mean_rewards = np.mean(
        aligned_rewards,
        axis=0,
    )

    return (
        np.asarray(common_timesteps),
        mean_rewards,
    )


def plot_mean_reward_panel(ax, runs, env_id):
    """
    Plot one mean reward curve per algorithm.

    For each algorithm:
      1. select its max-step run(s)
      2. average reward across seeds
    """

    env_runs = [
        run
        for run in runs
        if run["env_id"] == env_id
    ]

    by_algo = group_by(
        env_runs,
        ("algo",),
    )

    for algo, algo_runs in by_algo.items():
        curve = mean_reward_curve(
            algo_runs
        )

        if curve is None:
            continue

        timesteps, rewards = curve

        max_steps = max(
            run["steps"]
            for run in algo_runs
        )

        label = (
            f"{algo}_{env_id}:"
            f"{format_steps(max_steps)}"
        )

        ax.plot(
            timesteps,
            rewards,
            label=label,
        )

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Evaluation Reward")
    ax.set_title(env_id)
    ax.legend(fontsize=8)


def make_reward_plot_mean(runs):
    """
    Create the mean reward plot.

    For each (algorithm, environment):
      - select max-step runs
      - average rewards across seeds at matching timesteps

    Output:
        plots/mean/reward/reward_curves_mean.png
    """

    fig, axes = plt.subplots(
        1,
        len(ENVS),
        figsize=(14, 5),
    )

    for ax, env_id in zip(axes, ENVS):
        plot_mean_reward_panel(
            ax,
            runs,
            env_id,
        )

    fig.suptitle(
        "Evaluation Reward -- "
        "mean across seeds "
        "(max-step run per algorithm)"
    )

    plt.tight_layout()

    filename = "reward_curves_mean.png"

    output_path = os.path.join(
        MEAN_REWARD_DIR,
        filename,
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"saved {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(
        SEPARATE_REWARD_DIR,
        exist_ok=True,
    )

    os.makedirs(
        MEAN_REWARD_DIR,
        exist_ok=True,
    )

    runs = discover_runs()

    print(f"Found {len(runs)} run(s).")

    # --------------------------------------------------------
    # Separate
    # --------------------------------------------------------

    print("\n=== Separate reward plot ===")

    make_reward_plot(runs)

    # --------------------------------------------------------
    # Mean
    # --------------------------------------------------------

    print("\n=== Mean reward plot ===")

    make_reward_plot_mean(runs)


if __name__ == "__main__":
    main()