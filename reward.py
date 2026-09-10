"""
reward.py

Reward curves only (heatmap/snapshot logic split out of visualization.py).

For each (algo, env_id) group, only the run with the MOST training steps
is plotted -- so if you have both a 1M and a 5M CartPole DQN run, only
the 5M one shows up.

Output: reward_curves.png
    left subplot  -> CartPole-v1 runs
    right subplot -> Pendulum-v1 runs

Reward source per run, tried in order:
    1. <run_path>/eval/evaluations.npz   (SB3 EvalCallback)
    2. <run_path>/monitor/monitor.csv    (SB3 Monitor wrapper)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl_common import PLOT_DIR, discover_runs, group_by

ENVS = ["CartPole-v1", "Pendulum-v1"]


def best_run_per_algo(runs, env_id):
    """Among runs for this env_id, keep only the max-steps run per algo."""
    env_runs = [r for r in runs if r["env_id"] == env_id]
    by_algo = group_by(env_runs, ("algo",))
    return [max(group, key=lambda r: r["steps"]) for group in by_algo.values()]


def load_reward_curve(run):
    """Return (timesteps, mean_reward) or None if no reward data found."""

    eval_path = os.path.join(run["path"], "eval", "evaluations.npz")
    if os.path.exists(eval_path):
        data = np.load(eval_path)
        return data["timesteps"], data["results"].mean(axis=1)

    monitor_path = os.path.join(run["path"], "monitor", "monitor.csv")
    if os.path.exists(monitor_path):
        df = pd.read_csv(monitor_path, skiprows=1)  # skip SB3's json comment line
        timesteps = df["l"].cumsum().to_numpy()
        # smooth raw episode rewards -- monitor.csv is per-episode, much noisier
        # than eval-callback's averaged results.
        window = max(1, len(df) // 100)
        rewards = df["r"].rolling(window, min_periods=1).mean().to_numpy()
        return timesteps, rewards

    return None


def format_steps(steps):
    """5_000_000 -> '5M', 100_000 -> '100k'."""
    if steps >= 1_000_000:
        return f"{steps / 1_000_000:g}M"
    if steps >= 1_000:
        return f"{steps / 1_000:g}k"
    return str(steps)


def plot_reward_panel(ax, runs, env_id):
    for run in runs:
        curve = load_reward_curve(run)
        if curve is None:
            print(f"  no reward data for {run['algo']}_{env_id}_steps{run['steps']}, skipping")
            continue

        timesteps, rewards = curve
        label = f"{run['algo']}_{env_id}:{format_steps(run['steps'])}"
        ax.plot(timesteps, rewards, label=label)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Evaluation Reward")
    ax.set_title(env_id)
    ax.legend(fontsize=8)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    runs = discover_runs()
    print(f"Found {len(runs)} run(s).")

    fig, axes = plt.subplots(1, len(ENVS), figsize=(14, 5))

    for ax, env_id in zip(axes, ENVS):
        best_runs = best_run_per_algo(runs, env_id)
        #print(f"{env_id}: using {[f'{r['algo']}_steps{r['steps']}' for r in best_runs]}")
        plot_reward_panel(ax, best_runs, env_id)

    fig.suptitle("Evaluation Reward -- best (max-step) run per algorithm")
    plt.tight_layout()

    filename = "reward_curves.png"
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close()
    print(f"saved {filename} to {PLOT_DIR}/")


if __name__ == "__main__":
    main()