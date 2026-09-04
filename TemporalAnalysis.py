"""
temporalAnalysis.py

Temporal analysis: how does the learned Q-function evolve over training?

For each (algo, env, seed) group of runs (sorted by training steps),
picks 5 fixed initial states and plots, per state:
    x-axis: training steps
    y-axis: Q-value
    one curve per action

Output: temporal_<algo>_<env>_seed<seed>.png

Assumes multiple model directories exist per (algo, env, seed) at
different step counts (i.e. "steps" in the folder name is a training
checkpoint). If you only ever save one final model per seed, there is
nothing to plot here -- save intermediate checkpoints during training
instead (e.g. SB3's CheckpointCallback).
"""

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from rl_common import (
    ACTION_SETS,
    INITIAL_STATES,
    PLOT_DIR,
    action_label,
    discover_runs,
    group_by,
    load_model,
    q_value_for_action,
    set_state,
)


def make_temporal_plot(algo, env_id, seed, group_runs):
    group_runs = sorted(group_runs, key=lambda r: r["steps"])

    if len(group_runs) < 2:
        print(f"  skipping {algo}_{env_id}_seed{seed}: only one checkpoint")
        return

    steps_list = [r["steps"] for r in group_runs]
    states = INITIAL_STATES[env_id]
    actions = ACTION_SETS[env_id]

    env = gym.make(env_id)

    # q_values[state_idx, action_idx, checkpoint_idx]
    q_values = np.zeros((len(states), len(actions), len(group_runs)))

    for ci, r in enumerate(group_runs):
        model = load_model(algo, r["path"])

        for si, (x, y) in enumerate(states):
            obs = set_state(env_id, env, x, y)

            for ai, a in enumerate(actions):
                q_values[si, ai, ci] = q_value_for_action(model, obs, a)

    env.close()

    fig, axes = plt.subplots(1, len(states), figsize=(4 * len(states), 4), sharex=True)

    for si, (x, y) in enumerate(states):
        ax = axes[si]

        for ai, a in enumerate(actions):
            ax.plot(
                steps_list,
                q_values[si, ai, :],
                marker="o",
                markersize=3,
                label=action_label(a),
            )

        ax.set_title(f"state=({x:.2f}, {y:.2f})", fontsize=9)
        ax.set_xlabel("Training steps")
        if si == 0:
            ax.set_ylabel("Q-value")
        ax.legend(fontsize=7)

    fig.suptitle(f"{algo.upper()} ({env_id}, seed={seed}) -- Q-value evolution")
    plt.tight_layout()

    filename = f"temporal_{algo}_{env_id}_seed{seed}.png"
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close()
    print(f"  saved {filename}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    runs = discover_runs()
    groups = group_by(runs, ("algo", "env_id", "seed"))
    print(f"Found {len(groups)} (algo, env, seed) group(s).")

    for (algo, env_id, seed), group_runs in groups.items():
        print(f"Processing {algo}_{env_id}_seed{seed}...")
        make_temporal_plot(algo, env_id, seed, group_runs)

    print(f"Plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()