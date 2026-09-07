"""
temporalAnalysis.py

Temporal analysis: how does the learned Q-function evolve over training?

For each (algo, env, seed) *single run*, walks its mid-training
checkpoints (sorted by step count) and plots, per fixed state:
    x-axis: training steps
    y-axis: Q-value
    one curve per action

Output: temporal_<algo>_<env>_seed<seed>.png

Assumes each run directory has intermediate checkpoints saved during
its own training (see discover_checkpoints() in rl_common.py -- the
folder/filename convention there is a placeholder, confirm the real
one with your colleague). If you only ever save one final model per
run with no intermediate checkpoints, there is nothing to plot here --
save checkpoints during training instead (e.g. SB3's CheckpointCallback).
"""

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from rl_common import (
    ACTION_SETS,
    INITIAL_STATES,
    PLOT_DIR,
    action_label,
    discover_checkpoints,
    discover_runs,
    load_model_from_path,
    q_value_for_action,
    set_state,
)

# Fixed color per action so colors stay consistent across subplots.
ACTION_COLORS = plt.get_cmap("tab10").colors


def make_temporal_plot(run):
    algo, env_id, seed = run["algo"], run["env_id"], run["seed"]
    checkpoints = discover_checkpoints(run)

    if len(checkpoints) < 2:
        print(f"  skipping {algo}_{env_id}_seed{seed}: only one checkpoint")
        return

    steps_list = [c["steps"] for c in checkpoints]
    states = INITIAL_STATES[env_id]
    actions = ACTION_SETS[env_id]

    env = gym.make(env_id)

    # q_values[state_idx, action_idx, checkpoint_idx]
    q_values = np.zeros((len(states), len(actions), len(checkpoints)))

    for ci, c in enumerate(checkpoints):
        model = load_model_from_path(algo, c["path"])

        for si, (x, y) in enumerate(states):
            obs = set_state(env_id, env, x, y)

            for ai, a in enumerate(actions):
                q_values[si, ai, ci] = q_value_for_action(model, obs, a)

    env.close()

    fig, axes = plt.subplots(
        1, len(states), figsize=(4 * len(states), 4), sharex=True, sharey=True
    )

    for si, (x, y) in enumerate(states):
        ax = axes[si]

        for ai, a in enumerate(actions):
            ax.plot(
                steps_list,
                q_values[si, ai, :],
                marker="o",
                markersize=3,
                color=ACTION_COLORS[ai % len(ACTION_COLORS)],
                label=action_label(a),
            )

        # Mark the final checkpoint so it's obvious training stops there.
        ax.axvline(steps_list[-1], color="gray", linestyle="--", linewidth=0.8)

        ax.set_title(f"state=({x:.2f}, {y:.2f})", fontsize=9)
        ax.set_xlabel("Training steps")
        ax.xaxis.set_major_formatter(mticker.EngFormatter())
        if si == 0:
            ax.set_ylabel("Q-value")
        ax.legend(fontsize=7)

    fig.suptitle(f"{algo.upper()} ({env_id}, seed={seed}) -- Q-value evolution over training")
    plt.tight_layout()

    filename = f"temporal_{algo}_{env_id}_seed{seed}.png"
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close()
    print(f"  saved {filename}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    runs = discover_runs()
    print(f"Found {len(runs)} run(s).")

    for run in runs:
        print(f"Processing {run['algo']}_{run['env_id']}_seed{run['seed']}...")
        make_temporal_plot(run)

    print(f"Plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()