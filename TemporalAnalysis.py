"""
temporalAnalysis.py

Temporal analysis: how does the learned Q-function evolve over training?

Two types of plots are generated:

1. Separate:
   One temporal plot for each (algo, env, seed) run.

   Output:
       plots/separate/temporal/

2. Mean:
   For each (algo, env), select the runs with the maximum training
   step count and average Q-values across seeds at each checkpoint.

   Output:
       plots/mean/temporal/
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


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

SEPARATE_TEMPORAL_DIR = os.path.join(PLOT_DIR, "separate", "temporal")
MEAN_TEMPORAL_DIR = os.path.join(PLOT_DIR, "mean", "temporal")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def calculate_temporal_q_values(run):
    """
    Calculate Q-values for all states, actions and checkpoints of one run.

    Returns:
        steps_list:
            List of checkpoint training steps.

        q_values:
            Array with shape:
                (num_states, num_actions, num_checkpoints)
    """
    algo, env_id, seed = run["algo"], run["env_id"], run["seed"]
    checkpoints = discover_checkpoints(run)

    if len(checkpoints) < 2:
        print(
            f"  skipping {algo}_{env_id}_seed{seed}: "
            "only one checkpoint"
        )
        return None, None

    steps_list = [c["steps"] for c in checkpoints]
    states = INITIAL_STATES[env_id]
    actions = ACTION_SETS[env_id]

    env = gym.make(env_id)

    q_values = np.zeros(
        (len(states), len(actions), len(checkpoints))
    )

    for ci, checkpoint in enumerate(checkpoints):
        model = load_model_from_path(algo, checkpoint["path"])

        for si, (x, y) in enumerate(states):
            obs = set_state(env_id, env, x, y)

            for ai, action in enumerate(actions):
                q_values[si, ai, ci] = q_value_for_action(
                    model,
                    obs,
                    action,
                )

    env.close()

    return steps_list, q_values


# ---------------------------------------------------------------------------
# Separate plots
# ---------------------------------------------------------------------------


def make_temporal_plot(run):
    """
    Create a temporal Q-value plot for one individual seed/run.

    Output:
        plots/separate/temporal/
    """
    algo, env_id, seed = run["algo"], run["env_id"], run["seed"]

    steps_list, q_values = calculate_temporal_q_values(run)

    if steps_list is None:
        return

    states = INITIAL_STATES[env_id]
    actions = ACTION_SETS[env_id]

    fig, axes = plt.subplots(
        1,
        len(states),
        figsize=(4 * len(states), 4),
        sharex=True,
        sharey=True,
    )

    # Handle the case of only one state.
    if len(states) == 1:
        axes = [axes]

    for si, (x, y) in enumerate(states):
        ax = axes[si]

        for ai, action in enumerate(actions):
            ax.plot(
                steps_list,
                q_values[si, ai, :],
                marker="o",
                markersize=3,
                label=action_label(action),
            )

        # Mark the final checkpoint.
        ax.axvline(
            steps_list[-1],
            linestyle="--",
            linewidth=0.8,
        )

        ax.set_title(
            f"state=({x:.2f}, {y:.2f})",
            fontsize=9,
        )
        ax.set_xlabel("Training steps")
        ax.xaxis.set_major_formatter(mticker.EngFormatter())

        if si == 0:
            ax.set_ylabel("Q-value")

        ax.legend(fontsize=7)

    fig.suptitle(
        f"{algo.upper()} ({env_id}, seed={seed}) "
        "-- Q-value evolution over training"
    )

    plt.tight_layout()

    filename = (
        f"temporal_{algo}_{env_id}_"
        f"steps{run['steps']}_seed{seed}.png"
    )

    output_path = os.path.join(
        SEPARATE_TEMPORAL_DIR,
        filename,
    )

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  saved {output_path}")


# ---------------------------------------------------------------------------
# Mean plots
# ---------------------------------------------------------------------------


def make_temporal_plot_mean(runs):
    """
    Create one mean temporal Q-value plot for an (algo, env) group.

    `runs` should contain the runs for one specific (algo, env).

    The runs with the maximum training step count are selected.
    Q-values are then averaged across seeds at each checkpoint.
    """
    if not runs:
        return

    algo = runs[0]["algo"]
    env_id = runs[0]["env_id"]

    # ---------------------------------------------------------------
    # Select the longest training runs.
    # ---------------------------------------------------------------

    max_steps = max(run["steps"] for run in runs)

    selected_runs = [
        run
        for run in runs
        if run["steps"] == max_steps
    ]

    print(
        f"  {algo}_{env_id}: using "
        f"{len(selected_runs)} seed(s) at {max_steps} steps"
    )

    # ---------------------------------------------------------------
    # Calculate Q-values for every seed.
    # ---------------------------------------------------------------

    seed_results = []

    for run in selected_runs:
        steps_list, q_values = calculate_temporal_q_values(run)

        if steps_list is None:
            continue

        seed_results.append(
            {
                "seed": run["seed"],
                "steps": steps_list,
                "q_values": q_values,
            }
        )

    if not seed_results:
        print(
            f"  skipping mean {algo}_{env_id}: "
            "no valid runs"
        )
        return

    if len(seed_results) < 2:
        print(
            f"  warning: only {len(seed_results)} seed(s) "
            f"available for mean {algo}_{env_id}"
        )

    # ---------------------------------------------------------------
    # Align checkpoints by training step.
    #
    # This is safer than assuming checkpoint index i is always the
    # same training step across all seeds.
    # ---------------------------------------------------------------

    common_steps = set(seed_results[0]["steps"])

    for result in seed_results[1:]:
        common_steps &= set(result["steps"])

    common_steps = sorted(common_steps)

    if len(common_steps) < 2:
        print(
            f"  skipping mean {algo}_{env_id}: "
            "fewer than two common checkpoints"
        )
        return

    # ---------------------------------------------------------------
    # Collect Q-values corresponding to common checkpoints.
    # ---------------------------------------------------------------

    aligned_q_values = []

    for result in seed_results:
        step_to_index = {
            step: i
            for i, step in enumerate(result["steps"])
        }

        indices = [
            step_to_index[step]
            for step in common_steps
        ]

        aligned_q_values.append(
            result["q_values"][:, :, indices]
        )

    # Shape:
    #   (num_seeds, num_states, num_actions, num_checkpoints)
    aligned_q_values = np.stack(aligned_q_values, axis=0)

    # Mean over seeds:
    #   (num_states, num_actions, num_checkpoints)
    mean_q_values = np.mean(
        aligned_q_values,
        axis=0,
    )

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------

    states = INITIAL_STATES[env_id]
    actions = ACTION_SETS[env_id]

    fig, axes = plt.subplots(
        1,
        len(states),
        figsize=(4 * len(states), 4),
        sharex=True,
        sharey=True,
    )

    if len(states) == 1:
        axes = [axes]

    for si, (x, y) in enumerate(states):
        ax = axes[si]

        for ai, action in enumerate(actions):
            ax.plot(
                common_steps,
                mean_q_values[si, ai, :],
                marker="o",
                markersize=3,
                label=action_label(action),
            )

        # Mark final checkpoint.
        ax.axvline(
            common_steps[-1],
            linestyle="--",
            linewidth=0.8,
        )

        ax.set_title(
            f"state=({x:.2f}, {y:.2f})",
            fontsize=9,
        )
        ax.set_xlabel("Training steps")
        ax.xaxis.set_major_formatter(mticker.EngFormatter())

        if si == 0:
            ax.set_ylabel("Mean Q-value")

        ax.legend(fontsize=7)

    fig.suptitle(
        f"{algo.upper()} ({env_id}) "
        f"-- Mean Q-value evolution over training "
        f"({len(seed_results)} seeds)"
    )

    plt.tight_layout()

    filename = (
        f"temporal_{algo}_{env_id}_"
        f"steps{max_steps}_mean.png"
    )

    output_path = os.path.join(
        MEAN_TEMPORAL_DIR,
        filename,
    )

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(SEPARATE_TEMPORAL_DIR, exist_ok=True)
    os.makedirs(MEAN_TEMPORAL_DIR, exist_ok=True)

    runs = discover_runs()

    print(f"Found {len(runs)} run(s).")

    # ===============================================================
    # 1. Separate plots
    # ===============================================================

    print("\n=== Separate temporal plots ===")

    for run in runs:
        print(
            f"Processing "
            f"{run['algo']}_{run['env_id']}_seed{run['seed']}..."
        )

        make_temporal_plot(run)

    # ===============================================================
    # 2. Mean plots
    # ===============================================================

    print("\n=== Mean temporal plots ===")

    # Group by (algorithm, environment).
    groups = {}

    for run in runs:
        key = (
            run["algo"],
            run["env_id"],
        )

        groups.setdefault(key, []).append(run)

    for (algo, env_id), group_runs in groups.items():
        print(f"Processing mean {algo}_{env_id}...")

        make_temporal_plot_mean(group_runs)

    print("\nPlots saved to:")
    print(f"  Separate: {SEPARATE_TEMPORAL_DIR}/")
    print(f"  Mean:     {MEAN_TEMPORAL_DIR}/")


if __name__ == "__main__":
    main()