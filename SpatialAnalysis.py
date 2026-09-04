"""
spatialAnalysis.py

Spatial analysis: where in the state space is the error biggest?

Beyond the 2D heatmap, this sweeps one state dimension at a time
(freezing the other at 0) and plots, per sweep:
    left:  Q-value vs the free dimension, one curve per action
    right: bias (Q - MC return) vs the free dimension, one curve per action

Uses the highest-step (i.e. most-trained) run per (algo, env, seed).

Output: spatial_<algo>_<env>_seed<seed>_dim1.png / _dim2.png
"""

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from rl_common import (
    ACTION_SETS,
    GRID_SPECS,
    PLOT_DIR,
    action_label,
    discover_runs,
    load_model,
    q_value_for_action,
    rollout_return_forced_action,
    set_state,
)


def make_spatial_plots(algo, env_id, seed, run):
    model = load_model(algo, run["path"])
    env = gym.make(env_id)

    spec = GRID_SPECS[env_id]
    actions = ACTION_SETS[env_id]
    gamma = model.gamma

    sweeps = [
        ("dim1", spec["x"], 0.0, spec["x_label"]),  # vary x, freeze y=0
        ("dim2", spec["y"], 0.0, spec["y_label"]),  # vary y, freeze x=0
    ]

    for axis_name, sweep_vals, fixed_val, axis_label in sweeps:
        q_curves = np.zeros((len(actions), len(sweep_vals)))
        err_curves = np.zeros((len(actions), len(sweep_vals)))

        for vi, v in enumerate(sweep_vals):
            x, y = (v, fixed_val) if axis_name == "dim1" else (fixed_val, v)

            for ai, a in enumerate(actions):
                obs = set_state(env_id, env, x, y)
                q = q_value_for_action(model, obs, a)
                mc = rollout_return_forced_action(env_id, env, model, gamma, x, y, a)

                q_curves[ai, vi] = q
                err_curves[ai, vi] = q - mc

        fig, (ax_q, ax_err) = plt.subplots(1, 2, figsize=(12, 5))

        for ai, a in enumerate(actions):
            label = action_label(a)
            ax_q.plot(sweep_vals, q_curves[ai], marker="o", markersize=3, label=label)
            ax_err.plot(sweep_vals, err_curves[ai], marker="o", markersize=3, label=label)

        ax_q.set_xlabel(axis_label)
        ax_q.set_ylabel("Q-value")
        ax_q.set_title("Q-value")
        ax_q.legend(fontsize=7)

        ax_err.axhline(0, linestyle="--", linewidth=1, color="grey")
        ax_err.set_xlabel(axis_label)
        ax_err.set_ylabel("Q - MC return")
        ax_err.set_title("Bias")
        ax_err.legend(fontsize=7)

        fig.suptitle(f"{algo.upper()} ({env_id}, seed={seed}) -- {axis_name} sweep")
        plt.tight_layout()

        filename = f"spatial_{algo}_{env_id}_seed{seed}_{axis_name}.png"
        plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
        plt.close()
        print(f"  saved {filename}")

    env.close()


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    runs = discover_runs()

    # Most-trained run per (algo, env, seed).
    best = {}
    for r in runs:
        key = (r["algo"], r["env_id"], r["seed"])
        if key not in best or r["steps"] > best[key]["steps"]:
            best[key] = r

    print(f"Found {len(best)} (algo, env, seed) group(s).")

    for (algo, env_id, seed), run in best.items():
        print(f"Processing {algo}_{env_id}_seed{seed} (steps={run['steps']})...")
        make_spatial_plots(algo, env_id, seed, run)

    print(f"Plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()