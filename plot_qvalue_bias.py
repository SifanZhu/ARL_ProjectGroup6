"""
plot_qvalue_bias.py

Discovers all trained models in models/, computes mean predicted Q-value and
mean MC return at each training budget, and produces two plots per algo/env:
  - left:  Q-value vs MC return across training steps
  - right: bias (Q − MC) across training steps

Usage:
    python plot_qvalue_bias.py
    python plot_qvalue_bias.py --models-dir models --n-states 100 --seed 0
    python plot_qvalue_bias.py --out my_plot.png
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, SAC, TD3

from Qvalue_bias import estimate_bias_over_random_states
from train import DiscretizeActionWrapper

ALGO_CLS = {"dqn": DQN, "sac": SAC, "td3": TD3}
# matches folder names like dqn_CartPole-v1_steps10000_seed0
_RUN_RE = re.compile(
    r"^(?P<algo>dqn|sac|td3)_(?P<env>.+)_steps(?P<steps>\d+)_seed(?P<seed>\d+)$"
)


def discover_runs(models_dir: Path, seed: int) -> dict:
    """Return {(algo, env_id): [(steps, model_dir), ...]} sorted by steps."""
    runs = defaultdict(list)
    for d in sorted(models_dir.iterdir()):
        if not d.is_dir():
            continue
        m = _RUN_RE.match(d.name)
        if m and int(m.group("seed")) == seed and (d / "final_model.zip").exists():
            runs[(m.group("algo"), m.group("env"))].append(
                (int(m.group("steps")), d)
            )
    for key in runs:
        runs[key].sort(key=lambda x: x[0])
    return dict(runs)


def make_eval_env(algo: str, env_id: str) -> gym.Env:
    env = gym.make(env_id)
    # DQN on continuous envs was trained with DiscretizeActionWrapper
    if algo == "dqn" and isinstance(env.action_space, spaces.Box):
        env = DiscretizeActionWrapper(env)
    return env


def collect_stats(algo: str, env_id: str, step_model_pairs: list, n_states: int):
    steps_list, q_means, mc_means, bias_means = [], [], [], []
    for steps, model_dir in step_model_pairs:
        model = ALGO_CLS[algo].load(str(model_dir / "final_model"))
        env   = make_eval_env(algo, env_id)
        result = estimate_bias_over_random_states(model, env, n_states=n_states)
        env.close()

        steps_list.append(steps)
        q_means.append(result["q_preds"].mean())
        mc_means.append(result["mc_returns"].mean())
        bias_means.append(result["bias"].mean())
        print(
            f"  {algo.upper():3s} {env_id:15s} {steps:>10,} steps | "
            f"Q = {q_means[-1]:9.2f}   MC = {mc_means[-1]:9.2f}   "
            f"bias = {bias_means[-1]:+9.2f}"
        )
    return (
        np.array(steps_list),
        np.array(q_means),
        np.array(mc_means),
        np.array(bias_means),
    )


def plot_all(runs: dict, n_states: int, out_path: str) -> None:
    n_rows = len(runs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 3 * n_rows), squeeze=False)

    for row, ((algo, env_id), step_model_pairs) in enumerate(sorted(runs.items())):
        print(f"\n[{algo.upper()} / {env_id}]")
        steps, q_means, mc_means, bias_means = collect_stats(
            algo, env_id, step_model_pairs, n_states
        )

        # --- left: Q vs MC ---
        ax = axes[row, 0]
        ax.plot(steps, q_means,  marker="o", label="mean predicted Q")
        ax.plot(steps, mc_means, marker="s", linestyle="--", label="mean MC return")
        ax.set_xscale("log")
        ax.set_title(f"{algo.upper()} — {env_id}")
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(alpha=0.3)

        # --- right: bias ---
        ax = axes[row, 1]
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.plot(steps, bias_means, marker="o", color="darkorange", label="bias  (Q − MC)")
        ax.set_xscale("log")
        ax.set_title(f"{algo.upper()} — {env_id}  ·  bias")
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Q − MC return")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(
        "Q-value prediction vs. actual MC return across training budgets",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot Q-value bias for all trained models across timestep budgets."
    )
    p.add_argument("--models-dir", default="models", help="Root folder of trained models.")
    p.add_argument("--n-states", type=int, default=50, help="MC rollouts per budget.")
    p.add_argument("--seed", type=int, default=0, help="Which seed's models to load.")
    p.add_argument("--out", default="qvalue_bias_plot.png", help="Output image path.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    models_dir = Path(args.models_dir)
    runs = discover_runs(models_dir, seed=args.seed)
    if not runs:
        print(f"No trained models found in '{models_dir}'. Run train.py first.")
    else:
        plot_all(runs, n_states=args.n_states, out_path=args.out)
