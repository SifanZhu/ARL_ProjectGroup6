"""
analysis/plot_bias_performance_over_training.py

Zeitverlaufs-Version von plot_qvalue_bias_multiseed.py: statt nur das
final_model auszuwerten, werden ALLE gespeicherten Checkpoints
(models/<run>/checkpoints/model_<N>_steps.zip, siehe train.py's
CheckpointCallback) pro Seed geladen und ausgewertet. Für jeden Checkpoint
wird sowohl der relative Q-Wert-Bias als auch die Policy-Performance
(mittlerer, undiskontierter Episoden-Reward über N Eval-Episoden) berechnet,
jeweils gemittelt (+/- Std) über alle Seeds -- pro Algorithmus ein 2-Panel-
Plot (Bias links, Performance rechts), analog zum Referenz-Plot.

Achtung, Laufzeit: pro Algorithmus x Seed x Checkpoint wird ein Modell neu
geladen und n_episodes Rollouts gefahren (Default 50). Bei z.B. 20
Checkpoints x 5 Seeds x 50 Episoden sind das schnell mehrere tausend
Rollouts pro Algorithmus. Falls das zu lange dauert: --n-episodes oder
--max-checkpoints reduzieren.

Nutzung
-------
    python plot_bias_performance_over_training.py
    python plot_bias_performance_over_training.py --algos dqn
    python plot_bias_performance_over_training.py --algos dqn --dqn-env-id CartPole-v1
    python plot_bias_performance_over_training.py --n-episodes 20 --max-checkpoints 15
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter
from stable_baselines3 import DQN, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

from analysis.qvalue_bias import get_q_value
from train import DiscretizeActionWrapper

ALGO_CLASSES = {"dqn": DQN, "sac": SAC, "td3": TD3}
FIXED_ALGO_ENVS = {"sac": "Pendulum-v1", "td3": "Pendulum-v1"}
ALGO_TITLE_NAMES = {"dqn": "DQN", "sac": "SAC", "td3": "TD3"}

ENV_REWARD_BOUNDS = {
    "Pendulum-v1": (-200 * (np.pi**2 + 0.1 * 8**2 + 0.001 * 2**2), 0.0),
    "CartPole-v1": (0.0, 500.0),
}

CHECKPOINT_RE = re.compile(r"model_(\d+)_steps\.zip$")


def find_run_dir(models_dir: Path, algo: str, env_id: str, timesteps: int, seed: int) -> Optional[Path]:
    candidates = sorted(models_dir.glob(f"{algo}_{env_id}_steps{timesteps}_seed{seed}"))
    return candidates[0] if candidates else None


def list_checkpoints(run_dir: Path) -> Dict[int, Path]:
    """Gibt {step: path} fuer alle Checkpoints in run_dir/checkpoints zurueck."""
    ckpt_dir = run_dir / "checkpoints"
    result = {}
    if not ckpt_dir.exists():
        return result
    for path in ckpt_dir.glob("model_*_steps.zip"):
        match = CHECKPOINT_RE.search(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def make_eval_env(algo: str, env_id: str):
    env = gym.make(env_id)
    if algo == "dqn" and env_id == "Pendulum-v1":
        env = DiscretizeActionWrapper(env)
    return env


def rollout_episode(
    model: BaseAlgorithm, env, gamma: float, deterministic: bool = True, max_steps: int = 1000
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    obs, _ = env.reset()
    start_obs = obs.copy()
    start_action, _ = model.predict(start_obs, deterministic=deterministic)

    G, R, discount, steps, done = 0.0, 0.0, 1.0, 0, False
    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        G += discount * reward
        R += reward
        discount *= gamma
        done = terminated or truncated
        steps += 1

    return start_obs, start_action, G, R


def evaluate_checkpoint(
    model: BaseAlgorithm, env, algo: str, n_episodes: int, max_steps: int = 1000
) -> Tuple[float, float]:
    """Fuehrt n_episodes Rollouts, gibt (relative_bias, mean_raw_reward) zurueck."""
    gamma = model.gamma
    q_preds, discounted_returns, raw_returns = [], [], []

    for _ in range(n_episodes):
        start_obs, start_action, G, R = rollout_episode(model, env, gamma, max_steps=max_steps)
        q = get_q_value(model, start_obs, start_action)
        q_pred = q if algo == "dqn" else q["min"]

        q_preds.append(q_pred)
        discounted_returns.append(G)
        raw_returns.append(R)

    q_preds = np.array(q_preds)
    discounted_returns = np.array(discounted_returns)
    raw_returns = np.array(raw_returns)

    bias = q_preds - discounted_returns
    mean_return = discounted_returns.mean()
    denom = abs(mean_return) if abs(mean_return) > 1e-6 else np.nan
    relative_bias = bias.mean() / denom

    return relative_bias, raw_returns.mean()


def collect_over_training(
    models_dir: Path,
    algo: str,
    env_id: str,
    timesteps: int,
    seeds: List[int],
    n_episodes: int,
    max_checkpoints: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    algo_cls = ALGO_CLASSES[algo]

    # Checkpoints pro Seed sammeln, Vereinigung der Steps ueber alle Seeds bilden
    per_seed_ckpts = {}
    for seed in seeds:
        run_dir = find_run_dir(models_dir, algo, env_id, timesteps, seed)
        if run_dir is None:
            print(f"[warn] no run found: {algo} env={env_id} steps={timesteps} seed={seed}")
            continue
        ckpts = list_checkpoints(run_dir)
        if not ckpts:
            print(f"[warn] no checkpoints found in {run_dir}")
            continue
        per_seed_ckpts[seed] = ckpts

    if not per_seed_ckpts:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    all_steps = sorted(set().union(*[set(c.keys()) for c in per_seed_ckpts.values()]))
    if max_checkpoints is not None and len(all_steps) > max_checkpoints:
        idx = np.linspace(0, len(all_steps) - 1, max_checkpoints).round().astype(int)
        all_steps = [all_steps[i] for i in sorted(set(idx))]

    steps_out, bias_mean, bias_std, reward_mean, reward_std = [], [], [], [], []

    for step in all_steps:
        biases_this_step, rewards_this_step = [], []
        for seed, ckpts in per_seed_ckpts.items():
            if step not in ckpts:
                continue
            model = algo_cls.load(str(ckpts[step]))
            env = make_eval_env(algo, env_id)
            rel_bias, mean_reward = evaluate_checkpoint(model, env, algo, n_episodes)
            env.close()
            if not np.isnan(rel_bias):
                biases_this_step.append(rel_bias)
            rewards_this_step.append(mean_reward)

        if not rewards_this_step:
            continue

        steps_out.append(step)
        bias_mean.append(np.mean(biases_this_step) if biases_this_step else np.nan)
        bias_std.append(np.std(biases_this_step) if biases_this_step else np.nan)
        reward_mean.append(np.mean(rewards_this_step))
        reward_std.append(np.std(rewards_this_step))
        print(f"[{algo}] step {step:>9,}: rel. bias = {bias_mean[-1]:.2%}, "
              f"reward = {reward_mean[-1]:.1f}")

    return (
        np.array(steps_out), np.array(bias_mean), np.array(bias_std),
        np.array(reward_mean), np.array(reward_std),
    )


def format_steps(x, _pos=None) -> str:
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M".replace(".0M", "M")
    if x >= 1_000:
        return f"{x / 1_000:.0f}K"
    return f"{x:.0f}"


def plot_algo(
    models_dir: Path,
    algo: str,
    env_id: str,
    timesteps: int,
    seeds: List[int],
    n_episodes: int,
    max_checkpoints: Optional[int],
    out_path: Path,
) -> None:
    steps, bias_mean, bias_std, reward_mean, reward_std = collect_over_training(
        models_dir, algo, env_id, timesteps, seeds, n_episodes, max_checkpoints
    )

    if len(steps) == 0:
        print(f"[warn] nothing to plot for {algo} ({env_id})")
        return

    fig, (ax_bias, ax_perf) = plt.subplots(1, 2, figsize=(13, 5))

    ax_bias.plot(steps, bias_mean, marker="o", markersize=3, color="tab:blue", label=ALGO_TITLE_NAMES[algo])
    ax_bias.fill_between(steps, bias_mean - bias_std, bias_mean + bias_std, color="tab:blue", alpha=0.2)
    ax_bias.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    ax_bias.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_bias.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax_bias.set_xlabel("Timesteps")
    ax_bias.set_ylabel("Relative Q-value bias (bias / |mc_return_mean|)")
    ax_bias.set_title(f"Relative Q-value bias over training\n(mean ±1 std across {len(seeds)} seeds)")
    ax_bias.legend()
    ax_bias.grid(alpha=0.3)

    ax_perf.plot(steps, reward_mean, marker="o", markersize=3, color="tab:blue", label=ALGO_TITLE_NAMES[algo])
    ax_perf.fill_between(steps, reward_mean - reward_std, reward_mean + reward_std, color="tab:blue", alpha=0.2)
    if env_id in ENV_REWARD_BOUNDS:
        r_min, r_max = ENV_REWARD_BOUNDS[env_id]
        ax_perf.axhline(r_max, color="gray", linestyle=":", linewidth=1, label=f"true max ({r_max:.0f})")
        ax_perf.axhline(r_min, color="gray", linestyle=":", linewidth=1, label=f"true min ({r_min:.0f})")
    ax_perf.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax_perf.set_xlabel("Timesteps")
    ax_perf.set_ylabel(f"Re-eval mean reward (n={n_episodes})")
    ax_perf.set_title(f"{env_id} performance over training\n(mean ±1 std across {len(seeds)} seeds)")
    ax_perf.legend(fontsize=8)
    ax_perf.grid(alpha=0.3)

    fig.suptitle(f"{ALGO_TITLE_NAMES[algo]} ({env_id})", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Q-value bias & performance over training (checkpoints), averaged over seeds."
    )
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument(
        "--algos", type=str, nargs="+", default=["dqn", "sac", "td3"], choices=["dqn", "sac", "td3"],
        help="Which algorithm(s) to plot. One figure is created per algorithm.",
    )
    parser.add_argument(
        "--dqn-env-id", type=str, default="Pendulum-v1", choices=["CartPole-v1", "Pendulum-v1"],
        help="Which env DQN's checkpoints were trained on. SAC/TD3 are always Pendulum-v1.",
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-episodes", type=int, default=50, help="Rollouts per checkpoint per seed.")
    parser.add_argument(
        "--max-checkpoints", type=int, default=None,
        help="Subsample to at most this many checkpoints (evenly spaced) to limit runtime.",
    )
    parser.add_argument("--out-prefix", type=str, default="bias_performance")
    return parser.parse_args()


def main():
    args = parse_args()
    env_overrides = {"dqn": args.dqn_env_id}

    for algo in args.algos:
        env_id = env_overrides.get(algo, FIXED_ALGO_ENVS.get(algo))
        out_path = Path(f"{args.out_prefix}_{algo}.png")
        plot_algo(
            Path(args.models_dir), algo, env_id, args.timesteps, args.seeds,
            args.n_episodes, args.max_checkpoints, out_path,
        )


if __name__ == "__main__":
    main()