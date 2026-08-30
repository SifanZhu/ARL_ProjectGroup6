"""
"Lassen sich typische Fehler- oder Instabilitätsmuster identifizieren?"
based on the data currently available in models/ (3 algos x 3 training
budgets, seed 0).

Two building blocks per run:

  1. Eval curve instability (from eval/evaluations.npz):
     Is the peak of the eval performance during training higher than the
     performance at the end?

  2. Q-value bias (via Qvalue_bias.estimate_bias_over_random_states):
     Over-/underestimation of the final model per budget. Since 10k/100k/1M
     use the same seed, they are a rough (3-point) approximation of the
     time course of a single training run.

Output: results/model_evaluation.csv + one plot per algo/env in results/.
"""

import csv
import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, SAC, TD3

from Qvalue_bias import estimate_bias_over_random_states

ALGOS = {"dqn": DQN, "sac": SAC, "td3": TD3}

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
BIAS_N_STATES = 50
BIAS_SEED = 12345  # for comparing runs against each other


def eval_curve_stats(eval_npz_path: Path) -> dict:
    data = np.load(eval_npz_path)
    means = data["results"].mean(axis=1)
    timesteps = data["timesteps"]

    peak_idx = int(np.argmax(means))
    peak_timestep, peak_mean = int(timesteps[peak_idx]), float(means[peak_idx])
    final_timestep, final_mean = int(timesteps[-1]), float(means[-1])
    degradation = peak_mean - final_mean

    return {
        "peak_timestep": peak_timestep,
        "peak_mean": peak_mean,
        "final_timestep": final_timestep,
        "final_mean": final_mean,
        "degradation": degradation,
        "degradation_pct": degradation / abs(peak_mean) if peak_mean else float("nan"),
        "curve_std": float(means.std()),
        "eval_timesteps": timesteps,
        "eval_means": means,
    }


def bias_stats(algo: str, env_id: str, run_dir: Path) -> dict:
    model = ALGOS[algo].load(run_dir / "final_model")
    env = gym.make(env_id)
    env.reset(seed=BIAS_SEED)

    result = estimate_bias_over_random_states(model, env, n_states=BIAS_N_STATES)
    env.close()

    bias = result["bias"]
    bias_mean = float(bias.mean())
    mc_return_mean = float(result["mc_returns"].mean())
    return {
        "bias_mean": bias_mean,
        "bias_std": float(bias.std()),
        "q_pred_mean": float(result["q_preds"].mean()),
        "mc_return_mean": mc_return_mean,
        # bias normalized to the actual return magnitude, raw values are
        # not comparable across different budgets/envs, since the
        # achievable return scale itself varies a lot (e.g. untrained
        # 10k model vs. 1M model)
        "relative_bias": bias_mean / abs(mc_return_mean) if mc_return_mean else float("nan"),
    }


def evaluate_run(run_dir: Path) -> dict:
    with open(run_dir / "config.json") as f:
        config = json.load(f)

    row = {
        "run_name": run_dir.name,
        "algo": config["algo"],
        "env_id": config["env_id"],
        "timesteps": config["timesteps"],
        "seed": config["seed"],
    }
    row.update(eval_curve_stats(run_dir / "eval" / "evaluations.npz"))
    row.update(bias_stats(config["algo"], config["env_id"], run_dir))
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "run_name", "algo", "env_id", "timesteps", "seed",
        "peak_timestep", "peak_mean", "final_timestep", "final_mean",
        "degradation", "degradation_pct", "curve_std",
        "bias_mean", "bias_std", "relative_bias", "q_pred_mean", "mc_return_mean",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_algo(algo: str, env_id: str, rows: list[dict], out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: r["timesteps"])
    fig, (ax_curve, ax_bias) = plt.subplots(1, 2, figsize=(11, 4))

    for row in rows:
        ax_curve.plot(
            row["eval_timesteps"], row["eval_means"],
            label=f"budget={row['timesteps']}",
        )
        ax_curve.scatter([row["peak_timestep"]], [row["peak_mean"]], marker="^", color="green")
        ax_curve.scatter([row["final_timestep"]], [row["final_mean"]], marker="x", color="red")
    ax_curve.set_title(f"{algo.upper()} / {env_id}: Eval-Reward-Kurven")
    ax_curve.set_xlabel("Timesteps")
    ax_curve.set_ylabel("Mean eval reward")
    ax_curve.legend(fontsize=8)

    budgets = [r["timesteps"] for r in rows]
    bias_means = [r["bias_mean"] for r in rows]
    bias_stds = [r["bias_std"] for r in rows]
    relative_bias = [r["relative_bias"] for r in rows]

    ax_bias.errorbar(budgets, bias_means, yerr=bias_stds, marker="o", capsize=4, color="C0")
    ax_bias.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_bias.set_xscale("log")
    ax_bias.set_title(f"{algo.upper()} / {env_id}: Q-Wert-Bias über Trainings-Budgets")
    ax_bias.set_xlabel("Timesteps (log)")
    ax_bias.set_ylabel("Bias (Q_pred - MC_return)", color="C0")
    ax_bias.tick_params(axis="y", labelcolor="C0")

    # raw bias is not comparable across budgets, since the achievable
    # return scale itself changes a lot (untrained vs. maxed-out model),
    # therefore also show relative_bias (bias / mc_return_mean) as a second axis
    ax_bias_rel = ax_bias.twinx()
    ax_bias_rel.plot(budgets, relative_bias, marker="s", linestyle="--", color="C1")
    ax_bias_rel.set_ylabel("Relative bias (bias / mc_return_mean)", color="C1")
    ax_bias_rel.tick_params(axis="y", labelcolor="C1")
    ax_bias_rel.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_summary(rows: list[dict]) -> None:
    by_algo: dict[str, list[dict]] = {}
    for row in rows:
        by_algo.setdefault(row["algo"], []).append(row)

    for algo, algo_rows in by_algo.items():
        algo_rows = sorted(algo_rows, key=lambda r: r["timesteps"])
        print(f"\n=== {algo.upper()} ({algo_rows[0]['env_id']}) ===")
        for row in algo_rows:
            print(
                f"  steps={row['timesteps']:>8} | "
                f"peak={row['peak_mean']:8.1f}@{row['peak_timestep']:<7} "
                f"final={row['final_mean']:8.1f} "
                f"degradation={row['degradation']:+7.1f} ({row['degradation_pct']:+.1%}) | "
                f"bias_mean={row['bias_mean']:+8.2f} bias_std={row['bias_std']:7.2f} "
                f"relative_bias={row['relative_bias']:+7.1%}"
            )


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    run_dirs = sorted(p.parent for p in MODELS_DIR.glob("*/config.json"))
    rows = [evaluate_run(run_dir) for run_dir in run_dirs]

    write_csv(rows, RESULTS_DIR / "model_evaluation.csv")

    by_algo: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_algo.setdefault((row["algo"], row["env_id"]), []).append(row)
    for (algo, env_id), algo_rows in by_algo.items():
        plot_algo(algo, env_id, algo_rows, RESULTS_DIR / f"{algo}_{env_id}.png")

    print_summary(rows)
    print(f"\nCSV: {RESULTS_DIR / 'model_evaluation.csv'}")
    print(f"Plots: {RESULTS_DIR}/*.png")


if __name__ == "__main__":
    main()
