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

  3. Independent re-evaluation of best_model vs. final_model:
     The peak-vs-final degradation in evaluations.npz is based on only
     n_eval_episodes=5 per checkpoint, and since best_model/final_model
     share the same seed + eval schedule (SAC and TD3 in particular both
     peak/collapse at the exact same timestep), that gap could just be
     correlated eval-sampling noise rather than a real difference. This
     re-evaluates both checkpoints independently with a much larger
     episode count and a seed that is not the training-time eval seed, to
     check whether the gap survives under a cleaner test.

Output: results/model_evaluation.csv + one plot per algo/env in results/.
"""

import csv
import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy

from Qvalue_bias import estimate_bias_over_random_states
from train import DiscretizeActionWrapper

ALGOS = {"dqn": DQN, "sac": SAC, "td3": TD3}

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
BIAS_N_STATES = 50
BIAS_SEED = 12345  # for comparing runs against each other

RE_EVAL_N_EPISODES = 50
# deliberately different from the training-time eval seed (cfg.seed + 1000),
# so this isn't correlated with the original eval schedule
RE_EVAL_SEED = 777


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


def make_analysis_env(algo: str, env_id: str) -> gym.Env:
    """Builds an env matching how the model was actually trained. DQN on a
    continuous-action env (Pendulum) was trained through DiscretizeActionWrapper
    (see train.py), so bias/re-eval must use the same wrapper or actions
    won't match what the network expects.
    """
    env = gym.make(env_id)
    if algo == "dqn" and isinstance(env.action_space, spaces.Box):
        env = DiscretizeActionWrapper(env)
    return env


def bias_stats(algo: str, env_id: str, run_dir: Path) -> dict:
    model = ALGOS[algo].load(run_dir / "final_model")
    env = make_analysis_env(algo, env_id)
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


def re_evaluate(algo: str, env_id: str, model_path: Path) -> dict:
    """Independently re-evaluates a saved checkpoint: fresh seed (not the
    training-time eval seed), many more episodes than the n_eval_episodes=5
    used during training. Also reports the checkpoint's own num_timesteps,
    a free cross-check against the peak_timestep read from evaluations.npz.
    """
    model = ALGOS[algo].load(model_path)
    env = make_analysis_env(algo, env_id)
    env.reset(seed=RE_EVAL_SEED)

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=RE_EVAL_N_EPISODES, deterministic=True,
    )
    env.close()

    return {
        "reeval_mean": float(mean_reward),
        "reeval_std": float(std_reward),
        "saved_at_timesteps": int(model.num_timesteps),
    }


def collapse_check(algo: str, env_id: str, run_dir: Path) -> dict:
    best_path = run_dir / "best_model"
    if not (run_dir / "best_model.zip").exists():
        return {
            "best_saved_at_timesteps": None,
            "best_reeval_mean": float("nan"),
            "best_reeval_std": float("nan"),
            "final_reeval_mean": float("nan"),
            "final_reeval_std": float("nan"),
            "reeval_degradation": float("nan"),
            "reeval_degradation_sem_ratio": float("nan"),
        }

    best = re_evaluate(algo, env_id, best_path)
    final = re_evaluate(algo, env_id, run_dir / "final_model")

    degradation = best["reeval_mean"] - final["reeval_mean"]
    # standard error of the mean for each side, combined -- a rough signal-
    # to-noise indicator (how many "SEMs" apart the two means are), not a
    # formal significance test.
    sem_best = best["reeval_std"] / (RE_EVAL_N_EPISODES ** 0.5)
    sem_final = final["reeval_std"] / (RE_EVAL_N_EPISODES ** 0.5)
    combined_sem = (sem_best ** 2 + sem_final ** 2) ** 0.5

    return {
        "best_saved_at_timesteps": best["saved_at_timesteps"],
        "best_reeval_mean": best["reeval_mean"],
        "best_reeval_std": best["reeval_std"],
        "final_reeval_mean": final["reeval_mean"],
        "final_reeval_std": final["reeval_std"],
        "reeval_degradation": degradation,
        "reeval_degradation_sem_ratio": degradation / combined_sem if combined_sem else float("nan"),
    }


def evaluate_run(run_dir: Path) -> dict:
    with open(run_dir / "config.json") as f:
        config = json.load(f)

    row = {
        "run_name": run_dir.name,
        # "_old" runs share the exact same (algo, env_id, timesteps) as the
        # current ones (retrained on main with identical config)
        "variant": "old" if run_dir.name.endswith("_old") else "new",
        "algo": config["algo"],
        "env_id": config["env_id"],
        "timesteps": config["timesteps"],
        "seed": config["seed"],
    }
    row.update(eval_curve_stats(run_dir / "eval" / "evaluations.npz"))
    row.update(bias_stats(config["algo"], config["env_id"], run_dir))
    row.update(collapse_check(config["algo"], config["env_id"], run_dir))
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "run_name", "variant", "algo", "env_id", "timesteps", "seed",
        "peak_timestep", "peak_mean", "final_timestep", "final_mean",
        "degradation", "degradation_pct", "curve_std",
        "bias_mean", "bias_std", "relative_bias", "q_pred_mean", "mc_return_mean",
        "best_saved_at_timesteps", "best_reeval_mean", "best_reeval_std",
        "final_reeval_mean", "final_reeval_std",
        "reeval_degradation", "reeval_degradation_sem_ratio",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_algo(algo: str, env_id: str, rows: list[dict], out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: (r["variant"], r["timesteps"]))
    fig, (ax_curve, ax_bias) = plt.subplots(1, 2, figsize=(11, 4))

    for row in rows:
        variant_suffix = f" [{row['variant']}]" if row["variant"] != "new" else ""
        curve_linestyle = "--" if row["variant"] == "old" else "-"
        ax_curve.plot(
            row["eval_timesteps"], row["eval_means"],
            label=f"budget={row['timesteps']}{variant_suffix}",
            linestyle=curve_linestyle,
        )
        ax_curve.scatter([row["peak_timestep"]], [row["peak_mean"]], marker="^", color="green")
        ax_curve.scatter([row["final_timestep"]], [row["final_mean"]], marker="x", color="red")
        # independent re-evaluation (n=RE_EVAL_N_EPISODES, fresh seed) at the same
        # x-position, if these disagree a lot with the training-time markers
        # above, the recorded peak/final gap is likely eval-sampling noise.
        if not np.isnan(row["best_reeval_mean"]):
            ax_curve.scatter(
                [row["peak_timestep"]], [row["best_reeval_mean"]],
                marker="D", color="darkgreen", zorder=5,
            )
            ax_curve.scatter(
                [row["final_timestep"]], [row["final_reeval_mean"]],
                marker="P", color="darkred", zorder=5,
            )
    ax_curve.set_title(f"{algo.upper()} / {env_id}: Eval-Reward-Kurven")
    ax_curve.set_xlabel("Timesteps")
    ax_curve.set_ylabel("Mean eval reward")
    handles, labels = ax_curve.get_legend_handles_labels()
    handles += [
        plt.Line2D([], [], marker="^", color="green", linestyle="", label="peak (n=5, training eval)"),
        plt.Line2D([], [], marker="x", color="red", linestyle="", label="final (n=5, training eval)"),
        plt.Line2D([], [], marker="D", color="darkgreen", linestyle="", label=f"best_model re-eval (n={RE_EVAL_N_EPISODES})"),
        plt.Line2D([], [], marker="P", color="darkred", linestyle="", label=f"final_model re-eval (n={RE_EVAL_N_EPISODES})"),
    ]
    ax_curve.legend(handles=handles, fontsize=7)

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
    ax_bias_rel.set_ylabel("Relative bias (bias / mc_return_mean)", color="C1")
    ax_bias_rel.tick_params(axis="y", labelcolor="C1")
    ax_bias_rel.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")

    variants_present = sorted({r["variant"] for r in rows}, key=lambda v: v != "new")
    for variant, marker, bias_linestyle in zip(variants_present, ["o", "v"], ["-", "--"]):
        variant_rows = [r for r in rows if r["variant"] == variant]
        budgets = [r["timesteps"] for r in variant_rows]
        bias_means = [r["bias_mean"] for r in variant_rows]
        bias_stds = [r["bias_std"] for r in variant_rows]
        relative_bias = [r["relative_bias"] for r in variant_rows]
        suffix = "" if variant == "new" else f" [{variant}]"

        ax_bias.errorbar(
            budgets, bias_means, yerr=bias_stds, marker=marker, linestyle=bias_linestyle,
            capsize=4, color="C0", label=f"bias{suffix}",
        )
        ax_bias_rel.plot(
            budgets, relative_bias, marker=marker, linestyle=bias_linestyle,
            color="C1", label=f"relative_bias{suffix}",
        )

    if len(variants_present) > 1:
        lines1, labels1 = ax_bias.get_legend_handles_labels()
        lines2, labels2 = ax_bias_rel.get_legend_handles_labels()
        ax_bias.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_summary(rows: list[dict]) -> None:
    by_group: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_group.setdefault((row["algo"], row["env_id"]), []).append(row)

    for (algo, env_id), group_rows in by_group.items():
        group_rows = sorted(group_rows, key=lambda r: (r["variant"], r["timesteps"]))
        print(f"\n=== {algo.upper()} ({env_id}) ===")
        for row in group_rows:
            variant_suffix = f" [{row['variant']}]" if row["variant"] != "new" else ""
            print(
                f"  steps={row['timesteps']:>8}{variant_suffix} | "
                f"peak={row['peak_mean']:8.1f}@{row['peak_timestep']:<7} "
                f"final={row['final_mean']:8.1f} "
                f"degradation={row['degradation']:+7.1f} ({row['degradation_pct']:+.1%}) | "
                f"bias_mean={row['bias_mean']:+8.2f} bias_std={row['bias_std']:7.2f} "
                f"relative_bias={row['relative_bias']:+7.1%}"
            )
            if not np.isnan(row["best_reeval_mean"]):
                print(
                    f"             independent re-eval (n={RE_EVAL_N_EPISODES}): "
                    f"best={row['best_reeval_mean']:8.1f}±{row['best_reeval_std']:.1f} "
                    f"(saved@{row['best_saved_at_timesteps']}) "
                    f"final={row['final_reeval_mean']:8.1f}±{row['final_reeval_std']:.1f} "
                    f"| gap={row['reeval_degradation']:+7.1f} "
                    f"({row['reeval_degradation_sem_ratio']:+.1f} SEMs apart)"
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
