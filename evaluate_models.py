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

import argparse
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


def plot_cartpole_old_vs_new(rows: list[dict], out_path: Path) -> None:
    """Finding: two DQN/CartPole runs with byte-identical seed/config diverge
    sharply at 1M steps.
    """
    cp_rows = [r for r in rows if r["algo"] == "dqn" and r["env_id"] == "CartPole-v1"
               and r["variant"] in ("old", "new")]
    budgets = sorted({r["timesteps"] for r in cp_rows if r["timesteps"] <= 1_000_000})
    if not budgets:
        return

    fig, axes = plt.subplots(1, len(budgets), figsize=(4.2 * len(budgets), 4), sharey=True)
    if len(budgets) == 1:
        axes = [axes]

    for ax, budget in zip(axes, budgets):
        for variant, color in [("new", "C0"), ("old", "C1")]:
            row = next((r for r in cp_rows if r["timesteps"] == budget and r["variant"] == variant), None)
            if row is None:
                continue
            ax.plot(row["eval_timesteps"], row["eval_means"], label=variant, color=color)
        ax.set_title(f"{budget:,} steps")
        ax.set_xlabel("Timesteps")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean eval reward (n=5, training-time)")
    fig.suptitle("DQN/CartPole: old vs. retrained run -- same seed, same config.json")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bias_direction(rows: list[dict], out_path: Path) -> None:
    """Finding: DQN's Q-value bias direction is opposite on CartPole
    (underestimates for most of training) vs. Pendulum (overestimates for
    most of training).
    """
    groups = [("dqn", "CartPole-v1", "DQN / CartPole"), ("dqn", "Pendulum-v1", "DQN / Pendulum")]
    fig, ax = plt.subplots(figsize=(7, 5))

    for algo, env_id, label in groups:
        group_rows = sorted(
            (r for r in rows if r["algo"] == algo and r["env_id"] == env_id and r["variant"] == "new"),
            key=lambda r: r["timesteps"],
        )
        if not group_rows:
            continue
        budgets = [r["timesteps"] for r in group_rows]
        rel_bias = [r["relative_bias"] for r in group_rows]
        ax.plot(budgets, rel_bias, marker="o", label=label)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Training budget (timesteps, log scale)")
    ax.set_ylabel("Relative Q-value bias (bias / |mc_return_mean|)")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.set_title("DQN bias direction: overestimation (Pendulum) vs.\nunderestimation (CartPole) are the same algorithm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_collapse_signal_vs_noise(rows: list[dict], out_path: Path) -> None:
    """Finding: most recorded peak-to-final collapses (SAC/TD3, and DQN's
    own training curve in several cases) turn out to be eval-sampling noise
    once independently re-evaluated with n=50 instead of the training-time
    n=5. Only one run (DQN/CartPole 1M, new) survives as a real gap.
    """
    candidates = [r for r in rows if not np.isnan(r["reeval_degradation_sem_ratio"])]
    if not candidates:
        return
    candidates = sorted(candidates, key=lambda r: abs(r["reeval_degradation_sem_ratio"]))

    labels = [
        f"{r['algo']}/{r['env_id']} {r['timesteps']:,}" + ("" if r["variant"] == "new" else f" [{r['variant']}]")
        for r in candidates
    ]
    values = [r["reeval_degradation_sem_ratio"] for r in candidates]
    colors = ["crimson" if abs(v) >= 2 else "gray" for v in values]

    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(candidates))))
    ax.barh(labels, values, color=colors)
    ax.axvline(2, color="crimson", linestyle="--", linewidth=0.8)
    ax.axvline(-2, color="crimson", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("best_model vs. final_model re-eval gap, in SEMs\n(beyond ±2 = likely real, not noise)")
    ax.set_title("Which recorded peak-to-final collapses are real?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bias_consistency(rows: list[dict], out_path: Path) -> None:
    """Finding: bias_std shows a different kind of instability than bias
    direction/magnitude.
    """
    groups = [
        ("dqn", "CartPole-v1", "DQN / CartPole"),
        ("dqn", "Pendulum-v1", "DQN / Pendulum"),
        ("sac", "Pendulum-v1", "SAC / Pendulum"),
        ("td3", "Pendulum-v1", "TD3 / Pendulum"),
    ]
    fig, ax = plt.subplots(figsize=(7, 5))

    for algo, env_id, label in groups:
        group_rows = sorted(
            (r for r in rows if r["algo"] == algo and r["env_id"] == env_id and r["variant"] == "new"),
            key=lambda r: r["timesteps"],
        )
        if not group_rows:
            continue
        budgets = [r["timesteps"] for r in group_rows]
        bias_std = [r["bias_std"] for r in group_rows]
        ax.plot(budgets, bias_std, marker="o", label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training budget (timesteps, log scale)")
    ax.set_ylabel("bias_std across sampled states (log scale)")
    ax.set_title("Consistency of the Q-value miscalibration:\nsystematic (low) vs. erratic (high) across states")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_cartpole_5M_oscillation(rows: list[dict], out_path: Path) -> None:
    """Finding: the 5M-step DQN/CartPole run never stabilizes, it hits a
    perfect score as late as ~2M steps, then crashes and keeps oscillating
    through the very end of training.
    """
    row = next(
        (r for r in rows if r["algo"] == "dqn" and r["env_id"] == "CartPole-v1"
         and r["timesteps"] == 5_000_000 and r["variant"] == "new"),
        None,
    )
    if row is None:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(row["eval_timesteps"], row["eval_means"], color="C0", linewidth=0.9)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean eval reward (n=5)")
    ax.set_title("DQN/CartPole (5M steps): performance never stabilizes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def load_rows_from_csv(csv_path: Path, models_dir: Path) -> list[dict]:
    """Lets plots be regenerated/added without rerunning the
    expensive bias_stats/collapse_check computation.
    """
    int_fields = {"timesteps", "seed", "peak_timestep", "final_timestep", "best_saved_at_timesteps"}
    str_fields = {"run_name", "variant", "algo", "env_id"}

    rows = []
    with open(csv_path) as f:
        for raw in csv.DictReader(f):
            row = {}
            for key, value in raw.items():
                if key in str_fields:
                    row[key] = value
                elif key in int_fields:
                    row[key] = int(value) if value not in ("", "None") else None
                else:
                    row[key] = float(value) if value not in ("", "None") else float("nan")

            npz_path = models_dir / row["run_name"] / "eval" / "evaluations.npz"
            data = np.load(npz_path)
            row["eval_timesteps"] = data["timesteps"]
            row["eval_means"] = data["results"].mean(axis=1)
            rows.append(row)
    return rows

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


def make_plots(rows: list[dict]) -> None:
    plot_cartpole_old_vs_new(rows, RESULTS_DIR / "finding_cartpole_old_vs_new.png")
    plot_bias_direction(rows, RESULTS_DIR / "finding_bias_direction_cartpole_vs_pendulum.png")
    plot_collapse_signal_vs_noise(rows, RESULTS_DIR / "finding_collapse_signal_vs_noise.png")
    plot_cartpole_5M_oscillation(rows, RESULTS_DIR / "finding_cartpole_5M_oscillation.png")
    plot_bias_consistency(rows, RESULTS_DIR / "finding_bias_consistency.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Skip model loading/MC rollouts, regenerate plots from the existing "
             "results/model_evaluation.csv (+ cheap evaluations.npz reads) instead.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "model_evaluation.csv"

    if args.plots_only:
        rows = load_rows_from_csv(csv_path, MODELS_DIR)
    else:
        run_dirs = sorted(p.parent for p in MODELS_DIR.glob("*/config.json"))
        rows = [evaluate_run(run_dir) for run_dir in run_dirs]
        write_csv(rows, csv_path)

    make_plots(rows)

    print_summary(rows)
    print(f"\nCSV: {csv_path}")
    print(f"Plots: {RESULTS_DIR}/*.png")


if __name__ == "__main__":
    main()
