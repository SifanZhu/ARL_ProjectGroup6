"""
"Lassen sich typische Fehler- oder Instabilitätsmuster identifizieren?"

Two modes, sharing the same three measurement primitives:

  Default mode: compares final_model/best_model across separately-trained
  runs at different timestep budgets (10k/100k/1M/5M) for each (algo, env).
  Since each budget is a completely separate training run, this is only a
  coarse (3-4 point) approximation of a single run's time course.

  --checkpoints mode: walks every intermediate checkpoint
  (models/<run>/checkpoints/model_<N>_steps.zip) of every 1M-timestep run
  found under models/, giving a much finer-grained (every 10k-50k steps)
  view of how bias and performance evolve over one continuous run. More
  than one run can share an (algo, env_id); e.g. two independently
  trained runs of DQN/CartPole-v1 with the same config.json (see
  CHECKPOINT_TAG_LABELS).

The three measurement primitives, used by both modes:

  1. Training-time eval curve (eval_curve_stats, reading
     eval/evaluations.npz): SB3's EvalCallback, n_eval_episodes=5, run
     during training.

  2. Q-value bias via Monte-Carlo rollout (bias_stats_from_path /
     Qvalue_bias.estimate_bias_over_random_states): for n_states=50 random
     resets, rolls out one episode under the model's own deterministic
     policy and compares its own predicted Q-value against the return that
     rollout actually achieved.

  3. Independent re-evaluation (re_evaluate): evaluate_policy with
     n_eval_episodes=50 on a fresh seed never used during training,
     specifically to check whether an apparent training-time collapse
     (see collapse_check / plot_collapse_signal_vs_noise) survives
     independent scrutiny.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.ticker import FuncFormatter
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

# checkpoint filenames look like "model_10000_steps.zip", "model_50000_steps.zip", ...
CHECKPOINT_STEPS_RE = re.compile(r"(\d+)_steps")

CHECKPOINT_TAG_LABELS = {
    "run_A": "run A",
    "run_B": "run B",
    "old": "old",
}


# Core measurement primitives

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


def bias_stats_from_path(algo: str, env_id: str, model_path: Path) -> dict:
    """Q-value bias for an arbitrary saved checkpoint (final_model, best_model,
    or one of the intermediate checkpoints/ files)."""
    model = ALGOS[algo].load(model_path)
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
    """Default-mode measurement for one independently-trained run: training
    curve stats, Q-value bias of its final_model, and best-vs-final collapse
    check.
    """
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
    row.update(bias_stats_from_path(config["algo"], config["env_id"], run_dir / "final_model"))
    row.update(collapse_check(config["algo"], config["env_id"], run_dir))
    return row


def find_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Finds all checkpoint files in run_dir/checkpoints, sorted by training
    step extracted from the filename (model_10000_steps.zip, model_50000_steps.zip, ...).
    """
    ckpt_dir = run_dir / "checkpoints"
    checkpoints = []
    for path in ckpt_dir.glob("*.zip"):
        m = CHECKPOINT_STEPS_RE.search(path.stem)
        if m:
            checkpoints.append((int(m.group(1)), path))
    return sorted(checkpoints, key=lambda x: x[0])


def evaluate_checkpoints(run_dir: Path) -> list[dict]:
    """Walks every saved checkpoint of a single run and computes bias +
    re-eval performance at each one, giving a fine-grained (every 50k, plus
    an initial 10k point) view of how the run evolved -- rather than just
    the 3-point (10k/100k/1M) approximation across separate runs.
    """
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    algo, env_id = config["algo"], config["env_id"]

    rows = []
    for steps, ckpt_path in find_checkpoints(run_dir):
        row = {
            "run_name": run_dir.name,
            "algo": algo,
            "env_id": env_id,
            "checkpoint_steps": steps,
        }
        row.update(bias_stats_from_path(algo, env_id, ckpt_path))
        row.update(re_evaluate(algo, env_id, ckpt_path))
        rows.append(row)
    return rows


# CSV I/O

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


def write_checkpoint_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "run_name", "algo", "env_id", "checkpoint_steps",
        "bias_mean", "bias_std", "relative_bias", "q_pred_mean", "mc_return_mean",
        "reeval_mean", "reeval_std", "saved_at_timesteps",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _coerce_csv_row(raw: dict, str_fields: set, int_fields: set, nullable: bool = False) -> dict:
    """Shared type-coercion for a csv.DictReader row: fields in str_fields
    stay strings, fields in int_fields become int, everything else becomes
    float. With nullable=True, "" / "None" become None (int fields) or NaN
    (float fields) instead of raising.
    """
    row = {}
    for key, value in raw.items():
        if key in str_fields:
            row[key] = value
        elif key in int_fields:
            row[key] = int(value) if not (nullable and value in ("", "None")) else None
        else:
            row[key] = float(value) if not (nullable and value in ("", "None")) else float("nan")
    return row


def load_rows_from_csv(csv_path: Path, models_dir: Path) -> list[dict]:
    """Lets default-mode plots be regenerated/added without rerunning the
    expensive bias_stats/collapse_check computation.
    """
    int_fields = {"timesteps", "seed", "peak_timestep", "final_timestep", "best_saved_at_timesteps"}
    str_fields = {"run_name", "variant", "algo", "env_id"}

    rows = []
    with open(csv_path) as f:
        for raw in csv.DictReader(f):
            row = _coerce_csv_row(raw, str_fields, int_fields, nullable=True)
            npz_path = models_dir / row["run_name"] / "eval" / "evaluations.npz"
            data = np.load(npz_path)
            row["eval_timesteps"] = data["timesteps"]
            row["eval_means"] = data["results"].mean(axis=1)
            rows.append(row)
    return rows


def load_checkpoint_rows_from_csv(csv_path: Path) -> dict[str, list[dict]]:
    """Regenerates the checkpoint-progression plots from a previously written
    checkpoint_evaluation.csv, without reloading models or rerunning the
    MC rollouts / re-evaluation that produced it. Grouped by run_name, since
    more than one run_name can share an (algo, env_id).
    """
    int_fields = {"checkpoint_steps", "saved_at_timesteps"}
    str_fields = {"run_name", "algo", "env_id"}

    ckpt_rows_by_run: dict[str, list[dict]] = {}
    with open(csv_path) as f:
        for raw in csv.DictReader(f):
            row = _coerce_csv_row(raw, str_fields, int_fields)
            ckpt_rows_by_run.setdefault(row["run_name"], []).append(row)
    return ckpt_rows_by_run


# Plotting helpers

def _format_timestep_ticks(x: float, _pos=None) -> str:
    """Tick formatter for a linear timesteps axis: whole millions as '1M',
    whole thousands as '200K', otherwise the raw number.
    """
    x = int(x)
    if x == 0:
        return "0"
    if x % 1_000_000 == 0:
        return f"{x // 1_000_000}M"
    if x % 1_000 == 0:
        return f"{x // 1_000}K"
    return str(x)


def _set_timestep_xaxis(ax) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(_format_timestep_ticks))


def checkpoint_series_labels(ckpt_rows_by_run: dict[str, list[dict]]) -> dict[str, str]:
    """Maps each run_name to a plot label: plain "ALGO/ENV" when it's the
    only run for that (algo, env_id), or "ALGO/ENV (tag)" when multiple
    runs share it (see CHECKPOINT_TAG_LABELS).
    """
    run_names_by_group: dict[tuple[str, str], list[str]] = {}
    for run_name, rows in ckpt_rows_by_run.items():
        algo, env_id = rows[0]["algo"], rows[0]["env_id"]
        run_names_by_group.setdefault((algo, env_id), []).append(run_name)

    labels: dict[str, str] = {}
    for (algo, env_id), run_names in run_names_by_group.items():
        base = f"{algo.upper()}/{env_id}"
        if len(run_names) == 1:
            labels[run_names[0]] = base
            continue
        for run_name in run_names:
            m = re.search(r"_seed\d+_*(.*)$", run_name)
            tag = m.group(1) if m else ""
            tag_label = CHECKPOINT_TAG_LABELS.get(tag, tag or "unlabeled")
            labels[run_name] = f"{base} ({tag_label})"
    return labels


# Plots: default (budget-comparison) mode

def _aggregate_new_by_budget(
    rows: list[dict], algo: str, env_id: str, value_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Groups 'new'-variant rows for (algo, env_id) by training budget and
    aggregates value_key across seeds sharing that budget.

    Returns (budgets, means, lo, hi), all sorted by budget.
    """
    by_budget = defaultdict(list)
    for r in rows:
        if r["algo"] == algo and r["env_id"] == env_id and r["variant"] == "new":
            by_budget[r["timesteps"]].append(r[value_key])

    budgets = np.array(sorted(by_budget))
    means = np.array([np.mean(by_budget[b]) for b in budgets])
    lo = np.array([np.min(by_budget[b]) for b in budgets])
    hi = np.array([np.max(by_budget[b]) for b in budgets])
    return budgets, means, lo, hi


def plot_bias_direction(rows: list[dict], out_path: Path) -> None:
    """Finding: DQN's Q-value bias direction is opposite on CartPole
    (underestimates for most of training) vs. Pendulum (overestimates for
    most of training).

    Budgets with more than one seed are aggregated:
    mean line, shaded band spans the observed min-max across seeds.
    """
    groups = [("dqn", "CartPole-v1", "DQN / CartPole"), ("dqn", "Pendulum-v1", "DQN / Pendulum")]
    fig, ax = plt.subplots(figsize=(7, 5))

    for algo, env_id, label in groups:
        budgets, means, lo, hi = _aggregate_new_by_budget(rows, algo, env_id, "relative_bias")
        if len(budgets) == 0:
            continue
        line, = ax.plot(budgets, means, marker="o", label=label)
        ax.fill_between(budgets, lo, hi, color=line.get_color(), alpha=0.2, linewidth=0)

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


def plot_bias_consistency(rows: list[dict], out_path: Path) -> None:
    """Finding: bias_std shows a different kind of instability than bias
    direction/magnitude.

    Budgets with more than one seed are aggregated:
    mean line, shaded band spans the observed min-max across seeds.
    """
    groups = [
        ("dqn", "CartPole-v1", "DQN / CartPole"),
        ("dqn", "Pendulum-v1", "DQN / Pendulum"),
        ("sac", "Pendulum-v1", "SAC / Pendulum"),
        ("td3", "Pendulum-v1", "TD3 / Pendulum"),
    ]
    fig, ax = plt.subplots(figsize=(7, 5))

    for algo, env_id, label in groups:
        budgets, means, lo, hi = _aggregate_new_by_budget(rows, algo, env_id, "bias_std")
        if len(budgets) == 0:
            continue
        line, = ax.plot(budgets, means, marker="o", label=label)
        ax.fill_between(budgets, lo, hi, color=line.get_color(), alpha=0.2, linewidth=0)

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
    _set_timestep_xaxis(ax)
    ax.set_ylabel("Mean eval reward (n=5)")
    ax.set_title("DQN/CartPole (5M steps): performance never stabilizes")
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

    def base_label(r: dict) -> str:
        return f"{r['algo']}/{r['env_id']} {r['timesteps']:,}" + ("" if r["variant"] == "new" else f" [{r['variant']}]")

    # Budgets with more than one seed need the
    # seed appended so each one gets its own row.
    base_counts: dict[str, int] = defaultdict(int)
    for r in candidates:
        base_counts[base_label(r)] += 1
    labels = [
        base_label(r) + (f" (seed {r['seed']})" if base_counts[base_label(r)] > 1 else "")
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


def make_plots(rows: list[dict]) -> None:
    plot_bias_direction(rows, RESULTS_DIR / "finding_bias_direction_cartpole_vs_pendulum.png")
    plot_collapse_signal_vs_noise(rows, RESULTS_DIR / "finding_collapse_signal_vs_noise.png")
    plot_cartpole_5M_oscillation(rows, RESULTS_DIR / "finding_cartpole_5M_oscillation.png")
    plot_bias_consistency(rows, RESULTS_DIR / "finding_bias_consistency.png")


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


def run_default_mode(plots_only: bool) -> None:
    csv_path = RESULTS_DIR / "model_evaluation.csv"

    if plots_only:
        rows = load_rows_from_csv(csv_path, MODELS_DIR)
    else:
        run_dirs = sorted(p.parent for p in MODELS_DIR.glob("*/config.json"))
        rows = [evaluate_run(run_dir) for run_dir in run_dirs]
        write_csv(rows, csv_path)

    make_plots(rows)

    print_summary(rows)
    print(f"\nCSV: {csv_path}")
    print(f"Plots: {RESULTS_DIR}/*.png")


# Plots: --checkpoints mode

def plot_checkpoint_progression(all_ckpt_rows: dict[str, list[dict]], out_path: Path) -> None:
    """Tracks relative Q-value bias and re-eval performance across every
    saved checkpoint of the 1M-timestep runs, one line per (algo, env_id),
    averaged across every run_name sharing that group (e.g. seeds 0-4, or
    DQN/CartPole-v1's run_A/run_B/seed1-4), with a shaded +/-1 std band
    across those runs at each checkpoint_steps value. A checkpoint_steps
    value not shared by every run (e.g. run_A's coarser ~50k-step grid vs.
    others' 10k grid) is still averaged, just over however many runs
    actually report it. See plot_cartpole_old_vs_new_checkpoints for the
    individual per-run comparison this collapses.
    """
    groups: dict[tuple[str, str], dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for rows in all_ckpt_rows.values():
        for row in rows:
            groups[(row["algo"], row["env_id"])][row["checkpoint_steps"]].append(row)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for (algo, env_id), by_step in sorted(groups.items()):
        steps = sorted(by_step)
        label = f"{algo.upper()}/{env_id}"

        def mean_std(field: str) -> tuple[np.ndarray, np.ndarray]:
            values = [[r[field] for r in by_step[s]] for s in steps]
            return np.array([np.mean(v) for v in values]), np.array([np.std(v) for v in values])

        bias_mean, bias_std = mean_std("relative_bias")
        line1, = ax1.plot(steps, bias_mean, marker="o", label=label)
        ax1.fill_between(steps, bias_mean - bias_std, bias_mean + bias_std, color=line1.get_color(), alpha=0.2)

        reeval_mean, reeval_std = mean_std("reeval_mean")
        line2, = ax2.plot(steps, reeval_mean, marker="o", label=label)
        ax2.fill_between(steps, reeval_mean - reeval_std, reeval_mean + reeval_std, color=line2.get_color(), alpha=0.2)

    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Timesteps")
    _set_timestep_xaxis(ax1)
    ax1.set_ylabel("Relative Q-value bias (bias / |mc_return_mean|)")
    ax1.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax1.set_title("Relative Q-value bias over training\n(1M runs, mean ±1 std across runs)")
    ax1.legend()

    ax2.set_xlabel("Timesteps")
    _set_timestep_xaxis(ax2)
    ax2.set_ylabel(f"Re-eval mean reward (n={RE_EVAL_N_EPISODES})")
    ax2.set_title("Performance over training\n(1M runs, mean ±1 std across runs)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cartpole_old_vs_new_checkpoints(ckpt_rows_by_run: dict[str, list[dict]], out_path: Path) -> None:
    """Replaces the old plot_cartpole_old_vs_new (which compared runs using
    the training-time eval curve, n_eval_episodes=5) with one built from the
    checkpoint CSV's independent re-evaluation (n=50 episodes, a fresh seed
    not used anywhere during training) -- a far less noisy comparison of
    every DQN/CartPole-v1 run_name found in the checkpoint data, whatever
    they happen to be (an "_old" run, separately-trained runs sharing the
    same config, etc; see CHECKPOINT_TAG_LABELS).

    Finding: run A and run B are two separately-trained runs with identical
    config.json (same algo/env/timesteps/seed), confirmed not to be the same
    weights re-evaluated twice -- they diverge substantially at several
    checkpoints (e.g. step 1M: 457 vs 273) despite the identical config, the
    same training-instability signature as the original DQN/CartPole
    "_old"-vs-new comparison this plot replaces.

    Only covers the 1M-step budget, since that's what --checkpoints
    evaluates -- unlike the plot it replaces, it has no 10k/100k comparison.
    """
    labels = checkpoint_series_labels(ckpt_rows_by_run)
    cp_runs = {
        run_name: rows for run_name, rows in ckpt_rows_by_run.items()
        if rows and rows[0]["algo"] == "dqn" and rows[0]["env_id"] == "CartPole-v1"
    }
    if not cp_runs:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for run_name, rows in cp_runs.items():
        rows = sorted(rows, key=lambda r: r["checkpoint_steps"])
        steps = [r["checkpoint_steps"] for r in rows]
        ax.plot(steps, [r["reeval_mean"] for r in rows], marker="o", markersize=3, label=labels[run_name])

    ax.set_xlabel("Timesteps")
    _set_timestep_xaxis(ax)
    ax.set_ylabel(f"Re-eval mean reward (n={RE_EVAL_N_EPISODES})")
    ax.legend(fontsize=8)
    fig.suptitle("DQN/CartPole-v1: two separately-trained runs, same config (n=50 re-eval)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_checkpoint_summary(all_ckpt_rows: dict[str, list[dict]]) -> None:
    labels = checkpoint_series_labels(all_ckpt_rows)
    for run_name, rows in all_ckpt_rows.items():
        rows = sorted(rows, key=lambda r: r["checkpoint_steps"])
        print(f"\n=== {labels[run_name]} checkpoint progression ({run_name}) ===")
        for row in rows:
            print(
                f"  steps={row['checkpoint_steps']:>8} | "
                f"reeval={row['reeval_mean']:8.1f}±{row['reeval_std']:.1f} | "
                f"bias_mean={row['bias_mean']:+8.2f} bias_std={row['bias_std']:7.2f} "
                f"relative_bias={row['relative_bias']:+7.1%}"
            )


def run_checkpoints_mode(plots_only: bool = False, csv_path: Optional[Path] = None) -> None:
    """Evaluates every saved checkpoint of the 1M-timestep runs (grouped by
    run_name, since more than one run_name can share an (algo, env_id) --
    e.g. DQN/CartPole and DQN/Pendulum, or two independent re-evaluations of
    the same nominal run) and plots how bias and performance evolve over the
    course of training.

    plots_only: regenerate the plot from the existing checkpoint_evaluation.csv
    instead of reloading every checkpoint .zip and rerunning MC rollouts +
    re-evaluation (which requires the models/*/checkpoints/ directories to
    still exist on disk).

    csv_path: write/read the checkpoint CSV here instead of the default
    results/checkpoint_evaluation.csv -- e.g. so a teammate evaluating their
    own local checkpoints doesn't overwrite the shared CSV, and can hand back
    a separate file to be merged in by hand.
    """
    if csv_path is None:
        csv_path = RESULTS_DIR / "checkpoint_evaluation.csv"
        plot_path = RESULTS_DIR / "finding_checkpoint_progression.png"
    else:
        plot_path = csv_path.with_name(f"finding_{csv_path.stem}.png")

    if plots_only:
        ckpt_rows_by_run = load_checkpoint_rows_from_csv(csv_path)
    else:
        run_dirs_1m = []
        for cfg_path in MODELS_DIR.glob("*/config.json"):
            with open(cfg_path) as f:
                config = json.load(f)
            if config["timesteps"] == 1_000_000:
                run_dirs_1m.append(cfg_path.parent)

        ckpt_rows_by_run: dict[str, list[dict]] = {}
        all_ckpt_rows: list[dict] = []
        for run_dir in sorted(run_dirs_1m):
            rows_for_run = evaluate_checkpoints(run_dir)
            if not rows_for_run:
                print(f"Warnung: keine Checkpoints gefunden in {run_dir / 'checkpoints'}")
                continue
            ckpt_rows_by_run[run_dir.name] = rows_for_run
            all_ckpt_rows.extend(rows_for_run)

        write_checkpoint_csv(all_ckpt_rows, csv_path)

    plot_checkpoint_progression(ckpt_rows_by_run, plot_path)

    # Only meaningful against the full merged dataset (a teammate's partial
    # --output run won't have every DQN/CartPole-v1 variant to compare).
    if csv_path.name == "checkpoint_evaluation.csv":
        cartpole_plot_path = RESULTS_DIR / "finding_cartpole_old_vs_new.png"
        plot_cartpole_old_vs_new_checkpoints(ckpt_rows_by_run, cartpole_plot_path)
        print(f"Plot: {cartpole_plot_path}")

    print_checkpoint_summary(ckpt_rows_by_run)
    print(f"\nCSV: {csv_path}")
    print(f"Plot: {plot_path}")


# CLI

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Skip model loading/MC rollouts, regenerate plots from the existing "
             "results/model_evaluation.csv (+ cheap evaluations.npz reads) instead. "
             "Combined with --checkpoints, regenerates from "
             "results/checkpoint_evaluation.csv instead, with no models/ access at all.",
    )
    parser.add_argument(
        "--checkpoints", action="store_true",
        help="Evaluate every saved checkpoint (models/*/checkpoints/model_*_steps.zip) "
             "of the 1M-timestep runs found under models/ instead of the final/best "
             "models, and plot bias/performance over the course of training.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Only with --checkpoints: write the CSV (and its plot) to this path "
             "instead of results/checkpoint_evaluation.csv. Use this when evaluating "
             "your own local checkpoints for a run someone else already has results "
             "for, so you don't overwrite their CSV -- hand back the separate file "
             "to be merged in by hand instead.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.checkpoints:
        run_checkpoints_mode(
            plots_only=args.plots_only,
            csv_path=Path(args.output) if args.output else None,
        )
    else:
        run_default_mode(plots_only=args.plots_only)


if __name__ == "__main__":
    main()
