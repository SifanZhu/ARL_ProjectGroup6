"""
"Lassen sich typische Fehler- oder Instabilitätsmuster identifizieren?"

Evaluates every trained model under models/ in two ways:

  - Per-budget: final_model/best_model of every separately-trained run
    (10k/100k/1M/5M timesteps) for each  (algo, env) and written to
    model_evaluation.csv. This is only a coarse (3-4 point) approximation
    of a single run's time course, so it isn't plotted as a budget comparison
    (exception: single 5M-step CartPole run since its training curve is
    interesting by itself).

  - Checkpoint progression: every intermediate checkpoint
    (models/<run>/checkpoints/model_<N>_steps.zip) of the 1M-timestep runs,
    giving a finer-grained (every 10k-50k steps) view of how bias and
    performance evolve over one continuous run. More than one run can share
    an (algo, env_id) -> DQN/CartPole-v1 has two independently trained runs
    with the same config.

The measurement primitives:

  1. Training-time eval curve (eval_curve_stats, reading
     eval/evaluations.npz): SB3's EvalCallback, n_eval_episodes=5, ran
     during training. Per-budget only.

  2. Q-value bias via Monte-Carlo rollout (bias_stats_from_path /
     Qvalue_bias.estimate_bias_over_random_states): for n_states=50 random
     resets, rolls out one episode under the model's own deterministic
     policy and compares its own predicted Q-value against the return that
     rollout actually achieved.

  3. Independent re-evaluation (re_evaluate): evaluate_policy with
     n_eval_episodes=50 on a fresh seed never used during training.
     Checks whether an apparent collapse in the noisy training-time
     evaluation curve (best_model vs. final_model, as picked by SB3's
     EvalCallback) survives being re-checked with more episodes and a
     different seed. And, in checkpoint progression, its reeval_mean
     cross-checks primitive #2's mc_return_mean (since the two use
     different seeds and code paths, their agreement is what tells
     an apparent instability apart from a fluke of one evaluation.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

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
    "run_A": "seed 0",
    "run_B": "run B",
    "old": "old",
}

# True best-possible episode reward per environment (worst-possible only
# for CartPole-v1)
#   CartPole-v1: +1 reward/step, max_episode_steps=500 -> [0, 500]
#   Pendulum-v1: max reward is 0 as there are only negative "rewards"
ENV_REWARD_BOUNDS = {
    "CartPole-v1": (0.0, 500.0),
    "Pendulum-v1": (None, 0.0),
}


# Core measurement primitives

def make_analysis_env(algo: str, env_id: str) -> gym.Env:
    """Builds an env matching how the model was actually trained. DQN on a
    continuous-action env (Pendulum) was trained through DiscretizeActionWrapper,
    so bias/re-eval must use the same wrapper.
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
        # bias normalized to the actual return magnitude
        "relative_bias": bias_mean / abs(mc_return_mean) if mc_return_mean else float("nan"),
    }


def re_evaluate(algo: str, env_id: str, model_path: Path) -> dict:
    """Independently re-evaluates a saved checkpoint: fresh seed, more episodes
    than the n_eval_episodes=5 used during training. Also reports the checkpoint's
    own num_timesteps, a free cross-check against the peak_timestep read from
    evaluations.npz.
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
    # standard error of the mean for each side, combined
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
    an initial 10k point) view of how the run evolved.
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
    runs share it.
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
            m = re.search(r"_seed(\d+)_*(.*)$", run_name)
            seed_num, tag = (m.group(1), m.group(2)) if m else (None, "")
            if tag in CHECKPOINT_TAG_LABELS:
                tag_label = CHECKPOINT_TAG_LABELS[tag]
            elif not tag and seed_num is not None:
                tag_label = f"seed {seed_num}"
            else:
                tag_label = tag or "unlabeled"
            labels[run_name] = f"{base} ({tag_label})"
    return labels


# Plots: per-budget

def plot_cartpole_5M_oscillation(rows: list[dict], out_path: Path) -> None:
    """Finding: the 5M-step DQN/CartPole run never stabilizes, it hits a
    perfect score at around 2M steps, then crashes and keeps oscillating
    through the end of training.
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


# Plots: checkpoint progression

def _group_checkpoints_by_env_algo(
    all_ckpt_rows: dict[str, list[dict]],
) -> dict[str, dict[str, dict[int, list[dict]]]]:
    """(env_id -> algo -> checkpoint_steps -> rows), merging every run_name
    sharing an (algo, env_id) group (e.g. seeds 0-4, or DQN/CartPole-v1's
    run_A/run_B/seed1-4) so callers can average across them.
    """
    by_env: dict[str, dict[str, dict[int, list[dict]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rows in all_ckpt_rows.values():
        for row in rows:
            by_env[row["env_id"]][row["algo"]][row["checkpoint_steps"]].append(row)
    return by_env


def _mean_std_series(by_step: dict[int, list[dict]], steps: list[int], field: str) -> tuple[np.ndarray, np.ndarray]:
    values = [[r[field] for r in by_step[s]] for s in steps]
    return np.array([np.mean(v) for v in values]), np.array([np.std(v) for v in values])


def _env_slug(env_id: str) -> str:
    return re.sub(r"-v\d+$", "", env_id).lower()


def _env_plot_path(out_path: Path, env_id: str, suffix: str = "") -> Path:
    return out_path.with_name(f"{out_path.stem}{suffix}_{_env_slug(env_id)}{out_path.suffix}")


def plot_checkpoint_progression(all_ckpt_rows: dict[str, list[dict]], out_path: Path) -> list[Path]:
    """Tracks relative Q-value bias and re-eval performance across every
    saved checkpoint of the 1M-timestep runs. Algos sharing an environment
    plotted together on the same axes, each line averaged across every run_name
    sharing that (algo, env_id) group, with a shaded +/-1 std band across those
    runs at each checkpoint_steps value.

    The performance panel marks each environment's true best-possible
    episode reward and the worst-possible one for CartPole-v1 only.
    """
    written: list[Path] = []
    for env_id, by_algo in sorted(_group_checkpoints_by_env_algo(all_ckpt_rows).items()):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        for algo, by_step in sorted(by_algo.items()):
            steps = sorted(by_step)
            label = algo.upper()

            rel_mean, rel_std = _mean_std_series(by_step, steps, "relative_bias")
            line1, = ax1.plot(steps, rel_mean, marker="o", label=label)
            ax1.fill_between(steps, rel_mean - rel_std, rel_mean + rel_std, color=line1.get_color(), alpha=0.2)

            reeval_mean, reeval_std = _mean_std_series(by_step, steps, "reeval_mean")
            line2, = ax2.plot(steps, reeval_mean, marker="o", label=label)
            ax2.fill_between(steps, reeval_mean - reeval_std, reeval_mean + reeval_std, color=line2.get_color(), alpha=0.2)

        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax1.set_xlabel("Timesteps")
        _set_timestep_xaxis(ax1)
        ax1.set_ylabel("Relative Q-value bias (bias / |mc_return_mean|)")
        ax1.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
        ax1.set_title("Relative Q-value bias over training\n(mean ±1 std across runs)")
        ax1.legend()

        ax2.set_xlabel("Timesteps")
        _set_timestep_xaxis(ax2)
        ax2.set_ylabel(f"Re-eval mean reward (n={RE_EVAL_N_EPISODES})")
        ax2.set_title(f"{env_id} performance over training\n(mean ±1 std across runs)")
        if env_id in ENV_REWARD_BOUNDS:
            # Capture the data-driven range first, a true bound far outside
            # it would otherwise stretch the axis.
            data_ylim = ax2.get_ylim()
            true_lo, true_hi = ENV_REWARD_BOUNDS[env_id]
            ax2.axhline(true_hi, color="seagreen", linestyle=":", linewidth=1.2, label=f"true max ({true_hi:.0f})")
            if true_lo is not None:
                ax2.axhline(true_lo, color="crimson", linestyle=":", linewidth=1.2, label=f"true min ({true_lo:.0f})")
            ax2.set_ylim(data_ylim)
        ax2.legend()

        fig.tight_layout()
        env_path = _env_plot_path(out_path, env_id)
        fig.savefig(env_path, dpi=150)
        plt.close(fig)
        written.append(env_path)

    return written


def plot_checkpoint_absolute_bias(all_ckpt_rows: dict[str, list[dict]], out_path: Path) -> list[Path]:
    """Absolute (raw, un-normalized) Q-value bias over training, one figure
    per environment.
    """
    written: list[Path] = []
    for env_id, by_algo in sorted(_group_checkpoints_by_env_algo(all_ckpt_rows).items()):
        fig, ax = plt.subplots(figsize=(7, 5))

        for algo, by_step in sorted(by_algo.items()):
            steps = sorted(by_step)
            abs_mean, abs_std = _mean_std_series(by_step, steps, "bias_mean")
            line, = ax.plot(steps, abs_mean, marker="o", label=algo.upper())
            ax.fill_between(steps, abs_mean - abs_std, abs_mean + abs_std, color=line.get_color(), alpha=0.2)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Timesteps")
        _set_timestep_xaxis(ax)
        ax.set_ylabel("Absolute Q-value bias (bias_mean, raw reward units)")
        ax.set_title(f"{env_id}: absolute Q-value bias over training\n(mean ±1 std across runs)")
        ax.legend()

        fig.tight_layout()
        env_path = _env_plot_path(out_path, env_id, suffix="_absolute_bias")
        fig.savefig(env_path, dpi=150)
        plt.close(fig)
        written.append(env_path)

    return written


def plot_cartpole_seed_comparison(ckpt_rows_by_run: dict[str, list[dict]], out_path: Path) -> None:
    """Built from the checkpoint CSV's independent re-evaluation (n=50
    episodes, a fresh seed not used during training).
    """
    labels = checkpoint_series_labels(ckpt_rows_by_run)
    cp_runs = {
        run_name: rows for run_name, rows in ckpt_rows_by_run.items()
        if rows and rows[0]["algo"] == "dqn" and rows[0]["env_id"] == "CartPole-v1"
        and not run_name.endswith("__run_B")
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
    fig.suptitle("DQN/CartPole-v1: performance across seeds (n=50 re-eval)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_td3_seed0_collapse_case_study(ckpt_rows_by_run: dict[str, list[dict]], out_path: Path) -> bool:
    """Case study for the isolated TD3/Pendulum-v1 seed0 collapse (bias_mean,
    q_pred_mean, mc_return_mean and reeval_mean all spike simultaneously at
    one checkpoint, fully recovered by the next one 50k steps later).

    Left panel: reeval_mean +/-1 std across every checkpoint, isolating the
    single anomalous point. Right panel: raw training episode rewards in the
    window between the checkpoints just before/after the anomaly, with the
    worst episode marked.
    """
    run_name = "td3_Pendulum-v1_steps1000000_seed0"
    rows = ckpt_rows_by_run.get(run_name)
    if not rows:
        return False
    rows = sorted(rows, key=lambda r: r["checkpoint_steps"])
    steps = [r["checkpoint_steps"] for r in rows]
    reeval_mean = [r["reeval_mean"] for r in rows]
    reeval_std = [r["reeval_std"] for r in rows]

    med = float(np.median(reeval_std))
    mad = float(np.median([abs(v - med) for v in reeval_std])) or 1e-9
    outlier_idx = max(range(len(rows)), key=lambda i: abs(reeval_std[i] - med))
    if abs(reeval_std[outlier_idx] - med) < 6 * mad:
        return False
    outlier_step = steps[outlier_idx]

    monitor_path = MODELS_DIR / run_name / "monitor" / "monitor.csv"
    if not monitor_path.exists():
        return False

    with open(monitor_path) as f:
        f.readline()  # SB3 writes a json comment as the first line
        episode_rows = list(csv.DictReader(f))
    cum = 0
    episodes = []
    for r in episode_rows:
        cum += int(float(r["l"]))
        episodes.append((cum, float(r["r"])))

    window_lo = steps[outlier_idx - 1] if outlier_idx > 0 else max(0, outlier_step - 50_000)
    window_hi = steps[outlier_idx + 1] if outlier_idx + 1 < len(steps) else outlier_step + 50_000
    window = [(t, r) for t, r in episodes if window_lo <= t <= window_hi]
    if not window:
        return False
    worst_t, worst_r = min(window, key=lambda tr: tr[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.errorbar(steps, reeval_mean, yerr=reeval_std, marker="o", markersize=4, capsize=3)
    ax1.scatter([outlier_step], [reeval_mean[outlier_idx]], color="crimson", zorder=5, label="anomalous checkpoint")
    ax1.set_xlabel("Timesteps")
    _set_timestep_xaxis(ax1)
    ax1.set_ylabel(f"Re-eval mean reward ±1 std (n={RE_EVAL_N_EPISODES})")
    ax1.set_title(f"TD3/Pendulum-v1 seed0: checkpoint re-eval\n(anomaly at step {outlier_step:,})")
    ax1.legend()

    win_t = [t for t, _ in window]
    win_r = [r for _, r in window]
    ax2.plot(win_t, win_r, linewidth=0.8, color="C0")
    ax2.scatter([worst_t], [worst_r], color="crimson", zorder=5, label=f"worst episode ({worst_r:.0f})")
    ax2.axvline(outlier_step, color="gray", linestyle="--", linewidth=0.8, label=f"checkpoint saved ({outlier_step:,})")
    ax2.set_xlabel("Timesteps (training)")
    _set_timestep_xaxis(ax2)
    ax2.set_ylabel("Per-episode training reward")
    ax2.set_title("Raw training log around the anomaly\n(independent of any evaluation code)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


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


def run(plots_only: bool) -> None:
    """Evaluates every trained model under models/ (or, with plots_only,
    just reloads the two CSVs below) and regenerates every plot.

    Writes/reads results/model_evaluation.csv (final_model/best_model per
    budget) and results/checkpoint_evaluation.csv (every checkpoint of the
    1M-timestep runs).
    """
    model_csv_path = RESULTS_DIR / "model_evaluation.csv"
    checkpoint_csv_path = RESULTS_DIR / "checkpoint_evaluation.csv"

    if plots_only:
        rows = load_rows_from_csv(model_csv_path, MODELS_DIR)
        ckpt_rows_by_run = load_checkpoint_rows_from_csv(checkpoint_csv_path)
    else:
        run_dirs = sorted(p.parent for p in MODELS_DIR.glob("*/config.json"))
        rows = [evaluate_run(run_dir) for run_dir in run_dirs]
        write_csv(rows, model_csv_path)

        ckpt_rows_by_run: dict[str, list[dict]] = {}
        all_ckpt_rows: list[dict] = []
        for row in rows:
            if row["timesteps"] != 1_000_000:
                continue
            run_dir = MODELS_DIR / row["run_name"]
            rows_for_run = evaluate_checkpoints(run_dir)
            if not rows_for_run:
                print(f"Warnung: keine Checkpoints gefunden in {run_dir / 'checkpoints'}")
                continue
            ckpt_rows_by_run[run_dir.name] = rows_for_run
            all_ckpt_rows.extend(rows_for_run)

        write_checkpoint_csv(all_ckpt_rows, checkpoint_csv_path)

    plot_cartpole_5M_oscillation(rows, RESULTS_DIR / "finding_cartpole_5M_oscillation.png")
    checkpoint_plot_path = RESULTS_DIR / "finding_checkpoint_progression.png"
    written_plots = plot_checkpoint_progression(ckpt_rows_by_run, checkpoint_plot_path)
    written_plots += plot_checkpoint_absolute_bias(ckpt_rows_by_run, checkpoint_plot_path)

    cartpole_plot_path = RESULTS_DIR / "finding_cartpole_seeds.png"
    plot_cartpole_seed_comparison(ckpt_rows_by_run, cartpole_plot_path)
    written_plots.append(cartpole_plot_path)

    td3_case_study_path = RESULTS_DIR / "finding_td3_seed0_collapse.png"
    if plot_td3_seed0_collapse_case_study(ckpt_rows_by_run, td3_case_study_path):
        written_plots.append(td3_case_study_path)

    print_summary(rows)
    print_checkpoint_summary(ckpt_rows_by_run)
    print(f"\nCSVs: {model_csv_path}, {checkpoint_csv_path}")
    print(f"Plot: {RESULTS_DIR / 'finding_cartpole_5M_oscillation.png'}")
    for path in written_plots:
        print(f"Plot: {path}")


# CLI

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Skip model loading/MC rollouts/re-evaluation, regenerate plots from "
             "the existing results/model_evaluation.csv and "
             "results/checkpoint_evaluation.csv instead.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    run(plots_only=args.plots_only)


if __name__ == "__main__":
    main()
