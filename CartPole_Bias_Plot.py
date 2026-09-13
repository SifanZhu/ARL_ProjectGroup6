"""
analysis/plot_qvalue_bias_multiseed.py

Erweiterung von qvalue_bias.py: statt eines einzelnen Seeds werden fuer ein
festes Timestep-Budget (Default: 1_000_000) alle angegebenen Seeds pro
Algorithmus geladen, die Bias-Punkte (Q-Wert-Vorhersage vs. Monte-Carlo-Return)
ueber alle Seeds gepoolt, und in einem 1xN-Grid dargestellt (ein Panel pro
gewaehltem Algorithmus). Standardmaessig alle drei (DQN/Pendulum, SAC/Pendulum,
TD3/Pendulum) fuer einen fairen Vergleich auf demselben Environment; per
--algos laesst sich das auf einen oder zwei Algorithmen einschraenken, per
--dqn-env-id kann DQN stattdessen auf CartPole-v1 ausgewertet werden (ihr habt
DQN auf beiden Envs trainiert). Punkte werden pro Seed eingefaerbt, damit man
sieht ob einzelne Seeds systematisch abweichen.

Kein Retraining noetig: es werden nur die bereits gespeicherten
final_model.zip geladen und fuer die Bias-Schaetzung ausgewertet
(Inferenz + Rollouts), das Training selbst wird nicht wiederholt.

WICHTIG: Die Imports unten gehen davon aus, dass train.py und qvalue_bias.py
als Module importierbar sind (z.B. im gleichen Ordner liegen, oder
training/train.py bzw. analysis/qvalue_bias.py mit passenden __init__.py).
Falls euer Layout anders ist, passt die beiden Imports entsprechend an.

Nutzung
-------
    python plot_qvalue_bias_multiseed.py
    python plot_qvalue_bias_multiseed.py --algos dqn --dqn-env-id CartPole-v1
    python plot_qvalue_bias_multiseed.py --algos sac td3
    python plot_qvalue_bias_multiseed.py --n-states 100 --out qvalue_bias_1M.png
"""

import argparse
from pathlib import Path
from typing import List, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, SAC, TD3

# --- Anpassen, falls Modulpfade bei euch abweichen ---
from qvalue_bias import estimate_bias_over_random_states
from train import DiscretizeActionWrapper

ALGO_CLASSES = {"dqn": DQN, "sac": SAC, "td3": TD3}
# SAC/TD3 werden in diesem Projekt nur auf Pendulum-v1 trainiert (siehe train.py's
# DEFAULT_ENVS). DQN kann sowohl auf CartPole-v1 als auch auf Pendulum-v1
# trainiert sein -- welches gemeint ist, wird jetzt per --dqn-env-id gesteuert.
FIXED_ALGO_ENVS = {"sac": "Pendulum-v1", "td3": "Pendulum-v1"}
ALGO_TITLE_NAMES = {"dqn": "DQN", "sac": "SAC", "td3": "TD3"}
SEED_COLORS = plt.cm.tab10.colors  # bis zu 10 unterscheidbare Seed-Farben


def find_run_dir(models_dir: Path, algo: str, env_id: str, timesteps: int, seed: int) -> Optional[Path]:
    candidates = sorted(models_dir.glob(f"{algo}_{env_id}_steps{timesteps}_seed{seed}"))
    return candidates[0] if candidates else None


def make_eval_env(algo: str, env_id: str):
    env = gym.make(env_id)
    # DQN braucht bei kontinuierlichen Action-Spaces (Pendulum) den Discretize-Wrapper,
    # analog zu train.py's needs_discretize-Logik.
    if algo == "dqn" and env_id == "Pendulum-v1":
        env = DiscretizeActionWrapper(env)
    return env


def collect_bias_over_seeds(
    models_dir: Path, algo: str, env_id: str, timesteps: int, seeds: List[int], n_states: int
):
    """Laedt final_model pro Seed, sammelt (seed, q_preds, mc_returns, bias) je Seed."""
    algo_cls = ALGO_CLASSES[algo]

    per_seed = []
    for seed in seeds:
        run_dir = find_run_dir(models_dir, algo, env_id, timesteps, seed)
        if run_dir is None:
            print(f"[warn] no run found: {algo} env={env_id} steps={timesteps} seed={seed}")
            continue

        model = algo_cls.load(str(run_dir / "final_model"))
        env = make_eval_env(algo, env_id)

        result = estimate_bias_over_random_states(model, env, n_states=n_states)
        per_seed.append((seed, result["q_preds"], result["mc_returns"], result["bias"]))
        env.close()

    return per_seed


def plot_multiseed_bias(
    models_dir: Path,
    algos: List[str],
    env_overrides: dict,
    timesteps: int,
    seeds: List[int],
    n_states: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(algos), figsize=(6 * len(algos), 5.5), squeeze=False)
    axes = axes[0]

    for ax, algo in zip(axes, algos):
        env_id = env_overrides.get(algo, FIXED_ALGO_ENVS.get(algo))
        title_name = f"{ALGO_TITLE_NAMES[algo]} ({env_id})"

        per_seed = collect_bias_over_seeds(models_dir, algo, env_id, timesteps, seeds, n_states)

        if per_seed:
            for seed, q_preds, mc_returns, _ in per_seed:
                color = SEED_COLORS[seed % len(SEED_COLORS)]
                ax.scatter(mc_returns, q_preds, alpha=0.6, color=color, s=25, label=f"Seed {seed}")

            all_q = np.concatenate([p[1] for p in per_seed])
            all_mc = np.concatenate([p[2] for p in per_seed])
            lo, hi = min(all_q.min(), all_mc.min()), max(all_q.max(), all_mc.max())
            ax.plot([lo, hi], [lo, hi], "k--", label="perfect estimate")

            pooled_bias = np.concatenate([p[3] for p in per_seed])
            per_seed_means = np.array([p[3].mean() for p in per_seed])
            ax.set_title(
                f"{title_name}\n"
                f"pooled bias = {pooled_bias.mean():.2f}  |  "
                f"mean of per-seed means = {per_seed_means.mean():.2f} ± {per_seed_means.std():.2f}"
            )
        else:
            ax.set_title(f"{title_name}\n(no data)")

        ax.set_xlabel("Monte Carlo Return (true)")
        ax.set_ylabel("Estimated Q-Value")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle(f"Q-Value Bias over {len(seeds)} Seeds ({timesteps:,} Steps)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Q-Value bias plot (DQN/SAC/TD3), pooled over multiple seeds."
    )
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument(
        "--algos",
        type=str,
        nargs="+",
        default=["dqn", "sac", "td3"],
        choices=["dqn", "sac", "td3"],
        help="Which algorithm(s) to include, e.g. --algos dqn for DQN only.",
    )
    parser.add_argument(
        "--dqn-env-id",
        type=str,
        default="Pendulum-v1",
        choices=["CartPole-v1", "Pendulum-v1"],
        help="Which env DQN's models were trained on (you have both). "
        "Only relevant if 'dqn' is included in --algos. SAC/TD3 are always Pendulum-v1.",
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-states", type=int, default=50)
    parser.add_argument("--out", type=str, default="qvalue_bias_multiseed_1M.png")
    return parser.parse_args()


def main():
    args = parse_args()
    env_overrides = {"dqn": args.dqn_env_id}
    plot_multiseed_bias(
        Path(args.models_dir),
        args.algos,
        env_overrides,
        args.timesteps,
        args.seeds,
        args.n_states,
        Path(args.out),
    )


if __name__ == "__main__":
    main()