"""
visualization/plot_training_curves.py

Visualisiert die Reward-Kurven aller trainierten Modelle: 3 Algorithmen (DQN, SAC, TD3)
x 3 Trainings-Budgets (10k, 100k, 1M Steps) -> 9 Läufe insgesamt.

Erwartet die von training/train.py erzeugte Ordnerstruktur:
    models/<algo>_<env_id>_steps<N>_seed<seed>/monitor/monitor.csv

Erzeugt zwei Plots:
    1. reward_grid.png          -> 3x3 Grid, ein Panel pro (Algo, Budget)
    2. reward_grid_overlay.png  -> 1x3 Grid, pro Algo alle 3 Budgets überlagert (log-x)

Nutzung
-------
    python visualization/plot_training_curves.py
    python visualization/plot_training_curves.py --models-dir models --out reward_grid.png
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

ALGOS = ["dqn", "sac", "td3"]
TIMESTEPS = [10_000, 100_000, 1_000_000]
ALGO_COLORS = {"dqn": "tab:blue", "sac": "tab:orange", "td3": "tab:green"}
ALGO_LABELS = {"dqn": "DQN", "sac": "SAC", "td3": "TD3"}


def find_run_dir(models_dir: Path, algo: str, timesteps: int) -> Optional[Path]:
    """Findet den passenden Run-Ordner für (algo, timesteps), unabhängig von Env-Name/Seed
    im Ordnernamen -- z.B. models/dqn_CartPole-v1_steps100000_seed0.
    """
    candidates = sorted(models_dir.glob(f"{algo}_*_steps{timesteps}_seed*"))
    return candidates[0] if candidates else None


def load_reward_curve(run_dir: Path) -> Tuple[Optional[list], Optional[list]]:
    """Lädt die Trainings-Reward-Kurve aus monitor/monitor.csv (via SB3's Monitor-Format)."""
    monitor_dir = run_dir / "monitor"
    if not monitor_dir.exists() or not any(monitor_dir.glob("*.csv")):
        return None, None
    x, y = ts2xy(load_results(str(monitor_dir)), "timesteps")
    return x, y


def load_env_id(run_dir: Path) -> str:
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f).get("env_id", "?")
    return "?"


def plot_grid(models_dir: Path, out_path: Path) -> None:
    """3x3-Grid: eine Zeile pro Algo, eine Spalte pro Trainings-Budget."""
    fig, axes = plt.subplots(len(ALGOS), len(TIMESTEPS), figsize=(15, 10), squeeze=False)

    for row, algo in enumerate(ALGOS):
        for col, timesteps in enumerate(TIMESTEPS):
            ax = axes[row][col]
            run_dir = find_run_dir(models_dir, algo, timesteps)

            if run_dir is None:
                ax.set_title(f"{ALGO_LABELS[algo]} - {timesteps:,} Steps\n(Ordner fehlt)")
                ax.axis("off")
                continue

            x, y = load_reward_curve(run_dir)
            env_id = load_env_id(run_dir)

            if x is None or len(x) == 0:
                ax.set_title(f"{ALGO_LABELS[algo]} - {timesteps:,} Steps\n(keine Monitor-Daten)")
                ax.axis("off")
                continue

            ax.plot(x, y, color=ALGO_COLORS[algo], linewidth=1)
            ax.set_title(f"{ALGO_LABELS[algo]} ({env_id}) - {timesteps:,} Steps")
            ax.set_xlabel("Timesteps")
            if col == 0:
                ax.set_ylabel("Episode Reward")
            ax.grid(alpha=0.3)

    fig.suptitle("Reward-Kurven: Algorithmus × Trainings-Budget", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Gespeichert: {out_path}")
    plt.close(fig)


def plot_overlay_per_algo(models_dir: Path, out_path: Path) -> None:
    """Zusätzlich: pro Algo ein Panel mit allen 3 Budgets überlagert (log-x-Achse),
    zum direkten Vergleich der Lernkurven-Form unabhängig von der absoluten Länge.
    """
    fig, axes = plt.subplots(1, len(ALGOS), figsize=(15, 5), squeeze=False)
    axes = axes[0]

    for i, algo in enumerate(ALGOS):
        ax = axes[i]
        for timesteps in TIMESTEPS:
            run_dir = find_run_dir(models_dir, algo, timesteps)
            if run_dir is None:
                continue
            x, y = load_reward_curve(run_dir)
            if x is None or len(x) == 0:
                continue
            ax.plot(x, y, label=f"{timesteps:,} Steps", alpha=0.85)

        ax.set_title(ALGO_LABELS[algo])
        ax.set_xlabel("Timesteps")
        ax.set_xscale("log")
        if i == 0:
            ax.set_ylabel("Episode Reward")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Reward-Kurven pro Algorithmus, alle Budgets überlagert (log-x)", fontsize=14)
    plt.tight_layout()
    overlay_path = out_path.parent / f"{out_path.stem}_overlay{out_path.suffix}"
    plt.savefig(overlay_path, dpi=150)
    print(f"Gespeichert: {overlay_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualisiert Reward-Kurven aller trainierten Modelle (3 Algos x 3 Budgets)."
    )
    parser.add_argument("--models-dir", type=str, default="models", help="Ordner mit den Run-Ordnern.")
    parser.add_argument("--out", type=str, default="reward_grid.png", help="Dateiname für den Grid-Plot.")
    return parser.parse_args()


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    out_path = Path(args.out)

    plot_grid(models_dir, out_path)
    plot_overlay_per_algo(models_dir, out_path)


if __name__ == "__main__":
    main()