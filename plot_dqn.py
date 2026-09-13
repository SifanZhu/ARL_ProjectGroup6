"""
visualization/plot_training_curves_avg.py

Wie plot_training_curves.py, aber gemittelt ueber alle Seeds statt nur den
ersten Treffer zu nehmen (die urspruengliche find_run_dir() liefert per
sorted()-Glob immer nur EINEN Ordner, unabhaengig davon wie viele Seeds
existieren).

Fuer jedes (Algo, Timesteps)-Setting werden alle vorhandenen Seed-Ordner
geladen, die individuellen Reward-Kurven auf ein gemeinsames Timestep-Gitter
interpoliert (Episodenlaengen/-zeitpunkte unterscheiden sich zwischen Seeds),
und Mittelwert +/- Standardabweichung als Band geplottet.

Das ist rueckwirkend moeglich, ohne neu zu trainieren: SB3s Monitor-Wrapper
hat die komplette Trainings-Reward-Historie bereits in monitor/monitor.csv
persistiert, als train.py gelaufen ist.

Nutzung
-------
    python plot_training_curves_avg.py
    python plot_training_curves_avg.py --models-dir models --timesteps 1000000
    python plot_training_curves_avg.py --seeds 0 1 2 3 4   (Default ohnehin)
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy

ALGOS = ["dqn", "sac", "td3"]
ALGO_COLORS = {"dqn": "tab:blue", "sac": "tab:orange", "td3": "tab:green"}
ALGO_LABELS = {"dqn": "DQN", "sac": "SAC", "td3": "TD3"}

# Theoretical min/max achievable episode reward per environment, used to draw
# reference lines. Pendulum-v1: reward = -(theta^2 + 0.1*theta_dot^2 + 0.001*u^2)
# per step, theta in [-pi, pi], theta_dot clipped to [-8, 8], u clipped to
# [-2, 2], default episode length 200 steps -> max cost per step =
# pi^2 + 0.1*8^2 + 0.001*2^2 ~= 16.2736, so min reward = -200 * 16.2736.
# Max reward per step = 0 (theta=0, theta_dot=0, u=0) -> max episode reward = 0.
# CartPole-v1: +1 reward per surviving step, episode ends at 500 steps max
# (TimeLimit) or on failure -> min = 0 (fails immediately), max = 500.
ENV_REWARD_BOUNDS = {
    "Pendulum-v1": (-200 * (np.pi**2 + 0.1 * 8**2 + 0.001 * 2**2), 0.0),
    "CartPole-v1": (0.0, 500.0),
}

# SAC/TD3 are only ever trained on Pendulum-v1 in this project (see train.py's
# DEFAULT_ENVS), so we can assume that env for bounds/reference-line purposes
# even when no --env override is given. DQN needs an explicit override since
# it can be either CartPole-v1 or Pendulum-v1.
ASSUMED_ENV = {"sac": "Pendulum-v1", "td3": "Pendulum-v1"}


def find_seed_dirs(
    models_dir: Path, algo: str, timesteps: int, seeds: List[int], env_id: Optional[str] = None
) -> List[Path]:
    """Alle Seed-Ordner fuer (algo, timesteps).

    env_id=None: Env-Name im Ordnernamen wird per Wildcard offengelassen (Vorsicht bei
    DQN, das auf CartPole UND Pendulum trainiert sein kann -- sorted() waehlt dann
    implizit "CartPole-v1" vor "Pendulum-v1", rein alphabetisch).
    env_id="CartPole-v1" / "Pendulum-v1": explizit nur diesen Env-Ordner suchen.
    """
    dirs = []
    for seed in seeds:
        env_part = env_id if env_id else "*"
        candidates = sorted(models_dir.glob(f"{algo}_{env_part}_steps{timesteps}_seed{seed}"))
        if candidates:
            dirs.append(candidates[0])
        else:
            print(f"[warn] no folder found: {algo} env={env_part} steps={timesteps} seed={seed}")
    return dirs


def load_reward_curve(run_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    monitor_dir = run_dir / "monitor"
    if not monitor_dir.exists() or not any(monitor_dir.glob("*.csv")):
        return None, None
    x, y = ts2xy(load_results(str(monitor_dir)), "timesteps")
    return np.asarray(x), np.asarray(y)


def average_curves(
    run_dirs: List[Path], n_points: int = 200
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """Interpoliert jede Seed-Kurve auf ein gemeinsames Timestep-Gitter
    [0, min(max_x aller Seeds)] und gibt (x_grid, mean, std, n_seeds) zurueck.

    Wir schneiden bei min(max_x) ab, damit der Mittelwert an jeder Stelle des
    Gitters wirklich auf ALLEN Seeds basiert (kein Seed faellt vorzeitig aus
    der Mittelung, nur weil er (z.B. bei CartPole durch fruehe Terminierung)
    frueher endet als die anderen).
    """
    curves = [load_reward_curve(d) for d in run_dirs]
    curves = [(x, y) for x, y in curves if x is not None and len(x) >= 2]

    if not curves:
        return None

    max_x_common = min(x[-1] for x, _ in curves)
    if max_x_common <= 0:
        return None

    x_grid = np.linspace(0, max_x_common, n_points)
    interpolated = np.stack([np.interp(x_grid, x, y) for x, y in curves], axis=0)

    return x_grid, interpolated.mean(axis=0), interpolated.std(axis=0), len(curves)


def plot_averaged(
    models_dir: Path,
    algos: List[str],
    timesteps: int,
    seeds: List[int],
    out_path: Path,
    env_overrides: Optional[dict] = None,
) -> None:
    env_overrides = env_overrides or {}
    fig, ax = plt.subplots(figsize=(8, 5))

    envs_plotted = set()

    for algo in algos:
        env_id = env_overrides.get(algo)
        run_dirs = find_seed_dirs(models_dir, algo, timesteps, seeds, env_id=env_id)
        result = average_curves(run_dirs) if run_dirs else None

        if result is None:
            print(f"[warn] no usable monitor data for {algo} @ {timesteps} steps")
            continue

        x_grid, mean, std, n_seeds = result
        color = ALGO_COLORS[algo]
        env_label = f" [{env_id}]" if env_id else ""
        ax.plot(x_grid, mean, color=color, label=f"{ALGO_LABELS[algo]}{env_label} (n={n_seeds} seeds)")

        # mean +/- std can exceed the physical reward bounds even though no
        # individual episode ever does (it's a statistic, not a raw trajectory).
        # Clip the shaded band so it never overshoots visually.
        lower, upper = mean - std, mean + std
        bound_env = env_id if env_id else ASSUMED_ENV.get(algo)
        if bound_env in ENV_REWARD_BOUNDS:
            r_min, r_max = ENV_REWARD_BOUNDS[bound_env]
            lower = np.clip(lower, r_min, r_max)
            upper = np.clip(upper, r_min, r_max)
        ax.fill_between(x_grid, lower, upper, color=color, alpha=0.2)

        if bound_env:
            envs_plotted.add(bound_env)

    # Reference lines for theoretical min/max achievable episode reward, one
    # pair per distinct environment actually plotted (usually just one, e.g.
    # Pendulum-v1, if all three algos were trained on the same env).
    line_styles = [":", "--"]
    for i, env_id in enumerate(sorted(envs_plotted)):
        if env_id not in ENV_REWARD_BOUNDS:
            continue
        r_min, r_max = ENV_REWARD_BOUNDS[env_id]
        style = line_styles[i % len(line_styles)]
        ax.axhline(r_min, color="gray", linestyle=style, linewidth=1,
                   label=f"{env_id} min possible reward ({r_min:.1f})")
        ax.axhline(r_max, color="black", linestyle=style, linewidth=1,
                   label=f"{env_id} max possible reward ({r_max:.1f})")

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    algo_suffix = f" — {', '.join(a.upper() for a in algos)}" if len(algos) < len(ALGOS) else ""
    ax.set_title(f"Training reward, averaged over seeds ({timesteps:,} steps){algo_suffix}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reward-Kurven gemittelt über alle Seeds für ein festes Timestep-Budget."
    )
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument(
        "--algos", type=str, nargs="+", default=["dqn", "sac", "td3"], choices=["dqn", "sac", "td3"],
        help="Which algorithm(s) to include, e.g. --algos dqn for DQN only.",
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", type=str, default="reward_curves_avg.png")
    parser.add_argument(
        "--dqn-env-id",
        type=str,
        default="Pendulum-v1",
        choices=["CartPole-v1", "Pendulum-v1"],
        help="Welches DQN-Env in den Vergleich soll (ihr habt DQN auf beiden trainiert). "
        "Default Pendulum-v1, damit DQN/SAC/TD3 auf demselben Env verglichen werden. "
        "Nur relevant, wenn 'dqn' in --algos enthalten ist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env_overrides = {"dqn": args.dqn_env_id}
    plot_averaged(Path(args.models_dir), args.algos, args.timesteps, args.seeds, Path(args.out), env_overrides)


if __name__ == "__main__":
    main()