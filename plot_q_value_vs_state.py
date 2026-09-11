"""
plot_q_value_vs_state.py

Erzeugt fuer DQN/CartPole, DQN/Pendulum, SAC/Pendulum und TD3/Pendulum je einen Plot,
der reale (aus Trainings-Policy-Rollouts stammende) Q-Werte gegen eine vom Netz
gelernte, glatte Q-Funktions-Linie auftraegt -- beide gegen eine reale State-Variable
(Cart-Position fuer CartPole, Winkelgeschwindigkeit fuer Pendulum), getrennt nach
Aktions-Bucket. Jedes Modell wird mit mehreren Seeds trainiert (models/<run>_seed<N>);
fuer jeden Seed werden die realen Returns (gebinnt nach x-Wert) und die gelernte
Q-Kurve auf einem gemeinsamen Grid berechnet, und ueber die Seeds gemittelt (Mean +/-
Std), bevor geplottet wird. Die gelernte Linie wird auf einem Grid mit fixierten
Nebendimensionen ausgewertet (nicht an den realen, verrauschten States), damit sie
als glatte Kurve statt als Zickzack-Linie erscheint.

Ergaenzt estimate_bias_over_random_states in Qvalue_bias.py: dort wird der Bias auf
einen Skalar pro State reduziert, hier wird er ueber viele reale States hinweg
visualisiert.

CLI-Nutzung
-----------
    python plot_q_value_vs_state.py
    python plot_q_value_vs_state.py --dqn-steps 1000000 --sac-td3-steps 1000000 --n-episodes 20
    python plot_q_value_vs_state.py --seeds 0 1 2 3 4

    # Also plot one PNG per DQN/CartPole checkpoint found in checkpoints/model_<N>_steps.zip
    python plot_q_value_vs_state.py --dqn-cartpole-all-checkpoints

    # Only specific checkpoints of the chosen model (steps must match an existing model_<N>_steps.zip)
    python plot_q_value_vs_state.py --dqn-cartpole-checkpoints 100000 500000 1000000

    # Only the checkpoint sweep, skip the final_model plot
    python plot_q_value_vs_state.py --dqn-cartpole-all-checkpoints --dqn-cartpole-skip-final-model
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

from Qvalue_bias import get_q_value
from train import DiscretizeActionWrapper, N_ACTIONS


# ---------------------------------------------------------------------------
# 1. Reale Rollout-Daten: State, Aktion, Return-to-go pro Zeitschritt
# ---------------------------------------------------------------------------

def collect_rollout_samples(
    model: BaseAlgorithm,
    env: gym.Env,
    n_episodes: int = 20,
    max_steps: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sammelt (obs, Aktion, Return-to-go) ueber n_episodes echte Episoden.

    Reihenfolge der Rueckgabe = Sammelreihenfolge (Episode fuer Episode, Schritt fuer
    Schritt) -- das ist der spaetere State-Index auf der x-Achse der Plots.
    Return-to-go wird rueckwaerts INNERHALB jeder Episode berechnet -- kein
    zusaetzlicher Rollout noetig, da die Policy deterministisch der bereits
    gespielten Trajektorie folgt.
    """
    gamma = model.gamma
    obs_list, actions, returns = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_obs, ep_actions, ep_rewards = [], [], []
        done, steps = False, 0
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            ep_obs.append(obs.copy())
            ep_actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_rewards.append(reward)
            done = terminated or truncated
            steps += 1

        G = 0.0
        ep_returns = [0.0] * len(ep_rewards)
        for t in reversed(range(len(ep_rewards))):
            G = ep_rewards[t] + gamma * G
            ep_returns[t] = G

        obs_list.extend(ep_obs)
        actions.extend(ep_actions)
        returns.extend(ep_returns)

    return np.stack(obs_list), np.array(actions), np.array(returns)


# ---------------------------------------------------------------------------
# 2. State-Variablen-Extraktion, Aktions-Bucketing und gelernte Q-Werte je Sample
# ---------------------------------------------------------------------------

def cartpole_cart_position(obs: np.ndarray) -> float:
    return float(obs[0])  # obs = [cart_pos, cart_vel, pole_angle, pole_angvel]


def pendulum_angular_velocity(obs: np.ndarray) -> float:
    return float(obs[2])  # obs = [cos(theta), sin(theta), thetadot]


def bucket_action_discrete(action, n_actions: int) -> int:
    """0 = niedriger Aktionsindex (z.B. push left / negatives Drehmoment), 1 = hoch."""
    return 0 if float(action) < n_actions / 2 else 1


def bucket_action_continuous(action) -> int:
    return 0 if float(np.asarray(action).flatten()[0]) < 0 else 1


def cartpole_position_grid_obs(cart_position: float) -> np.ndarray:
    """Andere Dimensionen fixiert -> Q haengt nur noch von cart_position ab (glatte Linie)."""
    return np.array([cart_position, 0.0, 0.0, 0.0], dtype=np.float32)


def pendulum_angular_velocity_grid_obs(theta_dot: float) -> np.ndarray:
    """theta fixiert auf 0 -> Q haengt nur noch von theta_dot ab (glatte Linie)."""
    return np.array([1.0, 0.0, theta_dot], dtype=np.float32)


def learned_q_curve_discrete(model, grid: np.ndarray, grid_obs_fn, action_index: int) -> np.ndarray:
    return np.array([get_q_value(model, grid_obs_fn(x))[action_index] for x in grid])


def learned_q_curve_continuous(model, grid: np.ndarray, grid_obs_fn, action_value: float) -> np.ndarray:
    action = np.array([action_value], dtype=np.float32)
    return np.array([get_q_value(model, grid_obs_fn(x), action=action)["min"] for x in grid])


# ---------------------------------------------------------------------------
# 2b. Aggregation ueber mehrere Seeds
# ---------------------------------------------------------------------------

N_BINS = 20  # Anzahl x-Bins, ueber die reale Returns gemittelt werden (pro Seed und Bucket)


def real_q_bin_means(
    x_values: np.ndarray, buckets: np.ndarray, returns: np.ndarray, bin_edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Mittlerer realer Return je x-Bin, getrennt nach Aktions-Bucket (ein Seed).

    NaN in Bins ohne Sample fuer den jeweiligen Bucket -- wird spaeter beim
    Mitteln ueber Seeds per nanmean/nanstd ignoriert.
    """
    n_bins = len(bin_edges) - 1
    bin_idx = np.clip(np.digitize(x_values, bin_edges[1:-1]), 0, n_bins - 1)
    means = np.full((2, n_bins), np.nan)
    for bucket in (0, 1):
        mask_bucket = buckets == bucket
        for b in range(n_bins):
            sel = mask_bucket & (bin_idx == b)
            if sel.any():
                means[bucket, b] = returns[sel].mean()
    return means[0], means[1]


def nanmean_std(arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Mean/Std ueber Seeds (Achse 0), NaN-Bins bleiben bei genuegend anderen Seeds erhalten."""
    stacked = np.stack(arrays)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)


# ---------------------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------------------

def plot_q_vs_state(
    bin_centers: np.ndarray,
    real_low_mean: np.ndarray,
    real_low_std: np.ndarray,
    real_high_mean: np.ndarray,
    real_high_std: np.ndarray,
    grid: np.ndarray,
    learned_low_mean: np.ndarray,
    learned_low_std: np.ndarray,
    learned_high_mean: np.ndarray,
    learned_high_std: np.ndarray,
    labels: Tuple[str, str],
    xlabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Reale und gelernte Q-Werte sind je Mean +/- Std ueber alle Seeds (reale Werte
    zuvor pro Seed nach x-Bin gemittelt, damit sie ueber Seeds hinweg vergleichbar sind).
    """
    plt.figure(figsize=(8, 5))
    plt.errorbar(bin_centers, real_low_mean, yerr=real_low_std, fmt="o", ms=4, alpha=0.7,
                 color="tab:blue", ecolor="tab:blue", elinewidth=1, capsize=2, zorder=2,
                 label=f"{labels[0]} - real Q-values (mean +/- std over seeds)")
    plt.errorbar(bin_centers, real_high_mean, yerr=real_high_std, fmt="o", ms=4, alpha=0.7,
                 color="tab:orange", ecolor="tab:orange", elinewidth=1, capsize=2, zorder=2,
                 label=f"{labels[1]} - real Q-values (mean +/- std over seeds)")
    plt.plot(grid, learned_low_mean, color="green", linewidth=2.5, zorder=3,
             label=f"{labels[0]} - learned Q-function (mean over seeds)")
    plt.fill_between(grid, learned_low_mean - learned_low_std, learned_low_mean + learned_low_std,
                      color="green", alpha=0.2, zorder=1)
    plt.plot(grid, learned_high_mean, color="red", linewidth=2.5, zorder=3,
             label=f"{labels[1]} - learned Q-function (mean over seeds)")
    plt.fill_between(grid, learned_high_mean - learned_high_std, learned_high_mean + learned_high_std,
                      color="red", alpha=0.2, zorder=1)
    plt.xlabel(xlabel)
    plt.ylabel("Q-values")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 4. Pro Algo/Env: laden, samplen, plotten
# ---------------------------------------------------------------------------

CHECKPOINT_PATTERN = re.compile(r"model_(\d+)_steps\.zip$")


def discover_checkpoint_steps(run_dir: Path) -> List[int]:
    """Sortierte Liste der Timestep-Zahlen aller model_<N>_steps.zip in run_dir/checkpoints."""
    checkpoints_dir = run_dir / "checkpoints"
    matches = (CHECKPOINT_PATTERN.match(f.name) for f in checkpoints_dir.glob("model_*_steps.zip"))
    return sorted(int(m.group(1)) for m in matches if m is not None)


def plot_dqn_cartpole_model(models: List[DQN], n_episodes: int, out_path: Path, title: str) -> None:
    """models = ein DQN pro Seed; geplottet wird Mean +/- Std ueber diese Seeds."""
    env = gym.make("CartPole-v1")

    per_seed_x, per_seed_buckets, per_seed_returns = [], [], []
    for model in models:
        obs_arr, actions, returns = collect_rollout_samples(model, env, n_episodes)
        per_seed_x.append(np.array([cartpole_cart_position(obs) for obs in obs_arr]))
        per_seed_buckets.append(np.array([bucket_action_discrete(a, 2) for a in actions]))
        per_seed_returns.append(returns)
    env.close()

    x_min = min(x.min() for x in per_seed_x)
    x_max = max(x.max() for x in per_seed_x)
    bin_edges = np.linspace(x_min, x_max, N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    grid = np.linspace(x_min, x_max, 200)

    real_low_stack, real_high_stack, learned_low_stack, learned_high_stack = [], [], [], []
    for model, x_vals, buckets, returns in zip(models, per_seed_x, per_seed_buckets, per_seed_returns):
        low, high = real_q_bin_means(x_vals, buckets, returns, bin_edges)
        real_low_stack.append(low)
        real_high_stack.append(high)
        learned_low_stack.append(learned_q_curve_discrete(model, grid, cartpole_position_grid_obs, action_index=0))
        learned_high_stack.append(learned_q_curve_discrete(model, grid, cartpole_position_grid_obs, action_index=1))

    real_low_mean, real_low_std = nanmean_std(real_low_stack)
    real_high_mean, real_high_std = nanmean_std(real_high_stack)
    learned_low_mean, learned_low_std = nanmean_std(learned_low_stack)
    learned_high_mean, learned_high_std = nanmean_std(learned_high_stack)

    plot_q_vs_state(
        bin_centers, real_low_mean, real_low_std, real_high_mean, real_high_std,
        grid, learned_low_mean, learned_low_std, learned_high_mean, learned_high_std,
        labels=("push left", "push right"),
        xlabel="Cart position",
        title=title,
        out_path=out_path,
    )


def run_dqn_cartpole(
    models_dir: Path,
    steps: int,
    seeds: List[int],
    n_episodes: int,
    out_dir: Path,
    use_final_model: bool = True,
    all_checkpoints: bool = False,
    checkpoints: Optional[List[int]] = None,
) -> None:
    run_dirs = [models_dir / f"dqn_CartPole-v1_steps{steps}_seed{seed}" for seed in seeds]

    if use_final_model:
        models = [DQN.load(str(run_dir / "final_model")) for run_dir in run_dirs]
        plot_dqn_cartpole_model(
            models, n_episodes,
            out_path=out_dir / "dqn_cartpole_q_vs_cart_position.png",
            title="DQN (CartPole-v1)",
        )

    # explicit --dqn-cartpole-checkpoints selection takes precedence over --dqn-cartpole-all-checkpoints
    selected_checkpoints = checkpoints if checkpoints is not None else (
        discover_checkpoint_steps(run_dirs[0]) if all_checkpoints else []
    )
    for ckpt_steps in selected_checkpoints:
        models = [DQN.load(str(run_dir / "checkpoints" / f"model_{ckpt_steps}_steps")) for run_dir in run_dirs]
        plot_dqn_cartpole_model(
            models, n_episodes,
            out_path=out_dir / f"dqn_cartpole_q_vs_cart_position_step{ckpt_steps}.png",
            title=f"DQN (CartPole-v1, step {ckpt_steps})",
        )


def run_dqn_pendulum(models_dir: Path, steps: int, seeds: List[int], n_episodes: int, out_dir: Path) -> None:
    models, per_seed_x, per_seed_buckets, per_seed_returns = [], [], [], []
    for seed in seeds:
        run_dir = models_dir / f"dqn_Pendulum-v1_steps{steps}_seed{seed}"
        model = DQN.load(str(run_dir / "final_model"))
        env = DiscretizeActionWrapper(gym.make("Pendulum-v1"))
        obs_arr, actions, returns = collect_rollout_samples(model, env, n_episodes)
        env.close()

        models.append(model)
        per_seed_x.append(np.array([pendulum_angular_velocity(obs) for obs in obs_arr]))
        per_seed_buckets.append(np.array([bucket_action_discrete(a, N_ACTIONS) for a in actions]))
        per_seed_returns.append(returns)

    x_min = min(x.min() for x in per_seed_x)
    x_max = max(x.max() for x in per_seed_x)
    bin_edges = np.linspace(x_min, x_max, N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    grid = np.linspace(x_min, x_max, 200)

    real_low_stack, real_high_stack, learned_low_stack, learned_high_stack = [], [], [], []
    for model, x_vals, buckets, returns in zip(models, per_seed_x, per_seed_buckets, per_seed_returns):
        low, high = real_q_bin_means(x_vals, buckets, returns, bin_edges)
        real_low_stack.append(low)
        real_high_stack.append(high)
        learned_low_stack.append(
            learned_q_curve_discrete(model, grid, pendulum_angular_velocity_grid_obs, action_index=0)
        )
        learned_high_stack.append(
            learned_q_curve_discrete(model, grid, pendulum_angular_velocity_grid_obs, action_index=N_ACTIONS - 1)
        )

    real_low_mean, real_low_std = nanmean_std(real_low_stack)
    real_high_mean, real_high_std = nanmean_std(real_high_stack)
    learned_low_mean, learned_low_std = nanmean_std(learned_low_stack)
    learned_high_mean, learned_high_std = nanmean_std(learned_high_stack)

    plot_q_vs_state(
        bin_centers, real_low_mean, real_low_std, real_high_mean, real_high_std,
        grid, learned_low_mean, learned_low_std, learned_high_mean, learned_high_std,
        labels=("low torque bin", "high torque bin"),
        xlabel="Angular velocity (theta_dot)",
        title="DQN (Pendulum-v1, discretized)",
        out_path=out_dir / "dqn_pendulum_q_vs_angular_velocity.png",
    )


def run_continuous_pendulum(
    algo_cls, algo_name: str, models_dir: Path, steps: int, seeds: List[int], n_episodes: int, out_dir: Path
) -> None:
    models, per_seed_x, per_seed_buckets, per_seed_returns = [], [], [], []
    low_torque = high_torque = None
    for seed in seeds:
        run_dir = models_dir / f"{algo_name}_Pendulum-v1_steps{steps}_seed{seed}"
        model = algo_cls.load(str(run_dir / "final_model"))
        env = gym.make("Pendulum-v1")
        obs_arr, actions, returns = collect_rollout_samples(model, env, n_episodes)
        if low_torque is None:  # identical action range for every seed on this env
            low_torque, high_torque = float(env.action_space.low[0]), float(env.action_space.high[0])
        env.close()

        models.append(model)
        per_seed_x.append(np.array([pendulum_angular_velocity(obs) for obs in obs_arr]))
        per_seed_buckets.append(np.array([bucket_action_continuous(a) for a in actions]))
        per_seed_returns.append(returns)

    x_min = min(x.min() for x in per_seed_x)
    x_max = max(x.max() for x in per_seed_x)
    bin_edges = np.linspace(x_min, x_max, N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    grid = np.linspace(x_min, x_max, 200)

    real_low_stack, real_high_stack, learned_low_stack, learned_high_stack = [], [], [], []
    for model, x_vals, buckets, returns in zip(models, per_seed_x, per_seed_buckets, per_seed_returns):
        low, high = real_q_bin_means(x_vals, buckets, returns, bin_edges)
        real_low_stack.append(low)
        real_high_stack.append(high)
        learned_low_stack.append(
            learned_q_curve_continuous(model, grid, pendulum_angular_velocity_grid_obs, low_torque)
        )
        learned_high_stack.append(
            learned_q_curve_continuous(model, grid, pendulum_angular_velocity_grid_obs, high_torque)
        )

    real_low_mean, real_low_std = nanmean_std(real_low_stack)
    real_high_mean, real_high_std = nanmean_std(real_high_stack)
    learned_low_mean, learned_low_std = nanmean_std(learned_low_stack)
    learned_high_mean, learned_high_std = nanmean_std(learned_high_stack)

    plot_q_vs_state(
        bin_centers, real_low_mean, real_low_std, real_high_mean, real_high_std,
        grid, learned_low_mean, learned_low_std, learned_high_mean, learned_high_std,
        labels=("negative torque", "positive torque"),
        xlabel="Angular velocity (theta_dot)",
        title=f"{algo_name.upper()} (Pendulum-v1)",
        out_path=out_dir / f"{algo_name}_pendulum_q_vs_angular_velocity.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument(
        "--dqn-steps", type=int, default=1_000_000, # default model trained for 1M steps
        help="Timestep-Budget fuer die DQN-Modelle (CartPole und Pendulum).",
    )
    parser.add_argument(
        "--sac-td3-steps", type=int, default=1_000_000,
        help="Timestep-Budget fuer die SAC- und TD3-Modelle (Pendulum).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="Seeds, deren final_model (bzw. Checkpoints) geladen und gemittelt werden.",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=20, # default model trained for 1M steps
        help="Anzahl echter Rollouts pro Modell und Seed (Basis der realen Q-Werte).",
    )
    parser.add_argument("--out-dir", type=str, default="plots")
    parser.add_argument(
        "--dqn-cartpole-all-checkpoints", action="store_true",
        help="Erzeugt zusaetzlich je einen Plot pro vorhandenem Checkpoint in "
             "models/<run>/checkpoints/model_<N>_steps.zip fuer DQN/CartPole.",
    )
    parser.add_argument(
        "--dqn-cartpole-checkpoints", type=int, nargs="+", default=None, metavar="N",
        help="Plottet nur die angegebenen Checkpoint-Timesteps (z.B. --dqn-cartpole-checkpoints "
             "100000 500000), jeweils models/<run>/checkpoints/model_<N>_steps.zip. "
             "Ueberschreibt --dqn-cartpole-all-checkpoints, falls beide gesetzt sind.",
    )
    parser.add_argument(
        "--dqn-cartpole-skip-final-model", action="store_true",
        help="Ueberspringt den Plot fuer final_model.zip bei DQN/CartPole "
             "(z.B. wenn nur --dqn-cartpole-all-checkpoints gewuenscht ist).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dqn_cartpole(
        models_dir, args.dqn_steps, args.seeds, args.n_episodes, out_dir,
        use_final_model=not args.dqn_cartpole_skip_final_model,
        all_checkpoints=args.dqn_cartpole_all_checkpoints,
        checkpoints=args.dqn_cartpole_checkpoints,
    )
    run_dqn_pendulum(models_dir, args.dqn_steps, args.seeds, args.n_episodes, out_dir)
    run_continuous_pendulum(SAC, "sac", models_dir, args.sac_td3_steps, args.seeds, args.n_episodes, out_dir)
    run_continuous_pendulum(TD3, "td3", models_dir, args.sac_td3_steps, args.seeds, args.n_episodes, out_dir)

    print(f"Plots gespeichert in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
