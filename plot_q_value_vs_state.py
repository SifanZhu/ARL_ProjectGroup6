"""
plot_q_value_vs_state.py

Erzeugt fuer DQN/CartPole, DQN/Pendulum, SAC/Pendulum und TD3/Pendulum je einen Plot,
der reale (aus Trainings-Policy-Rollouts stammende) Q-Werte als Scatter gegen eine
vom Netz gelernte, glatte Q-Funktions-Linie auftraegt -- beide gegen eine reale
State-Variable (Cart-Position fuer CartPole, Winkelgeschwindigkeit fuer Pendulum),
getrennt nach Aktions-Bucket. Die gelernte Linie wird auf einem Grid mit fixierten
Nebendimensionen ausgewertet (nicht an den realen, verrauschten States), damit sie
als glatte Kurve statt als Zickzack-Linie erscheint.

Ergaenzt estimate_bias_over_random_states in Qvalue_bias.py: dort wird der Bias auf
einen Skalar pro State reduziert, hier wird er ueber viele reale States hinweg
visualisiert.

CLI-Nutzung
-----------
    python plot_q_value_vs_state.py
    python plot_q_value_vs_state.py --dqn-steps 5000000 --sac-td3-steps 1000000 --n-episodes 30
"""

import argparse
from pathlib import Path
from typing import Tuple

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
# 3. Plot
# ---------------------------------------------------------------------------

def plot_q_vs_state(
    x_values: np.ndarray,
    action_buckets: np.ndarray,
    real_returns: np.ndarray,
    grid: np.ndarray,
    learned_low: np.ndarray,
    learned_high: np.ndarray,
    labels: Tuple[str, str],
    xlabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Reale Q-Werte bleiben Scatter (viele, teils ueberlappende reale States) --
    Groesse/Deckkraft/Zorder so gewaehlt, dass die Punktwolke trotz Overplotting
    noch sichtbar bleibt; gelernte Q-Werte kommen von einem Grid mit fixierten
    Nebendimensionen und sind dadurch als glatte Linie darstellbar.
    """
    plt.figure(figsize=(8, 5))
    mask_low, mask_high = action_buckets == 0, action_buckets == 1
    plt.scatter(x_values[mask_low], real_returns[mask_low], s=16, alpha=0.45, color="tab:blue",
                marker="o", linewidths=0, zorder=2, label=f"{labels[0]} - real Q-values")
    plt.scatter(x_values[mask_high], real_returns[mask_high], s=16, alpha=0.45, color="tab:orange",
                marker="o", linewidths=0, zorder=2, label=f"{labels[1]} - real Q-values")
    plt.plot(grid, learned_low, color="green", linewidth=2.5, zorder=3, label=f"{labels[0]} - learned Q-function")
    plt.plot(grid, learned_high, color="red", linewidth=2.5, zorder=3, label=f"{labels[1]} - learned Q-function")
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

def run_dqn_cartpole(models_dir: Path, steps: int, seed: int, n_episodes: int, out_dir: Path) -> None:
    run_dir = models_dir / f"dqn_CartPole-v1_steps{steps}_seed{seed}"
    model = DQN.load(str(run_dir / "final_model"))
    env = gym.make("CartPole-v1")

    obs_arr, actions, returns = collect_rollout_samples(model, env, n_episodes)
    buckets = np.array([bucket_action_discrete(a, 2) for a in actions])
    cart_positions = np.array([cartpole_cart_position(obs) for obs in obs_arr])

    grid = np.linspace(cart_positions.min(), cart_positions.max(), 200)
    learned_low = learned_q_curve_discrete(model, grid, cartpole_position_grid_obs, action_index=0)
    learned_high = learned_q_curve_discrete(model, grid, cartpole_position_grid_obs, action_index=1)

    plot_q_vs_state(
        cart_positions, buckets, returns, grid, learned_low, learned_high,
        labels=("push left", "push right"),
        xlabel="Cart position",
        title="DQN (CartPole-v1)",
        out_path=out_dir / "dqn_cartpole_q_vs_cart_position.png",
    )
    env.close()


def run_dqn_pendulum(models_dir: Path, steps: int, seed: int, n_episodes: int, out_dir: Path) -> None:
    run_dir = models_dir / f"dqn_Pendulum-v1_steps{steps}_seed{seed}"
    model = DQN.load(str(run_dir / "final_model"))
    env = DiscretizeActionWrapper(gym.make("Pendulum-v1"))

    obs_arr, actions, returns = collect_rollout_samples(model, env, n_episodes)
    buckets = np.array([bucket_action_discrete(a, N_ACTIONS) for a in actions])
    angular_velocities = np.array([pendulum_angular_velocity(obs) for obs in obs_arr])

    grid = np.linspace(angular_velocities.min(), angular_velocities.max(), 200)
    learned_low = learned_q_curve_discrete(model, grid, pendulum_angular_velocity_grid_obs, action_index=0)
    learned_high = learned_q_curve_discrete(
        model, grid, pendulum_angular_velocity_grid_obs, action_index=N_ACTIONS - 1
    )

    plot_q_vs_state(
        angular_velocities, buckets, returns, grid, learned_low, learned_high,
        labels=("low torque bin", "high torque bin"),
        xlabel="Angular velocity (theta_dot)",
        title="DQN (Pendulum-v1, discretized)",
        out_path=out_dir / "dqn_pendulum_q_vs_angular_velocity.png",
    )
    env.close()


def run_continuous_pendulum(
    algo_cls, algo_name: str, models_dir: Path, steps: int, seed: int, n_episodes: int, out_dir: Path
) -> None:
    run_dir = models_dir / f"{algo_name}_Pendulum-v1_steps{steps}_seed{seed}"
    model = algo_cls.load(str(run_dir / "final_model"))
    env = gym.make("Pendulum-v1")

    obs_arr, actions, returns = collect_rollout_samples(model, env, n_episodes)
    buckets = np.array([bucket_action_continuous(a) for a in actions])
    angular_velocities = np.array([pendulum_angular_velocity(obs) for obs in obs_arr])

    low_torque, high_torque = float(env.action_space.low[0]), float(env.action_space.high[0])
    grid = np.linspace(angular_velocities.min(), angular_velocities.max(), 200)
    learned_low = learned_q_curve_continuous(model, grid, pendulum_angular_velocity_grid_obs, low_torque)
    learned_high = learned_q_curve_continuous(model, grid, pendulum_angular_velocity_grid_obs, high_torque)

    plot_q_vs_state(
        angular_velocities, buckets, returns, grid, learned_low, learned_high,
        labels=("negative torque", "positive torque"),
        xlabel="Angular velocity (theta_dot)",
        title=f"{algo_name.upper()} (Pendulum-v1)",
        out_path=out_dir / f"{algo_name}_pendulum_q_vs_angular_velocity.png",
    )
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument(
        "--dqn-steps", type=int, default=5_000_000,
        help="Timestep-Budget fuer die DQN-Modelle (CartPole und Pendulum).",
    )
    parser.add_argument(
        "--sac-td3-steps", type=int, default=1_000_000,
        help="Timestep-Budget fuer die SAC- und TD3-Modelle (Pendulum).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-episodes", type=int, default=20,
        help="Anzahl echter Rollouts pro Modell (Basis der Scatter-Punkte).",
    )
    parser.add_argument("--out-dir", type=str, default="plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dqn_cartpole(models_dir, args.dqn_steps, args.seed, args.n_episodes, out_dir)
    run_dqn_pendulum(models_dir, args.dqn_steps, args.seed, args.n_episodes, out_dir)
    run_continuous_pendulum(SAC, "sac", models_dir, args.sac_td3_steps, args.seed, args.n_episodes, out_dir)
    run_continuous_pendulum(TD3, "td3", models_dir, args.sac_td3_steps, args.seed, args.n_episodes, out_dir)

    print(f"Plots gespeichert in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
