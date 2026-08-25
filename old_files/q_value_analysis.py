import os

import gymnasium as gym
import numpy as np
import torch as th
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList


TOTAL_TIMESTEPS = 100_000
LOG_DIR_BASE = "./logs"


def make_env(env_id, log_dir):
    env = gym.make(env_id)
    env = Monitor(env, log_dir)
    return env


def train(algo_name, algo_cls, env_id, timesteps=TOTAL_TIMESTEPS):
    log_dir = f"{LOG_DIR_BASE}/{algo_name}_{env_id}"
    os.makedirs(log_dir, exist_ok=True)

    env = make_env(env_id, log_dir)

    eval_env = make_env(env_id, log_dir + "_eval")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=2000,
        n_eval_episodes=5,
        deterministic=True,
    )

    model = algo_cls("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    model.save(f"{log_dir}/final_model")

    return log_dir


def plot_reward_curve(log_dir, label):
    x, y = ts2xy(load_results(log_dir), "timesteps")
    plt.plot(x, y, label=label)


# 0. Training (DQN auf CartPole-v1, SAC/TD3 auf Pendulum-v1)

if __name__ == "__main__":
    runs = [
        ("DQN", DQN, "CartPole-v1"),
        ("SAC", SAC, "Pendulum-v1"),
        ("TD3", TD3, "Pendulum-v1"),
    ]

    log_dirs = []
    for name, cls, env_id in runs:
        print(f"Training {name} on {env_id} ...")
        log_dir = train(name, cls, env_id)
        log_dirs.append((log_dir, f"{name} ({env_id})"))

    plt.figure(figsize=(8, 5))
    for log_dir, label in log_dirs:
        plot_reward_curve(log_dir, label)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("Reward Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reward_curves.png")
    plt.show()

    # --- Reward-Kurven getrennt nach Umgebung (unterschiedliche Reward-Skalen) ---
    runs_split = [
        ("DQN", "CartPole-v1"),
        ("SAC", "Pendulum-v1"),
        ("TD3", "Pendulum-v1"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Subplot 1: DQN / CartPole ---
    name, env_id = runs_split[0]
    log_dir = f"{LOG_DIR_BASE}/{name}_{env_id}"
    x, y = ts2xy(load_results(log_dir), "timesteps")
    axes[0].plot(x, y, label=f"{name} ({env_id})", color="tab:blue")
    axes[0].set_title("CartPole-v1")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Episode Reward")
    axes[0].legend()

    # --- Subplot 2: SAC vs TD3 / Pendulum ---
    colors = {"SAC": "tab:orange", "TD3": "tab:green"}
    for name, env_id in runs_split[1:]:
        log_dir = f"{LOG_DIR_BASE}/{name}_{env_id}"
        x, y = ts2xy(load_results(log_dir), "timesteps")
        axes[1].plot(x, y, label=f"{name} ({env_id})", color=colors[name])
    axes[1].set_title("Pendulum-v1")
    axes[1].set_xlabel("Timesteps")
    axes[1].legend()

    fig.suptitle("Reward Curves (split by environment / reward scale)")
    plt.tight_layout()
    plt.savefig("reward_curves_split.png", dpi=150)
    plt.show()


    # 1. Trainiertes DQN-Modell laden
    def load_model(algo_name, algo_cls, env_id):
        log_dir = f"{LOG_DIR_BASE}/{algo_name}_{env_id}"
        model = algo_cls.load(f"{log_dir}/final_model")
        return model

    dqn_model = load_model("DQN", DQN, "CartPole-v1")

    # 2. Q-Werte extrahieren (nur DQN)
    @th.no_grad()
    def get_q_values_dqn(model: DQN, obs: np.ndarray) -> np.ndarray:
        """Gibt Q(s, a) für ALLE diskreten Aktionen zurueck. obs: (n_samples, obs_dim)"""
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        q_values = model.q_net(obs_tensor)          # shape: (n_samples, n_actions)
        return q_values.cpu().numpy()


    test_env_dqn = gym.make("CartPole-v1")
    obs_dqn, _ = test_env_dqn.reset()
    print("DQN Q(s, .) fuer alle Aktionen:", get_q_values_dqn(dqn_model, obs_dqn[np.newaxis, :]))


    # 3. Ground Truth: Monte-Carlo-Return als Vergleichsbasis (nur DQN)
    def rollout_discounted_return(model, env, gamma, deterministic=True, max_steps=1000):
        """Fuehrt eine Episode aus und gibt (start_obs, start_action, diskontierter_return) zurueck."""
        obs, _ = env.reset()
        start_obs = obs.copy()
        start_action, _ = model.predict(obs, deterministic=deterministic)

        G, discount, steps = 0.0, 1.0, 0
        done = False
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            G += discount * reward
            discount *= gamma
            done = terminated or truncated
            steps += 1

        return start_obs, start_action, G

    def estimate_q_bias_dqn(model, env, n_episodes=50):
        """Vergleicht geschaetzten Q-Wert am Episodenstart mit dem Monte-Carlo-Return (DQN)."""
        gamma = model.gamma
        q_preds, mc_returns = [], []

        for _ in range(n_episodes):
            s0, a0, G = rollout_discounted_return(model, env, gamma)

            q_all = get_q_values_dqn(model, s0[np.newaxis, :])   # (1, n_actions)
            q_pred = q_all[0, int(a0)]

            q_preds.append(q_pred)
            mc_returns.append(G)

        q_preds, mc_returns = np.array(q_preds), np.array(mc_returns)
        bias = q_preds - mc_returns
        return q_preds, mc_returns, bias

    # Eval-Env (nicht die Monitor-Trainings-Env verwenden!)
    dqn_eval_env = gym.make("CartPole-v1")

    q_dqn, mc_dqn, bias_dqn = estimate_q_bias_dqn(dqn_model, dqn_eval_env)

    print(f"DQN: mean bias = {bias_dqn.mean():.3f}  "
          f"(+ = Overestimation, - = Underestimation), std = {bias_dqn.std():.3f}")

    # 4. Visualisierung: geschätzt vs. tatsächlich (nur DQN)
    lo, hi = min(q_dqn.min(), mc_dqn.min()), max(q_dqn.max(), mc_dqn.max())
    plt.figure(figsize=(6, 5))
    plt.scatter(mc_dqn, q_dqn, alpha=0.6)
    plt.plot([lo, hi], [lo, hi], "k--", label="perfekte Schätzung")
    plt.xlabel("Monte-Carlo Return (wahr)")
    plt.ylabel("Geschätzter Q-Wert")
    plt.title("DQN (CartPole)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("q_value_bias_dqn.png", dpi=150)
    plt.show()

    # 5. Langzeitverhalten: Q-Funktion über den Trainingsverlauf
    def train_with_checkpoints(algo_name, algo_cls, env_id, timesteps=TOTAL_TIMESTEPS, save_freq=10_000):
        log_dir = f"{LOG_DIR_BASE}/{algo_name}_{env_id}"
        os.makedirs(log_dir, exist_ok=True)

        env = make_env(env_id, log_dir)
        eval_env = make_env(env_id, log_dir + "_eval")

        eval_callback = EvalCallback(
            eval_env, best_model_save_path=log_dir, log_path=log_dir,
            eval_freq=2000, n_eval_episodes=5, deterministic=True,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq, save_path=f"{log_dir}/checkpoints",
            name_prefix="model",
        )

        model = algo_cls("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=timesteps, callback=CallbackList([eval_callback, checkpoint_callback]))
        model.save(f"{log_dir}/final_model")
        return log_dir

    # Beispiel-Aufruf zum erneuten Trainieren mit Checkpoints (auskommentiert, da rechenintensiv):
    # train_with_checkpoints("DQN", DQN, "CartPole-v1")
