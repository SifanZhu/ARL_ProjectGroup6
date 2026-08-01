import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import DQN, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

TOTAL_TIMESTEPS = 10_000
LOG_DIR_BASE = "./logs"
EVAL_FREQ = 2000
N_REFERENCE_STATES = 50


def make_env(env_id, log_dir):
    env = gym.make(env_id)
    env = Monitor(env, log_dir)  # needed to log episode rewards to disk
    return env


def collect_reference_states(env_id, n_states=N_REFERENCE_STATES, seed=0):
    # fixed set of states sampled once via a random rollout, used as a stable
    # comparison to track how Q(s, ) evolves over training
    env = gym.make(env_id)
    states = []
    obs, _ = env.reset(seed=seed)
    for _ in range(n_states):
        states.append(obs)
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return np.array(states, dtype=np.float32)


class QValueLoggingCallback(BaseCallback):
    def __init__(self, reference_states, log_dir, log_freq=EVAL_FREQ, verbose=0):
        super().__init__(verbose)
        self.reference_states = reference_states
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.timesteps = []
        self.q_values = []

    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            self.timesteps.append(self.num_timesteps)
            self.q_values.append(self._estimate_q_value())
        return True

    def _estimate_q_value(self):
        obs_tensor, _ = self.model.policy.obs_to_tensor(self.reference_states)
        with torch.no_grad():
            if isinstance(self.model, DQN):
                # value of the greedy action, as estimated by the Q-network
                q_values = self.model.q_net(obs_tensor)
                q_estimate = q_values.max(dim=1).values.mean()
            else:
                # SAC/TD3: continuous action space, so Q needs an action —
                # use the action the current (deterministic) policy would take
                actions, _ = self.model.predict(self.reference_states, deterministic=True)
                actions_tensor = torch.as_tensor(actions, device=self.model.device, dtype=torch.float32)
                q_tuple = self.model.critic(obs_tensor, actions_tensor)
                q_estimate = torch.cat(q_tuple, dim=1).mean()  # average over twin critics
        return q_estimate.item()

    def _on_training_end(self):
        np.savez(
            f"{self.log_dir}/q_values.npz",
            timesteps=np.array(self.timesteps),
            q_values=np.array(self.q_values),
        )


def train(algo_name, algo_cls, env_id, timesteps=TOTAL_TIMESTEPS):
    log_dir = f"{LOG_DIR_BASE}/{algo_name}_{env_id}"
    import os
    os.makedirs(log_dir, exist_ok=True)

    env = make_env(env_id, log_dir)

    # eval env for periodic evaluation (optional but useful)
    eval_env = make_env(env_id, log_dir + "_eval")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=5,
        deterministic=True,
    )

    reference_states = collect_reference_states(env_id)
    q_value_callback = QValueLoggingCallback(reference_states, log_dir, log_freq=EVAL_FREQ)

    model = algo_cls("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps, callback=[eval_callback, q_value_callback])
    model.save(f"{log_dir}/final_model")

    return log_dir


def load_eval_results(log_dir):
    # deterministic evaluation rewards recorded periodically during training,
    # not raw training-time episode rewards (exploration noise is excluded)
    data = np.load(f"{log_dir}/evaluations.npz")
    timesteps = data["timesteps"]
    mean_rewards = data["results"].mean(axis=1)
    return timesteps, mean_rewards


def plot_reward_curve(log_dir, label):
    x, y = load_eval_results(log_dir)
    plt.plot(x, y, label=label)


def load_q_values(log_dir):
    data = np.load(f"{log_dir}/q_values.npz")
    return data["timesteps"], data["q_values"]


def plot_q_value_curve(log_dir, label):
    x, y = load_q_values(log_dir)
    plt.plot(x, y, label=label)


def main():
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
    plt.ylabel("Mean Evaluation Reward (deterministic)")
    plt.title("Deterministic Evaluation Reward Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reward_curves.png")

    plt.figure(figsize=(8, 5))
    for log_dir, label in log_dirs:
        plot_q_value_curve(log_dir, label)
    plt.xlabel("Timesteps")
    plt.ylabel("Estimated Q-Value (reference states)")
    plt.title("Q-Value Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("q_values.png")


def plot_split_reward_curves():
    runs = [
        ("DQN", "CartPole-v1"),
        ("SAC", "Pendulum-v1"),
        ("TD3", "Pendulum-v1"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Subplot 1: DQN / CartPole ---
    name, env_id = runs[0]
    log_dir = f"{LOG_DIR_BASE}/{name}_{env_id}"
    x, y = load_eval_results(log_dir)
    axes[0].plot(x, y, label=f"{name} ({env_id})", color="tab:blue")
    axes[0].set_title("CartPole-v1")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Mean Evaluation Reward (deterministic)")
    axes[0].legend()

    # --- Subplot 2: SAC vs TD3 / Pendulum ---
    colors = {"SAC": "tab:orange", "TD3": "tab:green"}
    for name, env_id in runs[1:]:
        log_dir = f"{LOG_DIR_BASE}/{name}_{env_id}"
        x, y = load_eval_results(log_dir)
        axes[1].plot(x, y, label=f"{name} ({env_id})", color=colors[name])
    axes[1].set_title("Pendulum-v1")
    axes[1].set_xlabel("Timesteps")
    axes[1].legend()

    fig.suptitle("Deterministic Evaluation Reward Curves (split by environment / reward scale)")
    plt.tight_layout()
    plt.savefig("reward_curves_split.png", dpi=150)


def plot_split_q_value_curves():
    runs = [
        ("DQN", "CartPole-v1"),
        ("SAC", "Pendulum-v1"),
        ("TD3", "Pendulum-v1"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # DQN / CartPole
    name, env_id = runs[0]
    log_dir = f"{LOG_DIR_BASE}/{name}_{env_id}"
    x, y = load_q_values(log_dir)
    axes[0].plot(x, y, label=f"{name} ({env_id})", color="tab:blue")
    axes[0].set_title("CartPole-v1")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Estimated Q-Value (reference states)")
    axes[0].legend()

    # SAC vs TD3 / Pendulum
    colors = {"SAC": "tab:orange", "TD3": "tab:green"}
    for name, env_id in runs[1:]:
        log_dir = f"{LOG_DIR_BASE}/{name}_{env_id}"
        x, y = load_q_values(log_dir)
        axes[1].plot(x, y, label=f"{name} ({env_id})", color=colors[name])
    axes[1].set_title("Pendulum-v1")
    axes[1].set_xlabel("Timesteps")
    axes[1].legend()

    fig.suptitle("Q-Value Curves (split by environment / reward scale)")
    plt.tight_layout()
    plt.savefig("q_values_split.png", dpi=150)


if __name__ == "__main__":
    main()
    plot_split_reward_curves()
    plot_split_q_value_curves()
    plt.show()
