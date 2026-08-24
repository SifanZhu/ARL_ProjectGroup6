# experiment.py

import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import DQN, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from environment import make_env
from callback import QValueLoggingCallback, load_q_values


# ============================================================
# Configuration
# ============================================================

TOTAL_TIMESTEPS = 100_000
EVAL_FREQ = 2_000

LOG_DIR_BASE = "./logs"

N_REFERENCE_STATES = 20
REFERENCE_SEED = 0


RUNS = [
    ("DQN_CartPole", DQN, "CartPole-v1"),
    ("DQN_PendulumDiscrete", DQN, "PendulumDiscrete-v1"),
    ("SAC_Pendulum", SAC, "Pendulum-v1"),
    ("TD3_Pendulum", TD3, "Pendulum-v1"),
]


# ============================================================
# Environment
# ============================================================

def make_monitor_env(env_id, log_dir):

    env = make_env(env_id)

    os.makedirs(log_dir, exist_ok=True)

    return Monitor(env, log_dir)


# ============================================================
# Reference states
# ============================================================

def collect_reference_states(
    env_id,
    n_states=N_REFERENCE_STATES,
    seed=REFERENCE_SEED,
):
    """
    Collect a fixed set of states.

    These states are used throughout training so that changes
    in the Q-value are attributable to learning rather than
    changing evaluation states.
    """

    env = make_env(env_id)

    states = []

    obs, _ = env.reset(seed=seed)

    for _ in range(n_states):

        states.append(obs.copy())

        action = env.action_space.sample()

        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    return np.asarray(states, dtype=np.float32)


# ============================================================
# Training
# ============================================================

def train(
    algo_name,
    algo_cls,
    env_id,
    timesteps=TOTAL_TIMESTEPS,
):

    log_dir = os.path.join(
        LOG_DIR_BASE,
        f"{algo_name}_{env_id}",
    )

    os.makedirs(log_dir, exist_ok=True)

    # Training environment
    env = make_monitor_env(
        env_id,
        log_dir,
    )

    # Evaluation environment
    eval_env = make_monitor_env(
        env_id,
        log_dir + "_eval",
    )

    # Fixed reference states
    reference_states = collect_reference_states(
        env_id
    )

    # Reward evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=5,
        deterministic=True,
    )

    # Q-value logging
    q_callback = QValueLoggingCallback(
        reference_states=reference_states,
        log_dir=log_dir,
        log_freq=EVAL_FREQ,
    )

    # Train
    model = algo_cls(
        "MlpPolicy",
        env,
        verbose=0,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=[
            eval_callback,
            q_callback,
        ],
    )

    model.save(
        os.path.join(
            log_dir,
            "final_model",
        )
    )

    env.close()
    eval_env.close()

    return log_dir


# ============================================================
# Reward evaluation
# ============================================================

def load_eval_results(log_dir):

    data = np.load(
        os.path.join(
            log_dir,
            "evaluations.npz",
        )
    )

    timesteps = data["timesteps"]

    mean_rewards = data["results"].mean(axis=1)

    return timesteps, mean_rewards


# ============================================================
# Reward visualization
# ============================================================

def plot_reward_curves(log_dirs):

    plt.figure(figsize=(9, 5))

    for log_dir, label in log_dirs:

        x, y = load_eval_results(log_dir)

        plt.plot(
            x,
            y,
            label=label,
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Evaluation Reward")
    plt.title("Deterministic Evaluation Reward")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "reward_curves.png",
        dpi=150,
    )

    plt.show()


# ============================================================
# Q-value visualization
# ============================================================

def plot_q_value_curves(log_dirs):

    plt.figure(figsize=(9, 5))

    for log_dir, label in log_dirs:

        x, y = load_q_values(log_dir)

        plt.plot(
            x,
            y,
            label=label,
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Q-value")
    plt.title(
        "Q-value Evolution on Fixed Reference States"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "q_value_curves.png",
        dpi=150,
    )

    plt.show()


# ============================================================
# Q-FUNCTION VISUALIZATION
# ============================================================

def extract_q_values_dqn(model, states):

    states_tensor, _ = model.policy.obs_to_tensor(states)

    with torch.no_grad():

        q_values = model.q_net(states_tensor)

    return q_values.cpu().numpy()


def plot_dqn_q_function(
    model,
    states,
    title,
):
    """
    Visualize Q(s,a) for individual reference states.

    x-axis: discrete action
    y-axis: Q-value

    One curve = one state.
    """

    q_values = extract_q_values_dqn(
        model,
        states,
    )

    plt.figure(figsize=(9, 5))

    for i in range(len(states)):

        plt.plot(
            np.arange(q_values.shape[1]),
            q_values[i],
            marker="o",
            alpha=0.6,
            label=f"state {i}",
        )

    plt.xlabel("Discrete Action")
    plt.ylabel("Q(s, a)")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(
        "dqn_q_function.png",
        dpi=150,
    )

    plt.show()


# ============================================================
# Q-FUNCTION EVOLUTION
# ============================================================

def compare_q_function_checkpoints(
    model_cls,
    model_path_template,
    states,
    checkpoints,
):
    """
    Plot how Q(s,a) changes during training.

    Each curve represents one training checkpoint.
    """

    plt.figure(figsize=(9, 5))

    for checkpoint in checkpoints:

        model = model_cls.load(
            model_path_template.format(
                checkpoint=checkpoint
            )
        )

        q_values = extract_q_values_dqn(
            model,
            states,
        )

        # Average across states
        mean_q = q_values.mean(axis=0)

        plt.plot(
            np.arange(len(mean_q)),
            mean_q,
            marker="o",
            label=f"{checkpoint} steps",
        )

    plt.xlabel("Discrete Action")
    plt.ylabel("Mean Q(s,a)")
    plt.title(
        "Q-function Evolution Across Training"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "q_function_evolution.png",
        dpi=150,
    )

    plt.show()


# ============================================================
# Monte-Carlo Q baseline
# ============================================================

def rollout_discounted_return(
    model,
    env,
    gamma,
    max_steps=1000,
):
    """
    Generate one episode following the model's
    deterministic policy and calculate the discounted return.
    """

    obs, _ = env.reset()

    start_obs = obs.copy()

    start_action, _ = model.predict(
        obs,
        deterministic=True,
    )

    G = 0.0
    discount = 1.0

    for _ in range(max_steps):

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, _ = env.step(
            action
        )

        G += discount * reward
        discount *= gamma

        if terminated or truncated:
            break

    return (
        start_obs,
        start_action,
        G,
    )


def get_dqn_q_values(
    model,
    obs,
):

    obs_tensor, _ = model.policy.obs_to_tensor(
        obs
    )

    with torch.no_grad():

        q_values = model.q_net(
            obs_tensor
        )

    return q_values.cpu().numpy()


def estimate_q_bias_dqn(
    model,
    env,
    n_episodes=50,
):
    """
    Compare Q(s0,a0) against the Monte-Carlo
    discounted return from the same state/action.
    """

    gamma = model.gamma

    q_predictions = []
    mc_returns = []

    for _ in range(n_episodes):

        s0, a0, G = rollout_discounted_return(
            model,
            env,
            gamma,
        )

        q_all = get_dqn_q_values(
            model,
            s0[np.newaxis, :],
        )

        q_pred = q_all[
            0,
            int(a0)
        ]

        q_predictions.append(q_pred)
        mc_returns.append(G)

    q_predictions = np.asarray(
        q_predictions
    )

    mc_returns = np.asarray(
        mc_returns
    )

    bias = q_predictions - mc_returns

    return (
        q_predictions,
        mc_returns,
        bias,
    )


def plot_q_bias(
    q_predictions,
    mc_returns,
    title,
):

    lo = min(
        q_predictions.min(),
        mc_returns.min(),
    )

    hi = max(
        q_predictions.max(),
        mc_returns.max(),
    )

    plt.figure(figsize=(6, 5))

    plt.scatter(
        mc_returns,
        q_predictions,
        alpha=0.6,
    )

    plt.plot(
        [lo, hi],
        [lo, hi],
        "k--",
        label="Perfect estimation",
    )

    plt.xlabel("Monte-Carlo Return")
    plt.ylabel("Estimated Q-value")
    plt.title(title)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "q_value_bias.png",
        dpi=150,
    )

    plt.show()


# ============================================================
# Main experiment
# ============================================================

def main():

    log_dirs = []

    for name, algo_cls, env_id in RUNS:

        print(
            f"Training {name} on {env_id}..."
        )

        log_dir = train(
            name,
            algo_cls,
            env_id,
        )

        log_dirs.append(
            (
                log_dir,
                f"{name} ({env_id})",
            )
        )

    # Reward
    plot_reward_curves(log_dirs)

    # Q-value evolution
    plot_q_value_curves(log_dirs)

    # --------------------------------------------------------
    # DQN Monte-Carlo analysis
    # --------------------------------------------------------

    dqn_log_dir = (
        f"{LOG_DIR_BASE}/"
        f"DQN_CartPole_CartPole-v1"
    )

    dqn_model_path = os.path.join(
        dqn_log_dir,
        "final_model",
    )

    # This path is only illustrative; use the actual
    # directory generated by train().
    if os.path.exists(dqn_model_path):

        model = DQN.load(
            dqn_model_path
        )

        env = make_env(
            "CartPole-v1"
        )

        q_pred, mc_return, bias = (
            estimate_q_bias_dqn(
                model,
                env,
            )
        )

        print(
            f"DQN mean Q bias: "
            f"{bias.mean():.3f}"
        )

        plot_q_bias(
            q_pred,
            mc_return,
            "DQN Q-value vs Monte-Carlo Return",
        )

        env.close()


if __name__ == "__main__":
    main()