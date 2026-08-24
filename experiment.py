import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch as th

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
PLOT_DIR = "./plots"

N_REFERENCE_STATES = 20


RUNS = [
    ("DQN", DQN, "CartPole-v1"),
    ("DQN", DQN, "PendulumDiscrete-v1"),
    ("SAC", SAC, "Pendulum-v1"),
    ("TD3", TD3, "Pendulum-v1"),
]

os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# Environment
# ============================================================

def make_logged_env(env_id, log_dir):
    env = make_env(env_id)
    return Monitor(env, log_dir)


def collect_reference_states(env_id, n_states=N_REFERENCE_STATES, seed=0):
    """
    Collect a fixed set of states used for Q-value logging.
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

def train(algo_name, algo_cls, env_id, timesteps=TOTAL_TIMESTEPS):

    log_dir = f"{LOG_DIR_BASE}/{algo_name}_{env_id}"
    os.makedirs(log_dir, exist_ok=True)

    env = make_logged_env(env_id, log_dir)
    eval_env = make_logged_env(env_id, log_dir + "_eval")

    reference_states = collect_reference_states(env_id)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=5,
        deterministic=True,
    )

    q_callback = QValueLoggingCallback(
        reference_states,
        log_dir,
        log_freq=EVAL_FREQ,
    )

    model = algo_cls("MlpPolicy", env, verbose=0)

    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, q_callback],
    )

    model.save(f"{log_dir}/final_model")

    env.close()
    eval_env.close()

    return log_dir


# ============================================================
# Monte-Carlo Q-value baseline
# ============================================================

@th.no_grad()
def get_q_values_dqn(model, obs):
    """
    Return Q(s,a) for all discrete actions.
    """

    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    q_values = model.q_net(obs_tensor)

    return q_values.cpu().numpy()


def rollout_discounted_return(
    model,
    env,
    gamma,
    max_steps=1000,
):
    """
    Run one deterministic episode and calculate
    the discounted Monte-Carlo return.
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

    return start_obs, start_action, G


def estimate_q_bias_dqn(model, env, n_episodes=50):

    gamma = model.gamma

    q_preds = []
    mc_returns = []

    for _ in range(n_episodes):

        s0, a0, G = rollout_discounted_return(
            model,
            env,
            gamma,
        )

        q_all = get_q_values_dqn(
            model,
            s0[np.newaxis, :],
        )

        q_pred = q_all[0, int(a0)]

        q_preds.append(q_pred)
        mc_returns.append(G)

    q_preds = np.asarray(q_preds)
    mc_returns = np.asarray(mc_returns)

    return q_preds, mc_returns, q_preds - mc_returns


def plot_q_bias(q_preds, mc_returns):

    lo = min(q_preds.min(), mc_returns.min())
    hi = max(q_preds.max(), mc_returns.max())

    plt.figure(figsize=(6, 5))

    plt.scatter(
        mc_returns,
        q_preds,
        alpha=0.6,
    )

    plt.plot(
        [lo, hi],
        [lo, hi],
        "k--",
        label="Perfect estimation",
    )

    plt.xlabel("Monte-Carlo Return")
    plt.ylabel("Estimated Q-Value")
    plt.title("DQN Q-Value vs Monte-Carlo Return")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/q_value_bias_dqn.png",
        dpi=150,
    )
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    log_dirs = []

    # Train all configurations
    for name, cls, env_id in RUNS:

        print(f"Training {name} on {env_id} ...")

        log_dir = train(
            name,
            cls,
            env_id,
        )

        log_dirs.append(
            (log_dir, f"{name} ({env_id})")
        )

    # Reward curves
    plot_reward_curves(log_dirs)

    # Q-value evolution
    plot_q_value_curves(log_dirs)

    # --------------------------------------------------------
    # DQN Monte-Carlo analysis
    # --------------------------------------------------------

    dqn_log_dir = log_dirs[0][0]

    dqn_model = DQN.load(
        f"{dqn_log_dir}/final_model"
    )

    dqn_eval_env = make_env("CartPole-v1")

    q_dqn, mc_dqn, bias_dqn = estimate_q_bias_dqn(
        dqn_model,
        dqn_eval_env,
    )

    print(
        f"DQN mean bias = {bias_dqn.mean():.3f} "
        f"(+ = overestimation, - = underestimation)"
    )

    plot_q_bias(
        q_dqn,
        mc_dqn,
    )

    dqn_eval_env.close()