"""
state_space_error_analysis.py

Analyze where in state space the learned Q-function deviates from the
Monte-Carlo return, for Pendulum (angle / angular velocity) and CartPole
(pole angle / pole angular velocity).

No retraining required -- loads the saved final_model.zip checkpoints and
forces the env into exact states.
"""

import os
import numpy as np
import torch as th
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, SAC, TD3

from environment import make_env  # reuse your existing env factory

LOG_DIR_BASE = "./logs"
PLOT_DIR = "./plots"
os.makedirs(PLOT_DIR, exist_ok=True)

MAX_STEPS = 200  # rollout horizon for MC return


# ============================================================
# Forcing exact states
# ============================================================

def pendulum_obs(theta, theta_dot):
    return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)


def set_pendulum_state(env, theta, theta_dot):
    env.reset()
    env.unwrapped.state = np.array([theta, theta_dot], dtype=np.float64)
    return pendulum_obs(theta, theta_dot)


def set_cartpole_state(env, theta, theta_dot, x=0.0, x_dot=0.0):
    env.reset()
    env.unwrapped.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
    return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)


# ============================================================
# Q-value queries
# ============================================================

@th.no_grad()
def q_value_discrete(model, obs):
    """DQN: max_a Q(s,a) -- the greedy state value."""
    obs_t, _ = model.policy.obs_to_tensor(obs[np.newaxis, :])
    q = model.q_net(obs_t)
    return q.max(dim=1).values.item()


@th.no_grad()
def q_value_continuous(model, obs):
    """SAC/TD3: Q(s, pi(s)), averaged over the critic ensemble."""
    action, _ = model.predict(obs, deterministic=True)
    obs_t, _ = model.policy.obs_to_tensor(obs[np.newaxis, :])
    act_t = th.as_tensor(action[np.newaxis, :], dtype=th.float32)
    q_vals = model.critic(obs_t, act_t)
    return th.cat(list(q_vals), dim=1).mean(dim=1).item()


# ============================================================
# Monte-Carlo return from a fixed starting state
# ============================================================

def mc_return_from_current_state(model, env, gamma, max_steps=MAX_STEPS):
    obs = env.unwrapped._get_obs() if hasattr(env.unwrapped, "_get_obs") else np.asarray(env.unwrapped.state, dtype=np.float32)

    G, discount = 0.0, 1.0
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        G += discount * reward
        discount *= gamma
        if terminated or truncated:
            break
    return G


# ============================================================
# Grid sweeps
# ============================================================

def sweep_pendulum(model, is_discrete, gamma, n_points=41):
    """Two 1D slices: theta-axis (theta_dot=0) and theta_dot-axis (theta=0)."""
    env = make_env("PendulumDiscrete-v1" if is_discrete else "Pendulum-v1")
    q_fn = q_value_discrete if is_discrete else q_value_continuous

    thetas = np.linspace(-np.pi, np.pi, n_points)
    theta_dots = np.linspace(-8.0, 8.0, n_points)

    # theta slice, theta_dot fixed at 0
    q_theta, mc_theta = [], []
    for theta in thetas:
        obs = set_pendulum_state(env, theta, 0.0)
        q_theta.append(q_fn(model, obs))
        mc_theta.append(mc_return_from_current_state(model, env, gamma))

    # theta_dot slice, theta fixed at 0
    q_tdot, mc_tdot = [], []
    for tdot in theta_dots:
        obs = set_pendulum_state(env, 0.0, tdot)
        q_tdot.append(q_fn(model, obs))
        mc_tdot.append(mc_return_from_current_state(model, env, gamma))

    env.close()
    return (thetas, np.array(q_theta), np.array(mc_theta)), \
           (theta_dots, np.array(q_tdot), np.array(mc_tdot))


def sweep_pendulum_grid(model, is_discrete, gamma, n_points=21):
    """Full 2D grid over (theta, theta_dot) for a heatmap."""
    env = make_env("PendulumDiscrete-v1" if is_discrete else "Pendulum-v1")
    q_fn = q_value_discrete if is_discrete else q_value_continuous

    thetas = np.linspace(-np.pi, np.pi, n_points)
    theta_dots = np.linspace(-8.0, 8.0, n_points)

    Q = np.zeros((n_points, n_points))
    MC = np.zeros((n_points, n_points))

    for i, theta in enumerate(thetas):
        for j, tdot in enumerate(theta_dots):
            obs = set_pendulum_state(env, theta, tdot)
            Q[j, i] = q_fn(model, obs)
            MC[j, i] = mc_return_from_current_state(model, env, gamma)

    env.close()
    return thetas, theta_dots, Q, MC


def sweep_cartpole_grid(model, gamma, n_points=21):
    """Full 2D grid over (pole angle, pole angular velocity) for CartPole,
    cart position/velocity fixed at 0."""
    env = make_env("CartPole-v1")

    thetas = np.linspace(-0.20, 0.20, n_points)
    theta_dots = np.linspace(-2.0, 2.0, n_points)

    Q = np.zeros((n_points, n_points))
    MC = np.zeros((n_points, n_points))

    for i, theta in enumerate(thetas):
        for j, tdot in enumerate(theta_dots):
            obs = set_cartpole_state(env, theta, tdot)
            Q[j, i] = q_value_discrete(model, obs)
            MC[j, i] = mc_return_from_current_state(model, env, gamma)

    env.close()
    return thetas, theta_dots, Q, MC


def sweep_cartpole(model, gamma, n_points=41):
    """Two 1D slices for CartPole: pole angle and pole angular velocity,
    cart position/velocity fixed at 0."""
    env = make_env("CartPole-v1")

    # CartPole-v1 terminates outside these ranges
    thetas = np.linspace(-0.20, 0.20, n_points)
    theta_dots = np.linspace(-2.0, 2.0, n_points)

    q_theta, mc_theta = [], []
    for theta in thetas:
        obs = set_cartpole_state(env, theta, 0.0)
        q_theta.append(q_value_discrete(model, obs))
        mc_theta.append(mc_return_from_current_state(model, env, gamma))

    q_tdot, mc_tdot = [], []
    for tdot in theta_dots:
        obs = set_cartpole_state(env, 0.0, tdot)
        q_tdot.append(q_value_discrete(model, obs))
        mc_tdot.append(mc_return_from_current_state(model, env, gamma))

    env.close()
    return (thetas, np.array(q_theta), np.array(mc_theta)), \
           (theta_dots, np.array(q_tdot), np.array(mc_tdot))


# ============================================================
# Plotting
# ============================================================

def plot_slice(x, q, mc, xlabel, title, fname_prefix):
    bias = q - mc

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(x, q, label="Q-value")
    axes[0].plot(x, mc, label="MC return")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Value")
    axes[0].set_title(f"{title}: Q vs MC")
    axes[0].legend()

    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].plot(x, bias, color="tab:red")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Q - MC (bias)")
    axes[1].set_title(f"{title}: signed error")

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname_prefix}.png", dpi=150)
    plt.close()


def plot_heatmap(thetas, theta_dots, Q, MC, title, fname_prefix):
    bias = Q - MC
    vmax = np.abs(bias).max()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(
        Q, origin="lower", aspect="auto",
        extent=[thetas.min(), thetas.max(), theta_dots.min(), theta_dots.max()],
        cmap="viridis",
    )
    axes[0].set_xlabel("theta (angle)")
    axes[0].set_ylabel("theta_dot (angular velocity)")
    axes[0].set_title(f"{title}: Q-value")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        bias, origin="lower", aspect="auto",
        extent=[thetas.min(), thetas.max(), theta_dots.min(), theta_dots.max()],
        cmap="coolwarm", vmin=-vmax, vmax=vmax,
    )
    axes[1].set_xlabel("theta (angle)")
    axes[1].set_ylabel("theta_dot (angular velocity)")
    axes[1].set_title(f"{title}: signed error (Q - MC)")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname_prefix}_heatmap.png", dpi=150)
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --- Pendulum, DQN (discrete) ---
    dqn_pendulum = DQN.load(f"{LOG_DIR_BASE}/DQN_PendulumDiscrete-v1/final_model")
    gamma = dqn_pendulum.gamma
    (th_, q_th, mc_th), (td_, q_td, mc_td) = sweep_pendulum(dqn_pendulum, True, gamma)
    plot_slice(th_, q_th, mc_th, "theta [rad]", "Pendulum DQN (theta_dot=0)", "pendulum_dqn_theta")
    plot_slice(td_, q_td, mc_td, "theta_dot [rad/s]", "Pendulum DQN (theta=0)", "pendulum_dqn_thetadot")
    thetas, theta_dots, Q, MC = sweep_pendulum_grid(dqn_pendulum, True, gamma)
    plot_heatmap(thetas, theta_dots, Q, MC, "Pendulum DQN", "pendulum_dqn")

    # --- Pendulum, SAC / TD3 (continuous) ---
    for name, cls in [("SAC", SAC), ("TD3", TD3)]:
        model = cls.load(f"{LOG_DIR_BASE}/{name}_Pendulum-v1/final_model")
        gamma = model.gamma
        (th_, q_th, mc_th), (td_, q_td, mc_td) = sweep_pendulum(model, False, gamma)
        plot_slice(th_, q_th, mc_th, "theta [rad]", f"Pendulum {name} (theta_dot=0)", f"pendulum_{name.lower()}_theta")
        plot_slice(td_, q_td, mc_td, "theta_dot [rad/s]", f"Pendulum {name} (theta=0)", f"pendulum_{name.lower()}_thetadot")
        thetas, theta_dots, Q, MC = sweep_pendulum_grid(model, False, gamma)
        plot_heatmap(thetas, theta_dots, Q, MC, f"Pendulum {name}", f"pendulum_{name.lower()}")

    # --- CartPole, DQN ---
    dqn_cartpole = DQN.load(f"{LOG_DIR_BASE}/DQN_CartPole-v1/final_model")
    gamma = dqn_cartpole.gamma
    (th_, q_th, mc_th), (td_, q_td, mc_td) = sweep_cartpole(dqn_cartpole, gamma)
    plot_slice(th_, q_th, mc_th, "pole angle [rad]", "CartPole DQN (theta_dot=0)", "cartpole_dqn_theta")
    plot_slice(td_, q_td, mc_td, "pole angular velocity [rad/s]", "CartPole DQN (theta=0)", "cartpole_dqn_thetadot")
    thetas, theta_dots, Q, MC = sweep_cartpole_grid(dqn_cartpole, gamma)
    plot_heatmap(thetas, theta_dots, Q, MC, "CartPole DQN", "cartpole_dqn")

    print(f"Plots saved to: {PLOT_DIR}/")