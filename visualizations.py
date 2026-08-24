import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

LOG_DIR_BASE = "./logs"
PLOT_DIR = "./plots"


RUNS = [
    ("DQN", "CartPole-v1"),
    ("DQN", "PendulumDiscrete-v1"),
    ("SAC", "Pendulum-v1"),
    ("TD3", "Pendulum-v1"),
]


# ============================================================
# Loading results
# ============================================================

def load_eval_results(log_dir):
    """
    Load deterministic evaluation rewards produced by EvalCallback.
    """

    data = np.load(f"{log_dir}/evaluations.npz")

    timesteps = data["timesteps"]
    mean_rewards = data["results"].mean(axis=1)

    return timesteps, mean_rewards


def load_q_values(log_dir):
    """
    Load Q-value evolution produced by QValueLoggingCallback.
    """

    data = np.load(f"{log_dir}/q_values.npz")

    timesteps = data["timesteps"]
    q_values = data["q_values"]

    return timesteps, q_values


# ============================================================
# Reward visualization
# ============================================================

def plot_reward_curves(log_dirs):
    """
    Plot all reward curves together.
    """

    os.makedirs(PLOT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for log_dir, label in log_dirs:
        x, y = load_eval_results(log_dir)

        plt.plot(
            x,
            y,
            label=label,
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Evaluation Reward")
    plt.title("Deterministic Evaluation Reward Curves")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/reward_curves.png",
        dpi=150,
    )

    plt.close()


def plot_split_reward_curves(log_dirs):
    """
    Plot reward curves split into:

    Left:
        DQN + CartPole-v1

    Right:
        DQN + PendulumDiscrete-v1
        SAC + Pendulum-v1
        TD3 + Pendulum-v1
    """

    os.makedirs(PLOT_DIR, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    # --------------------------------------------------------
    # CartPole
    # --------------------------------------------------------

    for log_dir, label in log_dirs:

        if "CartPole-v1" not in label:
            continue

        x, y = load_eval_results(log_dir)

        axes[0].plot(
            x,
            y,
            label=label,
        )

    axes[0].set_title("CartPole-v1")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Mean Evaluation Reward")
    axes[0].legend()


    # --------------------------------------------------------
    # Pendulum / Discrete Pendulum
    # --------------------------------------------------------

    for log_dir, label in log_dirs:

        if (
            "Pendulum-v1" not in label
            and "PendulumDiscrete-v1" not in label
        ):
            continue

        x, y = load_eval_results(log_dir)

        axes[1].plot(
            x,
            y,
            label=label,
        )

    axes[1].set_title("Pendulum-v1 / PendulumDiscrete-v1")
    axes[1].set_xlabel("Timesteps")
    axes[1].set_ylabel("Mean Evaluation Reward")
    axes[1].legend()


    fig.suptitle(
        "Deterministic Evaluation Reward Curves "
        "(split by environment)"
    )

    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/reward_curves_split.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# Q-value visualization
# ============================================================

def plot_q_value_curves(log_dirs):
    """
    Plot all Q-value curves together.
    """

    os.makedirs(PLOT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for log_dir, label in log_dirs:

        x, y = load_q_values(log_dir)

        plt.plot(
            x,
            y,
            label=label,
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Estimated Q-Value")
    plt.title("Q-Value Evolution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/q_values.png",
        dpi=150,
    )

    plt.close()


def plot_split_q_value_curves(log_dirs):
    """
    Plot Q-value curves split into:

    Left:
        DQN + CartPole-v1

    Right:
        DQN + PendulumDiscrete-v1
        SAC + Pendulum-v1
        TD3 + Pendulum-v1
    """

    os.makedirs(PLOT_DIR, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    # --------------------------------------------------------
    # CartPole
    # --------------------------------------------------------

    for log_dir, label in log_dirs:

        if "CartPole-v1" not in label:
            continue

        x, y = load_q_values(log_dir)

        axes[0].plot(
            x,
            y,
            label=label,
        )

    axes[0].set_title("CartPole-v1")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Estimated Q-Value")
    axes[0].legend()


    # --------------------------------------------------------
    # Pendulum / Discrete Pendulum
    # --------------------------------------------------------

    for log_dir, label in log_dirs:

        if (
            "Pendulum-v1" not in label
            and "PendulumDiscrete-v1" not in label
        ):
            continue

        x, y = load_q_values(log_dir)

        axes[1].plot(
            x,
            y,
            label=label,
        )

    axes[1].set_title("Pendulum-v1 / PendulumDiscrete-v1")
    axes[1].set_xlabel("Timesteps")
    axes[1].set_ylabel("Estimated Q-Value")
    axes[1].legend()


    fig.suptitle(
        "Q-Value Evolution "
        "(split by environment)"
    )

    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/q_values_split.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(PLOT_DIR, exist_ok=True)

    # Build log directory + label pairs
    log_dirs = [
        (
            f"{LOG_DIR_BASE}/{name}_{env_id}",
            f"{name} ({env_id})",
        )
        for name, env_id in RUNS
    ]

    print("Generating visualizations...")

    # All runs together
    plot_reward_curves(log_dirs)
    plot_q_value_curves(log_dirs)

    # Split by environment
    plot_split_reward_curves(log_dirs)
    plot_split_q_value_curves(log_dirs)

    print(f"Plots saved to: {PLOT_DIR}/")


if __name__ == "__main__":
    main()