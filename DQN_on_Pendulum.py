"""
DQN on Pendulum-v1
==================
Pendulum-v1 has a continuous action space (torque in [-2, 2]).
DQN requires a discrete action space, so two approaches are compared:

  1. Standard DQN: a fixed, hard-coded uniform grid of N_ACTIONS bins.
  2. Explicit Discretization DQN: a reusable ActionDiscretizer class that
     we can choose the number of bins (or supply custom breakpoints) and
     wraps any Gymnasium environment with a continuous 1-D action space.

Both agents share the same neural-network architecture and training loop.
The only difference is: How the continuous action is mapped to discrete bins.
"""

import random
import math
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Hyper-parameters
N_ACTIONS        = 11          # number of discrete torque bins
HIDDEN_SIZE      = 128
LR               = 1e-3         # stable learning, slightly slower convergence
GAMMA            = 0.99
BUFFER_SIZE      = 50_000
BATCH_SIZE       = 64
EPS_START        = 1.0
EPS_END          = 0.05
EPS_DECAY        = 500         # controls the rate of exponential decay of epsilon, higher means a slower decay
TARGET_UPDATE    = 10          # episodes between target-network syncs
N_EPISODES       = 300
MAX_STEPS        = 200         # Pendulum truncates at 200 steps

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# Shared components

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.memory = deque([],maxlen=capacity)

    # save a transition to the buffer
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    """Simple MLP Q-network."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = HIDDEN_SIZE):
        super().__init__()
        # encapsulates the network layers in a single nn.Sequential module
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    # call the network to get Q-values for all actions given a state: 
    # linear -> ReLU -> linear -> ReLU -> linear
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_action(q_net: QNetwork, state: np.ndarray, eps: float, n_actions: int) -> int:
    # with probability eps, select a random action (exploration)
    if random.random() < eps:
        return random.randrange(n_actions)
    
    # otherwise select the action with the highest Q-value (exploitation)
    with torch.no_grad(): # no gradients needed during action selection
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0) # convert state to tensor and add batch dimension
        return int(q_net(s).argmax(dim=1).item())


def train_step(
    q_net: QNetwork,
    target_net: QNetwork,
    memory: ReplayBuffer,
    optimizer: optim.Optimizer,
) -> float: # returns the loss value for logging
    # Don't train until we have at least 64 transitions stored
    if len(memory) < BATCH_SIZE:
        return 0.0

    transitions = memory.sample(BATCH_SIZE) # randomly sample a batch of transitions (64) from the replay buffer
    batch = Transition(*zip(*transitions)) # convert a list of 64 transitions (s, a, r, s', done) into one transition of 64 states, 64 actions, etc.

    states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(batch.action, dtype=torch.long, device=DEVICE).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE).unsqueeze(1) # terminal states are marked with 1.0, non-terminal with 0.0

    q_values = q_net(states).gather(1, actions)
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
        targets = rewards + GAMMA * next_q * (1.0 - dones)

    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()
    return loss.item()


def epsilon(step: int) -> float:
    # Exponential epsilon decay
    # Early on the agent explores randomly; over time it increasingly trusts its learned Q-values.
    return EPS_END + (EPS_START - EPS_END) * math.exp(-step / EPS_DECAY)


# ===========================================================================
# Approach 1 – Standard DQN
# ===========================================================================
# The standard approach hard-codes a uniform grid of N_ACTIONS bins covering
# the action range [-2, 2].  The environment is used as-is; the agent maps
# each integer action index to a torque value before calling env.step().

def make_action_map(n: int = N_ACTIONS, low: float = -2.0, high: float = 2.0) -> np.ndarray:
    """Return an array of n evenly-spaced torque values in [low, high]."""
    return np.linspace(low, high, n)


def run_standard_dqn(n_episodes: int = N_EPISODES, n_actions: int = N_ACTIONS):
    """Train a DQN agent on Pendulum-v1 using a hard-coded action grid."""
    print("\n" + "=" * 60)
    print("Approach 1: Standard DQN (hard-coded uniform discretization)")
    print("=" * 60)

    env = gym.make("Pendulum-v1")
    env.reset(seed=SEED)

    action_map = make_action_map(n_actions)
    obs_dim    = env.observation_space.shape[0]   # 3

    q_net      = QNetwork(obs_dim, n_actions).to(DEVICE)
    target_net = QNetwork(obs_dim, n_actions).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer    = ReplayBuffer(BUFFER_SIZE)

    episode_rewards = []
    step_count = 0

    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            eps    = epsilon(step_count)
            action = select_action(q_net, state, eps, n_actions)
            torque = action_map[action]

            next_state, reward, terminated, truncated, _ = env.step([torque])
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, float(done))
            train_step(q_net, target_net, buffer, optimizer)

            state        = next_state
            total_reward += reward
            step_count  += 1

            if done:
                break

        episode_rewards.append(total_reward)

        if (ep + 1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

        if (ep + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"  Episode {ep + 1:4d} | Avg reward (last 50): {avg:8.2f} | eps: {epsilon(step_count):.3f}")

    env.close()
    return episode_rewards


# ===========================================================================
# Approach 2 – DQN with Explicit Action-Space Discretization
# ===========================================================================
# Here we introduce a proper ActionDiscretizer wrapper that:
#   • inspects the environment's action_space at runtime,
#   • accepts either a number of uniform bins or an explicit array of breakpoints,
#   • wraps env.step() so the agent always works with integer actions,
#   • exposes the discrete action count via a standard `n` attribute.
#
# This makes the discretization reusable and easy to swap without touching
# the agent code.

class ActionDiscretizer(gym.Wrapper):
    """
    Gymnasium wrapper that discretizes a 1-D Box action space.

    Parameters
    ----------
    env       : Gymnasium environment with a 1-D continuous action space.
    n_bins    : Number of evenly-spaced bins.  Ignored if `breakpoints` given.
    breakpoints : Explicit 1-D array of action values to use as bins.
    """

    def __init__(self, env: gym.Env, n_bins: int = N_ACTIONS, breakpoints: np.ndarray = None):
        super().__init__(env)

        assert isinstance(env.action_space, gym.spaces.Box), (
            "ActionDiscretizer only supports Box action spaces."
        )
        assert env.action_space.shape == (1,), (
            "ActionDiscretizer currently supports 1-D action spaces."
        )

        if breakpoints is not None:
            self._bins = np.asarray(breakpoints, dtype=np.float32)
        else:
            low  = float(env.action_space.low[0])
            high = float(env.action_space.high[0])
            self._bins = np.linspace(low, high, n_bins, dtype=np.float32)

        self.action_space = gym.spaces.Discrete(len(self._bins))

    @property
    def n_actions(self) -> int:
        return len(self._bins)

    def step(self, action: int):
        continuous_action = [float(self._bins[action])]
        return self.env.step(continuous_action)

    def action_meanings(self) -> dict:
        return {i: round(float(v), 4) for i, v in enumerate(self._bins)}


def run_explicit_discretization_dqn(
    n_episodes: int = N_EPISODES,
    n_bins: int = N_ACTIONS,
    breakpoints: np.ndarray = None,
):
    """Train a DQN agent on Pendulum-v1 via the ActionDiscretizer wrapper."""
    print("\n" + "=" * 60)
    print("Approach 2: DQN with Explicit ActionDiscretizer wrapper")
    print("=" * 60)

    base_env = gym.make("Pendulum-v1")
    base_env.reset(seed=SEED)
    env      = ActionDiscretizer(base_env, n_bins=n_bins, breakpoints=breakpoints)

    print(f"  Action meanings: {env.action_meanings()}")

    n_actions = env.n_actions
    obs_dim   = env.observation_space.shape[0]   # 3

    q_net      = QNetwork(obs_dim, n_actions).to(DEVICE)
    target_net = QNetwork(obs_dim, n_actions).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer    = ReplayBuffer(BUFFER_SIZE)

    episode_rewards = []
    step_count = 0

    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            eps    = epsilon(step_count)
            action = select_action(q_net, state, eps, n_actions)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, float(done))
            train_step(q_net, target_net, buffer, optimizer)

            state        = next_state
            total_reward += reward
            step_count  += 1

            if done:
                break

        episode_rewards.append(total_reward)

        if (ep + 1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

        if (ep + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"  Episode {ep + 1:4d} | Avg reward (last 50): {avg:8.2f} | eps: {epsilon(step_count):.3f}")

    env.close()
    return episode_rewards


# ===========================================================================
# Plotting utility
# ===========================================================================

def smooth(values, window: int = 20) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_results(rewards_std: list, rewards_disc: list, window: int = 20):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, rewards, title in zip(
        axes,
        [rewards_std, rewards_disc],
        ["Standard DQN", "Explicit Discretization DQN"],
    ):
        episodes = np.arange(1, len(rewards) + 1)
        ax.plot(episodes, rewards, alpha=0.3, color="steelblue", label="raw")
        if len(rewards) >= window:
            smoothed = smooth(rewards, window)
            ax.plot(
                np.arange(window, len(rewards) + 1),
                smoothed,
                color="steelblue",
                linewidth=2,
                label=f"smoothed (w={window})",
            )
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("DQN on Pendulum-v1", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("dqn_pendulum_results.png", dpi=150)
    print("\nPlot saved to dqn_pendulum_results.png")
    plt.show()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    rewards_standard = run_standard_dqn(n_episodes=N_EPISODES, n_actions=N_ACTIONS)

    # Explicit discretization with the same 11 uniform bins (easy to swap)
    rewards_explicit = run_explicit_discretization_dqn(
        n_episodes=N_EPISODES,
        n_bins=N_ACTIONS,
        # To use custom breakpoints instead, uncomment and edit:
        # breakpoints=np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]),
    )

    plot_results(rewards_standard, rewards_explicit)

    # Final summary
    print("\n--- Final performance (last 50 episodes) ---")
    print(f"  Standard DQN:              {np.mean(rewards_standard[-50:]):8.2f}")
    print(f"  Explicit Discretization:   {np.mean(rewards_explicit[-50:]):8.2f}")
