"""
DQN on Pendulum-v1

Pendulum-v1 has a continuous action space (torque in [-2, 2]).
DQN requires a discrete action space, so a fixed, hard-coded uniform grid of
N_ACTIONS bins is used to discretize the continuous action space.
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
SEED  = 42
SEEDS = [42, 0, 1]   # seeds for multi-run evaluation
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Hyper-parameters
N_ACTIONS        = 11          # number of discrete torque bins
HIDDEN_SIZE      = 128
LR               = 1e-3         # stable learning, slightly slower convergence
GAMMA            = 0.99
BUFFER_SIZE      = 100_000      # maximum number of transitions to store in the replay buffer
BATCH_SIZE       = 64
EPS_START        = 1.0
EPS_END          = 0.05
EPS_DECAY        = 5000       # controls the rate of exponential decay of epsilon, higher means a slower decay
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
    # zeros out the future-reward term for terminal states
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
        targets = rewards + GAMMA * next_q * (1.0 - dones)
    # L1 loss (Huber loss) is less sensitive to outliers than MSE, which can help stabilize training
    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 10.0) # clip gradients to prevent exploding gradients
    optimizer.step()
    return loss.item()


def epsilon(step: int) -> float:
    # Exponential epsilon decay
    # Early on the agent explores randomly; over time it increasingly trusts its learned Q-values.
    return EPS_END + (EPS_START - EPS_END) * math.exp(-step / EPS_DECAY)


# The standard approach hard-codes a uniform grid of N_ACTIONS bins covering
# the action range [-2, 2].  The environment is used as-is; the agent maps
# each integer action index to a torque value before calling env.step().

def make_action_map(n: int = N_ACTIONS, low: float = -2.0, high: float = 2.0) -> np.ndarray:
    """Return an array of n evenly-spaced torque values in [low, high]."""
    return np.linspace(low, high, n)


def run_standard_dqn(n_episodes: int = N_EPISODES, n_actions: int = N_ACTIONS, seed: int = SEED):
    """Train a DQN agent on Pendulum-v1 using a hard-coded action grid."""
    print("\n" + "=" * 50)
    print(f"Standard DQN | seed={seed}")
    print("=" * 50)

    # set per-run seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Sets up the environment
    env = gym.make("Pendulum-v1")
    env.reset(seed=seed)

    # build the action map and initialize the Q-networks, optimizer, and replay buffer
    action_map = make_action_map(n_actions)
    obs_dim    = env.observation_space.shape[0]   

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
            torque = action_map[action] # map the discrete action index to a continuous torque value

            next_state, reward, terminated, truncated, _ = env.step([torque]) # pass the torque as a list to env.step()
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, float(done)) # store the transition in the replay buffer
            # perform a training step if we have enough transitions in the buffer
            if len(buffer) >= BATCH_SIZE:
                train_step(q_net, target_net, buffer, optimizer)

            state = next_state
            total_reward += reward
            step_count += 1

            if done:
                break

        episode_rewards.append(total_reward)

        if (ep + 1) % TARGET_UPDATE == 0: # update the target network to match the current Q-network
            target_net.load_state_dict(q_net.state_dict())

        if (ep + 1) % 50 == 0: # print the average reward over the last 50 episodes and the current epsilon value
            avg = np.mean(episode_rewards[-50:])
            print(f"  Episode {ep + 1:4d} | Avg reward (last 50): {avg:8.2f} | eps: {epsilon(step_count):.3f}")

    env.close()
    return episode_rewards, q_net, action_map


def run_multiple_seeds(
    seeds: list = SEEDS,
    n_episodes: int = N_EPISODES,
    n_actions: int = N_ACTIONS,
) -> tuple:
    """Run the DQN for each seed and return (all_rewards, best_q_net, action_map).
    
    all_rewards shape: (n_seeds, n_episodes)
    best_q_net: the trained network from the first seed (used for GIF recording)
    """
    all_rewards = []
    best_q_net      = None
    best_action_map = None
    best_score      = -float("inf")
    for s in seeds:
        rewards, q_net, action_map = run_standard_dqn(n_episodes=n_episodes, n_actions=n_actions, seed=s)
        all_rewards.append(rewards)
        score = np.mean(rewards[-50:])  # avg reward over last 50 episodes
        if score > best_score:
            best_score      = score
            best_q_net      = q_net
            best_action_map = action_map
    print(f"\nBest seed avg reward (last 50 eps): {best_score:.2f}")
    return np.array(all_rewards), best_q_net, best_action_map


# Plotting utility

# Smooth the rewards for better visualization
def smooth(values, window: int = 20) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_results(all_rewards: np.ndarray, window: int = 20):
    """Plot mean reward with min/max band across seeds.

    all_rewards: shape (n_seeds, n_episodes)
    """
    mean_r = np.mean(all_rewards, axis=0)
    min_r  = np.min(all_rewards, axis=0)
    max_r  = np.max(all_rewards, axis=0)
    n_seeds, n_episodes = all_rewards.shape
    episodes = np.arange(1, n_episodes + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    # faint individual seed traces
    for r in all_rewards:
        ax.plot(episodes, r, alpha=0.12, color="steelblue", linewidth=0.8)

    # min/max shaded band
    ax.fill_between(episodes, min_r, max_r, alpha=0.2, color="steelblue", label="min/max range")

    # smoothed mean
    if n_episodes >= window:
        smoothed_mean = smooth(mean_r, window)
        ax.plot(
            np.arange(window, n_episodes + 1),
            smoothed_mean,
            color="steelblue",
            linewidth=2,
            label=f"mean smoothed (w={window})",
        )
    else:
        ax.plot(episodes, mean_r, color="steelblue", linewidth=2, label="mean")

    ax.set_title(f"Standard DQN ({n_seeds} seeds)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle("DQN on Pendulum-v1", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("dqn_pendulum_results.png", dpi=150)
    print("\nPlot saved to dqn_pendulum_results.png")
    plt.show()


def record_gif(
    q_net: QNetwork,
    action_map: np.ndarray,
    n_actions: int = N_ACTIONS,
    seed: int = SEED,
    path: str = "dqn_pendulum.gif",
    fps: int = 30,
):
    """Run one greedy episode and save it as a GIF."""
    try:
        import imageio
    except ImportError:
        print("imageio not found. Install it with: pip install imageio")
        return

    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    state, _ = env.reset(seed=seed)
    frames = []

    for _ in range(MAX_STEPS):
        frames.append(env.render())
        action = select_action(q_net, state, eps=0.0, n_actions=n_actions)  # fully greedy
        torque = action_map[action]
        state, _, terminated, truncated, _ = env.step([torque])
        if terminated or truncated:
            break

    env.close()
    imageio.mimsave(path, frames, fps=fps)
    print(f"GIF saved to {path}")


# Entry point

if __name__ == "__main__":
    all_rewards, q_net, action_map = run_multiple_seeds(seeds=SEEDS, n_episodes=N_EPISODES, n_actions=N_ACTIONS)

    plot_results(all_rewards)

    # Final summary (mean over seeds, last 50 episodes)
    print("\n--- Final performance (last 50 episodes, mean across seeds) ---")
    print(f"  Standard DQN: {np.mean(all_rewards[:, -50:]):8.2f}")

    # Record a GIF of the trained agent (greedy policy, seed=42)
    record_gif(q_net, action_map, n_actions=N_ACTIONS, seed=SEED)
