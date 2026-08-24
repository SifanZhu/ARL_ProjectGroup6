# environment.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiscretePendulum(gym.Wrapper):
    """
    Pendulum-v1 with a discretized action space.

    Discrete action i is mapped to an evenly spaced torque in [-2, 2].
    """

    def __init__(self, n_actions: int = 11):
        env = gym.make("Pendulum-v1")
        super().__init__(env)

        self.n_actions = n_actions

        self.action_map = np.linspace(
            self.env.action_space.low[0],
            self.env.action_space.high[0],
            n_actions,
        )

        self.action_space = spaces.Discrete(n_actions)

    def step(self, action):
        torque = self.action_map[int(action)]
        return self.env.step(np.array([torque], dtype=np.float32))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def make_env(env_id: str):
    """
    Create an environment used by the experiments.
    """

    if env_id == "CartPole-v1":
        return gym.make(env_id)

    if env_id == "Pendulum-v1":
        return gym.make(env_id)

    if env_id == "PendulumDiscrete-v1":
        return DiscretePendulum(n_actions=11)

    raise ValueError(f"Unknown environment: {env_id}")