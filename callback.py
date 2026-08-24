# callback.py

import os

import numpy as np
import torch

from stable_baselines3 import DQN, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback


class QValueLoggingCallback(BaseCallback):
    """
    Logs a scalar Q-function summary over a fixed set of reference states.

    DQN:
        mean_s max_a Q(s, a)

    SAC / TD3:
        mean_s Q(s, pi(s))
        averaged across critics
    """

    def __init__(
        self,
        reference_states,
        log_dir,
        log_freq=2000,
        verbose=0,
    ):
        super().__init__(verbose)

        self.reference_states = np.asarray(
            reference_states,
            dtype=np.float32,
        )

        self.log_dir = log_dir
        self.log_freq = log_freq

        self.timesteps = []
        self.q_values = []

    def _on_step(self) -> bool:

        if self.n_calls % self.log_freq == 0:

            q_value = self._estimate_q_value()

            self.timesteps.append(self.num_timesteps)
            self.q_values.append(q_value)

        return True

    def _estimate_q_value(self):

        obs_tensor, _ = self.model.policy.obs_to_tensor(
            self.reference_states
        )

        with torch.no_grad():

            if isinstance(self.model, DQN):

                q_values = self.model.q_net(obs_tensor)

                # Q(s, a) for every action
                # choose greedy action
                q_estimate = q_values.max(dim=1).values.mean()

            elif isinstance(self.model, (SAC, TD3)):

                actions, _ = self.model.predict(
                    self.reference_states,
                    deterministic=True,
                )

                actions_tensor = torch.as_tensor(
                    actions,
                    device=self.model.device,
                    dtype=torch.float32,
                )

                q_tuple = self.model.critic(
                    obs_tensor,
                    actions_tensor,
                )

                # Average over critics and states
                q_estimate = torch.cat(
                    q_tuple,
                    dim=1,
                ).mean()

            else:
                raise TypeError(
                    f"Unsupported model type: {type(self.model)}"
                )

        return q_estimate.item()

    def _on_training_end(self):

        os.makedirs(self.log_dir, exist_ok=True)

        np.savez(
            os.path.join(self.log_dir, "q_values.npz"),
            timesteps=np.asarray(self.timesteps),
            q_values=np.asarray(self.q_values),
        )


def load_q_values(log_dir):

    data = np.load(
        os.path.join(log_dir, "q_values.npz")
    )

    return data["timesteps"], data["q_values"]