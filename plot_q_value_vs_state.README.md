# plot_q_value_vs_state.py

Produces one Q-value-bias plot per trained agent (DQN/CartPole, DQN/Pendulum,
SAC/Pendulum, TD3/Pendulum), each showing:

- **Real Q-values** (scatter, blue/orange): the actual discounted return-to-go
  computed from real policy rollouts, plotted against a real state variable (cart
  position for CartPole, angular velocity theta_dot for Pendulum) and colored by
  which action bucket was taken.
- **Learned Q-function** (line, green/red): the network's own Q(s, a) prediction,
  swept over a grid of the same state variable with all other state dimensions
  fixed (e.g. pole angle = 0, or pendulum theta = 0), so it renders as a smooth
  curve instead of a noisy zigzag.

Action buckets per run:
- DQN/CartPole: "push left" vs "push right" (the 2 native discrete actions).
- DQN/Pendulum: "low torque bin" vs "high torque bin" (lowest/highest of the
  `DiscretizeActionWrapper`'s discretized torque bins).
- SAC/TD3 (Pendulum): "negative torque" vs "positive torque" (sign of the
  continuous action).

This is a visual, per-action-bucket companion to `estimate_bias_over_random_states` in
[Qvalue_bias.py](Qvalue_bias.py), which instead reduces the bias to a single scalar
per state.

## Requirements

Same environment used for [train.py](train.py) (see [requirements.txt](requirements.txt)):
`gymnasium`, `stable-baselines3`, `torch`, `matplotlib`, `numpy`.

## Usage

```bash
# Defaults: DQN models at 5M timesteps, SAC/TD3 models at 1M timesteps, seed 0
python plot_q_value_vs_state.py

# Override budgets / sample size / output folder
python plot_q_value_vs_state.py --dqn-steps 5000000 --sac-td3-steps 1000000 \
    --n-episodes 30 --out-dir plots
```

### CLI arguments

| Argument            | Default     | Meaning                                                        |
|---------------------|-------------|-----------------------------------------------------------------|
| `--models-dir`       | `models`    | Root folder containing `<algo>_<env>_steps<N>_seed<seed>/` runs |
| `--dqn-steps`        | `5000000`   | Timestep budget used to pick the DQN run folders (CartPole + Pendulum) |
| `--sac-td3-steps`    | `1000000`   | Timestep budget used to pick the SAC and TD3 run folders (Pendulum) |
| `--seed`             | `0`         | Seed suffix of the run folders to load                          |
| `--n-episodes`       | `20`        | Number of real rollout episodes sampled for the scatter points  |
| `--out-dir`          | `plots`     | Output folder for the generated PNGs                            |

## Output

Four PNG files are written to `--out-dir`:

- `dqn_cartpole_q_vs_cart_position.png`
- `dqn_pendulum_q_vs_angular_velocity.png`
- `sac_pendulum_q_vs_angular_velocity.png`
- `td3_pendulum_q_vs_angular_velocity.png`

## Model loading

Each run loads `models/<run_name>/final_model.zip` (the model saved at the end of
training, not `best_model.zip` or an intermediate checkpoint).

## Known limitation

The script always loads `final_model.zip` for the selected `--dqn-steps` /
`--sac-td3-steps` run folder; it does not currently support plotting a specific
snapshot from `models/<run_name>/checkpoints/model_<N>_steps.zip` (e.g. to reproduce
a "policy from iteration N" style progression across training).
