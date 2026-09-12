# plot_q_value_vs_state.py

Produces one Q-value-bias plot per trained agent (DQN/CartPole, DQN/Pendulum,
SAC/Pendulum, TD3/Pendulum), each averaged over multiple training seeds
(`--seeds`, default `0 1 2 3 4`) and showing:

- **Real Q-values** (error bars, blue/orange): the actual discounted return-to-go
  MC computed from real policy rollouts. Per seed, samples are binned by a real
  state variable (cart position for CartPole, angular velocity theta_dot for
  Pendulum) and averaged within each bin; the plotted point/error bar is the
  mean +/- std of those per-seed bin means across all seeds, colored by which
  action bucket was taken.
- **Learned Q-function** (line + shaded band, green/red): the network's own
  Q(s, a) prediction, swept over a grid of the same state variable with all
  other state dimensions fixed (e.g. pole angle = 0, or pendulum theta = 0), so
  it renders as a smooth curve instead of a noisy zigzag. The line is the mean
  over seeds, the shaded band is +/- 1 std.

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
# Defaults: DQN models, SAC/TD3 models at 1M timesteps, seeds 0-4
python plot_q_value_vs_state.py

# Override budgets / sample size / output folder / seeds
python plot_q_value_vs_state.py --dqn-steps 1000000 --sac-td3-steps 1000000 \
    --n-episodes 20 --out-dir plots --seeds 0 1 2 3 4

# Also plot one PNG per DQN/CartPole checkpoint found in checkpoints/model_<N>_steps.zip
# (averaged across the same --seeds as the final_model plot)
python plot_q_value_vs_state.py --dqn-cartpole-all-checkpoints

# Only specific checkpoints of the chosen DQN/CartPole run (steps must match an existing model_<N>_steps.zip)
python plot_q_value_vs_state.py --dqn-cartpole-checkpoints 100000 500000 1000000

# Only the checkpoint sweep, skip the final_model plot (flag name kept for backward compatibility)
python plot_q_value_vs_state.py --dqn-cartpole-all-checkpoints --dqn-cartpole-skip-final-model
```

### CLI arguments

| Argument                          | Default         | Meaning                                                        |
|-----------------------------------|-----------------|-----------------------------------------------------------------|
| `--models-dir`                    | `models`        | Root folder containing `<algo>_<env>_steps<N>_seed<seed>/` runs |
| `--dqn-steps`                     | `1000000`       | Timestep budget used to pick the DQN run folders (CartPole + Pendulum) |
| `--sac-td3-steps`                 | `1000000`       | Timestep budget used to pick the SAC and TD3 run folders (Pendulum) |
| `--seeds`                         | `0 1 2 3 4`     | Seed suffixes of the run folders to load and average over        |
| `--n-episodes`                    | `20`            | Number of real rollout episodes sampled per model/seed           |
| `--out-dir`                       | `plots`         | Output folder for the generated PNGs                            |
| `--dqn-cartpole-all-checkpoints`  | off             | Also plot every checkpoint in the DQN/CartPole runs' `checkpoints/` folders |
| `--dqn-cartpole-checkpoints N...` | none            | Plot only these specific DQN/CartPole checkpoint timesteps (overrides `--dqn-cartpole-all-checkpoints`) |
| `--dqn-cartpole-skip-final-model` | off             | Skip the non-checkpoint `final_model.zip` plot for DQN/CartPole   |

Each run folder actually used is `models/<run>_seed<seed>/` for every `seed` in
`--seeds`, e.g. with the defaults: `dqn_CartPole-v1_steps1000000_seed0` ...
`dqn_CartPole-v1_steps1000000_seed4`. All of these must exist on disk.

## Output

By default, four PNG files are written to `--out-dir`:

- `dqn_cartpole_q_vs_cart_position.png`
- `dqn_pendulum_q_vs_angular_velocity.png`
- `sac_pendulum_q_vs_angular_velocity.png`
- `td3_pendulum_q_vs_angular_velocity.png`

If `--dqn-cartpole-all-checkpoints` or `--dqn-cartpole-checkpoints` is used, one
additional `dqn_cartpole_q_vs_cart_position_step<N>.png` is written per selected
checkpoint.

## Model loading

For every seed in `--seeds`, DQN/Pendulum, SAC, and TD3 always load
`models/<run_name>_seed<seed>/best_model.zip` (the snapshot with the highest
`EvalCallback` score seen during training, not `final_model.zip` or an
intermediate checkpoint). The resulting real-Q bin means and learned-Q curves
are then averaged across seeds before plotting (see "Multi-seed averaging"
below).

DQN/CartPole loads `final_model.zip` per seed by default too, but can
additionally (or instead) load snapshots from
`models/<run_name>_seed<seed>/checkpoints/model_<N>_steps.zip` via
`--dqn-cartpole-all-checkpoints` / `--dqn-cartpole-checkpoints` /
`--dqn-cartpole-skip-final-model`, e.g. to reproduce a "policy from iteration N"
style progression across training -- each checkpoint plot is itself averaged
over all `--seeds`. Checkpoint files under `checkpoints/` are periodic,
unevaluated training snapshots (`CheckpointCallback`), independent of
`final_model.zip`.

When `--dqn-cartpole-all-checkpoints` is used, only checkpoint steps present in
**every** selected seed's `checkpoints/` folder are plotted
(`discover_common_checkpoint_steps`) -- this matters because seeds can have
been trained with a different `checkpoint_freq` (e.g. seed 0 at 10k vs. seeds
1-4 at 50k steps), so not every seed has a file for every step.

## Multi-seed averaging

For each model type, one model is loaded per seed and rolled out independently
(`--n-episodes` episodes each). Because different seeds visit different real
states, the per-seed real-Q samples are first binned into 20 equal-width bins
over the combined x-range (`real_q_bin_means`), separately per action bucket.
The learned Q-function is evaluated on a shared grid (same x-range) so it's
directly comparable across seeds. Both the binned real-Q means and the learned
curves are then reduced to a per-bin/per-grid-point mean and std across seeds
(`nanmean_std`) -- bins with no samples for a given seed are treated as NaN and
excluded from that bin's average instead of skewing it toward zero.

## Known limitation

Checkpoint selection (`--dqn-cartpole-all-checkpoints` /
`--dqn-cartpole-checkpoints`) is currently only implemented for DQN/CartPole;
DQN/Pendulum, SAC, and TD3 still always load `final_model.zip`.
