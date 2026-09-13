# ARL_ProjectGroup6

Trains DQN, SAC and TD3 on CartPole-v1 / Pendulum-v1
across several timestep budgets and seeds, then analyzes the resulting
Q-function estimates for systematic error and instability patterns.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+ (developed against `gymnasium==1.0.0`,
`stable-baselines3==2.4.1`).

## Training

`train.py` trains models and writes each run to its own folder under
`models/`, including checkpoints, the final/best model, and SB3's
`EvalCallback` monitor logs.

```bash
# All three algos, all three default timestep budgets, seed 0 (defaults):
python train.py

# Only DQN (on CartPole-v1 and Pendulum-v1), two budgets, three seeds:
python train.py --algo dqn --timesteps 10000 100000 --seeds 0 1 2

# SAC and TD3 on a different environment:
python train.py --algo sac td3 --env Pendulum-v1 --timesteps 100000 1000000
```

The checkpoints of the trained models are missing, as they take up too much space.

## Instability analysis (`instability_analysis/evaluate_models.py`)

Aims to answer "Can typical error/instability patterns be identified?" by
re-evaluating every trained model in `models/`:

- **Training-time eval curve stats**: from each run's `eval/evaluations.npz`.
- **Q-value bias**: Monte Carlo rollout comparing a model's own predicted
  Q-value against its actually achieved return (`Qvalue_bias.py`).
- **Independent re-evaluation**: checks whether an apparent best-vs-final
  collapse in the training-time eval curve survives re-evaluation on more
  episodes with a fresh seed.

### Needed Files

- `Qvalue_bias.py` and `train.py` at the repo root
- in default mode: `models/` at the repo root
- in `--plots-only` mode: `checkpoint_evaluation.csv` and `model_evaluation.csv`
  in `instability_analysis/results`

### How to run it

Run it from inside `instability_analysis/`, so its `results/` output lands
next to the script and `../models` correctly resolves to the repo-root
`models/` directory:

```bash
cd instability_analysis
python evaluate_models.py
```

This writes `results/model_evaluation.csv` and
`results/checkpoint_evaluation.csv`, plus a set of `results/finding_*.png`
plots.

To only regenerate the plots from already-computed CSVs (skips reloading
models / rerunning Monte Carlo rollouts / re-evaluation):

```bash
python evaluate_models.py --plots-only
```

## Folder "Q-value_VS_State"

- Used to produce a Q-value-bias plot per trained agent (DQN/CartPole, DQN/Pendulum,
SAC/Pendulum, TD3/Pendulum).

- Each agent is trained with 1M time steps over multiple training seeds
(`--seeds`, default `0 1 2 3 4`).

- Then the plot is created by averaging the result over 5 seeds. 

To run it, please read the plot_q_value-state.README.md.

## Folder "analysis"

### analysis/qvalue_bias.py

Core module for Q-value bias analysis. Contains:
- `get_q_value(...)`: Extracts Q-values from a trained model (DQN, SAC, TD3).
- `estimate_bias_over_random_states(...)`: Compares Q-value predictions vs.
  Monte Carlo returns across random start states (1 rollout per state).
- `estimate_return_at_state(...)`: Precise MC estimate from the same start state
  (multiple rollouts, for heatmap/grid analyses).

Imported by the other scripts, but can also be run directly (loads two example
models and prints the mean bias):

```bash
python -m analysis.qvalue_bias
```

---

### analysis/_plot_qvalue_bias_multiseed.py

Plots Q-value bias (prediction vs. actual return) for the final models, pooled
across multiple seeds. One panel per algorithm, points colored by seed. Only
loads already-trained `final_model.zip` files, no retraining.

```bash
python -m analysis._plot_qvalue_bias_multiseed
python -m analysis._plot_qvalue_bias_multiseed --algos dqn --dqn-env-id CartPole-v1
python -m analysis._plot_qvalue_bias_multiseed --algos sac td3
python -m analysis._plot_qvalue_bias_multiseed --n-states 100 --out qvalue_bias_1M.png
```

---

### analysis/plot_training_curves_avg.py

Plots training reward curves (from SB3's `monitor.csv`), averaged across all
seeds ± standard deviation as a band. One combined plot for all selected
algorithms, including reference lines for min/max achievable reward.

```bash
python -m analysis.plot_training_curves_avg
python -m analysis.plot_training_curves_avg --models-dir models --timesteps 1000000
python -m analysis.plot_training_curves_avg --algos dqn
python -m analysis.plot_training_curves_avg --dqn-env-id CartPole-v1
```

---

### analysis/plot_bias_performance_over_training.py

Time-series version of `_plot_qvalue_bias_multiseed.py`: instead of only the
final model, this script evaluates **all saved checkpoints** per seed.
Produces a 2-panel plot per algorithm (Q-value bias on the left, policy
performance on the right) over the course of training, averaged across seeds.

⚠️ Runtime-heavy: for every algo × seed × checkpoint, `n-episodes` rollouts are
run. With many checkpoints/seeds, reduce `--n-episodes` or `--max-checkpoints`.

```bash
python -m analysis.plot_bias_performance_over_training
python -m analysis.plot_bias_performance_over_training --algos dqn
python -m analysis.plot_bias_performance_over_training --algos dqn --dqn-env-id CartPole-v1
python -m analysis.plot_bias_performance_over_training --n-episodes 20 --max-checkpoints 15
```

---

**Note:** All scripts are run from the project root (`python -m analysis.<name>`), not directly from inside the `analysis` folder, since they import `train.py` from the root.

## Q-Function Temporal Analysis
`Q_Function_Evolution/TemporalAnalysis.py` produces the evolution of Q-function plots of our models under `models/` using their checkpoints.
- `Plots/separate` contains plots of all available models.
- `Plots/mean` contains the "means-over-seeds" plots.

