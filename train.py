"""
training/train.py

Modulares Trainings-Skript für das RL-Praktikum: trainiert DQN/SAC/TD3 auf
konfigurierbaren Environments für unterschiedliche Timestep-Budgets und Seeds,
und speichert jedes Modell in einer eindeutigen Ordnerstruktur ab.

CLI-Nutzung
-----------
    # Alle drei Algos, alle drei Timestep-Budgets, 1 Seed (Defaults):
    python train.py

    # Nur DQN, zwei Budgets, drei Seeds:
    python train.py --algo dqn --timesteps 10000 100000 --seeds 0 1 2

    # SAC und TD3 auf einem anderen Environment:
    python train.py --algo sac td3 --env Pendulum-v1 --timesteps 100000 1000000

Als Modul (z.B. aus einem Analyse-/Notebook-Skript)
-----------------------------------------------------
    from training.train import TrainConfig, train_model, train_batch

    cfg = TrainConfig(algo="dqn", env_id="CartPole-v1", timesteps=100_000, seed=0)
    model_path = train_model(cfg)

    # oder gleich mehrere Kombinationen:
    results = train_batch(algos=["dqn"], timesteps_list=[10_000, 100_000], seeds=[0, 1])
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
from stable_baselines3 import DQN, SAC, TD3
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

ALGOS = {"dqn": DQN, "sac": SAC, "td3": TD3}
DEFAULT_ENVS = {"dqn": "CartPole-v1", "sac": "Pendulum-v1", "td3": "Pendulum-v1"}


@dataclass
class TrainConfig:
    algo: str                      # "dqn" | "sac" | "td3"
    env_id: str                    # z.B. "CartPole-v1"
    timesteps: int                 # Trainings-Budget in Timesteps
    seed: int = 0
    out_dir: str = "models"
    checkpoint_freq: int = 10_000  # alle N Steps ein Checkpoint (für Q-Funktions-Zeitverlauf, Thema 6)
    eval_freq: int = 2_000
    n_eval_episodes: int = 5
    policy: str = "MlpPolicy"
    algo_kwargs: Optional[dict] = None  # zusätzliche Kwargs an den SB3-Konstruktor (z.B. gamma, lr)

    @property
    def run_name(self) -> str:
        return f"{self.algo}_{self.env_id}_steps{self.timesteps}_seed{self.seed}"

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name


def make_env(env_id: str, log_dir: Path, seed: int) -> Monitor:
    env = gym.make(env_id)
    env.reset(seed=seed)
    env = Monitor(env, str(log_dir))
    return env


def train_model(cfg: TrainConfig) -> Path:
    """Trainiert ein einzelnes Modell gemäß cfg und speichert alles unter cfg.run_dir:

        models/<run_name>/
            final_model.zip
            best_model.zip          (von EvalCallback, falls besser als final)
            config.json              (die verwendete TrainConfig, für Reproduzierbarkeit)
            monitor/                 (Trainings-Episoden-Logs)
            eval/                    (Evaluations-Logs)
            checkpoints/              (Zwischenstände, z.B. model_10000_steps.zip)

    Rückgabe: Pfad zum finalen Modell (ohne .zip-Endung, SB3-Konvention für .load()).
    """
    if cfg.algo not in ALGOS:
        raise ValueError(f"Unbekannter Algo '{cfg.algo}'. Erlaubt: {list(ALGOS)}")

    algo_cls = ALGOS[cfg.algo]
    run_dir = cfg.run_dir
    (run_dir / "monitor").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_env = make_env(cfg.env_id, run_dir / "monitor", cfg.seed)
    eval_env = make_env(cfg.env_id, run_dir / "eval", cfg.seed + 1000)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir / "eval"),
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.checkpoint_freq,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="model",
    )

    algo_kwargs = cfg.algo_kwargs or {}
    model = algo_cls(cfg.policy, train_env, seed=cfg.seed, verbose=0, **algo_kwargs)
    model.learn(
        total_timesteps=cfg.timesteps,
        callback=CallbackList([eval_callback, checkpoint_callback]),
    )

    final_model_path = run_dir / "final_model"
    model.save(str(final_model_path))

    # Config mitspeichern -> spätere Analyse-Skripte wissen, mit welchen Settings
    # (Env, Algo, Timesteps, Seed) jedes Modell trainiert wurde, ohne Namen zu parsen.
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_env.close()
    eval_env.close()

    return final_model_path


def train_batch(
    algos: List[str],
    timesteps_list: List[int],
    seeds: List[int],
    env_overrides: Optional[Dict[str, str]] = None,
    out_dir: str = "models",
) -> List[Tuple[TrainConfig, Path]]:
    """Trainiert alle Kombinationen aus algos x timesteps_list x seeds.

    env_overrides: optionales dict {algo: env_id}; sonst wird DEFAULT_ENVS verwendet.
    Gibt eine Liste (TrainConfig, model_path) für jeden Run zurück.
    """
    env_overrides = env_overrides or {}
    results = []
    for algo in algos:
        env_id = env_overrides.get(algo, DEFAULT_ENVS[algo])
        for timesteps in timesteps_list:
            for seed in seeds:
                cfg = TrainConfig(
                    algo=algo, env_id=env_id, timesteps=timesteps, seed=seed, out_dir=out_dir
                )
                print(f"[train] {cfg.run_name} ...")
                model_path = train_model(cfg)
                results.append((cfg, model_path))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trainiert DQN/SAC/TD3-Modelle für das RL-Praktikum (Q-Funktions-Analyse)."
    )
    parser.add_argument(
        "--algo", nargs="+", choices=list(ALGOS) + ["all"], default=["all"],
        help="Welche Algorithmen trainiert werden sollen (mehrere möglich, oder 'all').",
    )
    parser.add_argument(
        "--env", type=str, default=None,
        help="Environment-ID; überschreibt für ALLE gewählten Algos die Default-Envs.",
    )
    parser.add_argument(
        "--timesteps", type=int, nargs="+", default=[10_000, 100_000, 1_000_000],
        help="Liste der Timestep-Budgets, z.B. --timesteps 10000 100000 1000000",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0],
        help="Liste der Seeds, z.B. --seeds 0 1 2",
    )
    parser.add_argument("--out-dir", type=str, default="models", help="Ausgabeverzeichnis.")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--eval-freq", type=int, default=2_000)
    return parser.parse_args()


def main():
    args = parse_args()
    algos = list(ALGOS) if "all" in args.algo else args.algo
    env_overrides = {a: args.env for a in algos} if args.env else None

    train_batch(
        algos=algos,
        timesteps_list=args.timesteps,
        seeds=args.seeds,
        env_overrides=env_overrides,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()