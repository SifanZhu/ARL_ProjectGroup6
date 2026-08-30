"""
analysis/qvalue_bias.py

Kombiniert zwei zusammengehörige Bausteine der Q-Funktions-Analyse:

  1. Q-Wert-Extraktion aus einem trainierten Modell (get_q_value)
     -> algo-agnostisch: funktioniert für DQN (diskret, z.B. CartPole oder Pendulum
        mit DiscretizeActionWrapper) UND SAC/TD3 (kontinuierlich, inkl. Twin-Critics
        Q1/Q2/min).

  2. Monte-Carlo-Schätzung des "wahren" Returns (rollout_return,
     estimate_bias_over_random_states) -> um die Netz-Vorhersage (1) gegen die
     tatsächliche Performance zu prüfen (Über-/Unterschätzung).

Zwei unterschiedliche MC-Anwendungsfälle sind hier bewusst getrennt:

  - estimate_return_at_state(...):
        N Wiederholungen AB DEMSELBEN State -> ein präziser, rauscharmer
        Ground-Truth-Wert für GENAU EINEN Punkt im Zustandsraum.
        Das ist der Baustein für spätere Grid-/Heatmap-Analysen
        (z.B. x=Winkel, feste Winkelgeschwindigkeit).

  - estimate_bias_over_random_states(...):
        1 Rollout pro ZUFÄLLIGEM Start-State, über N verschiedene States.
        Das ist der direkte, algo-agnostische Nachfolger von
        `estimate_q_bias_dqn` aus dem ursprünglichen Skript: liefert eine
        Verteilung von Bias-Werten über typische Start-Zustände (günstig,
        weil nur 1 statt N Rollouts pro State nötig sind).

Nutzung
-------
    from analysis.qvalue_bias import estimate_bias_over_random_states
    from train import DiscretizeActionWrapper

    # DQN auf CartPole (natives diskretes Action-Space):
    result = estimate_bias_over_random_states(model, env, n_states=50)

    # DQN auf Pendulum (kontinuierliches Action-Space via DiscretizeActionWrapper):
    result = estimate_bias_over_random_states(
        model_p, DiscretizeActionWrapper(gym.make("Pendulum-v1")), n_states=50
    )
    print(result["bias"].mean(), result["bias"].std())
"""

from typing import Callable, Dict, Optional, Union

import numpy as np
import torch as th
from stable_baselines3 import DQN, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

DISCRETE_ALGOS = (DQN,)
CONTINUOUS_ALGOS = (SAC, TD3)

# Ein reset_fn bringt die Env in einen gewünschten State und gibt die Start-Obs zurück.
# Signatur: reset_fn(env) -> obs (np.ndarray). Env-spezifisch (z.B. CartPole vs Pendulum
# haben unterschiedliche interne State-Repräsentationen) -> lebt NICHT in diesem Modul,
# wird von außen übergeben (z.B. aus envs/state_setters.py, sobald wir die Grid-Analyse
# angehen).


# ---------------------------------------------------------------------------
# 1. Q-Wert-Extraktion
# ---------------------------------------------------------------------------

@th.no_grad()
def get_q_value(
    model: BaseAlgorithm,
    obs: np.ndarray,
    action: Optional[np.ndarray] = None,
) -> Union[np.ndarray, float, Dict[str, float]]:
    """Gibt den/die Q-Wert(e) für einen State (und ggf. eine Aktion) zurück.

    DQN (diskrete Aktionen):
        action=None -> Q-Werte für ALLE Aktionen, np.ndarray shape (n_actions,)
        action=a    -> skalarer Q-Wert Q(s,a) als float

    SAC/TD3 (kontinuierliche Aktionen, Twin-Critics):
        action=None -> es wird die Aktion genommen, die die Policy deterministisch
                        wählen würde (model.predict)
        action=a    -> Q(s,a) für genau diese Aktion
        Rückgabe immer ein dict: {"q_values": [Q1, Q2], "min": ..., "mean": ...}
        "min" ist der Wert, den SB3 intern auch für den Trainings-Zielwert nutzt.

    obs: 1D-Array (ein einzelner State, kein Batch).
    """
    obs_batch = obs[np.newaxis, :] if obs.ndim == 1 else obs

    if isinstance(model, DISCRETE_ALGOS):
        obs_tensor, _ = model.policy.obs_to_tensor(obs_batch)
        q_values = model.q_net(obs_tensor).cpu().numpy()[0]  # shape (n_actions,)
        if action is None:
            return q_values
        return float(q_values[int(action)])

    elif isinstance(model, CONTINUOUS_ALGOS):
        if action is None:
            action, _ = model.predict(obs_batch, deterministic=True)

        action_batch = np.asarray(action, dtype=np.float32)
        if action_batch.ndim == 1:
            action_batch = action_batch[np.newaxis, :]

        obs_tensor, _ = model.policy.obs_to_tensor(obs_batch)
        action_tensor = th.as_tensor(action_batch, dtype=th.float32, device=model.device)

        critic_outputs = model.critic(obs_tensor, action_tensor)  # tuple, 1 Tensor pro Critic-Netz
        q_list = [float(q.cpu().numpy()[0, 0]) for q in critic_outputs]

        return {
            "q_values": q_list,          # z.B. [Q1, Q2]
            "min": min(q_list),           # das, was SB3 intern für den Zielwert nutzt
            "mean": float(np.mean(q_list)),
        }

    else:
        raise TypeError(
            f"Nicht unterstützter Modelltyp: {type(model)}. "
            f"Erwartet DQN, SAC oder TD3."
        )


# ---------------------------------------------------------------------------
# 2. Monte-Carlo-Return
# ---------------------------------------------------------------------------

def rollout_return(
    model: BaseAlgorithm,
    env,
    gamma: float,
    deterministic: bool = True,
    max_steps: int = 1000,
    reset_fn: Optional[Callable] = None,
):
    """Führt EINE Episode aus und gibt (start_obs, start_action, diskontierter_return) zurück.

    reset_fn: optionaler Callback reset_fn(env) -> obs, um die Env in einen bestimmten
    State zu versetzen statt eines zufälligen env.reset(). Ohne reset_fn: normaler
    zufälliger Reset.
    """
    if reset_fn is not None:
        obs = reset_fn(env)
    else:
        obs, _ = env.reset()

    start_obs = obs.copy()
    start_action, _ = model.predict(start_obs, deterministic=deterministic)

    G, discount, steps, done = 0.0, 1.0, 0, False
    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        G += discount * reward
        discount *= gamma
        done = terminated or truncated
        steps += 1

    return start_obs, start_action, G


def estimate_return_at_state(
    model: BaseAlgorithm,
    env,
    gamma: float,
    reset_fn: Callable,
    n_episodes: int = 50,
    deterministic: bool = True,
    max_steps: int = 1000,
) -> Dict:
    """Präzise MC-Schätzung des Returns AB DEMSELBEN State (via reset_fn), gemittelt
    über n_episodes Wiederholungen. Für Grid-/Heatmap-Analysen (ein Punkt im
    Zustandsraum, geringes Rauschen gewünscht).
    """
    returns = []
    start_obs_ref, start_action_ref = None, None
    for _ in range(n_episodes):
        start_obs, start_action, G = rollout_return(
            model, env, gamma, deterministic, max_steps, reset_fn
        )
        returns.append(G)
        if start_obs_ref is None:
            start_obs_ref, start_action_ref = start_obs, start_action

    returns = np.array(returns)
    return {
        "start_obs": start_obs_ref,
        "start_action": start_action_ref,
        "returns": returns,
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "n_episodes": n_episodes,
    }


# ---------------------------------------------------------------------------
# 3. Bias über viele (zufällige) States -- Nachfolger von estimate_q_bias_dqn
# ---------------------------------------------------------------------------

def estimate_bias_over_random_states(
    model: BaseAlgorithm,
    env,
    n_states: int = 50,
    deterministic: bool = True,
    max_steps: int = 1000,
) -> Dict:
    """Vergleicht für n_states zufällige Episoden-Starts jeweils den vorhergesagten
    Q-Wert mit dem tatsächlich erzielten (einzelnen) Return -- algo-agnostisch
    (DQN, SAC, TD3). Direkter Nachfolger von `estimate_q_bias_dqn` im alten Skript.

    Rückgabe (dict):
        q_preds: np.ndarray, ein Q-Wert pro State (bei SAC/TD3: der "min"-Wert)
        mc_returns: np.ndarray, ein MC-Return pro State
        bias: np.ndarray, q_preds - mc_returns (+ = Überschätzung, - = Unterschätzung)
        raw_q: Liste der vollen get_q_value()-Rückgaben (bei SAC/TD3 inkl. Q1/Q2 einzeln)
    """
    gamma = model.gamma
    q_preds, mc_returns, raw_q = [], [], []

    for _ in range(n_states):
        start_obs, start_action, G = rollout_return(model, env, gamma, deterministic, max_steps)
        q = get_q_value(model, start_obs, start_action)

        if isinstance(model, DISCRETE_ALGOS):
            q_pred = q  # get_q_value gibt hier direkt den Skalar zurück (action übergeben)
        else:
            q_pred = q["min"]  # konsistent mit dem, was SB3 intern fürs Training nutzt

        q_preds.append(q_pred)
        mc_returns.append(G)
        raw_q.append(q)

    q_preds, mc_returns = np.array(q_preds), np.array(mc_returns)
    bias = q_preds - mc_returns

    return {
        "q_preds": q_preds,
        "mc_returns": mc_returns,
        "bias": bias,
        "raw_q": raw_q,
        "n_states": n_states,
    }


if __name__ == "__main__":
    # Kleiner Selbsttest / Beispiel-Aufruf, analog zum alten Skript.
    import gymnasium as gym
    from train import DiscretizeActionWrapper

    model = DQN.load("models/dqn_CartPole-v1_steps10000_seed0/final_model")
    eval_env = gym.make("CartPole-v1")

    result = estimate_bias_over_random_states(model, eval_env, n_states=50)
    print(
        f"DQN CartPole: mean bias = {result['bias'].mean():.3f} "
        f"(+ = Überschätzung, - = Unterschätzung), std = {result['bias'].std():.3f}"
    )
    eval_env.close()

    # DiscretizeActionWrapper maps discrete action indices back to continuous torques
    model_p = DQN.load("models/dqn_Pendulum-v1_steps10000_seed0/final_model")
    eval_env_p = DiscretizeActionWrapper(gym.make("Pendulum-v1"))

    result_p = estimate_bias_over_random_states(model_p, eval_env_p, n_states=50)
    print(
        f"DQN Pendulum: mean bias = {result_p['bias'].mean():.3f} "
        f"(+ = Überschätzung, - = Unterschätzung), std = {result_p['bias'].std():.3f}"
    )
    eval_env_p.close()