from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from delta_girr import DeltaGIRRCalculator


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, float]


class DeltaHedgeEnv:
    """
    Minimal RL-style environment for delta-hedge optimization (no external deps).

    This environment is intentionally lightweight so it can be wrapped later by
    Gymnasium/Stable-Baselines if desired.

    State: current delta vector (ordered as df_base rows) concatenated with current capital.
    Action: delta increment vector (same length as state delta part), scaled by step_size.
    Transition: delta_{t+1} = clip(delta_t + step_size * action, delta_bounds)

    Reward: -capital_{t+1} - cost_lambda * ||action||_1 - smooth_lambda * smoothness_penalty
    """

    def __init__(
        self,
        calc: DeltaGIRRCalculator,
        df_base: pd.DataFrame,
        step_size: float = 0.05,
        delta_bounds: Tuple[float, float] = (-1e7, 1e7),
        horizon: int = 25,
        use_three_corr_scenarios: bool = True,
        scenario_scales: Sequence[float] = (0.75, 1.0, 1.25),
        cost_lambda: float = 1e-6,
        smooth_lambda: float = 1e-18,
        seed: int = 0,
    ) -> None:
        required = {"ccy", "curve", "tenor", "ftype", "delta"}
        missing = required - set(df_base.columns)
        if missing:
            raise ValueError(f"df_base missing columns: {sorted(missing)}")

        self.calc = calc
        self.df_base = df_base.copy()
        self.df_base["tenor"] = self.df_base["tenor"].astype(float)
        self.df_base["delta"] = self.df_base["delta"].astype(float)

        self.n = len(self.df_base)
        self.step_size = float(step_size)
        self.delta_lo, self.delta_hi = float(delta_bounds[0]), float(delta_bounds[1])
        self.horizon = int(horizon)

        self.use_three_corr_scenarios = bool(use_three_corr_scenarios)
        self.scenario_scales = tuple(float(x) for x in scenario_scales)

        self.cost_lambda = float(cost_lambda)
        self.smooth_lambda = float(smooth_lambda)

        self.rng = np.random.default_rng(int(seed))

        self.t = 0
        self.delta = self.df_base["delta"].to_numpy(float).copy()

    def _capital(self, delta_vec: np.ndarray) -> float:
        df = self.df_base.copy()
        df["delta"] = delta_vec.astype(float)
        if self.use_three_corr_scenarios:
            return float(self.calc.capital_max_corr_scenarios(df, scenario_scales=self.scenario_scales))
        return float(self.calc.capital(df))

    def reset(self, noise_scale: float = 0.0) -> np.ndarray:
        self.t = 0
        self.delta = self.df_base["delta"].to_numpy(float).copy()
        if noise_scale != 0.0:
            self.delta = self.delta + self.rng.normal(0.0, float(noise_scale), size=self.n)
        K = self._capital(self.delta)
        return np.concatenate([self.delta, np.array([K], float)])

    def step(self, action: np.ndarray) -> StepResult:
        action = np.asarray(action, float).reshape(-1)
        if action.shape[0] != self.n:
            raise ValueError(f"action length {action.shape[0]} != n {self.n}")

        self.t += 1
        # apply action
        self.delta = np.clip(self.delta + self.step_size * action, self.delta_lo, self.delta_hi)

        K = self._capital(self.delta)

        # simple smoothness: penalty on delta differences across adjacent tenors per group
        df_tmp = self.df_base.copy()
        df_tmp["delta"] = self.delta
        smooth_pen = 0.0
        for _, g in df_tmp.groupby(["ccy", "curve", "ftype"], sort=False):
            gg = g.sort_values("tenor")
            d = gg["delta"].to_numpy(float)
            if len(d) >= 2:
                smooth_pen += float(np.sum(np.diff(d) ** 2))

        cost_pen = float(np.sum(np.abs(action)))

        reward = -float(K) - self.cost_lambda * cost_pen - self.smooth_lambda * smooth_pen
        done = bool(self.t >= self.horizon)

        state = np.concatenate([self.delta, np.array([K], float)])
        info = {"K": float(K), "cost_pen": float(cost_pen), "smooth_pen": float(smooth_pen)}
        return StepResult(state=state, reward=float(reward), done=done, info=info)


def evaluate_random_policy(
    env: DeltaHedgeEnv,
    n_episodes: int = 100,
    action_scale: float = 1.0,
    noise_scale_reset: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Evaluate a simple random policy to produce baseline RL metrics.

    This function is useful to validate the environment and to generate a
    *baseline* table (to be improved later with a trained RL agent).

    Metrics reported (per episode):
      - K_final: capital at the end of the episode
      - K_max  : same as K_final if env.use_three_corr_scenarios=True (already max over Low/Normal/High)
      - turnover_action: accumulated L1 norm of actions (proxy for trading/turnover)
      - smooth_final: smoothness penalty at final state
      - total_reward: sum of rewards over the episode
    """
    rng = np.random.default_rng(int(seed))

    rows = []
    for ep in range(int(n_episodes)):
        state = env.reset(noise_scale=float(noise_scale_reset))
        delta0 = state[:-1].copy()
        total_reward = 0.0
        turnover_action = 0.0

        done = False
        last_info: Dict[str, float] = {}
        while not done:
            action = rng.normal(loc=0.0, scale=float(action_scale), size=env.n)
            turnover_action += float(np.sum(np.abs(action)))
            step = env.step(action)
            total_reward += float(step.reward)
            done = bool(step.done)
            last_info = step.info

        deltaT = step.state[:-1].copy()
        K_final = float(last_info.get("K", step.state[-1]))
        smooth_final = float(last_info.get("smooth_pen", 0.0))
        turnover_delta = float(np.sum(np.abs(deltaT - delta0)))

        rows.append(
            {
                "episode": int(ep),
                "K_final": float(K_final),
                "K_max": float(K_final),
                "turnover_action": float(turnover_action),
                "turnover_delta": float(turnover_delta),
                "smooth_final": float(smooth_final),
                "total_reward": float(total_reward),
            }
        )

    return pd.DataFrame(rows)
