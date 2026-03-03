from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from delta_girr import DeltaGIRRCalculator


@dataclass
class GAResult:
    best_multipliers: np.ndarray
    best_df: pd.DataFrame
    best_objective: float
    best_capital: float
    history: pd.DataFrame  # columns: gen, best_objective, best_capital


def _smoothness_penalty(df: pd.DataFrame) -> float:
    """
    Penalize large jumps between adjacent tenors within each (ccy, curve, ftype) group.
    """
    pen = 0.0
    for _, g in df.groupby(["ccy", "curve", "ftype"], sort=False):
        gg = g.sort_values("tenor")
        d = gg["delta"].to_numpy(float)
        if len(d) >= 2:
            pen += float(np.sum(np.diff(d) ** 2))
    return float(pen)


def _sign_pattern_penalty(df: pd.DataFrame, break_tenor: float = 2.0) -> float:
    """
    Encourage delta < 0 in short tenors and delta > 0 in long tenors (heuristic).

    Note: sign conventions depend on how delta is computed (PV01/DV01). Use this
    penalty only when the sign convention is consistent and economically intended.
    """
    pen = 0.0
    for _, g in df.groupby(["ccy", "curve", "ftype"], sort=False):
        gg = g.copy()
        short = gg[gg["tenor"] <= float(break_tenor)]["delta"].to_numpy(float)
        long = gg[gg["tenor"] > float(break_tenor)]["delta"].to_numpy(float)
        # penalty if short deltas are positive (want negative)
        pen += float(np.sum(np.maximum(short, 0.0) ** 2))
        # penalty if long deltas are negative (want positive)
        pen += float(np.sum(np.maximum(-long, 0.0) ** 2))
    return float(pen)


def _build_df_from_multipliers(df_base: pd.DataFrame, m: np.ndarray) -> pd.DataFrame:
    df = df_base.copy()
    df["delta"] = df["delta"].to_numpy(float) * m.astype(float)
    return df


def optimize_delta_ga(
    calc: DeltaGIRRCalculator,
    df_base: pd.DataFrame,
    population_size: int = 200,
    n_generations: int = 200,
    bounds: Tuple[float, float] = (-2.0, 2.0),
    tournament_k: int = 3,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.2,
    mutation_sigma: float = 0.10,
    elitism: int = 2,
    lambda_smooth: float = 1e-12,
    lambda_turnover: float = 1e-3,
    lambda_sign: float = 0.0,
    sign_break_tenor: float = 2.0,
    use_three_corr_scenarios: bool = True,
    scenario_scales: Sequence[float] = (0.75, 1.0, 1.25),
    seed: int = 0,
) -> GAResult:
    """
    Real-coded Genetic Algorithm to minimize SA GIRR Delta capital.

    Individuals are multiplicative scalings of the base deltas: delta_i = m_i * delta0_i.
    This supports sign flips (m_i < 0) and bounded search space.

    The objective includes optional penalties:
      - smoothness across adjacent tenors
      - turnover (distance from m=1)
      - sign-pattern heuristic (short negative, long positive)

    Returns the best individual found and a simple convergence history.
    """
    required = {"ccy", "curve", "tenor", "ftype", "delta"}
    missing = required - set(df_base.columns)
    if missing:
        raise ValueError(f"df_base missing columns: {sorted(missing)}")

    df_base = df_base.copy()
    df_base["tenor"] = df_base["tenor"].astype(float)
    df_base["delta"] = df_base["delta"].astype(float)

    n = len(df_base)
    lo, hi = float(bounds[0]), float(bounds[1])
    rng = np.random.default_rng(int(seed))

    # init population around 1.0
    pop = rng.normal(loc=1.0, scale=0.25, size=(int(population_size), n))
    pop = np.clip(pop, lo, hi)

    def capital_of(df: pd.DataFrame) -> float:
        if use_three_corr_scenarios:
            return float(calc.capital_max_corr_scenarios(df, scenario_scales=scenario_scales))
        return float(calc.capital(df))

    def objective(m: np.ndarray) -> Tuple[float, float]:
        df = _build_df_from_multipliers(df_base, m)
        K = capital_of(df)
        smooth = _smoothness_penalty(df)
        turn = float(np.sum(np.abs(m - 1.0)))
        sign_pen = _sign_pattern_penalty(df, break_tenor=float(sign_break_tenor)) if lambda_sign != 0.0 else 0.0
        J = float(K + lambda_smooth * smooth + lambda_turnover * turn + lambda_sign * sign_pen)
        return J, K

    # evaluate initial
    objs = np.zeros(pop.shape[0], float)
    caps = np.zeros(pop.shape[0], float)
    for i in range(pop.shape[0]):
        objs[i], caps[i] = objective(pop[i])

    hist_rows: List[Dict[str, float]] = []

    def tournament_select() -> int:
        idxs = rng.integers(0, pop.shape[0], size=int(tournament_k))
        best = idxs[np.argmin(objs[idxs])]
        return int(best)

    for gen in range(int(n_generations)):
        # sort for elitism
        order = np.argsort(objs)
        pop = pop[order]
        objs = objs[order]
        caps = caps[order]

        hist_rows.append({"gen": float(gen), "best_objective": float(objs[0]), "best_capital": float(caps[0])})

        new_pop = [pop[i].copy() for i in range(min(int(elitism), pop.shape[0]))]

        while len(new_pop) < pop.shape[0]:
            p1 = pop[tournament_select()]
            p2 = pop[tournament_select()]

            child = p1.copy()
            if rng.random() < float(crossover_prob):
                alpha = rng.random(size=n)
                child = alpha * p1 + (1.0 - alpha) * p2

            # mutation
            if rng.random() < float(mutation_prob):
                child = child + rng.normal(loc=0.0, scale=float(mutation_sigma), size=n)

            child = np.clip(child, lo, hi)
            new_pop.append(child)

        pop = np.asarray(new_pop, float)

        # evaluate
        for i in range(pop.shape[0]):
            objs[i], caps[i] = objective(pop[i])

    # final best
    best_idx = int(np.argmin(objs))
    best_m = pop[best_idx].copy()
    best_df = _build_df_from_multipliers(df_base, best_m)
    best_obj, best_cap = float(objs[best_idx]), float(caps[best_idx])

    return GAResult(
        best_multipliers=best_m,
        best_df=best_df,
        best_objective=best_obj,
        best_capital=best_cap,
        history=pd.DataFrame(hist_rows),
    )
