from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import heapq
import math

from delta_girr import DeltaGIRRCalculator


@dataclass
class MonteCarloScenarioResult:
    """Container for Monte Carlo scenario outputs."""
    results: pd.DataFrame  # columns: scenario_id, K (and optionally K_low/K_mid/K_high)
    scenarios: Optional[List[pd.DataFrame]] = None  # optional list of scenario DataFrames


def _exp_kernel(tenors: np.ndarray, kappa: float) -> np.ndarray:
    """Exponential kernel for tenor-to-tenor correlation."""
    t = tenors.reshape(-1, 1).astype(float)
    return np.exp(-float(kappa) * np.abs(t - t.T))


def generate_delta_scenarios_mc(
    df_base: pd.DataFrame,
    n_scenarios: int,
    sigma_base: float = 0.20,
    kappa: float = 0.50,
    seed: int = 0,
    store_scenarios: bool = False,
) -> MonteCarloScenarioResult:
    """
    Generate Monte Carlo scenarios for delta sensitivities.

    The generator perturbs deltas *within each* (ccy, curve, ftype) group using a
    correlated multivariate normal across tenors. This is a synthetic generator
    intended to explore candidate sensitivity configurations, not to forecast P&L.

    Parameters
    ----------
    df_base:
        Base sensitivities in long format with columns:
        ccy, curve, tenor, ftype, delta
    n_scenarios:
        Number of scenarios to generate.
    sigma_base:
        Base noise level (relative). Per tenor volatility is sigma_base * |delta_base|.
    kappa:
        Decay parameter for tenor kernel exp(-kappa*|Ti-Tj|).
    seed:
        RNG seed for reproducibility.
    store_scenarios:
        If True, returns the full list of scenario DataFrames (may be large).

    Returns
    -------
    MonteCarloScenarioResult
        results DataFrame + optionally the list of scenario DataFrames.
    """
    required = {"ccy", "curve", "tenor", "ftype", "delta"}
    missing = required - set(df_base.columns)
    if missing:
        raise ValueError(f"df_base missing columns: {sorted(missing)}")

    df_base = df_base.copy()
    df_base["tenor"] = df_base["tenor"].astype(float)
    df_base["delta"] = df_base["delta"].astype(float)

    rng = np.random.default_rng(int(seed))

    # Pre-split groups for speed and deterministic ordering.
    grouped: List[Tuple[Tuple[str, str, str], pd.DataFrame]] = []
    for key, g in df_base.groupby(["ccy", "curve", "ftype"], sort=True):
        gg = g.sort_values("tenor").reset_index(drop=True)
        grouped.append((key, gg))

    scenario_dfs: List[pd.DataFrame] = []
    # We'll build results in a list and convert to DataFrame.
    res_rows: List[Dict[str, float]] = []

    for s in range(int(n_scenarios)):
        parts: List[pd.DataFrame] = []
        for _, g in grouped:
            tenors = g["tenor"].to_numpy(float)
            d0 = g["delta"].to_numpy(float)

            # Correlation across tenors, scaled by per-tenor sigma.
            C = _exp_kernel(tenors, kappa=float(kappa))
            sig = float(sigma_base) * np.abs(d0)
            # Avoid zero-variance issues
            sig = np.where(sig == 0.0, float(sigma_base), sig)

            Sigma = (sig.reshape(-1, 1) * C) * sig.reshape(1, -1)

            eps = rng.multivariate_normal(mean=np.zeros(len(d0)), cov=Sigma, method="cholesky")
            d1 = d0 + eps

            gg = g.copy()
            gg["delta"] = d1
            parts.append(gg)

        df_s = pd.concat(parts, ignore_index=True)
        df_s["delta"] = df_s["delta"].astype(float)

        if store_scenarios:
            scenario_dfs.append(df_s)

        res_rows.append({"scenario_id": float(s)})

    results = pd.DataFrame(res_rows)
    return MonteCarloScenarioResult(results=results, scenarios=scenario_dfs if store_scenarios else None)


def simulate_capital_over_scenarios(
    calc: DeltaGIRRCalculator,
    df_base: pd.DataFrame,
    n_scenarios: int,
    sigma_base: float = 0.20,
    kappa: float = 0.50,
    seed: int = 0,
    use_three_corr_scenarios: bool = True,
    scenario_scales: Sequence[float] = (0.75, 1.0, 1.25),
) -> pd.DataFrame:
    """
    Generate Monte Carlo delta scenarios and compute capital for each scenario.

    Parameters
    ----------
    calc:
        A calibrated DeltaGIRRCalculator (theta, RW, etc.).
    df_base:
        Base sensitivities.
    n_scenarios:
        Number of MC scenarios.
    sigma_base, kappa, seed:
        Scenario generation parameters.
    use_three_corr_scenarios:
        If True, capital per scenario is max across correlation scales.
    scenario_scales:
        Correlation scale multipliers (Low/Normal/High).

    Returns
    -------
    pd.DataFrame
        Columns: scenario_id, K (and if use_three_corr_scenarios: K_low/K_mid/K_high)
    """
    mc = generate_delta_scenarios_mc(
        df_base=df_base,
        n_scenarios=n_scenarios,
        sigma_base=sigma_base,
        kappa=kappa,
        seed=seed,
        store_scenarios=True,
    )
    assert mc.scenarios is not None

    out_rows: List[Dict[str, float]] = []
    for s, df_s in enumerate(mc.scenarios):
        if use_three_corr_scenarios:
            Ks = []
            for sc in scenario_scales:
                K_sc = DeltaGIRRCalculator(
                    corr=calc.corr.__class__(
                        theta=float(calc.corr.theta),
                        scenario_scale=float(sc),
                        rho_curve_diff=float(calc.corr.rho_curve_diff),
                        gamma_ccy=float(calc.corr.gamma_ccy),
                        floor_tenor=float(calc.corr.floor_tenor),
                    ),
                    rws=calc.rws,
                ).capital(df_s)
                Ks.append(float(K_sc))
            out_rows.append(
                {
                    "scenario_id": float(s),
                    "K_low": float(Ks[0]),
                    "K_mid": float(Ks[1]) if len(Ks) > 1 else float(Ks[0]),
                    "K_high": float(Ks[2]) if len(Ks) > 2 else float(Ks[-1]),
                    "K": float(max(Ks)),
                }
            )
        else:
            out_rows.append({"scenario_id": float(s), "K": float(calc.capital(df_s))})

    return pd.DataFrame(out_rows)


def simulate_capital_over_scenarios_select_top(
    calc: DeltaGIRRCalculator,
    df_base: pd.DataFrame,
    n_scenarios: int,
    top_n: int = 50,
    sigma_base: float = 0.20,
    kappa: float = 0.50,
    seed: int = 0,
    use_three_corr_scenarios: bool = True,
    scenario_scales: Sequence[float] = (0.75, 1.0, 1.25),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Monte Carlo deltas + capital per scenario, storing only the best (lowest-capital) scenarios.

    This helper is designed for notebooks/experiments where we need:
      1) A full per-scenario result table (for summaries, histograms, etc.)
      2) A *small* set of best scenarios (top_n) to inspect the delta shapes

    Compared to simulate_capital_over_scenarios(..., store_scenarios=True), this
    function avoids keeping *all* scenario DataFrames in memory.

    Returns
    -------
    results_df:
        One row per scenario with capital (and optionally K_low/K_mid/K_high) and
        additional evaluation metrics:
          - smooth_pen : stability by tenor
          - turnover_abs, turnover_rel : turnover vs base deltas
    top_scenarios_df:
        Long-format table with the same columns as df_base plus:
          - scenario_id, K (and K_low/K_mid/K_high)
        for the best top_n scenarios (lowest K).
    """
    required = {"ccy", "curve", "tenor", "ftype", "delta"}
    missing = required - set(df_base.columns)
    if missing:
        raise ValueError(f"df_base missing columns: {sorted(missing)}")

    # Stable ordering (important for reproducibility and for interpreting the delta vector).
    df0 = df_base.copy()
    df0["tenor"] = df0["tenor"].astype(float)
    df0["delta"] = df0["delta"].astype(float)
    df0["ccy"] = df0["ccy"].astype(str)
    df0["curve"] = df0["curve"].astype(str)
    df0["ftype"] = df0["ftype"].astype(str)
    df0 = df0.sort_values(["ccy", "curve", "ftype", "tenor"]).reset_index(drop=True)

    delta0 = df0["delta"].to_numpy(float)
    base_abs = float(np.sum(np.abs(delta0))) + 1e-12

    rng = np.random.default_rng(int(seed))

    # Pre-split groups (indices are already sorted by tenor because of df0 sorting).
    group_idxs: List[np.ndarray] = []
    group_tenors: List[np.ndarray] = []
    group_L: List[np.ndarray] = []

    for _, g in df0.groupby(["ccy", "curve", "ftype"], sort=False):
        idx = g.index.to_numpy()
        ten = g["tenor"].to_numpy(float)

        d0 = delta0[idx]
        C = _exp_kernel(ten, kappa=float(kappa))
        sig = float(sigma_base) * np.abs(d0)
        sig = np.where(sig == 0.0, float(sigma_base), sig)
        Sigma = (sig.reshape(-1, 1) * C) * sig.reshape(1, -1)

        # Cholesky for fast sampling
        L = np.linalg.cholesky(Sigma + 1e-18 * np.eye(Sigma.shape[0]))

        group_idxs.append(idx)
        group_tenors.append(ten)
        group_L.append(L)

    # Precompute risk weights vector (same logic as DeltaGIRRCalculator.weighted_sensitivities).
    def _rw_row(ftype: str, tenor: float) -> float:
        if ftype == "inflation":
            return float(calc.rws.rw_infl())
        return float(calc.rws.rw_rate(float(tenor)))

    rw = np.array([_rw_row(ft, tn) for ft, tn in zip(df0["ftype"].to_numpy(), df0["tenor"].to_numpy())], float)

    # Precompute correlation matrices for scenario scales.
    factors = [
        (r.ccy, r.curve, float(r.tenor), "inflation" if r.ftype == "inflation" else "rate")
        for r in df0.itertuples(index=False)
    ]

    R_list: List[np.ndarray] = []
    for sc in scenario_scales:
        corr_sc = calc.corr.__class__(
            theta=float(calc.corr.theta),
            scenario_scale=float(sc),
            rho_curve_diff=float(calc.corr.rho_curve_diff),
            gamma_ccy=float(calc.corr.gamma_ccy),
            floor_tenor=float(calc.corr.floor_tenor),
        )
        R_list.append(np.array(corr_sc.matrix(factors), dtype=float))

    # Results + top-N heap (max-heap by using -K)
    rows: List[Dict[str, float]] = []
    heap: List[Tuple[float, int, np.ndarray, Dict[str, float]]] = []

    for s in range(int(n_scenarios)):
        delta = delta0.copy()

        # Sample each group
        for idx, L in zip(group_idxs, group_L):
            z = rng.normal(0.0, 1.0, size=L.shape[0])
            eps = L @ z
            delta[idx] = delta0[idx] + eps

        w = delta * rw

        # smoothness penalty by group (adjacent tenor diffs)
        smooth = 0.0
        for idx in group_idxs:
            d = delta[idx]
            if d.shape[0] >= 2:
                smooth += float(np.sum(np.diff(d) ** 2))

        turnover_abs = float(np.sum(np.abs(delta - delta0)))
        turnover_rel = float(turnover_abs / base_abs)

        if use_three_corr_scenarios:
            Ks = []
            for R in R_list:
                k2 = float(w @ (R @ w))
                Ks.append(float(math.sqrt(max(k2, 0.0))))
            K_low = float(Ks[0])
            K_mid = float(Ks[1]) if len(Ks) > 1 else float(Ks[0])
            K_high = float(Ks[2]) if len(Ks) > 2 else float(Ks[-1])
            K = float(max(Ks))
            row = {
                "scenario_id": float(s),
                "K_low": K_low,
                "K_mid": K_mid,
                "K_high": K_high,
                "K": K,
                "smooth_pen": float(smooth),
                "turnover_abs": float(turnover_abs),
                "turnover_rel": float(turnover_rel),
            }
        else:
            k2 = float(w @ (R_list[0] @ w))
            K = float(math.sqrt(max(k2, 0.0)))
            row = {
                "scenario_id": float(s),
                "K": K,
                "smooth_pen": float(smooth),
                "turnover_abs": float(turnover_abs),
                "turnover_rel": float(turnover_rel),
            }

        rows.append(row)

        # Update top-N
        item_info = {"K": float(K)}
        if "K_low" in row:
            item_info.update({"K_low": float(row["K_low"]), "K_mid": float(row["K_mid"]), "K_high": float(row["K_high"])})
        if len(heap) < int(top_n):
            heapq.heappush(heap, (-float(K), int(s), delta.copy(), item_info))
        else:
            if float(K) < -heap[0][0]:
                heapq.heapreplace(heap, (-float(K), int(s), delta.copy(), item_info))

    results_df = pd.DataFrame(rows)

    # Build top scenarios df (long format)
    heap_sorted = sorted(heap, key=lambda x: -x[0])  # ascending K
    top_parts: List[pd.DataFrame] = []
    for rank, (negK, sid, delta_vec, info) in enumerate(heap_sorted, start=1):
        dft = df0.copy()
        dft["delta"] = delta_vec.astype(float)
        dft.insert(0, "scenario_rank", int(rank))
        dft.insert(1, "scenario_id", int(sid))
        dft["K"] = float(info["K"])
        if "K_low" in info:
            dft["K_low"] = float(info["K_low"])
            dft["K_mid"] = float(info["K_mid"])
            dft["K_high"] = float(info["K_high"])
        top_parts.append(dft)

    top_df = pd.concat(top_parts, ignore_index=True) if top_parts else df0.head(0).copy()
    return results_df, top_df
