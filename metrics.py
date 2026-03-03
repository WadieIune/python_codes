from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


def smoothness_penalty(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("ccy", "curve", "ftype"),
    tenor_col: str = "tenor",
    delta_col: str = "delta",
) -> float:
    """Penalty for jumps between adjacent tenors (sum of squared first differences).

    This metric is used as a proxy for *stability by tenor* (smooth curve exposure).
    It is computed within each (ccy, curve, ftype) group after sorting by tenor.

    Returns
    -------
    float
        Sum_{groups} Sum_{j} (delta_{j+1}-delta_{j})^2
    """
    required = set(group_cols) | {tenor_col, delta_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"smoothness_penalty: missing columns: {sorted(missing)}")

    pen = 0.0
    for _, g in df.groupby(list(group_cols), sort=False):
        gg = g.sort_values(tenor_col)
        d = gg[delta_col].to_numpy(float)
        if len(d) >= 2:
            pen += float(np.sum(np.diff(d) ** 2))
    return float(pen)


def turnover_abs(
    df_base: pd.DataFrame,
    df_new: pd.DataFrame,
    delta_col: str = "delta",
) -> float:
    """Absolute turnover between two delta tables (sum |delta_new - delta_base|).

    Notes
    -----
    Assumes the two DataFrames have the same rows in the same order.
    Use a stable sorting convention before calling if needed.
    """
    if len(df_base) != len(df_new):
        raise ValueError("turnover_abs: df_base and df_new must have same length")
    d0 = df_base[delta_col].to_numpy(float)
    d1 = df_new[delta_col].to_numpy(float)
    return float(np.sum(np.abs(d1 - d0)))


def turnover_rel(
    df_base: pd.DataFrame,
    df_new: pd.DataFrame,
    delta_col: str = "delta",
    eps: float = 1e-12,
) -> float:
    """Relative turnover = sum|Δ1-Δ0| / (sum|Δ0| + eps)."""
    num = turnover_abs(df_base, df_new, delta_col=delta_col)
    den = float(np.sum(np.abs(df_base[delta_col].to_numpy(float)))) + float(eps)
    return float(num / den)
