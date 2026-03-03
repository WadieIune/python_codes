from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd


def capital_summary(
    df: pd.DataFrame,
    capital_col: str = "K",
    percentiles: Sequence[float] = (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99),
) -> pd.DataFrame:
    """Compute a compact summary table for a capital column.

    Returns a 1-row DataFrame with mean/std/min/max and selected percentiles.

    Notes
    -----
    This is intended to be exported to CSV and/or LaTeX for the TFM.
    """
    s = df[capital_col].astype(float)
    out = {
        "n": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
    }
    for p in percentiles:
        out[f"p{int(round(100*p))}"] = float(s.quantile(float(p)))
    return pd.DataFrame([out])


def df_to_latex(
    df: pd.DataFrame,
    out_path: str,
    index: bool = False,
    float_format: str = "%.4f",
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> None:
    """Write a LaTeX tabular to file (for \input{} in Overleaf)."""
    latex = df.to_latex(index=index, float_format=lambda x: float_format % x, escape=False)
    if caption or label:
        # wrap into table environment if requested
        parts = ["\\begin{table}[h]", "\\centering", latex]
        if caption:
            parts.append(f"\\caption{{{caption}}}")
        if label:
            parts.append(f"\\label{{{label}}}")
        parts.append("\\end{table}")
        latex = "\n".join(parts)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)
