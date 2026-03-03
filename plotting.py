from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    out_path: str,
    bins: int = 50,
    title: Optional[str] = None,
) -> None:
    """Save a histogram of a numeric column to out_path."""
    x = df[column].astype(float).to_numpy()
    plt.figure()
    plt.hist(x, bins=int(bins))
    plt.xlabel(column)
    plt.ylabel("Frecuencia")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_convergence(
    history: pd.DataFrame,
    out_path: str,
    x_col: str = "gen",
    y_col: str = "best_capital",
    title: str = "Convergencia del algoritmo genético",
) -> None:
    """Save a simple line plot of GA convergence to out_path."""
    plt.figure()
    plt.plot(history[x_col].astype(float).to_numpy(), history[y_col].astype(float).to_numpy())
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
