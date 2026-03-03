from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from config import (
    DEFAULT_GAMMA_CCY,
    DEFAULT_RHO_CURVE_DIFF,
    DEFAULT_RW_INFLATION,
    DEFAULT_RW_RATES,
)
from correlations import GIRRCorrelationModel
from delta_girr import DeltaGIRRCalculator
from utils import load_corr_inputs_from_excel, parse_delta_sheet_from_excel
from risk_weights import RiskWeights


def replicate_delta_girr_examples(xlsx_path: str) -> pd.DataFrame:
    xlsx_path = str(xlsx_path)
    corr_inputs = load_corr_inputs_from_excel(xlsx_path, sheet_name="correlaciones")

    corr_model = GIRRCorrelationModel(
        theta=corr_inputs.theta,
        scenario_scale=corr_inputs.scenario_scale,
        rho_curve_diff=DEFAULT_RHO_CURVE_DIFF,
        gamma_ccy=DEFAULT_GAMMA_CCY,
    )
    rws = RiskWeights(
        rw_rates=DEFAULT_RW_RATES,
        rw_inflation=DEFAULT_RW_INFLATION,
        premium_adjustment=True,  # Excel uses 'SI'
    )
    calc = DeltaGIRRCalculator(corr=corr_model, rws=rws)

    sheets = [
        "delta GIRR 1",
        "delta GIRR 2",
        "delta GIRR 3",
        "delta GIRR 4",
        "delta GIRR 5",
        "delta GIRR 6",
        "ejercicio resuelto",
    ]

    out_rows = []
    for sh in sheets:
        df = parse_delta_sheet_from_excel(xlsx_path, sh, default_ccy="EUR")
        k = calc.capital(df)
        out_rows.append({"sheet": sh, "capital": k})
    return pd.DataFrame(out_rows)


if __name__ == "__main__":
    # Example usage:
    #   python -m frtb_sa_girr.examples_replication /path/to/Delta_GIRR_Sol.xlsx
    import sys

    if len(sys.argv) < 2:
        print("Usage: python examples_replication.py <Delta_GIRR_Sol.xlsx>")
        raise SystemExit(2)

    df = replicate_delta_girr_examples(sys.argv[1])
    pd.set_option("display.float_format", lambda x: f"{x:,.6f}")
    print(df)
