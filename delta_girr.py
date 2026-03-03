from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from correlations import FactorType, GIRRCorrelationModel, RiskFactor
from risk_weights import RiskWeights


@dataclass
class DeltaGIRRCalculator:
    """Compute FRTB SA GIRR Delta capital (as per provided Excel).

    Input format
    ------------
    A DataFrame with columns:
        - ccy   : str   (e.g., 'EUR', 'USD')
        - curve : str   (e.g., '1M', '3M', 'OIS', 'Inflacion')
        - tenor : float (e.g., 0.25, 0.5, 1, 2, 3, 5)
        - ftype : 'rate' | 'inflation'
        - delta : float

    Notes
    -----
    * Missing deltas should be represented as 0.
    * Correlation model matches the Excel construction (including scenario scaling).
    * The calculator returns:
        - total capital K = sqrt(w^T R w)
      where w is the vector of weighted sensitivities.
    """

    corr: GIRRCorrelationModel
    rws: RiskWeights

    def weighted_sensitivities(self, sensi: pd.DataFrame) -> pd.DataFrame:
        required = {"ccy", "curve", "tenor", "ftype", "delta"}
        missing = required - set(sensi.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        df = sensi.copy()
        # ensure correct dtypes
        df["ccy"] = df["ccy"].astype(str)
        df["curve"] = df["curve"].astype(str)
        df["tenor"] = df["tenor"].astype(float)
        df["ftype"] = df["ftype"].astype(str)
        df["delta"] = df["delta"].astype(float)

        # Apply risk weights
        def _rw(row: pd.Series) -> float:
            if row["ftype"] == "inflation":
                return self.rws.rw_infl()
            return self.rws.rw_rate(float(row["tenor"]))

        df["rw"] = df.apply(_rw, axis=1)
        df["ws"] = df["delta"] * df["rw"]
        return df

    def _ordered_factors_and_vector(self, ws_df: pd.DataFrame) -> Tuple[List[RiskFactor], np.ndarray]:
        factors: List[RiskFactor] = [
            (r.ccy, r.curve, float(r.tenor), "inflation" if r.ftype == "inflation" else "rate")
            for r in ws_df.itertuples(index=False)
        ]
        w = ws_df["ws"].to_numpy(dtype=float)
        return factors, w

    def correlation_matrix(self, factors: Sequence[RiskFactor]) -> np.ndarray:
        return np.array(self.corr.matrix(factors), dtype=float)

    def capital(self, sensi: pd.DataFrame) -> float:
        ws_df = self.weighted_sensitivities(sensi)
        factors, w = self._ordered_factors_and_vector(ws_df)
        R = self.correlation_matrix(factors)
        k2 = float(w.T @ R @ w)
        return math.sqrt(max(0.0, k2))

    def capital_with_diagnostics(self, sensi: pd.DataFrame) -> dict:
        ws_df = self.weighted_sensitivities(sensi)
        factors, w = self._ordered_factors_and_vector(ws_df)
        R = self.correlation_matrix(factors)
        k2 = float(w.T @ R @ w)
        return {
            "capital": math.sqrt(max(0.0, k2)),
            "ws_df": ws_df,
            "factors": factors,
            "corr_matrix": R,
            "ws_vector": w,
        }

    def capital_max_corr_scenarios(
            self,
            df: pd.DataFrame,
            scenario_scales: Sequence[float] = (0.75, 1.0, 1.25),
        ) -> float:
            """
            Compute capital as the maximum across correlation scenarios (Low/Normal/High).

            FRTB SA sensitivity-based capital is typically evaluated under three correlation
            scenarios: Low (rho*0.75), Medium/Normal (rho), and High (rho*1.25), taking
            the maximum. This helper mirrors that requirement while reusing the same
            risk weights and theta.

            Parameters
            ----------
            df:
                Sensitivities in long format.
            scenario_scales:
                Multipliers applied to correlations (scenario_scale). Default (0.75, 1.0, 1.25).

            Returns
            -------
            float
                Max capital across the provided scenario scales.
            """
            vals: List[float] = []
            for s in scenario_scales:
                corr_s = GIRRCorrelationModel(
                    theta=float(self.corr.theta),
                    scenario_scale=float(s),
                    rho_curve_diff=float(self.corr.rho_curve_diff),
                    gamma_ccy=float(self.corr.gamma_ccy),
                    floor_tenor=float(self.corr.floor_tenor),
                )
                calc_s = DeltaGIRRCalculator(corr=corr_s, rws=self.rws)
                vals.append(calc_s.capital(df))
            return float(max(vals)) if vals else 0.0
