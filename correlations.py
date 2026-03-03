from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Literal, Sequence, Tuple

FactorType = Literal["rate", "inflation"]
RiskFactor = Tuple[str, str, float, FactorType]  # (ccy, curve, tenor, ftype)


def tenor_correlation(t_i: float, t_j: float, theta: float, floor: float = 0.4) -> float:
    """Tenor correlation for rate curves in the Excel 'correlaciones' sheet.

    Excel formula (base):
        MAX(EXP(-theta * ABS(Ti-Tj)/MIN(Ti,Tj)), 40%)

    Returns
    -------
    float
        In [floor, 1].
    """
    ti = float(t_i)
    tj = float(t_j)
    if ti == tj:
        return 1.0
    raw = math.exp(-theta * abs(ti - tj) / min(ti, tj))
    return max(raw, floor)


@dataclass(frozen=True)
class GIRRCorrelationModel:
    """Correlation model matching the provided Excel calculator."""

    theta: float = 0.03
    scenario_scale: float = 1.0  # correlaciones!Y2 in the Excel
    rho_curve_diff: float = 0.999  # 99.9% for different curves within same currency
    gamma_ccy: float = 0.5  # cross-currency correlation multiplier
    floor_tenor: float = 0.4  # 40% floor for rate tenor correlations

    def rho(self, a: RiskFactor, b: RiskFactor) -> float:
        """Pairwise correlation between two risk factors."""
        ccy1, curve1, t1, ft1 = a
        ccy2, curve2, t2, ft2 = b

        # Diagonal
        if a == b:
            return 1.0

        # Cross-currency: constant 0.5 * scenario
        if ccy1 != ccy2:
            return float(self.gamma_ccy) * float(self.scenario_scale)

        # Same currency:
        if ft1 == "inflation" and ft2 == "inflation":
            # In the Excel examples, inflation vertices are treated as perfectly correlated (all ones).
            return 1.0

        if ft1 == "inflation" or ft2 == "inflation":
            # Inflation vs rates: constant 0.4 * scenario (capped at 1)
            return min(1.0, 0.4 * float(self.scenario_scale))

        # Rate vs rate:
        base = tenor_correlation(t1, t2, theta=float(self.theta), floor=float(self.floor_tenor))
        if curve1 != curve2:
            base *= float(self.rho_curve_diff)

        # Scenario scaling + cap at 1 (Excel: IF(base*scenario>1,1,base*scenario))
        return min(1.0, base * float(self.scenario_scale))

    def matrix(self, factors: Sequence[RiskFactor]) -> List[List[float]]:
        """Build the full correlation matrix for the provided ordered factor list."""
        n = len(factors)
        out = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                out[i][j] = self.rho(factors[i], factors[j])
        return out
