from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RiskWeights:
    """Risk weights container.

    Parameters
    ----------
    rw_rates:
        Mapping tenor -> RW (normal) for rate curves.
    rw_inflation:
        RW (normal) for inflation risk factors.
    premium_adjustment:
        If True, apply the Excel adjustment RW_premium = RW_normal / sqrt(2).
        In the Excel, this is controlled by the 'SI' flag.
    """
    rw_rates: Mapping[float, float]
    rw_inflation: float
    premium_adjustment: bool = True

    def rw_rate(self, tenor: float) -> float:
        t = float(tenor)
        if t not in self.rw_rates:
            raise KeyError(f"Missing RW for rate tenor={t}. Available={sorted(self.rw_rates)}")
        rw = float(self.rw_rates[t])
        return rw / math.sqrt(2.0) if self.premium_adjustment else rw

    def rw_infl(self) -> float:
        rw = float(self.rw_inflation)
        return rw / math.sqrt(2.0) if self.premium_adjustment else rw
