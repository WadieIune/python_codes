from __future__ import annotations

# --- Default parameters (matched to the provided Excel calculator) ---

# Tenor risk weights used in the examples (RW normal).
# NOTE: Extend this dict if you add more tenors (10Y, 15Y, 20Y, 30Y, etc.).
DEFAULT_RW_RATES = {
    0.25: 170.0,
    0.5: 170.0,
    1.0: 160.0,
    2.0: 130.0,
    3.0: 120.0,
    5.0: 110.0,
}

# Inflation RW (normal). In the Excel: 160, then divided by sqrt(2) if premium adjustment is enabled.
DEFAULT_RW_INFLATION = 160.0

# Correlation decay parameter (theta) in correlaciones!B2
DEFAULT_THETA = 0.03

# Same-currency, different-curve multiplier (99.9% in the Excel formulas)
DEFAULT_RHO_CURVE_DIFF = 0.999

# Cross-currency correlation (Excel uses 0.5 * scenario)
DEFAULT_GAMMA_CCY = 0.5
