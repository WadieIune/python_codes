"""FRTB SA - GIRR Delta calculator (simplified).

This package replicates the Excel calculator in Delta_GIRR_Sol.xlsx for:
- Example 1..6
- Exercise resolved

Scope:
- Standardised Approach (SA) only
- GIRR Delta only (no IMA)
- Correlations and RWs aligned to the provided Excel.
"""

from config import DEFAULT_RW_RATES, DEFAULT_RW_INFLATION, DEFAULT_THETA, DEFAULT_RHO_CURVE_DIFF, DEFAULT_GAMMA_CCY
from utils import load_corr_inputs_from_excel, parse_delta_sheet_from_excel
from delta_girr import DeltaGIRRCalculator
from scenario_mc import generate_delta_scenarios_mc, simulate_capital_over_scenarios
from optimizer_ga import optimize_delta_ga, GAResult
from rl_env import DeltaHedgeEnv

from scenario_mc import simulate_capital_over_scenarios_select_top
from metrics import smoothness_penalty, turnover_abs, turnover_rel
from plotting import plot_histogram, plot_convergence
from reporting import capital_summary, df_to_latex
from rl_env import evaluate_random_policy
