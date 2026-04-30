"""Mechanism tests for Plan A: does PBoC narrative ambiguity / confidence
predict realized forecast uncertainty and translate into wider predictive
intervals?

Two regressions, both with Newey-West HAC standard errors:

    (1) |e_t|            = α + β·TXT_Ambiguity_{t}  + γ·TXT_Confidence_{t}
                              + δ·controls + u_t
        — does narrative uncertainty forecast the magnitude of realized
        prediction errors? (uses errors from the enhanced QRF median.)

    (2) Δwidth_t         = α + β·TXT_Ambiguity_{t}  + γ·TXT_Confidence_{t}
                              + δ·controls + u_t
        where Δwidth_t = width80(enhanced) - width80(baseline).
        — does the enhanced model conditionally widen its PI when narrative
        is ambiguous, even if average widths look similar?

Inputs read from outputs/density/{baseline_qrf,enhanced_qrf}_quantiles.csv
and the merged CPI + NLP frame (publication-lag-corrected).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from utils.data_utils import load_enhanced_data, OUTPUT_DIR
from utils.density_eval import ols_newey_west, format_nw_table


DENSITY_DIR = os.path.join(OUTPUT_DIR, 'density')
NLP_COLS = ['TXT_Ambiguity', 'TXT_Confidence']
CONTROLS = ['Inflation_lag1', 'Inflation_lag3', 'PMI_OutputPrices_lag1',
            'M2_Growth_lag1']


def _build_panel(publication_lag_months: int = 2) -> pd.DataFrame:
    df, target = load_enhanced_data(stationarize=True,
                                    publication_lag_months=publication_lag_months)

    # Build minimal lags we'll use as controls
    df = df.copy()
    df['Inflation_lag1'] = df['Inflation'].shift(1)
    df['Inflation_lag3'] = df['Inflation'].shift(3)
    if 'PMI_OutputPrices' in df.columns:
        df['PMI_OutputPrices_lag1'] = df['PMI_OutputPrices'].shift(1)
    if 'M2_Growth' in df.columns:
        df['M2_Growth_lag1'] = df['M2_Growth'].shift(1)
    return df


def _load_quantiles(name: str) -> pd.DataFrame:
    path = os.path.join(DENSITY_DIR, f'{name}_quantiles.csv')
    return pd.read_csv(path, index_col=0, parse_dates=True)


def regression_abs_error_on_ambiguity():
    print("\n=== Reg 1: |e_t| ~ TXT_Ambiguity + TXT_Confidence + controls ===")
    qenh = _load_quantiles('enhanced_qrf')
    df = _build_panel()

    qenh = qenh.copy()
    qenh['y_true'] = df['Inflation'].reindex(qenh.index).values
    qenh['abs_err'] = (qenh['y_true'] - qenh['q50']).abs()
    qenh['width80'] = qenh['q90'] - qenh['q10']

    panel = qenh.join(df[NLP_COLS + [c for c in CONTROLS if c in df.columns]],
                      how='left').dropna()
    print(f"  n = {len(panel)}")

    regressors = ['TXT_Ambiguity', 'TXT_Confidence'] \
                 + [c for c in CONTROLS if c in panel.columns]
    y = panel['abs_err'].values
    X = np.column_stack([np.ones(len(panel))] +
                        [panel[c].values for c in regressors])
    res = ols_newey_west(y, X)
    table = format_nw_table(res, ['intercept'] + regressors)
    print(table.to_string())
    print(f"  R² = {res['r2']:.3f}   NW lag = {res['lag']}")

    out = os.path.join(DENSITY_DIR, 'reg_abs_error_on_ambiguity.csv')
    table.to_csv(out)
    print(f"  → {out}")
    return table, res


def regression_delta_width_on_ambiguity():
    print("\n=== Reg 2: Δwidth_t ~ TXT_Ambiguity + TXT_Confidence + controls ===")
    qb = _load_quantiles('baseline_qrf')
    qe = _load_quantiles('enhanced_qrf')
    common = qb.index.intersection(qe.index)
    df = _build_panel()

    panel = pd.DataFrame(index=common)
    panel['width_base'] = (qb.loc[common, 'q90'] - qb.loc[common, 'q10']).values
    panel['width_enh'] = (qe.loc[common, 'q90'] - qe.loc[common, 'q10']).values
    panel['delta_width'] = panel['width_enh'] - panel['width_base']
    panel = panel.join(df[NLP_COLS + [c for c in CONTROLS if c in df.columns]],
                       how='left').dropna()
    print(f"  n = {len(panel)}")
    print(f"  mean Δwidth = {panel['delta_width'].mean():+.4f}, "
          f"sd = {panel['delta_width'].std():.4f}")

    regressors = ['TXT_Ambiguity', 'TXT_Confidence'] \
                 + [c for c in CONTROLS if c in panel.columns]
    y = panel['delta_width'].values
    X = np.column_stack([np.ones(len(panel))] +
                        [panel[c].values for c in regressors])
    res = ols_newey_west(y, X)
    table = format_nw_table(res, ['intercept'] + regressors)
    print(table.to_string())
    print(f"  R² = {res['r2']:.3f}   NW lag = {res['lag']}")

    out = os.path.join(DENSITY_DIR, 'reg_delta_width_on_ambiguity.csv')
    table.to_csv(out)
    print(f"  → {out}")
    return table, res


def regression_baseline_width_on_ambiguity():
    """Sanity check — does ambiguity track baseline width too?
    If yes, ambiguity is just a proxy for volatility regimes. If no, the
    enhanced-only loading on ambiguity reflects a true narrative channel."""
    print("\n=== Reg 3 (placebo): baseline width ~ TXT_Ambiguity + controls ===")
    qb = _load_quantiles('baseline_qrf')
    df = _build_panel()
    qb = qb.copy()
    qb['width80'] = qb['q90'] - qb['q10']
    panel = qb.join(df[NLP_COLS + [c for c in CONTROLS if c in df.columns]],
                    how='left').dropna()
    regressors = ['TXT_Ambiguity', 'TXT_Confidence'] \
                 + [c for c in CONTROLS if c in panel.columns]
    y = panel['width80'].values
    X = np.column_stack([np.ones(len(panel))] +
                        [panel[c].values for c in regressors])
    res = ols_newey_west(y, X)
    table = format_nw_table(res, ['intercept'] + regressors)
    print(table.to_string())
    print(f"  R² = {res['r2']:.3f}")
    return table, res


def main():
    regression_abs_error_on_ambiguity()
    regression_delta_width_on_ambiguity()
    regression_baseline_width_on_ambiguity()


if __name__ == "__main__":
    main()
