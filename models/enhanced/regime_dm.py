"""Regime-conditional DM tests + 2022 commodity-shock case study.

Reviewer Critique V.1: the model under-shoots the 2022 commodity-driven CPI
spike even though DRV_Supply rises — why? Reviewer Critique III: when does
narrative help and when doesn't it? This module:

1. Splits the 56-obs test window by regime indicators (DRV_Supply high/low,
   PMI input prices high/low, supply-shock period 2021-09 to 2022-12) and
   runs DM tests separately on each subset.
2. Produces a 2022 case-study figure: actual CPI, baseline forecast,
   enhanced forecast, with DRV_Supply overlaid.

Outputs in outputs/regime/:
- regime_dm.csv         — per-regime DM stats and RMSE comparisons
- case_2022_commodity.png  — case-study figure
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.data_utils import (
    load_enhanced_data, OUTPUT_DIR, PREDICTIONS_DIR, NLP_FEATURE_COLS,
)
from utils.dm_test import dm_test


REGIME_DIR = os.path.join(OUTPUT_DIR, 'regime')
os.makedirs(REGIME_DIR, exist_ok=True)


def _load_pred(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(PREDICTIONS_DIR, f"{name}.csv"),
                       index_col='date', parse_dates=True)


def _build_regimes(test_index: pd.DatetimeIndex) -> pd.DataFrame:
    """For each test date, build a row of regime indicators.

    Uses the publication-lag-corrected NLP join so the indicator is the
    information available at forecast time.
    """
    df, _ = load_enhanced_data(stationarize=True, publication_lag_months=2)
    df = df.reindex(test_index)
    out = pd.DataFrame(index=test_index)
    if 'DRV_Supply' in df.columns:
        out['DRV_Supply'] = df['DRV_Supply']
        out['supply_high'] = (df['DRV_Supply']
                              > df['DRV_Supply'].median()).astype(int)
    if 'PMI_OutputPrices' in df.columns:
        out['PMI_OutputPrices'] = df['PMI_OutputPrices']
        out['pmi_high'] = (df['PMI_OutputPrices']
                           > df['PMI_OutputPrices'].median()).astype(int)
    if 'Inflation' in df.columns:
        out['Inflation'] = df['Inflation']
        out['high_inflation'] = (df['Inflation'] > 0).astype(int)
    # Supply-shock window
    shock_start = pd.Timestamp('2021-09-01')
    shock_end   = pd.Timestamp('2022-12-31')
    out['shock_2021_22'] = ((test_index >= shock_start) &
                            (test_index <= shock_end)).astype(int)
    # Post-shock deflation
    out['deflation_2023_25'] = (test_index >= pd.Timestamp('2023-01-01')).astype(int)
    return out


def regime_dm_for_pair(base: str, enh: str) -> pd.DataFrame:
    a = _load_pred(base)
    b = _load_pred(enh)
    idx = a.index.intersection(b.index)
    y = a.loc[idx, 'y_true'].values
    e_a = y - a.loc[idx, 'y_pred'].values
    e_b = y - b.loc[idx, 'y_pred'].values

    regimes = _build_regimes(idx)

    rows = []
    rows.append(_summary_row(idx, e_a, e_b, mask=np.ones(len(idx), bool),
                             label='full_sample', base=base, enh=enh))
    for col in regimes.columns:
        if col in ('DRV_Supply', 'PMI_OutputPrices', 'Inflation'):
            continue
        m = regimes[col].astype(int).values.astype(bool)
        if m.sum() < 6 or (~m).sum() < 6:
            continue
        rows.append(_summary_row(idx, e_a, e_b, mask=m,
                                 label=f'{col}=1', base=base, enh=enh))
        rows.append(_summary_row(idx, e_a, e_b, mask=~m,
                                 label=f'{col}=0', base=base, enh=enh))
    return pd.DataFrame(rows)


def _summary_row(idx, e_a, e_b, mask, label, base, enh):
    ea = e_a[mask]
    eb = e_b[mask]
    n = int(mask.sum())
    rmse_a = float(np.sqrt(np.mean(ea ** 2)))
    rmse_b = float(np.sqrt(np.mean(eb ** 2)))
    if n >= 4:
        dm, p = dm_test(ea, eb)
    else:
        dm, p = float('nan'), float('nan')
    return {
        'baseline': base, 'enhanced': enh, 'regime': label, 'n': n,
        'rmse_baseline': rmse_a, 'rmse_enhanced': rmse_b,
        'pct_improvement': 100 * (rmse_a - rmse_b) / rmse_a if rmse_a > 0 else float('nan'),
        'dm_stat': dm, 'p_value': p,
        'better': enh if rmse_b < rmse_a else base,
    }


def run_all_pairs():
    pairs = [
        ('lasso_baseline',         'lasso_enhanced'),
        ('elastic_net_baseline',   'elastic_net_enhanced'),
        ('random_forest_baseline', 'random_forest_enhanced'),
        ('random_forest_baseline', 'random_forest_publag2m'),
        ('pca_baseline',           'pca_enhanced'),
        ('comb_baseline',          'comb_enhanced'),
    ]
    out = []
    for a, b in pairs:
        try:
            df = regime_dm_for_pair(a, b)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue
        out.append(df)
    summary = pd.concat(out, ignore_index=True)
    csv_path = os.path.join(REGIME_DIR, 'regime_dm.csv')
    summary.to_csv(csv_path, index=False)
    print("\n=== Regime-conditional DM ===")
    print(summary.to_string(index=False))
    print(f"\n→ {csv_path}")
    return summary


def case_study_2022(base: str = 'random_forest_baseline',
                    enh: str = 'random_forest_enhanced'):
    a = _load_pred(base)
    b = _load_pred(enh)
    idx = a.index.intersection(b.index)
    df, _ = load_enhanced_data(stationarize=True, publication_lag_months=2)

    plt_start = pd.Timestamp('2021-01-01')
    plt_end   = pd.Timestamp('2023-06-30')
    sel = idx[(idx >= plt_start) & (idx <= plt_end)]
    if len(sel) == 0:
        print("  case study: no test obs in 2021-2023 window; skipping")
        return

    y = a.loc[sel, 'y_true']
    pred_b = a.loc[sel, 'y_pred']
    pred_e = b.loc[sel, 'y_pred']
    drv = df.reindex(sel).get('DRV_Supply', pd.Series(index=sel, dtype=float))

    fig, ax1 = plt.subplots(figsize=(13, 5.6))
    ax1.plot(y.index, y.values, color='black', linewidth=2.2, label='Actual CPI MoM')
    ax1.plot(pred_b.index, pred_b.values, color='#1f77b4', linewidth=1.6,
             linestyle='-.', label='Baseline RF forecast')
    ax1.plot(pred_e.index, pred_e.values, color='#d62728', linewidth=1.6,
             linestyle='--', label='Enhanced RF forecast')
    ax1.axhline(0, color='gray', linewidth=0.7, alpha=0.6)
    ax1.axvspan(pd.Timestamp('2021-09-01'), pd.Timestamp('2022-12-31'),
                color='#ffe6cc', alpha=0.45,
                label='Commodity-shock window (2021-09 to 2022-12)')
    ax1.set_ylabel('CPI MoM growth (%)')
    ax1.set_xlabel('Date')

    if drv.notna().any():
        ax2 = ax1.twinx()
        ax2.plot(drv.index, drv.values, color='#2ca02c', linewidth=1.4,
                 alpha=0.85, label='DRV_Supply (PBoC narrative)')
        ax2.set_ylabel('DRV_Supply')
        l1, lab1 = ax1.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lab1 + lab2, loc='upper left', fontsize=9)
    else:
        ax1.legend(loc='upper left', fontsize=9)

    plt.title('2022 Commodity Shock Case Study — Where Both Models Under-shoot')
    plt.tight_layout()
    out = os.path.join(REGIME_DIR, 'case_2022_commodity.png')
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"  case study figure → {out}")
    return out


def case_study_deflation(base: str = 'random_forest_baseline',
                         enh: str = 'random_forest_enhanced'):
    """2023-2025 deflation regime: where the regime-DM finds the real gain."""
    a = _load_pred(base)
    b = _load_pred(enh)
    idx = a.index.intersection(b.index)
    df, _ = load_enhanced_data(stationarize=True, publication_lag_months=2)

    plt_start = pd.Timestamp('2023-01-01')
    plt_end   = pd.Timestamp('2025-06-30')
    sel = idx[(idx >= plt_start) & (idx <= plt_end)]
    if len(sel) == 0:
        return

    y = a.loc[sel, 'y_true']
    pred_b = a.loc[sel, 'y_pred']
    pred_e = b.loc[sel, 'y_pred']
    err_b = (y - pred_b).abs()
    err_e = (y - pred_e).abs()
    txt_conf = df.reindex(sel).get('TXT_Confidence', pd.Series(index=sel, dtype=float))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5),
                                   gridspec_kw={'height_ratios': [3, 2]},
                                   sharex=True)
    ax1.plot(y.index, y.values, color='black', linewidth=2.2, label='Actual CPI MoM')
    ax1.plot(pred_b.index, pred_b.values, color='#1f77b4', linewidth=1.6,
             linestyle='-.', label='Baseline RF forecast')
    ax1.plot(pred_e.index, pred_e.values, color='#d62728', linewidth=1.6,
             linestyle='--', label='Enhanced RF forecast')
    ax1.axhline(0, color='gray', linewidth=0.7, alpha=0.6)
    ax1.set_ylabel('CPI MoM growth (%)')
    ax1.legend(loc='best', fontsize=9)
    ax1.set_title('2023–2025 Deflation Regime — Where Narrative Features Help Most '
                  f'(ΔRMSE = {100*(np.sqrt((y-pred_b).pow(2).mean()) - np.sqrt((y-pred_e).pow(2).mean())) / np.sqrt((y-pred_b).pow(2).mean()):+.2f}%)')

    width = 12
    x = np.arange(len(sel))
    ax2.bar(sel - pd.Timedelta(days=4), err_b.values, width=8,
            color='#1f77b4', alpha=0.65, label='|err| Baseline')
    ax2.bar(sel + pd.Timedelta(days=4), err_e.values, width=8,
            color='#d62728', alpha=0.65, label='|err| Enhanced')
    ax2.set_ylabel('|forecast error|')
    ax2.legend(loc='upper left', fontsize=9)

    if txt_conf.notna().any():
        ax2b = ax2.twinx()
        ax2b.plot(txt_conf.index, txt_conf.values, color='#2ca02c',
                  linewidth=1.4, alpha=0.85, label='TXT_Confidence (PBoC)')
        ax2b.set_ylabel('TXT_Confidence')
        l2, lab2 = ax2.get_legend_handles_labels()
        l2b, lab2b = ax2b.get_legend_handles_labels()
        ax2.legend(l2 + l2b, lab2 + lab2b, loc='upper left', fontsize=9)

    plt.tight_layout()
    out = os.path.join(REGIME_DIR, 'case_2023_25_deflation.png')
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"  deflation case study figure → {out}")
    return out


if __name__ == "__main__":
    run_all_pairs()
    case_study_2022()
    case_study_deflation()
