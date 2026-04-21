"""Diebold-Mariano test.

Compares forecast accuracy of two models using squared-error loss by default.
Pairs every baseline model with its enhanced counterpart (by reading the
prediction CSVs written by each model to outputs/predictions/).

Usage:
    python -m utils.dm_test
"""
import os
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from utils.data_utils import PREDICTIONS_DIR, OUTPUT_DIR


BASELINE_ENHANCED_PAIRS = [
    ('lasso_baseline',         'lasso_enhanced'),
    ('elastic_net_baseline',   'elastic_net_enhanced'),
    ('random_forest_baseline', 'random_forest_enhanced'),
    ('pca_baseline',           'pca_enhanced'),
    ('pls_baseline',           'pls_enhanced'),
    ('comb_baseline',          'comb_enhanced'),
]

# Enhanced-only models (no natural baseline twin) — compared vs each baseline for reference.
ENHANCED_ONLY = ['lstm_enhanced', 'xgboost_enhanced']


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2
            ) -> Tuple[float, float]:
    """Harvey-Leybourne-Newbold small-sample-corrected Diebold-Mariano.

    H0: the two forecasts have equal expected loss.
    Positive DM means model-1 has larger loss (model-2 is better).

    Parameters
    ----------
    e1, e2 : arrays of forecast errors (y_true - y_pred) aligned in time.
    h      : forecast horizon (for autocovariance bandwidth).
    power  : loss exponent — 2 for MSE, 1 for MAE.
    """
    e1, e2 = np.asarray(e1, dtype=float), np.asarray(e2, dtype=float)
    if e1.shape != e2.shape:
        raise ValueError("forecast error arrays must align")

    d = np.abs(e1) ** power - np.abs(e2) ** power
    T = len(d)
    mean_d = d.mean()

    # Newey-West style long-run variance truncated at h-1 lags.
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for k in range(1, h):
        cov = np.cov(d[k:], d[:-k], ddof=0)[0, 1]
        var_d += 2 * cov
    var_d /= T
    if var_d <= 0:
        return float('nan'), float('nan')

    dm = mean_d / np.sqrt(var_d)
    # HLN small-sample correction
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat = dm * correction
    p_value = 2 * (1 - student_t.cdf(abs(dm_stat), df=T - 1))
    return float(dm_stat), float(p_value)


def _load_pred(name: str) -> pd.DataFrame:
    path = os.path.join(PREDICTIONS_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing predictions for {name}: {path}")
    return pd.read_csv(path, index_col='date', parse_dates=True)


def _pair_errors(a: str, b: str) -> Tuple[np.ndarray, np.ndarray, int]:
    da, db = _load_pred(a), _load_pred(b)
    idx = da.index.intersection(db.index)
    if len(idx) == 0:
        raise ValueError(f"no overlapping dates between {a} and {b}")
    y_true = da.loc[idx, 'y_true'].values
    # y_true should match between the two frames — assert softly
    if not np.allclose(y_true, db.loc[idx, 'y_true'].values):
        raise ValueError(f"y_true mismatch between {a} and {b}")
    e1 = y_true - da.loc[idx, 'y_pred'].values
    e2 = y_true - db.loc[idx, 'y_pred'].values
    return e1, e2, len(idx)


def run_dm_all(output_csv: str = None):
    output_csv = output_csv or os.path.join(OUTPUT_DIR, 'dm_test_results.csv')

    rows = []

    for base, enh in BASELINE_ENHANCED_PAIRS:
        try:
            e1, e2, n = _pair_errors(base, enh)
        except FileNotFoundError as ex:
            print(f"[skip] {base} vs {enh}: {ex}")
            continue
        dm, p = dm_test(e1, e2)
        rows.append({'model_1': base, 'model_2': enh, 'n': n,
                     'dm_statistic': dm, 'p_value': p,
                     'better_model': enh if dm > 0 else base})

    # Cross-pair: each enhanced-only model vs its most natural counterpart (random_forest_baseline).
    for extra in ENHANCED_ONLY:
        for base in ['random_forest_baseline']:
            try:
                e1, e2, n = _pair_errors(base, extra)
            except FileNotFoundError as ex:
                print(f"[skip] {base} vs {extra}: {ex}")
                continue
            dm, p = dm_test(e1, e2)
            rows.append({'model_1': base, 'model_2': extra, 'n': n,
                         'dm_statistic': dm, 'p_value': p,
                         'better_model': extra if dm > 0 else base})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nDM test summary written to {output_csv}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run_dm_all()
