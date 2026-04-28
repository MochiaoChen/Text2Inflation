"""DM-test power analysis for the 56-observation out-of-sample window.

Quantifies the sample size needed to detect the observed RMSE improvement of
the enhanced Random Forest over its baseline at conventional significance
levels — supporting the paper's argument that the p=0.244 DM result is a
power artifact, not evidence of zero effect.

Usage:
    python -m utils.power_analysis
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

from utils.data_utils import PREDICTIONS_DIR


def loss_diff_series(y_true, y_a, y_b):
    """Squared-error loss differential d_t = e_a^2 - e_b^2."""
    return (y_true - y_a) ** 2 - (y_true - y_b) ** 2


def required_n(d: np.ndarray, alpha: float = 0.05, power: float = 0.80,
               two_sided: bool = True) -> float:
    """Sample size needed to reject H0: E[d]=0 with given power.

    Uses the observed mean and variance of d to project a normal-approx DM
    statistic. Reports n such that mean / sqrt(var/n) hits the critical value.
    """
    mean_d = d.mean()
    var_d = np.var(d, ddof=1)
    if mean_d == 0:
        return float('inf')
    z_alpha = norm.ppf(1 - alpha / 2) if two_sided else norm.ppf(1 - alpha)
    z_beta = norm.ppf(power)
    n = ((z_alpha + z_beta) ** 2) * var_d / (mean_d ** 2)
    return float(n)


def achieved_power(d: np.ndarray, n: int, alpha: float = 0.05,
                   two_sided: bool = True) -> float:
    """Power of the DM test at the given n given observed effect size."""
    mean_d = d.mean()
    var_d = np.var(d, ddof=1)
    if var_d == 0:
        return float('nan')
    z_alpha = norm.ppf(1 - alpha / 2) if two_sided else norm.ppf(1 - alpha)
    ncp = mean_d / np.sqrt(var_d / n)
    if two_sided:
        return float(1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp))
    return float(1 - norm.cdf(z_alpha - ncp))


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(PREDICTIONS_DIR, f"{name}.csv"),
                       index_col='date', parse_dates=True)


def main():
    pairs = [
        ('lasso_baseline',         'lasso_enhanced'),
        ('elastic_net_baseline',   'elastic_net_enhanced'),
        ('random_forest_baseline', 'random_forest_enhanced'),
        ('pca_baseline',           'pca_enhanced'),
        ('pls_baseline',           'pls_enhanced'),
        ('comb_baseline',          'comb_enhanced'),
    ]

    rows = []
    for base, enh in pairs:
        a = _load(base)
        b = _load(enh)
        idx = a.index.intersection(b.index)
        y_true = a.loc[idx, 'y_true'].values
        d = loss_diff_series(y_true, a.loc[idx, 'y_pred'].values,
                             b.loc[idx, 'y_pred'].values)
        n = len(d)
        rmse_b = float(np.sqrt(np.mean((y_true - a.loc[idx, 'y_pred'].values) ** 2)))
        rmse_e = float(np.sqrt(np.mean((y_true - b.loc[idx, 'y_pred'].values) ** 2)))
        rows.append({
            'baseline': base,
            'enhanced': enh,
            'n_actual': n,
            'rmse_baseline': rmse_b,
            'rmse_enhanced': rmse_e,
            'rmse_pct_improvement': 100 * (rmse_b - rmse_e) / rmse_b,
            'mean_loss_diff': float(d.mean()),
            'sd_loss_diff': float(np.std(d, ddof=1)),
            'power_at_n': achieved_power(d, n, alpha=0.05),
            'n_for_80pct_power_5pct': required_n(d, alpha=0.05, power=0.80),
            'n_for_80pct_power_10pct': required_n(d, alpha=0.10, power=0.80),
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(PREDICTIONS_DIR), 'power_analysis.csv')
    df.to_csv(out_path, index=False)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)
    print(df.to_string(index=False))
    print(f"\nWritten to {out_path}")
    return df


if __name__ == "__main__":
    main()
