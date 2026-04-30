"""Giacomini-Rossi (2010, JAE) Fluctuation test for time-varying forecast accuracy.

Why this matters here: reviewer Critique II argued the 56-obs DM result is
weak. The Fluctuation test asks the *opposite* question — is the forecast
ranking stable over time, or are there windows where the enhanced model
clearly dominates and others where it does not? This is what supports the
regime-conditional story (Plan B).

Method: rolling DM-style statistic
    F_t = sqrt(m) * mean(d_{t-m/2 : t+m/2}) / σ̂_LR
where d_τ = L(e_baseline,τ) - L(e_enhanced,τ), L = squared error, σ̂_LR = a
HAC long-run sd estimated on the full sample. Reject H_0: equal accuracy at
all t if max_t |F_t| > k_α(μ), where μ = m / T.

GR (2010, Table 1) two-sided critical values for selected μ:
    μ = 0.1 : 1% 3.690 / 5% 3.176 / 10% 2.928
    μ = 0.2 : 1% 3.482 / 5% 2.978 / 10% 2.706
    μ = 0.3 : 1% 3.354 / 5% 2.882 / 10% 2.587

Outputs:
- outputs/fluctuation/{pair}_fluctuation.csv   (date, F_t, threshold)
- outputs/fluctuation/{pair}_fluctuation.png

Usage:
    python -m utils.fluctuation_test
"""
from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.data_utils import OUTPUT_DIR, PREDICTIONS_DIR


FLUCT_DIR = os.path.join(OUTPUT_DIR, 'fluctuation')
os.makedirs(FLUCT_DIR, exist_ok=True)

# Two-sided GR critical values (Giacomini-Rossi 2010 Table 1).
GR_CV = {
    0.1: {0.01: 3.690, 0.05: 3.176, 0.10: 2.928},
    0.2: {0.01: 3.482, 0.05: 2.978, 0.10: 2.706},
    0.3: {0.01: 3.354, 0.05: 2.882, 0.10: 2.587},
    0.4: {0.01: 3.261, 0.05: 2.805, 0.10: 2.503},
    0.5: {0.01: 3.181, 0.05: 2.745, 0.10: 2.443},
}

PAIRS = [
    ('lasso_baseline',         'lasso_enhanced'),
    ('elastic_net_baseline',   'elastic_net_enhanced'),
    ('random_forest_baseline', 'random_forest_enhanced'),
    ('random_forest_baseline', 'random_forest_publag2m'),
    ('pca_baseline',           'pca_enhanced'),
    ('comb_baseline',          'comb_enhanced'),
]


def _nearest_mu_cv(mu: float, alpha: float = 0.05) -> Tuple[float, float]:
    keys = sorted(GR_CV.keys())
    nearest = min(keys, key=lambda k: abs(k - mu))
    return nearest, GR_CV[nearest][alpha]


def _hac_lrvar(d: np.ndarray, lag: int = None) -> float:
    """Newey-West long-run variance of a scalar series."""
    d = np.asarray(d, dtype=float).ravel()
    T = len(d)
    if lag is None:
        lag = max(1, int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0))))
    dm = d - d.mean()
    s2 = float(np.mean(dm ** 2))
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        s2 += 2 * w * float(np.mean(dm[k:] * dm[:-k]))
    return s2


def fluctuation_series(d: np.ndarray, m: int) -> np.ndarray:
    """Centered rolling fluctuation statistic.

    d : (T,) loss differential e_a^2 - e_b^2  (positive = b better)
    m : window size (must be even-ish; we use floor(m/2) on each side)
    Returns F_t aligned with the centers of the rolling window. Endpoints
    where the window does not fit are np.nan.
    """
    d = np.asarray(d, dtype=float).ravel()
    T = len(d)
    half = m // 2
    s_lr = np.sqrt(_hac_lrvar(d))
    if s_lr <= 0:
        return np.full(T, np.nan)

    F = np.full(T, np.nan)
    for t in range(half, T - half):
        seg = d[t - half: t - half + m]
        F[t] = np.sqrt(m) * seg.mean() / s_lr
    return F


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(PREDICTIONS_DIR, f"{name}.csv"),
                       index_col='date', parse_dates=True)


def run_pair(base: str, enh: str, m_frac: float = 0.3) -> pd.DataFrame:
    try:
        a, b = _load(base), _load(enh)
    except FileNotFoundError as e:
        print(f"  [skip] {e}")
        return None

    idx = a.index.intersection(b.index)
    y = a.loc[idx, 'y_true'].values
    e_a = y - a.loc[idx, 'y_pred'].values
    e_b = y - b.loc[idx, 'y_pred'].values
    d = e_a ** 2 - e_b ** 2  # positive => enhanced has lower loss

    T = len(d)
    m = max(8, int(round(m_frac * T)))
    if m % 2 == 1:
        m += 1
    mu = m / T
    nearest_mu, cv = _nearest_mu_cv(mu, alpha=0.05)

    F = fluctuation_series(d, m)
    df = pd.DataFrame({'F_t': F, 'd_t': d, 'cv_5pct': cv,
                       'cv_10pct': _nearest_mu_cv(mu, 0.10)[1]},
                      index=idx)
    df.attrs['m'] = m
    df.attrs['T'] = T
    df.attrs['mu'] = mu
    df.attrs['mu_nearest'] = nearest_mu

    out_csv = os.path.join(FLUCT_DIR, f'{base}__vs__{enh}.csv')
    df.to_csv(out_csv)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(df.index, df['F_t'], color='#d62728', linewidth=1.8,
            label=f'Fluctuation F_t (m={m}, T={T})')
    ax.axhline(cv, color='black', linestyle='--', linewidth=1.0,
               label=f'GR 5% CV ≈ ±{cv:.2f} (μ≈{nearest_mu})')
    ax.axhline(-cv, color='black', linestyle='--', linewidth=1.0)
    ax.axhline(0, color='gray', linewidth=0.7, alpha=0.6)
    ax.fill_between(df.index, -cv, cv, color='gray', alpha=0.06)
    ax.set_title(f'Fluctuation Test: {base}  vs  {enh}\n'
                 f'positive = enhanced lower MSE; reject equal accuracy when |F_t| > CV')
    ax.set_ylabel('Fluctuation statistic')
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    out_png = os.path.join(FLUCT_DIR, f'{base}__vs__{enh}.png')
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  {base} vs {enh}: m={m} mu={mu:.2f} cv5%≈{cv:.2f}  "
          f"max|F|={np.nanmax(np.abs(F)):.2f}  → {out_png}")
    return df


def main():
    summary = []
    for a, b in PAIRS:
        d = run_pair(a, b)
        if d is None:
            continue
        F = d['F_t'].dropna().values
        if F.size == 0:
            continue
        max_abs = float(np.nanmax(np.abs(F)))
        cv5 = float(d['cv_5pct'].iloc[0])
        cv10 = float(d['cv_10pct'].iloc[0])
        summary.append({
            'baseline': a, 'enhanced': b,
            'T': d.attrs['T'], 'm': d.attrs['m'],
            'mu': d.attrs['mu'], 'mu_nearest': d.attrs['mu_nearest'],
            'max_abs_F': max_abs,
            'cv_5pct': cv5, 'cv_10pct': cv10,
            'reject_5pct': bool(max_abs > cv5),
            'reject_10pct': bool(max_abs > cv10),
            'date_at_max': d['F_t'].abs().idxmax().strftime('%Y-%m-%d'),
        })
    summary_df = pd.DataFrame(summary)
    out = os.path.join(FLUCT_DIR, 'fluctuation_summary.csv')
    summary_df.to_csv(out, index=False)
    print("\n=== Fluctuation summary ===")
    print(summary_df.to_string(index=False))
    print(f"→ {out}")
    return summary_df


if __name__ == "__main__":
    main()
