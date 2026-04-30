"""Quantile Random Forest density forecasts — baseline vs enhanced.

QRF approximation: each tree's leaf-mean prediction is treated as one ensemble
draw. With 500 trees we get a 500-point empirical predictive distribution per
forecast date — enough to estimate central quantiles (5th–95th) reliably.
This avoids depending on `sklearn-quantile` while remaining methodologically
defensible (Meinshausen 2006 uses leaf samples; using leaf means is the
"forest of trees" approximation common in econometric forecasting, e.g.
Coulombe et al. 2022).

Outputs (under outputs/density/):
- {model}_quantiles.csv     — date × τ table of predicted quantiles
- {model}_samples.npz       — (T, n_trees) ensemble matrix (for CRPS)
- density_metrics.csv       — CRPS, pinball, coverage by model
- christoffersen.csv        — coverage tests
- pi_width_with_ambiguity.png — figure 1 of plan A
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from utils.data_utils import (
    setup_plot_style, load_and_clean_data, load_enhanced_data,
    create_lag_features, OUTPUT_DIR,
)
from utils.density_eval import (
    crps_series, pinball_loss, coverage_rate, average_interval_width,
    christoffersen_test, dm_crps,
)


DENSITY_DIR = os.path.join(OUTPUT_DIR, 'density')
os.makedirs(DENSITY_DIR, exist_ok=True)

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
N_TREES = 500
START_RATIO = 0.8


def _fit_predict_ensemble(X_tr, y_tr, X_te) -> np.ndarray:
    """Fit RF, return per-tree predictions on X_te → shape (n_test, n_trees)."""
    rf = RandomForestRegressor(
        n_estimators=N_TREES, max_depth=None, min_samples_split=3,
        n_jobs=-1, random_state=42,
    )
    rf.fit(X_tr, y_tr)
    # stack each tree's prediction → (n_trees, n_test) → transpose
    per_tree = np.stack([est.predict(X_te) for est in rf.estimators_], axis=0)
    return per_tree.T  # (n_test, n_trees)


def expanding_window_qrf(data_final: pd.DataFrame, target_col: str,
                         start_ratio: float = START_RATIO):
    """Expanding-window QRF. Returns (y_true, samples_matrix, dates).

    samples_matrix : (T_test, n_trees) — ensemble predictions per forecast date.
    """
    X_all = data_final.drop(columns=[target_col]).values
    y_all = data_final[target_col].values
    dates = data_final.index

    split = int(len(data_final) * start_ratio)
    truths, samples_rows, idx = [], [], []
    for t in range(split, len(data_final)):
        X_tr = X_all[:t]
        y_tr = y_all[:t]
        X_te = X_all[t:t + 1]
        ens = _fit_predict_ensemble(X_tr, y_tr, X_te)  # (1, n_trees)
        truths.append(float(y_all[t]))
        samples_rows.append(ens[0])
        idx.append(dates[t])

    samples = np.vstack(samples_rows)
    y_true = pd.Series(truths, index=pd.DatetimeIndex(idx), name='y_true')
    return y_true, samples


def quantiles_from_samples(samples: np.ndarray, taus=QUANTILES) -> pd.DataFrame:
    qs = np.quantile(samples, taus, axis=1).T  # (T, len(taus))
    return pd.DataFrame(qs, columns=[f'q{int(100*t):02d}' for t in taus])


@dataclass
class DensityRun:
    name: str
    y_true: pd.Series
    samples: np.ndarray
    quantiles_df: pd.DataFrame  # indexed by date, cols q05..q95


def run_one(name: str, df: pd.DataFrame, target_col: str) -> DensityRun:
    print(f"\n>>> [{name}] expanding-window QRF...")
    data_final = create_lag_features(df, target_col)
    y_true, samples = expanding_window_qrf(data_final, target_col)
    qdf = quantiles_from_samples(samples)
    qdf.index = y_true.index

    # persist
    qdf.to_csv(os.path.join(DENSITY_DIR, f'{name}_quantiles.csv'))
    np.savez_compressed(os.path.join(DENSITY_DIR, f'{name}_samples.npz'),
                        samples=samples, y_true=y_true.values,
                        dates=np.array([str(d) for d in y_true.index]))
    return DensityRun(name=name, y_true=y_true, samples=samples,
                      quantiles_df=qdf)


def metrics_for_run(run: DensityRun) -> dict:
    y = run.y_true.values
    crps = crps_series(run.samples, y)
    qdf = run.quantiles_df
    out = {
        'model': run.name,
        'n': len(y),
        'mean_crps': float(np.mean(crps)),
        'rmse_median': float(np.sqrt(np.mean((y - qdf['q50'].values) ** 2))),
        'mae_median': float(np.mean(np.abs(y - qdf['q50'].values))),
    }
    # pinball at every τ
    for tau, col in zip(QUANTILES, qdf.columns):
        out[f'pinball_{col}'] = pinball_loss(y, qdf[col].values, tau)
    # 80% PI: q10–q90, 90% PI: q05–q95
    out['cov80'] = coverage_rate(y, qdf['q10'].values, qdf['q90'].values)
    out['cov90'] = coverage_rate(y, qdf['q05'].values, qdf['q95'].values)
    out['width80'] = average_interval_width(qdf['q10'].values, qdf['q90'].values)
    out['width90'] = average_interval_width(qdf['q05'].values, qdf['q95'].values)
    return out, crps


def christoffersen_for_run(run: DensityRun):
    y = run.y_true.values
    qdf = run.quantiles_df
    rows = []
    for level, lo, hi in [(0.80, 'q10', 'q90'), (0.90, 'q05', 'q95')]:
        hits = ((y < qdf[lo].values) | (y > qdf[hi].values)).astype(int)
        target = 1 - level
        r = christoffersen_test(hits, alpha_target=target)
        r['model'] = run.name
        r['pi_level'] = level
        rows.append(r)
    return rows


def figure_pi_width_vs_ambiguity(base_run: DensityRun, enh_run: DensityRun,
                                 enh_df: pd.DataFrame):
    """Plot 80% PI width over time for baseline vs enhanced, overlay TXT_Ambiguity."""
    setup_plot_style()
    qb, qe = base_run.quantiles_df, enh_run.quantiles_df
    width_b = qb['q90'] - qb['q10']
    width_e = qe['q90'] - qe['q10']

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.plot(width_b.index, width_b.values, label='Baseline 80% PI width',
             color='#1f77b4', linewidth=2)
    ax1.plot(width_e.index, width_e.values, label='Enhanced 80% PI width',
             color='#d62728', linewidth=2, linestyle='--')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('80% Prediction Interval Width (pct points)')
    ax1.grid(True, linestyle='--', alpha=0.5)

    if 'TXT_Ambiguity' in enh_df.columns:
        amb = enh_df['TXT_Ambiguity'].reindex(width_e.index, method='ffill')
        ax2 = ax1.twinx()
        ax2.plot(amb.index, amb.values, color='gray', alpha=0.6,
                 linewidth=1.2, label='TXT_Ambiguity (PBoC narrative)')
        ax2.set_ylabel('TXT_Ambiguity (0–1)')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')

    plt.title('Predictive Interval Width vs. Narrative Ambiguity', fontsize=14)
    plt.tight_layout()
    out = os.path.join(DENSITY_DIR, 'pi_width_vs_ambiguity.png')
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"    figure → {out}")


def main():
    setup_plot_style()

    # Baseline (no NLP)
    df_b, target_b = load_and_clean_data(stationarize=True)
    base_run = run_one('baseline_qrf', df_b, target_b)

    # Enhanced (with publication-lag-corrected NLP join)
    df_e, target_e = load_enhanced_data(stationarize=True,
                                        publication_lag_months=2)
    enh_run = run_one('enhanced_qrf', df_e, target_e)

    # Align overlap (baseline uses fewer rows than enhanced after NLP merge)
    common_idx = base_run.y_true.index.intersection(enh_run.y_true.index)
    print(f"\nOverlap: baseline {len(base_run.y_true)} obs, "
          f"enhanced {len(enh_run.y_true)} obs, common {len(common_idx)}")

    def _trim(run: DensityRun) -> DensityRun:
        mask = run.y_true.index.isin(common_idx)
        return DensityRun(
            name=run.name,
            y_true=run.y_true[mask],
            samples=run.samples[mask],
            quantiles_df=run.quantiles_df.loc[mask],
        )
    base_run = _trim(base_run)
    enh_run = _trim(enh_run)

    # Metrics
    rows = []
    crps_a = crps_b = None
    for run in (base_run, enh_run):
        m, c = metrics_for_run(run)
        rows.append(m)
        if run.name == 'baseline_qrf':
            crps_a = c
        else:
            crps_b = c

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(DENSITY_DIR, 'density_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n>>> Density metrics → {metrics_path}")
    print(metrics_df.to_string(index=False))

    # DM on CRPS
    dm_stat, p = dm_crps(crps_a, crps_b)
    print(f"\nDM on CRPS (baseline vs enhanced): stat={dm_stat:.3f}  p={p:.3f}")
    pd.DataFrame([{
        'metric': 'CRPS', 'baseline_mean': float(np.mean(crps_a)),
        'enhanced_mean': float(np.mean(crps_b)),
        'pct_improvement': 100 * (np.mean(crps_a) - np.mean(crps_b)) / np.mean(crps_a),
        'dm_stat': dm_stat, 'p_value': p, 'n': len(crps_a),
    }]).to_csv(os.path.join(DENSITY_DIR, 'dm_crps.csv'), index=False)

    # Christoffersen
    cs_rows = []
    for run in (base_run, enh_run):
        cs_rows.extend(christoffersen_for_run(run))
    cs_df = pd.DataFrame(cs_rows)
    cs_path = os.path.join(DENSITY_DIR, 'christoffersen.csv')
    cs_df.to_csv(cs_path, index=False)
    print(f"\n>>> Christoffersen tests → {cs_path}")
    print(cs_df.to_string(index=False))

    # Headline figure
    figure_pi_width_vs_ambiguity(base_run, enh_run, df_e)

    return metrics_df, cs_df


if __name__ == "__main__":
    main()
