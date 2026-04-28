"""Random Forest Enhanced — publication-lag-adjusted robustness check.

Refits the enhanced RF spec but assumes each PBoC quarterly report is only
available `publication_lag_months` months after the quarter ends, removing the
mild forward-looking artifact in the default expanding-window join.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from utils.data_utils import (
    setup_plot_style, load_enhanced_data, create_lag_features,
    expanding_window_predict, evaluate_and_plot, save_predictions,
    save_metrics_to_csv, ENHANCED_OUTPUT_DIR, PREDICTIONS_DIR,
)
from utils.dm_test import dm_test, clark_west_test


def run(publication_lag_months: int = 2, file_path=None):
    setup_plot_style()
    df, target_col = load_enhanced_data(
        file_path, stationarize=True,
        publication_lag_months=publication_lag_months)
    data_final = create_lag_features(df, target_col)

    factory = lambda: RandomForestRegressor(n_estimators=300, max_depth=None,
                                            min_samples_split=3, n_jobs=-1,
                                            random_state=42)
    y_true, y_pred, _ = expanding_window_predict(
        data_final, target_col, factory, scale=False)

    name = f'random_forest_publag{publication_lag_months}m'
    metrics = evaluate_and_plot(
        y_true, y_pred, f'Random Forest (Enhanced, +{publication_lag_months}m lag)',
        os.path.join(ENHANCED_OUTPUT_DIR, f'2_random_forest_publag{publication_lag_months}m.png'))
    save_predictions(name, y_true, y_pred)
    save_metrics_to_csv(name, **metrics)

    # Compare against baseline RF predictions if available.
    base_path = os.path.join(PREDICTIONS_DIR, 'random_forest_baseline.csv')
    if os.path.exists(base_path):
        base = pd.read_csv(base_path, index_col='date', parse_dates=True)
        idx = base.index.intersection(y_true.index)
        yt = base.loc[idx, 'y_true'].values
        y_b = base.loc[idx, 'y_pred'].values
        y_e = pd.Series(y_pred, index=y_true.index).loc[idx].values
        e1 = yt - y_b
        e2 = yt - y_e
        dm, p = dm_test(e1, e2)
        cw, cw_p = clark_west_test(yt, y_b, y_e)
        print(f"\nDM stat = {dm:.3f} (p={p:.3f})  CW stat = {cw:.3f} (p={cw_p:.3f})")
        rmse_b = float(np.sqrt(np.mean(e1 ** 2)))
        rmse_e = float(np.sqrt(np.mean(e2 ** 2)))
        print(f"RMSE baseline = {rmse_b:.4f}  RMSE pub-lag enhanced = {rmse_e:.4f}  "
              f"({100*(rmse_b-rmse_e)/rmse_b:+.2f}%)")
    return metrics


if __name__ == "__main__":
    lag = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    run(publication_lag_months=lag)
