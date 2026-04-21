"""LASSO baseline — expanding-window CPI 环比增长率预测。"""
import os
from sklearn.linear_model import LassoCV

from utils.data_utils import (
    setup_plot_style, load_and_clean_data, create_lag_features,
    expanding_window_predict, evaluate_and_plot, save_predictions,
    save_metrics_to_csv, BASELINE_OUTPUT_DIR,
)

MODEL_NAME = 'lasso_baseline'


def run_lasso_baseline(file_path=None):
    setup_plot_style()
    df, target_col = load_and_clean_data(file_path, stationarize=True)
    data_final = create_lag_features(df, target_col)

    factory = lambda: LassoCV(cv=5, random_state=42, max_iter=10000)
    y_true, y_pred, _ = expanding_window_predict(data_final, target_col, factory)

    metrics = evaluate_and_plot(y_true, y_pred, 'LASSO (Baseline)',
                                os.path.join(BASELINE_OUTPUT_DIR, '1_lasso.png'))
    save_predictions(MODEL_NAME, y_true, y_pred)
    save_metrics_to_csv(MODEL_NAME, **metrics)
    return metrics


if __name__ == "__main__":
    run_lasso_baseline()
