"""Random Forest Enhanced — expanding-window。"""
import os
from sklearn.ensemble import RandomForestRegressor

from utils.data_utils import (
    setup_plot_style, load_enhanced_data, create_lag_features,
    expanding_window_predict, evaluate_and_plot, save_predictions,
    save_metrics_to_csv, ENHANCED_OUTPUT_DIR,
)

MODEL_NAME = 'random_forest_enhanced'


def run_random_forest_enhanced(file_path=None):
    setup_plot_style()
    df, target_col = load_enhanced_data(file_path, stationarize=True)
    data_final = create_lag_features(df, target_col)

    factory = lambda: RandomForestRegressor(n_estimators=300, max_depth=None,
                                            min_samples_split=3, n_jobs=-1,
                                            random_state=42)
    y_true, y_pred, _ = expanding_window_predict(data_final, target_col, factory, scale=False)

    metrics = evaluate_and_plot(y_true, y_pred, 'Random Forest (Enhanced)',
                                os.path.join(ENHANCED_OUTPUT_DIR, '2_random_forest_enhanced.png'))
    save_predictions(MODEL_NAME, y_true, y_pred)
    save_metrics_to_csv(MODEL_NAME, **metrics)
    return metrics


if __name__ == "__main__":
    run_random_forest_enhanced()
