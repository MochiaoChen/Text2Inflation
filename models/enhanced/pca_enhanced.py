"""PCA + OLS Enhanced — expanding-window。"""
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from utils.data_utils import (
    setup_plot_style, load_enhanced_data, create_lag_features,
    expanding_window_predict, evaluate_and_plot, save_predictions,
    save_metrics_to_csv, ENHANCED_OUTPUT_DIR,
)

MODEL_NAME = 'pca_enhanced'


def run_pca_enhanced(file_path=None):
    setup_plot_style()
    df, target_col = load_enhanced_data(file_path, stationarize=True)
    data_final = create_lag_features(df, target_col)

    factory = lambda: Pipeline([('pca', PCA(n_components=0.95)),
                                ('ols', LinearRegression())])
    y_true, y_pred, _ = expanding_window_predict(data_final, target_col, factory)

    metrics = evaluate_and_plot(y_true, y_pred, 'PCA + OLS (Enhanced)',
                                os.path.join(ENHANCED_OUTPUT_DIR, '4_pca_enhanced.png'))
    save_predictions(MODEL_NAME, y_true, y_pred)
    save_metrics_to_csv(MODEL_NAME, **metrics)
    return metrics


if __name__ == "__main__":
    run_pca_enhanced()
