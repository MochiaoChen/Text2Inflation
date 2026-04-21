"""PLS Enhanced — expanding-window。"""
import os
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, RegressorMixin

from utils.data_utils import (
    setup_plot_style, load_enhanced_data, create_lag_features,
    expanding_window_predict, evaluate_and_plot, save_predictions,
    save_metrics_to_csv, ENHANCED_OUTPUT_DIR,
)

MODEL_NAME = 'pls_enhanced'


class _PLSWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=10):
        self.n_components = n_components

    def fit(self, X, y):
        k = max(1, min(self.n_components, X.shape[1], X.shape[0] - 1))
        self._pls = PLSRegression(n_components=k)
        self._pls.fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self._pls.predict(X)).ravel()


def run_pls_enhanced(file_path=None):
    setup_plot_style()
    df, target_col = load_enhanced_data(file_path, stationarize=True)
    data_final = create_lag_features(df, target_col)

    factory = lambda: _PLSWrapper(n_components=10)
    y_true, y_pred, _ = expanding_window_predict(data_final, target_col, factory)

    metrics = evaluate_and_plot(y_true, y_pred, 'PLS (Enhanced)',
                                os.path.join(ENHANCED_OUTPUT_DIR, '5_pls_enhanced.png'))
    save_predictions(MODEL_NAME, y_true, y_pred)
    save_metrics_to_csv(MODEL_NAME, **metrics)
    return metrics


if __name__ == "__main__":
    run_pls_enhanced()
