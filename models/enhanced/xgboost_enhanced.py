"""XGBoost Enhanced — expanding-window，首次用 GridSearchCV 选超参后锁定复用。"""
import os
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from utils.data_utils import (
    setup_plot_style, load_enhanced_data, create_lag_features,
    expanding_window_predict, evaluate_and_plot, save_predictions,
    save_metrics_to_csv, ENHANCED_OUTPUT_DIR,
)

MODEL_NAME = 'xgboost_enhanced'

PARAM_GRID = {
    'n_estimators': [200, 400],
    'max_depth': [3, 5],
    'learning_rate': [0.03, 0.1],
    'subsample': [0.8, 1.0],
}


def _tune_once(data_final, target_col, start_ratio=0.8):
    """在初始训练集上用 TimeSeriesSplit + GridSearch 选一次超参，避免每步重复搜索。"""
    split_index = int(len(data_final) * start_ratio)
    X = data_final.iloc[:split_index].drop(columns=[target_col])
    y = data_final.iloc[:split_index][target_col]
    base = XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror',
                         tree_method='hist')
    gs = GridSearchCV(base, PARAM_GRID, cv=TimeSeriesSplit(n_splits=5),
                      scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=0)
    gs.fit(X.values, y.values)
    print(f">>> XGBoost best params: {gs.best_params_}  (CV RMSE={-gs.best_score_:.4f})")
    return gs.best_params_


def run_xgboost_enhanced(file_path=None):
    setup_plot_style()
    df, target_col = load_enhanced_data(file_path, stationarize=True)
    data_final = create_lag_features(df, target_col)

    best = _tune_once(data_final, target_col)

    factory = lambda: XGBRegressor(random_state=42, n_jobs=-1,
                                   objective='reg:squarederror',
                                   tree_method='hist', **best)
    y_true, y_pred, _ = expanding_window_predict(data_final, target_col, factory, scale=False)

    metrics = evaluate_and_plot(y_true, y_pred, 'XGBoost (Enhanced)',
                                os.path.join(ENHANCED_OUTPUT_DIR, '8_xgboost_enhanced.png'))
    save_predictions(MODEL_NAME, y_true, y_pred)
    save_metrics_to_csv(MODEL_NAME, **metrics)
    return metrics


if __name__ == "__main__":
    run_xgboost_enhanced()
