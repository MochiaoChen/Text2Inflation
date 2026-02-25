"""
PLS (偏最小二乘) 回归模型预测通货膨胀率 (Enhanced with NLP Features)
使用交叉验证选择最优潜变量数量
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Adjust path to import utils from code root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_utils import (
    setup_plot_style, load_enhanced_data, create_lag_features,
    split_and_scale, evaluate_and_plot, OUTPUT_DIR
)


def run_pls_enhanced(file_path=None):
    setup_plot_style()

    # 数据加载与平稳化 (Enhanced)
    df, target_col = load_enhanced_data(file_path, stationarize=True)

    # 特征工程
    data_final = create_lag_features(df, target_col)

    # 划分与标准化
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = \
        split_and_scale(data_final, target_col)

    # 交叉验证选择最优潜变量数量
    print("\n>>> 通过交叉验证选择最优潜变量数量...")
    max_components = min(X_train_scaled.shape[1], 20)
    rmse_scores = []
    cv_splitter = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    for n_comp in range(1, max_components + 1):
        pls = PLSRegression(n_components=n_comp)
        scores = cross_val_score(pls, X_train_scaled, y_train,
                                 cv=cv_splitter, scoring='neg_mean_squared_error')
        rmse = np.mean(np.sqrt(-scores))
        rmse_scores.append(rmse)

    optimal_n_components = np.argmin(rmse_scores) + 1
    print(f"    交叉验证确定的最优潜变量数量: {optimal_n_components} 个")

    # 模型训练
    print("\n>>> 模型训练 (PLS Enhanced)...")
    pls_model = PLSRegression(n_components=optimal_n_components)
    pls_model.fit(X_train_scaled, y_train)

    # 输出结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 预测与评估
    y_pred = pls_model.predict(X_test_scaled).flatten()

    # 额外评估：与基准模型对比
    y_test_values = y_test.values.ravel()
    r2 = r2_score(y_test_values, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_values, y_pred))
    mae = mean_absolute_error(y_test_values, y_pred)

    y_mean = y_train.mean()
    rmse_avg = np.sqrt(mean_squared_error(y_test_values, np.full_like(y_test_values, y_mean)))
    rRMSE = rmse / rmse_avg

    print(f"\n    --- 基准对比 ---")
    print(f" 基准历史平均 RMSE: {rmse_avg:.4f}")
    print(f" 相对 RMSE: {rRMSE:.4f}")

    output_path = os.path.join(OUTPUT_DIR, '5.PLS_enhanced.png')
    evaluate_and_plot(y_test, y_pred, 'PLS_Enhanced', output_path)


if __name__ == "__main__":
    run_pls_enhanced()
