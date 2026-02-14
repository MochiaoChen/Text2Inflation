"""
组合预测 (Combination / Comb) 模型
对 LASSO、Elastic Net、PCA+OLS 三个基准模型的预测结果取简单平均
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, ElasticNetCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import (
    setup_plot_style, load_and_clean_data, create_lag_features,
    split_and_scale, evaluate_and_plot, OUTPUT_DIR
)


def run_comb(file_path=None):
    setup_plot_style()

    # 数据加载（Comb 原始脚本未做平稳化，保持原逻辑）
    df, target_col = load_and_clean_data(file_path, stationarize=False)

    # 特征工程
    data_final = create_lag_features(df, target_col)

    # 划分与标准化
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = \
        split_and_scale(data_final, target_col)

    # 初始化预测结果存储
    predictions = pd.DataFrame(index=y_test.index)

    # --- LASSO ---
    print("\n>>> 训练 LASSO 模型...")
    lasso_model = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso_model.fit(X_train_scaled, y_train)
    predictions['LASSO'] = lasso_model.predict(X_test_scaled)

    # --- Elastic Net ---
    print(">>> 训练 Elastic Net 模型...")
    enet_model = ElasticNetCV(l1_ratio=[.1, .5, .9, 1.0], cv=5, random_state=42, max_iter=10000)
    enet_model.fit(X_train_scaled, y_train)
    predictions['ENet'] = enet_model.predict(X_test_scaled)

    # --- PCA + OLS ---
    print(">>> 训练 PCA + OLS 模型...")
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    lr_model = LinearRegression()
    lr_model.fit(X_train_pca, y_train)
    predictions['PCA'] = lr_model.predict(X_test_pca)

    print(f"    完成训练 {len(predictions.columns)} 个基准模型: {predictions.columns.tolist()}")

    # Comb: 简单平均
    print("\n>>> 预测组合 (COMB) — 简单平均...")
    predictions['COMB_Mean'] = predictions.mean(axis=1)
    y_pred_comb = predictions['COMB_Mean']

    # 各子模型 RMSE 对比
    print("\n    ** 基准模型 RMSE 对比 **")
    for model_name in predictions.columns[:-1]:
        rmse_single = np.sqrt(mean_squared_error(y_test, predictions[model_name]))
        print(f"    - {model_name} RMSE: {rmse_single:.4f}")

    evaluate_and_plot(y_test, y_pred_comb.values, 'COMB (Mean)', '6_comb.png')


if __name__ == "__main__":
    run_comb()
