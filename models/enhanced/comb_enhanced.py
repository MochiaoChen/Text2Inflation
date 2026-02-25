"""
组合预测 (Combination / Comb) 模型 (Enhanced with NLP Features)
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

# Adjust path to import utils from code root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.data_utils import (
    setup_plot_style, load_enhanced_data, create_lag_features,
    split_and_scale, evaluate_and_plot
)


def run_comb_enhanced(file_path=None):
    setup_plot_style()

    # 数据加载 (Comb 原始脚本未做平稳化?? 检查原始代码... 是的，原始代码 stationarize=False)
    # 但基准模型都是 True。这里为了Enhanced对比，我们保持和原始 Comb 一致。
    # 不过通常预测通胀率（Rate）比 CPI Index 更稳健。
    # 原始 baseline/comb.py 使用 stationarize=False，这意味着它是在预测 CPI Index 本身?
    # 让我们再检查一下 utils/data_utils.py。stationarize=False 返回原始 CPI。
    # 但其他模型如 Lasso 是 stationarize=True。
    # 组合预测如果基于不同目标（Rate vs Index）混合其实是有问题的。
    # *仔细检查 baseline/comb.py*: Load data -> stationarize=False. 
    # 但是 LassoCV fit 的是 X_train_scaled, y_train.
    # 如果 stationarize=False, y_train 是 CPI Index (e.g. 102.3).
    # 如果 stationarize=True, y_train 是 Inflation Rate (e.g. 0.5%).
    
    # 用户的原始 baseline comb.py 确实写了 stationarize=False。
    # 这可能是一个 BUG，或者用户意图如此。
    # 为了保持 "Enhanced" 与 "Baseline" 的可比性，我应该遵循 Baseline 的配置。
    # 尽管我觉得应该用 True。
    # 让我们再次确认 baseline 的 comb.py
    # ... df, target_col = load_and_clean_data(file_path, stationarize=False) ...
    # 是的。
    # 为了严谨，我将在 Enhanced 中也使用 False，保持控制变量。
    
    df, target_col = load_enhanced_data(file_path, stationarize=False)

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
    print("\n>>> 预测组合 (COMB Enhanced) — 简单平均...")
    predictions['COMB_Mean'] = predictions.mean(axis=1)
    y_pred_comb = predictions['COMB_Mean']

    # 各子模型 RMSE 对比
    print("\n    ** 基准模型 RMSE 对比 **")
    for model_name in predictions.columns[:-1]:
        rmse_single = np.sqrt(mean_squared_error(y_test, predictions[model_name]))
        print(f"    - {model_name} RMSE: {rmse_single:.4f}")


    evaluate_and_plot(y_test, y_pred_comb.values, 'COMB (Mean) Enhanced', 'enhanced/6_comb_enhanced.png')


if __name__ == "__main__":
    run_comb_enhanced()
