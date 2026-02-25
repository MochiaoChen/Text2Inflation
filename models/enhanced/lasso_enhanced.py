"""
LASSO 回归模型预测通货膨胀率 (Enhanced with NLP Features)
使用 LassoCV 自动选择正则化参数
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV

# Adjust path to import utils from code root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.data_utils import (
    setup_plot_style, load_enhanced_data, create_lag_features,
    split_and_scale, evaluate_and_plot
)


def run_lasso_enhanced(file_path=None):
    setup_plot_style()

    # 数据加载与平稳化 (Enhanced)
    df, target_col = load_enhanced_data(file_path, stationarize=True)

    # 特征工程
    data_final = create_lag_features(df, target_col)

    # 划分与标准化
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = \
        split_and_scale(data_final, target_col)

    # 模型训练
    print("\n>>> 模型训练 (LASSO Enhanced)...")
    lasso_model = LassoCV(cv=5, random_state=42, max_iter=10000, verbose=False)
    lasso_model.fit(X_train_scaled, y_train)

    print(f"    最佳 Alpha 参数: {lasso_model.alpha_:.6f}")

    # 打印特征重要性
    coefs = pd.Series(lasso_model.coef_, index=feature_names)
    important_feats = coefs[coefs != 0].sort_values(ascending=False)
    print("    LASSO 筛选出的关键预测因子 (Top 5 正向 & Top 5 负向):")
    if len(important_feats) > 10:
        print(pd.concat([important_feats.head(5), important_feats.tail(5)]))
    else:
        print(important_feats)

    # 预测与评估
    y_pred = lasso_model.predict(X_test_scaled)
    evaluate_and_plot(y_test, y_pred, 'LASSO Enhanced', 'enhanced/1_lasso_enhanced.png')


if __name__ == "__main__":
    run_lasso_enhanced()
