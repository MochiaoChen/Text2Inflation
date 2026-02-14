"""
Elastic Net 回归模型预测通货膨胀率 (Enhanced with NLP Features)
使用 ElasticNetCV 自动选择 alpha 和 l1_ratio
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV

# Adjust path to import utils from code root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.data_utils import (
    setup_plot_style, load_enhanced_data, create_lag_features,
    split_and_scale, evaluate_and_plot, OUTPUT_DIR
)


def run_elastic_net_enhanced(file_path=None):
    setup_plot_style()

    # 数据加载与平稳化 (Enhanced)
    df, target_col = load_enhanced_data(file_path, stationarize=True)

    # 特征工程
    data_final = create_lag_features(df, target_col)

    # 划分与标准化
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = \
        split_and_scale(data_final, target_col)

    # 模型训练
    print("\n>>> 模型训练 (Elastic Net Enhanced)...")
    enet_model = ElasticNetCV(
        l1_ratio=[.1, .5, .7, .9, .95, .99, 1.0],
        cv=5,
        random_state=42,
        max_iter=10000,
        n_jobs=-1
    )
    enet_model.fit(X_train_scaled, y_train)

    print(f"    最佳 Alpha（正则化强度）: {enet_model.alpha_:.6f}")
    print(f"    最佳 L1 比例（l1_ratio）: {enet_model.l1_ratio_:.2f}")

    # 打印特征重要性
    coefs = pd.Series(enet_model.coef_, index=feature_names)
    important_feats = coefs[coefs.abs() > 1e-4].sort_values(ascending=False)
    print("\n    Elastic Net 筛选的关键预测因子（Top 5 正向 & Top 5 负向）:")
    if len(important_feats) > 10:
        print(pd.concat([important_feats.head(5), important_feats.tail(5)]))
    else:
        print(important_feats)

    # 确保输出目录存在
    enhanced_output_dir = os.path.join(OUTPUT_DIR, 'enhanced')
    os.makedirs(enhanced_output_dir, exist_ok=True)

    # 预测与评估
    y_pred = enet_model.predict(X_test_scaled)
    evaluate_and_plot(y_test, y_pred, 'Elastic Net Enhanced', 'enhanced/2_elastic_net_enhanced.png')


if __name__ == "__main__":
    run_elastic_net_enhanced()
