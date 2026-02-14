"""
随机森林回归模型预测通货膨胀率
使用 RandomForestRegressor 集成学习方法
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import (
    setup_plot_style, load_and_clean_data, create_lag_features,
    split_and_scale, evaluate_and_plot
)


def run_random_forest(file_path=None):
    setup_plot_style()

    # 数据加载与平稳化
    df, target_col = load_and_clean_data(file_path, stationarize=True)

    # 特征工程
    data_final = create_lag_features(df, target_col)

    # 划分与标准化
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = \
        split_and_scale(data_final, target_col)

    # 模型训练
    print("\n>>> 模型训练 (随机森林)...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)

    # 特征重要性
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(10)
    print("\n    随机森林特征重要性 Top 10:")
    print(top_features)

    # 预测与评估
    y_pred = rf_model.predict(X_test_scaled)
    evaluate_and_plot(y_test, y_pred, '随机森林', '3_random_forest.png')


if __name__ == "__main__":
    run_random_forest()
