"""
PCA 降维 + 线性回归模型预测通货膨胀率
先用 PCA 提取主成分，再用 OLS 拟合
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import (
    setup_plot_style, load_and_clean_data, create_lag_features,
    split_and_scale, evaluate_and_plot, OUTPUT_DIR
)


def run_pca(file_path=None):
    setup_plot_style()

    # 数据加载与平稳化
    df, target_col = load_and_clean_data(file_path, stationarize=True)

    # 特征工程
    data_final = create_lag_features(df, target_col)

    # 划分与标准化
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = \
        split_and_scale(data_final, target_col)

    # PCA 降维（保留 95% 方差）
    print("\n>>> PCA 降维...")
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    n_components_kept = pca.n_components_
    print(f"    原始特征数: {X_train_scaled.shape[1]} -> 保留主成分数: {n_components_kept}")

    # 绘制碎石图
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, n_components_kept + 1), pca.explained_variance_ratio_,
            alpha=0.7, align='center', label='单个成分解释方差')
    plt.step(range(1, n_components_kept + 1), np.cumsum(pca.explained_variance_ratio_),
             where='mid', color='red', label='累积解释方差')
    plt.ylabel('解释方差比率')
    plt.xlabel('主成分')
    plt.title('PCA 碎石图：成分对信息的保留程度')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_pca_scree.png'))
    plt.show()

    # 线性回归
    print("\n>>> 模型训练 (PCA + OLS)...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_pca, y_train)

    # 预测与评估
    y_pred = lr_model.predict(X_test_pca)
    evaluate_and_plot(y_test, y_pred, 'PCA + OLS', '4_pca.png')


if __name__ == "__main__":
    run_pca()
