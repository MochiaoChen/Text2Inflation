import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, ElasticNetCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from utils import save_metrics_to_csv

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

def model_comb(file_path):
    print(">>> 1. 正在读取和清洗数据...")
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df.sort_values('date').set_index('date')
    except Exception as e:
        print(f"读取文件出错，请检查文件名和路径: {e}")
        return

    # **数据清洗与预处理（保持之前的健壮性）**
    df.columns = df.columns.str.strip() 
    original_target_col = 'CPI(2001-01=100)'
    target_col = 'CPI'
    if original_target_col in df.columns:
        df.rename(columns={original_target_col: target_col}, inplace=True)
    else:
        print(f"错误：数据集中找不到原始目标列 '{original_target_col}'。")
        return
    
    feature_cols = [col for col in df.columns if col != target_col]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') 

    df = df.ffill().bfill()
    df.dropna(axis=1, how='all', inplace=True)
    
    print(f"    数据加载完成，时间跨度: {df.index.min().date()} 到 {df.index.max().date()}")

    print("\n>>> 2. 特征工程 (构造滞后变量)...")
    current_feature_cols = [col for col in df.columns if col != target_col]
    feature_dfs = []
    lags = [1, 2, 3, 6, 12] 
    
    for col in current_feature_cols:
        for lag in lags:
            name = f"{col}_lag{lag}"
            if col in df.columns:
                 feature_dfs.append(df[col].shift(lag).rename(name))
            
    df_features = pd.concat(feature_dfs, axis=1)
    data_final = pd.concat([df[target_col], df_features], axis=1)
    data_final.dropna(inplace=True)
    
    print(f"    特征构造完成。最终特征数量: {data_final.shape[1]-1} 个（不含目标列）")

    print("\n>>> 3. 划分训练集和测试集...")
    split_index = int(len(data_final) * 0.8)
    
    X_train = data_final.iloc[:split_index].drop(columns=[target_col])
    y_train = data_final.iloc[:split_index][target_col]
    X_test = data_final.iloc[split_index:].drop(columns=[target_col])
    y_test = data_final.iloc[split_index:][target_col]
    
    print(f"    训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

    print("\n>>> 4. 数据标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 初始化一个 DataFrame 来存储所有模型的预测结果
    predictions = pd.DataFrame(index=y_test.index)
    
    # --- 训练 LASSO 模型 ---
    print("\n>>> 5.1 训练 LASSO 模型...")
    lasso_model = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso_model.fit(X_train_scaled, y_train)
    predictions['LASSO'] = lasso_model.predict(X_test_scaled)

    # --- 训练 Elastic Net 模型 ---
    print(">>> 5.2 训练 Elastic Net 模型...")
    enet_model = ElasticNetCV(l1_ratio=[.1, .5, .9, 1.0], cv=5, random_state=42, max_iter=10000)
    enet_model.fit(X_train_scaled, y_train)
    predictions['ENet'] = enet_model.predict(X_test_scaled)
    
    # --- 训练 PCA + OLS 模型 ---
    print(">>> 5.3 训练 PCA + OLS 模型...")
    # 重新在训练集上进行 PCA 降维，保留 95% 方差
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_pca, y_train)
    predictions['PCA'] = lr_model.predict(X_test_pca)
    
    print(f"    完成训练 {len(predictions.columns)} 个基准模型: {predictions.columns.tolist()}")

    print("\n>>> 6. 预测组合 (Comb) 模型计算...")
    # 关键步骤：计算简单平均预测值 (Simple Mean Combination)
    predictions['COMB_Mean'] = predictions.mean(axis=1) 
    y_pred_comb = predictions['COMB_Mean']

    # 计算 Comb 模型的误差指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_comb))
    mae = mean_absolute_error(y_test, y_pred_comb)
    r2 = r2_score(y_test, y_pred_comb)
    
    print(f"    COMB 模型结果分析：")
    print(f"RMSE： {rmse:.4f}")
    print(f"MAE：: {mae:.4f}")
    print(f"R²： {r2:.4f}")
    metrics_csv_path = "../../outputs/Outputs.csv"
    save_metrics_to_csv("Comb", rmse, mae, r2, metrics_csv_path)
    
    print("基准模型结果对比：")
    # 打印所有基准模型的性能对比
    for model_name in predictions.columns[:-1]:
         rmse_single = np.sqrt(mean_squared_error(y_test, predictions[model_name]))
         print(f"·{model_name} RMSE：{rmse_single:.4f}")

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='真实值', color='blue', linewidth=2)
    plt.plot(y_test.index, y_pred_comb, label='预测值', color='darkred', linestyle='--', linewidth=2)
    plt.title('CPI 预测结果对比（Comb）', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('CPI 指数', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    output_dir = "../../outputs/baseline"
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在，不存在则创建

    # 保存图表
    output_path = os.path.join(output_dir, "6.Comb.png")
    plt.savefig(output_path)
    print(f"图表已保存至：{output_path}")
    
    # 展示图表
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_file = "../../data/CPI_Data.csv"
    model_comb(csv_file)