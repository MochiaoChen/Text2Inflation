import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 引入PLS回归模型、交叉验证工具及相关预处理/评估工具
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler # 必须引入标准化
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from utils import save_metrics_to_csv

# ===================== 全局绘图配置 =====================
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def process_and_forecast_inflation_pls(file_path):
    """
    基于PLS回归模型的通货膨胀率预测函数 (预测目标：环比通胀率)
    """
    # ===================== 1. 数据读取与基础清洗 =====================
    print(">>> 1. 正在读取和清洗原始数据...")
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df.sort_values('date').set_index('date')
    except Exception as e:
        print(f"读取错误: {e}（请检查文件路径和格式）")
        return

    df.columns = df.columns.str.strip()
    
    if 'CPI(2001-01=100)' in df.columns:
        df.rename(columns={'CPI(2001-01=100)': 'CPI_Index'}, inplace=True)
    elif 'CPI' not in df.columns:
        print("错误: 找不到 CPI 指数列 ('CPI(2001-01=100)' 或 'CPI_Index')")
        return
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce') 
    
    df = df.ffill().bfill()
    df.dropna(axis=1, how='all', inplace=True)

    # ===================== 2. 特征工程：构造增长率与滞后特征 =====================
    print(">>> 2. 正在进行特征工程（环比增长率与滞后特征）...")
    
    # 核心目标变量：CPI环比增长率（通胀率，%）
    df['Inflation'] = df['CPI_Index'].pct_change() * 100 
    
    feature_cols = [col for col in df.columns if col != 'CPI_Index' and col != 'Inflation']
    
    # 特征标准化：所有特征转换为环比增长率
    for col in feature_cols:
        # 重命名为 Growth，同时移除原始 Level 特征
        df[col] = df[col].pct_change() * 100
    
    # 构建建模数据集：仅保留增长率特征和通胀率
    df_modeling = df.drop(columns=['CPI_Index']) # 移除原始指数
    df_modeling = df_modeling.replace([np.inf, -np.inf], np.nan)
    
    # 构造滞后特征
    target_col = 'Inflation'
    current_feature_cols = [col for col in df_modeling.columns if col != target_col]
    feature_dfs = []
    lags = [1, 2, 3, 6, 12]
    
    for col in current_feature_cols:
        for lag in lags:
            lag_feature = df_modeling[col].shift(lag).rename(f"{col}_lag{lag}")
            feature_dfs.append(lag_feature)
            
    df_features = pd.concat(feature_dfs, axis=1)
    data_final = pd.concat([df_modeling[target_col], df_features], axis=1)
    data_final.dropna(inplace=True)
    print(f"    特征构造完成。最终特征数量: {data_final.shape[1]-1} 个 (不含目标列)")

    # ===================== 3. 数据集划分与标准化预处理 =====================
    print("\n>>> 3. 正在划分数据集并标准化...")
    split_index = int(len(data_final) * 0.8)
    
    X_train = data_final.iloc[:split_index].drop(columns=[target_col])
    y_train = data_final.iloc[:split_index][target_col]
    X_test = data_final.iloc[split_index:].drop(columns=[target_col])
    y_test = data_final.iloc[split_index:][target_col]
    
    # 确保数据集有效
    if X_train.empty or X_test.empty:
        print("\n!!! 错误：数据清洗后训练集或测试集为空。请检查原始数据质量。")
        return

    # **PLS需要标准化**
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_test_values = y_test.values.ravel()
    
    print(f"    训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
    
    # ===================== 4. 最优潜变量选择（n_components） =====================
    print("\n>>> 4. 正在通过交叉验证选择最优潜变量数量...")
    max_components = min(X_train_scaled.shape[1], 20)
    rmse_scores = []
    cv_splitter = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    for n_comp in range(1, max_components + 1):
        pls = PLSRegression(n_components=n_comp)
        # 计算交叉验证的负均方误差（转为RMSE）
        scores = cross_val_score(pls, X_train_scaled, y_train, 
                                 cv=cv_splitter, scoring='neg_mean_squared_error')
        rmse = np.mean(np.sqrt(-scores))
        rmse_scores.append(rmse)

    optimal_n_components = np.argmin(rmse_scores) + 1
    print(f"    交叉验证确定的最优潜变量数量: {optimal_n_components} 个")


    # ===================== 5. PLS 模型训练与预测 =====================
    print("\n>>> 5. PLS 模型训练...")
    pls_model = PLSRegression(n_components=optimal_n_components)
    pls_model.fit(X_train_scaled, y_train)
    
    print("    PLS回归模型训练完成")
    
    # 测试集预测
    y_pred = pls_model.predict(X_test_scaled).flatten()
    
    # ===================== 6. 模型评估与基准对比 =====================
    print("\n>>> 6. 模型预测与评估（R² 与 相对 RMSE）...")

    # 1. 评估模型性能
    r2 = r2_score(y_test_values, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_values, y_pred))
    mae = mean_absolute_error(y_test_values, y_pred)

    # 2. 计算基准模型 (历史平均) 的RMSE
    y_mean = y_train.mean() 
    rmse_avg = np.sqrt(mean_squared_error(y_test_values, np.full_like(y_test_values, y_mean)))
    rRMSE = rmse / rmse_avg # 相对RMSE (rRMSE < 1.0 表示模型好于基准)
    
    # 输出评估结果
    print(" *评估结果：")
    print(f"MAE： {mae:.4f}")
    print(f"RMSE： {rmse:.4f}")
    print(f"R²： {r2:.4f}")
    print(f"【基准对比】基准历史平均RMSE: {rmse_avg:.4f}")
    print(f"【基准对比】相对RMSE: {rRMSE:.4f}") # 关注此值是否小于 1.0
    metrics_csv_path = "../../outputs/Outputs.csv"
    save_metrics_to_csv("PLS", rmse, mae, r2, metrics_csv_path)


    # ===================== 7. 结果可视化 =====================
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test_values, label='真实值', color='blue')
    plt.plot(y_test.index, y_pred, label='预测值', color='purple', linestyle='--')
    
    plt.title('通货膨胀率预测结果对比（PLS回归）', fontsize=16)
    plt.xlabel('日期')
    plt.ylabel('CPI 环比增长率 (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_dir = "../../outputs/baseline"
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在，不存在则创建

    # 保存图表
    output_path = os.path.join(output_dir, "5.PLS.png")
    plt.savefig(output_path)
    print(f"图表已保存至：{output_path}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_file = "../../data/CPI_Data.csv"
    process_and_forecast_inflation_pls(csv_file)