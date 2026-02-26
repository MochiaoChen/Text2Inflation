import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from utils import save_metrics_to_csv

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def process_and_forecast_cpi(file_path):
    print(">>> 正在读取和清洗数据...")
    try:
        # 1.1 读取CSV，解析日期
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df.sort_values('date').set_index('date')
    except Exception as e:
        print(f"读取文件出错，请检查文件名和路径: {e}")
        return

    # 强制清理列名空格
    df.columns = df.columns.str.strip() 
    
    # 确保目标列名正确
    original_target_col = 'CPI(2001-01=100)'
    target_col_raw = 'CPI_Index' # 暂存原始指数列名
    
    if original_target_col in df.columns:
        df.rename(columns={original_target_col: target_col_raw}, inplace=True)
    elif 'CPI' in df.columns: # 兼容可能已经被改过名的文件
        df.rename(columns={'CPI': target_col_raw}, inplace=True)
    else:
        print(f"错误：数据集中找不到原始目标列。请检查列名。")
        return
    
    # 获取特征列
    feature_cols = [col for col in df.columns if col != target_col_raw]

    # **【优化点 2】强制转换为数值类型**
    for col in feature_cols + [target_col_raw]:
        df[col] = pd.to_numeric(df[col], errors='coerce') 

    # 简单填充
    df = df.ffill().bfill()
    # **【优化点 3】丢弃全空列**
    df.dropna(axis=1, how='all', inplace=True)
    
    
    # 1. 构造目标变量：通胀率 (Inflation) = CPI指数的环比增长率
    df['Inflation'] = df[target_col_raw].pct_change() * 100
    
    # 2. 对所有特征列也求增长率/差分，确保特征和目标处于同一个"量级"
    # 注意：如果某些列本身就是百分比（如利率），再求差分代表"变化量"，也是合理的
    # 为了保证 LASSO 效果，最好让所有数据都是平稳的
    cols_to_transform = [c for c in df.columns if c != 'Inflation' and c != target_col_raw]
    
    for col in cols_to_transform:
        # 这里统一用增长率，如果是负数或0可能会有问题，但在宏观数据中通常用差分或增长率
        # 简单起见，我们对所有列做百分比变化。如果列中有0，pct_change会产生inf，需要处理
        df[col] = df[col].pct_change() * 100
        
    # 移除原始的 Level 列（CPI指数），只保留 Inflation 和转换后的特征
    df = df.drop(columns=[target_col_raw])
    
    # 处理因为差分产生的 NaN (第一行) 和 Inf
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    target_col = 'Inflation' # 新的目标变量名
    
    print(f"    数据变换完成。当前预测目标: {target_col} (CPI 环比增长率)")
    print(f"    数据时间跨度: {df.index.min().date()} 到 {df.index.max().date()}")

    # ==============================================================================

    print("\n>>> 特征工程 (构造滞后变量)...")
    
    current_feature_cols = [col for col in df.columns if col != target_col]
    feature_dfs = []
    lags = [1, 2, 3, 6, 12] 
    
    for col in current_feature_cols:
        for lag in lags:
            name = f"{col}_lag{lag}"
            feature_dfs.append(df[col].shift(lag).rename(name))
            
    # 合并特征
    df_features = pd.concat(feature_dfs, axis=1)
    data_final = pd.concat([df[target_col], df_features], axis=1)
    
    # 去除滞后产生的空值
    data_final.dropna(inplace=True)
    
    print(f"    特征构造完成。最终特征数量: {data_final.shape[1]-1} 个")

    print("\n>>> 划分训练集和测试集...")
    split_index = int(len(data_final) * 0.8)
    
    train_data = data_final.iloc[:split_index]
    test_data = data_final.iloc[split_index:]
    
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    print(f"    训练集: {len(X_train)}, 测试集: {len(X_test)}")

    print("\n>>> 数据标准化与模型训练 (LASSO)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # LassoCV 自动选择 Alpha
    lasso_model = LassoCV(cv=5, random_state=42, max_iter=10000, verbose=False)
    lasso_model.fit(X_train_scaled, y_train)
    
    print(f"    最佳 Alpha 参数: {lasso_model.alpha_:.6f}")
    
    # 打印特征重要性
    coefs = pd.Series(lasso_model.coef_, index=X_train.columns)
    important_feats = coefs[coefs != 0].sort_values(ascending=False)
    print("    LASSO筛选出的关键预测因子 (Top 5 正向 & Top 5 负向):")
    if len(important_feats) > 10:
         print(pd.concat([important_feats.head(5), important_feats.tail(5)]))
    else:
         print(important_feats)

    print("\n>>> 预测与评估...")
    y_pred = lasso_model.predict(X_test_scaled)
    
    # 计算误差指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE： {rmse:.4f}")
    print(f"MAE： {mae:.4f}")
    print(f"R²： {r2:.4f}")
    metrics_csv_path = "../../outputs/Outputs.csv"
    save_metrics_to_csv("LASSO", rmse, mae, r2, metrics_csv_path)


    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='真实值', color='blue', linewidth=2)
    plt.plot(y_test.index, y_pred, label='预测值', color='red', linestyle='--', linewidth=2)
    plt.title('通货膨胀率预测结果对比（LASSO）', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('CPI 环比增长率（%）', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    output_dir = "../../outputs/baseline"
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在，不存在则创建

    # 保存图表
    output_path = os.path.join(output_dir, "1.LASSO.png")
    plt.savefig(output_path)
    print(f"图表已保存至：{output_path}")


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_file = "../../data/CPI_Data.csv"
    process_and_forecast_cpi(csv_file)