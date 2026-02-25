import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

# 设置绘图风格与中文字体
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def process_and_forecast_inflation_pca(file_path):
    """
    基于PCA降维+线性回归的通货膨胀率预测函数 (修正版)
    """
    print(">>> 读取、清洗与平稳化数据...")
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df.sort_values('date').set_index('date')
    except Exception as e:
        print(f"文件读取错误: {e}")
        return

    df.columns = df.columns.str.strip()
    original_target_col = 'CPI(2001-01=100)'
    target_col_raw = 'CPI_Index'
    
    if original_target_col in df.columns:
        df.rename(columns={original_target_col: target_col_raw}, inplace=True)
    elif 'CPI' in df.columns:
        df.rename(columns={'CPI': target_col_raw}, inplace=True)
    else:
        print(f"错误：未找到目标列。")
        return
    
    # 强制转换为数值类型
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce') 

    # 填充缺失值
    df = df.ffill().bfill()
    df.dropna(axis=1, how='all', inplace=True)
    
    print(f"    数据加载完成，时间跨度: {df.index.min().date()} 至 {df.index.max().date()}")

    
    # 目标变量：通胀率 (Inflation) = CPI环比增长率
    df['Inflation'] = df[target_col_raw].pct_change() * 100
    
    # 特征变量：对所有其他特征也做增长率/差分处理，使其平稳
    cols_to_transform = [c for c in df.columns if c != 'Inflation' and c != target_col_raw]
    for col in cols_to_transform:
        df[col] = df[col].pct_change() * 100
        
    # 移除原始的 CPI 指数
    df = df.drop(columns=[target_col_raw])
    
    # 处理因差分产生的 NaN (第一行) 和 Inf
    df_modeling = df.replace([np.inf, -np.inf], np.nan).dropna()
    target_col = 'Inflation' 
    
    print(f"    数据变换完成。当前预测目标: {target_col} (CPI 环比增长率)")
    # ==============================================================================


    print("\n>>> 特征工程（构造滞后变量）...")
    current_feature_cols = [col for col in df_modeling.columns if col != target_col]
    feature_dfs = []
    lags = [1, 2, 3, 6, 12] 
    
    for col in current_feature_cols:
        for lag in lags:
            name = f"{col}_lag{lag}"
            feature_dfs.append(df_modeling[col].shift(lag).rename(name))
            
    df_features = pd.concat(feature_dfs, axis=1)
    data_final = pd.concat([df_modeling[target_col], df_features], axis=1)
    data_final.dropna(inplace=True)
    
    print(f"    特征构造完成，最终特征数量: {data_final.shape[1]-1} 个")

    print("\n>>> 划分训练集与测试集...")
    split_index = int(len(data_final) * 0.8) 
    
    X_train = data_final.iloc[:split_index].drop(columns=[target_col])
    y_train = data_final.iloc[:split_index][target_col]
    X_test = data_final.iloc[split_index:].drop(columns=[target_col])
    y_test = data_final.iloc[split_index:][target_col]
    
    print(f"    训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

    print("\n>>> 数据标准化与PCA降维...")
    
    # 5.1 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5.2 PCA降维（保留95%方差的主成分）
    pca = PCA(n_components=0.95) 
    X_train_pca = pca.fit_transform(X_train_scaled) 
    X_test_pca = pca.transform(X_test_scaled) 
    
    n_components_kept = pca.n_components_
    print(f"    PCA降维完成。原始特征数: {X_train.shape[1]} -> 保留主成分数: {n_components_kept}")
    
    # 绘制碎石图 
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, n_components_kept + 1), pca.explained_variance_ratio_, 
             alpha=0.7, align='center', label='单个成分解释方差')
    plt.step(range(1, n_components_kept + 1), np.cumsum(pca.explained_variance_ratio_), 
              where='mid', color='red', label='累积解释方差')
    plt.ylabel('解释方差比率')
    plt.xlabel('主成分')
    plt.title('PCA碎石图：成分对信息的保留程度')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n>>> 模型训练（线性回归）...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_pca, y_train)
    print("    基于PCA因子的线性回归模型训练完成")

    print("\n>>> 预测与评估...")
    y_pred = lr_model.predict(X_test_pca) 
    
    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"    测试集RMSE（均方根误差）: {rmse:.4f}")
    print(f"    测试集MAE（平均绝对误差）: {mae:.4f}")
    print(f"    测试集R²（拟合优度）: {r2:.4f}") # R² 应为正数

    # 绘制预测结果对比图
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='真实值', color='blue', linewidth=2)
    plt.plot(y_test.index, y_pred, label='预测值', color='green', linestyle='--', linewidth=2)
    plt.title('通货膨胀率预测结果对比（PCA因子模型）', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('CPI 环比增长率 (%)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    output_dir = "../../outputs/baseline"
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在，不存在则创建

    # 保存图表
    output_path = os.path.join(output_dir, "4.PCA.png")
    plt.savefig(output_path)
    print(f"图表已保存至：{output_path}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_file = "../../data/CPI_Data.csv"
    process_and_forecast_inflation_pca(csv_file)