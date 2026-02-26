import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 引入弹性网络模型及相关预处理/评估工具
from sklearn.linear_model import ElasticNetCV 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from utils import save_metrics_to_csv

# 设置绘图风格与中文字体
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def process_and_forecast_cpi(file_path):
    """
    基于Elastic Net回归的CPI预测函数
    步骤包括：数据读取与清洗、特征工程、数据集划分、标准化、模型训练、预测评估及可视化
    """
    print(">>> 1. 读取与清洗数据...")
    try:
        # 读取CSV文件并解析日期列，按日期排序后设为索引
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df.sort_values('date').set_index('date')
    except Exception as e:
        print(f"文件读取错误，请检查路径和文件名: {e}")
        return

    # 数据清洗与预处理
    df.columns = df.columns.str.strip()  # 去除列名前后空格
    original_target_col = 'CPI(2001-01=100)'
    target_col = 'CPI'
    # 重命名目标列（若存在原始目标列）
    if original_target_col in df.columns:
        df.rename(columns={original_target_col: target_col}, inplace=True)
    else:
        print(f"错误：未找到目标列 '{original_target_col}'")
        return
    
    # 提取特征列并转换为数值类型（无效值转为NaN）
    feature_cols = [col for col in df.columns if col != target_col]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') 

    # 填充缺失值（前向填充+后向填充），删除全为空的列
    df = df.ffill().bfill()
    df.dropna(axis=1, how='all', inplace=True)
    
    print(f"    数据加载完成，时间跨度: {df.index.min().date()} 至 {df.index.max().date()}")

    print("\n>>> 2. 特征工程（构造滞后变量）...")
    current_feature_cols = [col for col in df.columns if col != target_col]
    feature_dfs = []
    lags = [1, 2, 3, 6, 12]  # 滞后阶数（1/2/3/6/12期）
    
    # 为每个特征构造指定阶数的滞后变量
    for col in current_feature_cols:
        for lag in lags:
            name = f"{col}_lag{lag}"
            if col in df.columns:
                feature_dfs.append(df[col].shift(lag).rename(name))
            
    # 合并特征与目标变量，删除滞后产生的缺失值
    df_features = pd.concat(feature_dfs, axis=1)
    data_final = pd.concat([df[target_col], df_features], axis=1)
    data_final.dropna(inplace=True)
    
    print(f"    特征构造完成，最终特征数量: {data_final.shape[1]-1} 个（不含目标列）")

    print("\n>>> 3. 划分训练集与测试集...")
    split_index = int(len(data_final) * 0.8)  # 8:2划分比例
    
    train_data = data_final.iloc[:split_index]
    test_data = data_final.iloc[split_index:]
    
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    print(f"    训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

    print("\n>>> 4. 数据标准化与模型训练（Elastic Net）...")
    
    # 标准化特征（Elastic Net对尺度敏感）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 使用ElasticNetCV自动选择最优参数（alpha和l1_ratio）
    # l1_ratio：L1正则化比例（0为纯Ridge，1为纯LASSO）
    enet_model = ElasticNetCV(
        l1_ratio=[.1, .5, .7, .9, .95, .99, 1.0],  # 尝试不同L1/L2比例
        cv=5, 
        random_state=42, 
        max_iter=10000, 
        n_jobs=-1  # 使用所有CPU核心加速
    )
    
    # 训练模型
    enet_model.fit(X_train_scaled, y_train)
    
    print(f"    最佳Alpha（正则化强度）: {enet_model.alpha_:.6f}")
    print(f"    最佳L1比例（l1_ratio）: {enet_model.l1_ratio_:.2f}")
    
    # 提取并展示重要特征（系数绝对值大于阈值的特征）
    coefs = pd.Series(enet_model.coef_, index=X_train.columns)
    important_feats = coefs[coefs.abs() > 1e-4].sort_values(ascending=False)
    
    print("\n    Elastic Net筛选的关键预测因子（Top 5正向 & Top 5负向）:")
    if len(important_feats) > 10:
        print(pd.concat([important_feats.head(5), important_feats.tail(5)]))
    else:
        print(important_feats)


    print("\n>>> 5. 预测与评估...")
    y_pred = enet_model.predict(X_test_scaled)  # 测试集预测
    
    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE： {rmse:.4f}")
    print(f"MAE： {mae:.4f}")
    print(f"R²： {r2:.4f}")
    metrics_csv_path = "../../outputs/Outputs.csv"
    save_metrics_to_csv("Elastic Net", rmse, mae, r2, metrics_csv_path)

    # 绘制预测结果对比图
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='真实值', color='blue', linewidth=2)
    plt.plot(y_test.index, y_pred, label='预测值', color='red', linestyle='--', linewidth=2)
    plt.title('CPI预测结果对比（Elastic Net 模型）', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('CPI指数', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    output_dir = "../../outputs/baseline"
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在，不存在则创建

    # 保存图表
    output_path = os.path.join(output_dir, "3.Elastic Net.png")
    plt.savefig(output_path)
    print(f"图表已保存至：{output_path}")

    # 展示图表
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_file = "../../data/CPI_Data.csv"
    process_and_forecast_cpi(csv_file)