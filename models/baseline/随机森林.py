import os   # 文件路径操作（检查文件存在、创建目录、构建保存路径）
import pandas as pd  # 数据处理（读取、清洗、转换）
import numpy as np   # 数值计算（数组操作、数学运算、异常值处理）
import matplotlib.pyplot as plt  # 数据可视化（绘制预测结果对比图）
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归模型（核心预测模型）
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 模型评估指标
import seaborn as sns  # 美化绘图风格
from utils import save_metrics_to_csv  # 自定义工具函数：保存评估指标到CSV文件（便于结果记录和对比）

# ===================== 全局绘图配置 =====================
# 设置seaborn绘图风格（白色网格），提升图表可读性
sns.set(style="whitegrid")
# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常问题

def process_and_forecast_inflation(file_path):
    """
    基于随机森林回归模型的通货膨胀率预测函数
    核心流程：数据读取清洗 → 特征工程（增长率+滞后特征）→ 数据集划分 → 模型训练 → 评估 → 可视化
    
    Parameters:
        file_path (str): CPI原始数据CSV文件的路径（需包含date日期列和CPI指数列）
    
    Returns:
        None: 函数直接输出评估指标并保存可视化图表
    """
    # ===================== 1. 数据读取与基础清洗 =====================
    print(">>> 正在读取和清洗原始数据...")
    try:
        # 读取CSV文件，解析date列为日期类型
        df = pd.read_csv(file_path, parse_dates=['date'])
        # 按日期排序并设置为索引（时间序列数据必须保证时序性）
        df = df.sort_values('date').set_index('date')
    except Exception as e:
        print(f"读取错误: {e}（请检查文件路径和格式）")
        return

    # 去除列名首尾空格（避免因列名含空格导致的索引错误）
    df.columns = df.columns.str.strip()
    
    # 统一CPI指数列名（兼容不同命名格式）
    if 'CPI(2001-01=100)' in df.columns:
        df.rename(columns={'CPI(2001-01=100)': 'CPI_Index'}, inplace=True)
    else:
        print("错误: 找不到 CPI 指数列 ('CPI(2001-01=100)' 或 'CPI_Index')")
        return
    
    # 将所有列转换为数值类型（无法转换的设为NaN，处理隐性非数值缺失值）
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce') 
    
    # 缺失值处理：前向填充→后向填充（时间序列优先时序填充，保留趋势）
    df = df.ffill().bfill()
    # 删除全为空值的列（无有效信息的特征无需保留）
    df.dropna(axis=1, how='all', inplace=True)

    # ===================== 2. 特征工程：构造增长率与滞后特征 =====================
    # 计算CPI环比增长率（通胀率，%）：核心目标变量
    df['Inflation'] = df['CPI_Index'].pct_change() * 100 
    
    # 提取特征列（排除目标变量相关列）
    feature_cols = [col for col in df.columns if col != 'CPI_Index' and col != 'Inflation']
    # 特征标准化：所有特征转换为环比增长率（保证特征与目标变量量纲一致，提升模型效果）
    for col in feature_cols:
        df[col + '_Growth'] = df[col].pct_change() * 100
        
    # 构建建模数据集：仅保留增长率特征和通胀率（移除原始指数和特征，避免信息冗余）
    df_modeling = df.drop(columns=feature_cols + ['CPI_Index'])
    # 替换无穷值为NaN（pct_change可能因分母为0产生inf/-inf，需处理）
    df_modeling = df_modeling.replace([np.inf, -np.inf], np.nan)
    
    # 构造滞后特征（捕捉时间序列的滞后效应，通胀率受历史周期影响显著）
    target_col = 'Inflation'  # 目标变量：通胀率
    current_cols = df_modeling.columns
    feature_dfs = []
    lags = [1, 2, 3, 6, 12]  # 滞后阶数：1/2/3个月（短期）、6个月（中期）、12个月（年度周期）
    
    for col in current_cols:
        if col != target_col:
            for lag in lags:
                # 为每个特征构造不同滞后阶数的特征（命名格式：特征名_lag阶数）
                lag_feature = df_modeling[col].shift(lag).rename(f"{col}_lag{lag}")
                feature_dfs.append(lag_feature)
    
    # 合并所有滞后特征
    df_features = pd.concat(feature_dfs, axis=1)
    # 合并目标变量和滞后特征，形成最终建模数据集
    data_final = pd.concat([df_modeling[target_col], df_features], axis=1)
    # 删除滞后特征引入的NaN（滞后12期会导致前12行数据缺失）
    data_final.dropna(inplace=True)
    print(f"    特征构造完成。最终特征数量: {data_final.shape[1]-1} 个 (不含目标列)")

    # ===================== 3. 数据集划分与最终清洗 =====================
    # 时序数据划分：8:2比例划分训练集/测试集（不随机划分，保留时序性）
    split_index = int(len(data_final) * 0.8)
    
    X_train = data_final.iloc[:split_index].drop(columns=[target_col])  # 训练集特征
    y_train = data_final.iloc[:split_index][target_col]                # 训练集标签
    X_test = data_final.iloc[split_index:].drop(columns=[target_col])  # 测试集特征
    y_test = data_final.iloc[split_index:][target_col]                 # 测试集标签
    
    print(f"    训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

    # 最终清洗：移除残留的NaN行（防止模型训练报错）
    X_train = X_train.dropna(axis=0)
    y_train = y_train.loc[X_train.index]  # 同步标签索引，保证样本一一对应
    
    X_test = X_test.dropna(axis=0)
    y_test = y_test.loc[X_test.index]     # 同步标签索引

    # 校验数据集有效性：空数据集直接终止流程
    if X_train.empty or X_test.empty:
        print("\n!!! 错误：数据清洗后训练集或测试集为空。请检查原始数据质量。")
        return
    
    # 格式转换：确保标签为1D数组（适配sklearn模型输入要求）
    y_train = y_train.values.ravel()
    y_test_values = y_test.values.ravel()

    # ===================== 4. 随机森林模型训练 =====================
    # 初始化随机森林回归模型（参数经经验调优，平衡拟合效果与过拟合风险）
    rf_model = RandomForestRegressor(
        n_estimators=200,        # 决策树数量：200棵保证模型稳定性
        max_depth=5,             # 树最大深度：限制复杂度，防止过拟合
        min_samples_leaf=5,      # 叶节点最小样本数：降低噪声影响
        random_state=42,         # 随机种子：保证结果可复现
        n_jobs=-1                # 并行计算：使用所有CPU核心加速训练
    )
    # 模型拟合训练集
    rf_model.fit(X_train, y_train)

    # ===================== 5. 模型预测与评估 =====================
    # 测试集预测
    y_pred = rf_model.predict(X_test)
    
    # 计算评估指标（多角度衡量预测准确性）
    r2 = r2_score(y_test_values, y_pred)          # 决定系数：解释数据变异的能力（0~1，越接近1越好）
    rmse = np.sqrt(mean_squared_error(y_test_values, y_pred))  # 均方根误差：衡量平均偏差（越小越好）
    mae = mean_absolute_error(y_test_values, y_pred)           # 平均绝对误差：衡量绝对偏差（越小越好）
    
    # 输出评估结果
    print(f"MAE： {mae:.4f}")
    print(f"RMSE： {rmse:.4f}")
    print(f"R²： {r2:.4f}")
    metrics_csv_path = "../../outputs/Outputs.csv"
    save_metrics_to_csv("随机森林", rmse, mae, r2, metrics_csv_path)

    # ===================== 6. 结果可视化 =====================
    plt.figure(figsize=(12, 6))  # 设置图表尺寸
    # 绘制真实通胀率曲线
    plt.plot(y_test.index, y_test_values, label='真实值', color='blue')
    # 绘制预测通胀率曲线（虚线区分）
    plt.plot(y_test.index, y_pred, label='预测值', color='red', linestyle='--')
    
    # 图表样式配置
    plt.title('通货膨胀率预测结果对比（随机森林）', fontsize=16)
    plt.xlabel('日期')
    plt.ylabel('CPI 环比增长率 (%)')
    plt.legend()  # 显示图例
    plt.grid(True, linestyle='--', alpha=0.5)  # 网格线（半透明虚线）
    
    output_dir = "../../outputs/baseline"
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在，不存在则创建

    # 保存图表
    output_path = os.path.join(output_dir, "2.随机森林.png")
    plt.savefig(output_path)
    print(f"图表已保存至：{output_path}")

    # 展示图表
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_file = "../../data/CPI_Data.csv" 
    process_and_forecast_inflation(csv_file)