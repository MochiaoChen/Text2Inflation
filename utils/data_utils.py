"""
Text2Inflation 公共数据处理模块

提供所有模型脚本共用的函数：
- 绘图风格设置
- 数据读取、清洗与平稳化
- 滞后特征构造
- 训练/测试集划分与标准化

Author: Text2Inflation Team
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# ===================== 路径常量 =====================

# 获取 code/ 目录的绝对路径（本文件位于 code/utils/）
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_CODE_DIR, 'data')
OUTPUT_DIR = os.path.join(_CODE_DIR, 'outputs')

# 默认数据文件路径
DEFAULT_CPI_PATH = os.path.join(DATA_DIR, 'CPI_Data.csv')


# ===================== 绘图配置 =====================

def setup_plot_style():
    """设置 matplotlib 中文绘图风格（黑体 + whitegrid）"""
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


# ===================== 数据加载与清洗 =====================

def load_and_clean_data(file_path=None, stationarize=True):
    """
    读取 CPI 数据 CSV，执行清洗和（可选的）平稳化处理。

    Parameters
    ----------
    file_path : str, optional
        CSV 文件路径，默认使用 data/CPI_Data.csv
    stationarize : bool, default True
        是否将 CPI 指数转换为环比通胀率，并对特征做增长率变换。
        - True:  返回的目标列为 'Inflation'（环比增长率 %）
        - False: 返回的目标列为 'CPI'（原始指数值）

    Returns
    -------
    df : pd.DataFrame
        处理后的 DataFrame（DatetimeIndex），包含目标列和特征列
    target_col : str
        目标列名（'Inflation' 或 'CPI'）
    """
    if file_path is None:
        file_path = DEFAULT_CPI_PATH

    print(">>> 1. 正在读取和清洗数据...")
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df.sort_values('date').set_index('date')
    except Exception as e:
        raise RuntimeError(f"读取文件出错，请检查文件名和路径: {e}")

    # 清理列名
    df.columns = df.columns.str.strip()

    # 识别并重命名目标列
    original_target_col = 'CPI(2001-01=100)'
    target_col_raw = 'CPI_Index'

    if original_target_col in df.columns:
        df.rename(columns={original_target_col: target_col_raw}, inplace=True)
    elif 'CPI' in df.columns:
        df.rename(columns={'CPI': target_col_raw}, inplace=True)
    else:
        raise ValueError("错误：数据集中找不到 CPI 目标列。请检查列名。")

    # 强制转换为数值类型
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 填充缺失值 & 丢弃全空列
    df = df.ffill().bfill()
    df.dropna(axis=1, how='all', inplace=True)

    print(f"    数据加载完成，时间跨度: {df.index.min().date()} 至 {df.index.max().date()}")

    if stationarize:
        # 将 CPI 指数 → 环比通胀率
        print("\n>>> 2. 核心修正：转换为增长率（通货膨胀率）进行建模...")
        df['Inflation'] = df[target_col_raw].pct_change() * 100

        cols_to_transform = [c for c in df.columns if c not in ('Inflation', target_col_raw)]
        for col in cols_to_transform:
            df[col] = df[col].pct_change() * 100

        df = df.drop(columns=[target_col_raw])
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        target_col = 'Inflation'
        print(f"    数据变换完成。当前预测目标: {target_col} (CPI 环比增长率)")
    else:
        # 保持原始 CPI 指数
        target_col = target_col_raw
        df.rename(columns={target_col_raw: 'CPI'}, inplace=True)
        target_col = 'CPI'


def load_enhanced_data(file_path=None, nlp_file_path=None, stationarize=True):
    """
    读取 CPI 数据并合并 NLP 提取的通胀叙事特征。
    NLP 特征是季度的，将通过 forward fill 填充到月度数据中。

    Parameters
    ----------
    file_path : str, optional
        CPI 数据路径
    nlp_file_path : str, optional
        NLP 特征 CSV 路径，默认 data/nlp_features.csv
    stationarize : bool, default True
        是否平稳化

    Returns
    -------
    df_enhanced : pd.DataFrame
        包含 CPI + NLP 特征的 DataFrame
    target_col : str
        目标列名
    """
    if nlp_file_path is None:
        nlp_file_path = os.path.join(DATA_DIR, 'nlp_features.csv')

    # 1. 加载基础数据
    df, target_col = load_and_clean_data(file_path, stationarize)

    # 2. 加载 NLP 特征
    if not os.path.exists(nlp_file_path):
        print(f"Warning: NLP feature file not found at {nlp_file_path}. Returning baseline data.")
        return df, target_col
    
    print(f"\n>>> 加载 NLP 增强特征: {nlp_file_path}")
    nlp_df = pd.read_csv(nlp_file_path)
    
    # 解析 report_period (YYYY-QN) 到日期
    # 将季度末作为特征发布的日期，或者季度初？
    # PBoC 报告通常在季度结束后 1-2 个月发布（如 Q1 报告在 5 月发布）。
    # 为了避免未来函数，我们假设特征在报告发布月份生效。
    # 简化处理：YYYY-Q1 -> YYYY-05-01, Q2 -> 08-01, Q3 -> 11-01, Q4 -> Next Year-02-01
    # 实际上，更为稳健的做法是：
    # Q1 (Jan-Mar) 的数据 -> 此处对应 Q1 报告 -> Q1 报告发布于 Q2 中期 (May) -> 只能用于预测 Q2 后半段及以后
    # 这里我们采用简单映射，将 NLP 特征映射到该季度末的下一个月，然后 ffill
    
    quarter_map = {
        'Q1': '-04-01', # Q1 结束是 3.31，这里近似为 4.1
        'Q2': '-07-01',
        'Q3': '-10-01',
        'Q4': '-12-31'  # 保持 20xx 年内
    }
    # 但中国央行报告发布有滞后。Q1报告通常5月发。
    # 如果我们在做实时预测，只能用已发布的。
    # 这是一个回测项目，我们假设：
    # 报告是对本季度的总结，包含了对未来的展望。
    # 我们将 report_period 解析为该季度的最后一天。
    
    nlp_dates = []
    for p in nlp_df['report_period']:
        year, q = p.split('-')
        if q == 'Q1': date_str = f"{year}-03-31"
        elif q == 'Q2': date_str = f"{year}-06-30"
        elif q == 'Q3': date_str = f"{year}-09-30"
        elif q == 'Q4': date_str = f"{year}-12-31"
        nlp_dates.append(pd.Timestamp(date_str))
        
    nlp_df.index = nlp_dates
    nlp_df = nlp_df.sort_index()
    nlp_df = nlp_df.drop(columns=['report_period'])
    
    # 3. 合并到月度数据
    # 将 NLP 特征 join 到 df，并 ffill
    # 注意：df 的索引是月度。
    
    # 先创建一个包含所有 df 日期的空模板
    df_enhanced = df.copy()
    
    # 将 NLP 特征 reindex 到 df 的时间轴，使用 ffill 填充缺失月
    # method='ffill' 会将 Q1 (3.31) 的值填充到 4月, 5月, 6月... 直到用于 Q2 (6.30) 的新值出现
    nlp_reindexed = nlp_df.reindex(df.index, method='ffill')
    
    # 合并
    df_enhanced = pd.concat([df_enhanced, nlp_reindexed], axis=1)
    
    # 去除可能产生的 NaN (尤其是早期没有 NLP 数据的时间段)
    df_enhanced.dropna(inplace=True)
    
    print(f"    增强后特征维度: {df_enhanced.shape[1]-1} (新增 {nlp_df.shape[1]} 个 NLP 特征)")
    return df_enhanced, target_col




# ===================== 特征工程 =====================

def create_lag_features(df, target_col, lags=None):
    """
    为所有特征列构造滞后变量。

    Parameters
    ----------
    df : pd.DataFrame
        包含目标列和特征列的 DataFrame
    target_col : str
        目标列名
    lags : list of int, optional
        滞后阶数列表，默认 [1, 2, 3, 6, 12]

    Returns
    -------
    data_final : pd.DataFrame
        包含目标列和所有滞后特征的 DataFrame（已去除 NaN）
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12]

    print("\n>>> 特征工程（构造滞后变量）...")
    current_feature_cols = [col for col in df.columns if col != target_col]
    feature_dfs = []

    for col in current_feature_cols:
        for lag in lags:
            name = f"{col}_lag{lag}"
            feature_dfs.append(df[col].shift(lag).rename(name))

    df_features = pd.concat(feature_dfs, axis=1)
    data_final = pd.concat([df[target_col], df_features], axis=1)
    data_final.dropna(inplace=True)

    print(f"    特征构造完成。最终特征数量: {data_final.shape[1]-1} 个")
    return data_final


# ===================== 数据集划分与标准化 =====================

def split_and_scale(data_final, target_col, ratio=0.8):
    """
    按时间顺序划分训练/测试集并标准化特征。

    Parameters
    ----------
    data_final : pd.DataFrame
        包含目标列和特征的 DataFrame
    target_col : str
        目标列名
    ratio : float
        训练集占比，默认 0.8

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled : np.ndarray
    y_train : pd.Series
    y_test : pd.Series
    scaler : StandardScaler
        已 fit 的标准化器（可用于后续逆变换）
    """
    print("\n>>> 划分训练集与测试集...")
    split_index = int(len(data_final) * ratio)

    X_train = data_final.iloc[:split_index].drop(columns=[target_col])
    y_train = data_final.iloc[:split_index][target_col]
    X_test = data_final.iloc[split_index:].drop(columns=[target_col])
    y_test = data_final.iloc[split_index:][target_col]

    if X_train.empty or X_test.empty:
        raise ValueError("错误：训练集或测试集为空。请检查原始数据质量。")

    print(f"    训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

    print("\n>>> 数据标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train.columns


# ===================== 评估与可视化 =====================

def evaluate_and_plot(y_test, y_pred, model_name, output_filename):
    """
    计算评估指标并绘制预测对比图。

    Parameters
    ----------
    y_test : pd.Series
        真实值
    y_pred : array-like
        预测值
    model_name : str
        模型名称（用于标题和图例）
    output_filename : str
        输出 PNG 文件名（保存到 outputs/ 目录）
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n>>> 评估结果 ({model_name}):")
    print(f"    RMSE (均方根误差): {rmse:.4f}")
    print(f"    MAE  (平均绝对误差): {mae:.4f}")
    print(f"    R²   (拟合优度): {r2:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='真实值 (Actual)', color='blue', linewidth=2)
    plt.plot(y_test.index, y_pred, label=f'{model_name} 预测值', color='red', linestyle='--', linewidth=2)
    plt.title(f'通货膨胀率预测结果对比（{model_name}）', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('CPI 环比增长率 (%)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path)
    print(f"\n>>> 结果图表已保存为 '{output_path}'")
    plt.show()

    return {'rmse': rmse, 'mae': mae, 'r2': r2}
