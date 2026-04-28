"""
Text2Inflation 公共数据处理模块

提供所有模型脚本共用的函数：
- 绘图风格设置
- 数据读取、清洗与平稳化
- 滞后特征构造
- 训练/测试集划分与标准化
- Expanding window 滚动预测

Author: Text2Inflation Team
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ===================== 路径常量 =====================

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, 'data')
OUTPUT_DIR = os.path.join(_REPO_ROOT, 'outputs')
BASELINE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'cpi_growth_baseline')
ENHANCED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'cpi_growth_enhanced')
SHAP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'shap')
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')

DEFAULT_CPI_PATH = os.path.join(DATA_DIR, 'CPI_Data.csv')
DEFAULT_NLP_PATH = os.path.join(DATA_DIR, 'nlp_features.csv')

# 10 维 NLP 叙事特征列名
NLP_FEATURE_COLS = [
    'INF_Sentiment', 'INF_Duration',
    'DRV_Demand', 'DRV_Supply', 'DRV_External', 'DRV_Monetary',
    'POL_Tone', 'POL_Priority',
    'TXT_Ambiguity', 'TXT_Confidence',
]


def _ensure_dirs():
    for d in (OUTPUT_DIR, BASELINE_OUTPUT_DIR, ENHANCED_OUTPUT_DIR,
              SHAP_OUTPUT_DIR, PREDICTIONS_DIR):
        os.makedirs(d, exist_ok=True)


_ensure_dirs()


# ===================== 绘图配置 =====================

def setup_plot_style():
    """设置 matplotlib 中文绘图风格（黑体 + whitegrid）"""
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False


# ===================== 数据加载与清洗 =====================

def load_and_clean_data(file_path=None, stationarize=True):
    """
    读取 CPI 数据 CSV，执行清洗和（可选的）平稳化处理。

    Returns
    -------
    df : pd.DataFrame
    target_col : str
    """
    if file_path is None:
        file_path = DEFAULT_CPI_PATH

    print(">>> 1. 正在读取和清洗数据...")
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.sort_values('date').set_index('date')
    df.columns = df.columns.str.strip()

    original_target_col = 'CPI(2001-01=100)'
    target_col_raw = 'CPI_Index'
    if original_target_col in df.columns:
        df.rename(columns={original_target_col: target_col_raw}, inplace=True)
    elif 'CPI' in df.columns:
        df.rename(columns={'CPI': target_col_raw}, inplace=True)
    else:
        raise ValueError("错误：数据集中找不到 CPI 目标列。")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.ffill().bfill()
    df.dropna(axis=1, how='all', inplace=True)

    print(f"    数据加载完成，时间跨度: {df.index.min().date()} 至 {df.index.max().date()}")

    if stationarize:
        print(">>> 2. 转换为环比通胀率进行建模...")
        df['Inflation'] = df[target_col_raw].pct_change() * 100
        for col in [c for c in df.columns if c not in ('Inflation', target_col_raw)]:
            df[col] = df[col].pct_change() * 100
        df = df.drop(columns=[target_col_raw])
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        target_col = 'Inflation'
        print(f"    预测目标: {target_col} (CPI 环比增长率)")
    else:
        df.rename(columns={target_col_raw: 'CPI'}, inplace=True)
        target_col = 'CPI'

    return df, target_col


def load_enhanced_data(file_path=None, nlp_file_path=None, stationarize=True,
                       publication_lag_months=0):
    """
    读取 CPI 数据并合并 NLP 叙事特征（季度 → 月度 ffill）。

    publication_lag_months : int
        Push each report's effective availability forward by this many months
        to mimic PBoC publication delay (reports for quarter Q typically appear
        ~1-2 months after Q ends). Default 0 reproduces the original (mildly
        forward-looking) join.
    """
    if nlp_file_path is None:
        nlp_file_path = DEFAULT_NLP_PATH

    df, target_col = load_and_clean_data(file_path, stationarize)

    if not os.path.exists(nlp_file_path):
        print(f"Warning: NLP feature file not found at {nlp_file_path}. Returning baseline data.")
        return df, target_col

    print(f">>> 加载 NLP 增强特征: {nlp_file_path} (publication_lag_months={publication_lag_months})")
    nlp_df = pd.read_csv(nlp_file_path)

    nlp_dates = []
    for p in nlp_df['report_period']:
        year, q = p.split('-')
        month_day = {'Q1': '-03-31', 'Q2': '-06-30', 'Q3': '-09-30', 'Q4': '-12-31'}[q]
        nlp_dates.append(pd.Timestamp(f"{year}{month_day}"))

    nlp_df.index = pd.DatetimeIndex(nlp_dates)
    nlp_df = nlp_df.sort_index().drop(columns=['report_period'])

    if publication_lag_months:
        nlp_df.index = nlp_df.index + pd.DateOffset(months=publication_lag_months)

    nlp_reindexed = nlp_df.reindex(df.index, method='ffill')
    df_enhanced = pd.concat([df, nlp_reindexed], axis=1).dropna()

    print(f"    增强后特征维度: {df_enhanced.shape[1]-1} (新增 {nlp_df.shape[1]} 个 NLP 特征)")
    return df_enhanced, target_col


# ===================== 特征工程 =====================

def create_lag_features(df, target_col, lags=None):
    """为所有非目标列构造滞后变量。"""
    if lags is None:
        lags = [1, 2, 3, 6, 12]

    print(">>> 特征工程（构造滞后变量）...")
    current_feature_cols = [col for col in df.columns if col != target_col]
    feature_dfs = [df[col].shift(lag).rename(f"{col}_lag{lag}")
                   for col in current_feature_cols for lag in lags]

    df_features = pd.concat(feature_dfs, axis=1)
    data_final = pd.concat([df[target_col], df_features], axis=1).dropna()
    print(f"    特征构造完成。最终特征数量: {data_final.shape[1]-1} 个")
    return data_final


# ===================== 数据集划分与标准化 =====================

def split_and_scale(data_final, target_col, ratio=0.8):
    """按时间顺序 80/20 划分并标准化（保留给旧模型用）。"""
    print(">>> 划分训练/测试集...")
    split_index = int(len(data_final) * ratio)

    X_train = data_final.iloc[:split_index].drop(columns=[target_col])
    y_train = data_final.iloc[:split_index][target_col]
    X_test = data_final.iloc[split_index:].drop(columns=[target_col])
    y_test = data_final.iloc[split_index:][target_col]

    if X_train.empty or X_test.empty:
        raise ValueError("错误：训练集或测试集为空。")

    print(f"    训练集: {len(X_train)}, 测试集: {len(X_test)}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train.columns


# ===================== Expanding window 滚动预测 =====================

def expanding_window_predict(data_final, target_col, model_factory,
                             start_ratio=0.8, step=1, scale=True, verbose=True):
    """
    按 expanding window 逐期重新训练 + 一步外推预测。

    Parameters
    ----------
    data_final : pd.DataFrame
        目标 + 特征 DataFrame（已去 NaN，时间索引）
    target_col : str
    model_factory : callable () -> estimator
        返回一个未 fit 的 sklearn 兼容估计器。每步重新调用以获得新实例。
    start_ratio : float
        初始训练集占比。测试从此点开始向后展开。
    step : int
        预测步长（默认每期一次）。
    scale : bool
        是否对特征做 StandardScaler（每步用截至当前的训练集重新 fit）。

    Returns
    -------
    y_true : pd.Series, y_pred : pd.Series  （索引对齐测试期日期）
    """
    if verbose:
        print(f">>> Expanding-window 预测 (start_ratio={start_ratio}, step={step})...")

    split_index = int(len(data_final) * start_ratio)
    X_all = data_final.drop(columns=[target_col])
    y_all = data_final[target_col]
    feature_names = X_all.columns

    preds, truths, idx = [], [], []
    for t in range(split_index, len(data_final), step):
        X_tr = X_all.iloc[:t]
        y_tr = y_all.iloc[:t]
        X_te = X_all.iloc[t:t+1]
        y_te = y_all.iloc[t:t+1]

        if scale:
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
        else:
            X_tr_s = X_tr.values
            X_te_s = X_te.values

        model = model_factory()
        model.fit(X_tr_s, y_tr)
        yhat = model.predict(X_te_s)

        preds.append(float(np.asarray(yhat).ravel()[0]))
        truths.append(float(y_te.iloc[0]))
        idx.append(y_te.index[0])

    y_true = pd.Series(truths, index=pd.DatetimeIndex(idx), name=target_col)
    y_pred = pd.Series(preds, index=pd.DatetimeIndex(idx), name='y_pred')
    if verbose:
        print(f"    完成 {len(y_pred)} 步滚动预测。")
    return y_true, y_pred, feature_names


# ===================== 预测结果保存 =====================

def save_predictions(model_name, y_true, y_pred):
    """将 y_true / y_pred 保存到 outputs/predictions/{model_name}.csv 供 DM/SHAP 使用。"""
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    path = os.path.join(PREDICTIONS_DIR, f"{model_name}.csv")
    out = pd.DataFrame({'y_true': y_true.values, 'y_pred': y_pred.values},
                       index=y_true.index)
    out.index.name = 'date'
    out.to_csv(path)
    print(f"    预测结果保存: {path}")
    return path


# ===================== 评估与可视化 =====================

def evaluate_and_plot(y_test, y_pred, model_name, output_path):
    """计算 RMSE/MAE/R² 并绘制对比图，保存到 output_path（可为绝对路径或仅文件名）。"""
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f">>> 评估结果 ({model_name}):  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    if not os.path.isabs(output_path):
        output_path = os.path.join(OUTPUT_DIR, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual', color='blue', linewidth=2)
    plt.plot(y_test.index, np.asarray(y_pred), label='Predicted', color='red', linestyle='--', linewidth=2)
    plt.title(f'CPI Inflation Forecast vs. Actual ({model_name})', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('CPI MoM Growth (%)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"    图表保存: {output_path}")
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def save_metrics_to_csv(model_name, rmse, mae, r2, csv_path=None):
    """追加一行到 outputs/Outputs.csv（model, rmse, mae, r2）。"""
    if csv_path is None:
        csv_path = os.path.join(OUTPUT_DIR, 'Outputs.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row = pd.DataFrame([{'model': model_name, 'rmse': rmse, 'mae': mae, 'r2': r2}])
    if os.path.exists(csv_path):
        try:
            existing = pd.read_csv(csv_path)
            if 'model' in existing.columns:
                existing = existing[existing['model'] != model_name]
                row = pd.concat([existing[['model', 'rmse', 'mae', 'r2']], row], ignore_index=True)
        except Exception:
            pass
    row.to_csv(csv_path, index=False)
    return csv_path
