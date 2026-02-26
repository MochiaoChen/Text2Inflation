# 通用工具函数
import os
import pandas as pd

def save_metrics_to_csv(model_name, rmse, mae, r2, csv_path):
    # 确保CSV所在目录存在（避免路径不存在报错）
    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    # 构造指标数据
    metrics_data = {
        '模型名称': [model_name],
        'RMSE': [round(rmse, 4)],
        'MAE': [round(mae, 4)],
        'R²': [round(r2, 4)]
    }
    df_metrics = pd.DataFrame(metrics_data)

    # 写入CSV：文件不存在则新建（带表头），存在则追加（不带表头）
    if os.path.exists(csv_path):
        df_metrics.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8')
    else:
        df_metrics.to_csv(csv_path, mode='w', header=True, index=False, encoding='utf-8')
    
    print(f"模型指标已保存至：{csv_path}")