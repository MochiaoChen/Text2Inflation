import os
import json
import pandas as pd
import glob

def process_jsons(json_dir, output_csv):
    """读取所有 JSON 文件并转换为 nlp_features.csv"""
    print(f">>> 正在读取 JSON 文件目录: {json_dir}")
    files = glob.glob(os.path.join(json_dir, "*.json"))
    data_list = []
    
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as jfile:
                content = json.load(jfile)
                # 提取 report_period 和 quantitative_metrics 中的所有字段
                row = {"report_period": content["report_period"]}
                if "quantitative_metrics" in content:
                    row.update(content["quantitative_metrics"])
                data_list.append(row)
        except Exception as e:
            print(f"Error processing {f}: {e}")
    
    df = pd.DataFrame(data_list)
    # 按报告周期排序
    df = df.sort_values("report_period")
    df.to_csv(output_csv, index=False)
    print(f"    成功将 {len(df)} 条记录保存到 {output_csv}")
    return df

def merge_with_cpi(cpi_csv, nlp_csv, output_enhanced_csv):
    """将 NLP 特征与 CPI 月度数据合并，生成 enhanced_data.csv"""
    print(f"\n>>> 正在合并数据 (使用 merge_asof 避免未来函数): {cpi_csv} + {nlp_csv}")
    
    # 1. 加载 CPI 数据
    cpi_df = pd.read_csv(cpi_csv)
    cpi_df['date'] = pd.to_datetime(cpi_df['date'])
    cpi_df = cpi_df.sort_values('date')
    
    # 2. 加载 NLP 特征
    nlp_df = pd.read_csv(nlp_csv)
    
    # 将季度周期映射为该季度最后一天
    def q_to_date(q_str):
        try:
            year, q = q_str.split('-')
            if q == 'Q1': return f"{year}-03-31"
            if q == 'Q2': return f"{year}-06-30"
            if q == 'Q3': return f"{year}-09-30"
            if q == 'Q4': return f"{year}-12-31"
        except:
            return None
        return None
    
    nlp_df['date'] = pd.to_datetime(nlp_df['report_period'].apply(q_to_date))
    nlp_df = nlp_df.drop(columns=['report_period']).sort_values('date')
    
    # 3. 使用 merge_asof 合并
    # direction='backward' 意味着每个 CPI 日期将匹配它之前的最近一个 NLP 日期
    # 例如 2001-04-01 会匹配到 2001-03-31 的 Q1 特征
    merged = pd.merge_asof(cpi_df, nlp_df, on='date', direction='backward')
    
    # 4. 保存结果
    # 如果用户想剔除没有 NLP 数据的前期部分，可以取消下面这一行的注释
    # merged.dropna(subset=[nlp_df.columns[0]], inplace=True)
    
    merged.to_csv(output_enhanced_csv, index=False)
    print(f"    成功将增强后的数据保存到 {output_enhanced_csv}")
    print(f"    最终数据维度: {merged.shape}")

if __name__ == "__main__":
    # 路径配置
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    JSON_DIR = os.path.join(BASE_DIR, "data", "2001-2025年中国货币政策执行报告JSON文件")
    NLP_CSV = os.path.join(BASE_DIR, "data", "nlp_features.csv")
    CPI_CSV = os.path.join(BASE_DIR, "data", "CPI_Data.csv")
    ENHANCED_CSV = os.path.join(BASE_DIR, "data", "enhanced_data.csv")
    
    # 执行处理逻辑
    process_jsons(JSON_DIR, NLP_CSV)
    merge_with_cpi(CPI_CSV, NLP_CSV, ENHANCED_CSV)
