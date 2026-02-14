"""
xlsx_to_csv.py — 将 Excel 原始数据转换为 CSV 格式
"""

import os
import pandas as pd

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

specified_files = [os.path.join(DATA_DIR, "raw_data.xlsx")]

for file_path in specified_files:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    sheets = pd.read_excel(file_path, sheet_name=None)
    for sheet_name, df in sheets.items():
        output_file = os.path.join(DATA_DIR, f"{base_name}_{sheet_name}.csv")
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"{os.path.basename(file_path)} 的工作表 {sheet_name} 转换成 {output_file}")