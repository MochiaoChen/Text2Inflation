import os
import re
import glob
import shutil
 
# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, 'data', 'reports')

def parse_filename_to_period(filename):
    """
    将中文文件名转换为 standard period 字符串 (YYYY-QN)
    例如: "2001年第一季度中国货币政策执行报告.pdf" -> "2001-Q1"
    """
    basename = os.path.basename(filename)
    match = re.search(r'(\d{4})年第([一二三四])季度', basename)
    if match:
        year = match.group(1)
        quarter_cn = match.group(2)
        quarter_map = {'一': '1', '二': '2', '三': '3', '四': '4'}
        return f"{year}-Q{quarter_map[quarter_cn]}"
    return None

def rename_reports():
    print(f"Scanning reports in: {REPORTS_DIR}")
    pdf_files = glob.glob(os.path.join(REPORTS_DIR, "*.pdf"))
    pdf_files.sort()
    
    count = 0
    errors = 0
    
    for pdf_path in pdf_files:
        period = parse_filename_to_period(pdf_path)
        
        if period:
            new_filename = f"{period}.pdf"
            new_path = os.path.join(REPORTS_DIR, new_filename)
            
            # 如果文件名已经是目标格式，跳过
            if os.path.basename(pdf_path) == new_filename:
                continue
                
            try:
                os.rename(pdf_path, new_path)
                print(f"Renamed: {os.path.basename(pdf_path)} -> {new_filename}")
                count += 1
            except Exception as e:
                print(f"Error renaming {os.path.basename(pdf_path)}: {e}")
                errors += 1
        else:
            # 检查是否已经是标准格式 YYYY-QN.pdf
            if re.match(r'\d{4}-Q[1-4]\.pdf', os.path.basename(pdf_path)):
                continue
            print(f"Skipping unknown format: {os.path.basename(pdf_path)}")
            
    print(f"\nSummary: Renamed {count} files. {errors} errors.")

if __name__ == "__main__":
    rename_reports()
