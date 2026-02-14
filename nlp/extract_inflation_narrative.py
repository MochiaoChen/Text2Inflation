import os
import re
import json
import time
import glob
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置 Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables. Please set it in .env file.")

client = genai.Client(api_key=API_KEY)

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
REPORTS_DIR = os.path.join(DATA_DIR, 'reports')
OUTPUT_JSON_DIR = os.path.join(DATA_DIR, 'nlp_results')
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, 'nlp_features.csv')

# 确保输出目录存在
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

MODEL_NAME = "gemini-2.0-flash" 

SYSTEM_PROMPT = """
# Role Definition
你是一位拥有20年经验的宏观经济学家和计量语言学专家。你专门研究中国人民银行（PBoC）的货币政策传导机制。你精通通过解读央行《货币政策执行报告》中的措辞变化（Central Bank Communication）来预测宏观经济变量。

# Task Context
我将上传一份完整的 PBoC《货币政策执行报告》（PDF）。你需要通读全文，重点关注“宏观经济分析”、“货币政策趋势”、“价格走势”等相关章节。

# Objective
请将报告中的非结构化“通胀叙事”转化为可用于时间序列回归分析的**高维结构化数据**。

# Extraction Dimension (Strict Definitions)
请对以下每一个维度进行评分或参数提取。请务必基于**经济学逻辑**而非简单的关键词匹配。

## Group A: Inflation Outlook (通胀前景)
1.  **`INF_Sentiment` (-10 to 10):** 央行对未来物价走势的总体判断。
    * -10: 极度担忧通缩/物价下行（如“需求收缩”、“物价下行压力巨大”）。
    * 0: 物价基本平稳/合意区间。
    * 10: 极度担忧通胀/过热（如“物价上涨压力大”、“防止通胀反弹”）。
2.  **`INF_Duration` (0 to 1):** 央行认为当前价格波动的持续性。
    * 0: 暂时的/阶段性的（Transitory）。
    * 1: 长期的/结构性的（Persistent/Structural）。

## Group B: Attribution Analysis (归因叙事 - 菲利普斯曲线视角)
*在此组中，评估央行将通胀/通缩压力归咎于哪些因素的**权重 (0-10)**。*
3.  **`DRV_Demand`:** 来源于总需求（消费、投资）的强度。
4.  **`DRV_Supply`:** 来源于供给侧（猪肉、蔬菜、产能瓶颈）的强度。
5.  **`DRV_External`:** 来源于外部环境（国际油价、主要经济体通胀）的强度。
6.  **`DRV_Monetary`:** 来源于货币供应量（M2、信贷规模）的强度。

## Group C: Policy Stance (政策立场)
7.  **`POL_Tone` (-10 to 10):** 针对价格稳定的货币政策基调。
    * -10: 极度宽松/鸽派（强调降准降息、刺激需求）。
    * 0: 稳健中性（强调流动性合理充裕）。
    * 10: 极度紧缩/鹰派（强调闸门、去杠杆、抑制泡沫）。
8.  **`POL_Priority` (0 to 10):** “稳定物价”在当前所有政策目标（增长、就业、收支平衡、金融稳定）中的相对优先级。
    * 10代表“把稳定物价作为当前首要任务”。

## Group D: Narrative Features (叙事特征)
9.  **`TXT_Ambiguity` (0 to 10):** 文本的模糊程度。央行是否使用了大量模棱两可的词汇（如“有待观察”、“不确定性增加”）来规避明确承诺？
10. **`TXT_Confidence` (0 to 10):** 央行对自身预测能力的自信程度。

# Output Format (JSON Only)
不要输出任何 Markdown 前缀或解释性文字，仅输出合法的 JSON 对象。
结构如下：
{
  "report_period": "YYYY-QN",
  "quantitative_metrics": {
    "INF_Sentiment": <float>,
    "INF_Duration": <float>,
    "DRV_Demand": <float>,
    "DRV_Supply": <float>,
    "DRV_External": <float>,
    "DRV_Monetary": <float>,
    "POL_Tone": <float>,
    "POL_Priority": <float>,
    "TXT_Ambiguity": <float>,
    "TXT_Confidence": <float>
  },
  "evidence_chain": {
    "INF_Sentiment_Reasoning": "引用原文... 说明原因...",
    "POL_Tone_Reasoning": "引用原文... 说明原因..."
  },
  "key_narrative_shifts": "简述本期报告相对于常规表述最大的叙事转变（例如：不再提'跨周期调节'，转而强调'逆周期调节'）"
}

# Constraint
- 必须基于 PBoC 的官方语境（PBoC-speak），例如准确区分“跨周期”与“逆周期”，“精准滴灌”与“大水漫灌”的含义差异。
- 如果文中未明确提及某维度，请基于上下文推断隐性态度，实在无法推断则填 null，但尽量避免。
"""



def parse_filename_to_period(filename):
    """
    将文件名转换为 standard period 字符串 (YYYY-QN)
    支持: 
    1. "2001年第一季度中国货币政策执行报告.pdf" -> "2001-Q1"
    2. "2001-Q1.pdf" -> "2001-Q1"
    """
    basename = os.path.basename(filename)
    
    # 尝试匹配 YYYY-QN.pdf 格式
    match_std = re.match(r'(\d{4}-Q[1-4])\.pdf', basename)
    if match_std:
        return match_std.group(1)

    # 尝试匹配中文格式
    match_cn = re.search(r'(\d{4})年第([一二三四])季度', basename)
    if match_cn:
        year = match_cn.group(1)
        quarter_cn = match_cn.group(2)
        quarter_map = {'一': '1', '二': '2', '三': '3', '四': '4'}
        return f"{year}-Q{quarter_map[quarter_cn]}"
    
    return None


def extract_json_from_response(text):
    """
    从 LLM 返回的文本中提取 JSON 对象。
    处理 Markdown 代码块 (```json ... ```) 
    """
    # 尝试找到 JSON 代码块
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # 尝试直接找 {} 包裹的内容
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            return None
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from text: {text[:100]}...")
        return None

def process_single_report(pdf_path):
    """
    处理单个 PDF 报告
    """
    period = parse_filename_to_period(pdf_path)
    if not period:
        print(f"Skipping file with unknown format: {pdf_path}")
        return None

    json_output_path = os.path.join(OUTPUT_JSON_DIR, f"{period}.json")
    
    # 如果已经处理过，直接读取
    if os.path.exists(json_output_path):
        print(f"  [Skipping] Already processed: {period}")
        with open(json_output_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    print(f"  [Processing] {period} ...")

    try:
        # Upload the file to Gemini
        file_upload = client.files.upload(file=pdf_path)

        # Wait for the file to be active
        while file_upload.state.name == "PROCESSING":
            time.sleep(2)
            file_upload = client.files.get(name=file_upload.name)

        if file_upload.state.name != "ACTIVE":
            raise Exception(f"File upload failed with state: {file_upload.state.name}")
        
        # Generate content
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=file_upload.uri,
                            mime_type=file_upload.mime_type,
                        ),
                        types.Part.from_text(text=SYSTEM_PROMPT),
                    ],
                ),
            ],
        )

        result_json = extract_json_from_response(response.text)
        
        if result_json:
            # 强制覆盖 report_period 以确保一致性
            result_json['report_period'] = period
            
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, ensure_ascii=False, indent=2)
            print(f"  [Success] Saved to {json_output_path}")
            return result_json
        else:
            print(f"  [Error] Failed to parse JSON for {period}")
            return None

    except Exception as e:
        print(f"  [Error] Processing {period} failed: {e}")
        return None


def main():
    if not API_KEY:
        print("Please set GEMINI_API_KEY in .env file to run this script.")
        # Create a dummy CSV for testing if no API key
        if not os.path.exists(OUTPUT_CSV_PATH):
             print("Creating dummy CSV for testing purposes...")
             df = pd.DataFrame(columns=['report_period', 'INF_Sentiment', 'INF_Duration', 'DRV_Demand', 'DRV_Supply', 'DRV_External', 'DRV_Monetary', 'POL_Tone', 'POL_Priority', 'TXT_Ambiguity', 'TXT_Confidence'])
             df.to_csv(OUTPUT_CSV_PATH, index=False)
        return

    pdf_files = glob.glob(os.path.join(REPORTS_DIR, "*.pdf"))
    pdf_files.sort()
    
    all_metrics = []
    
    print(f"Found {len(pdf_files)} PDF reports.")
    
    for pdf_path in pdf_files:
        result = process_single_report(pdf_path)
        if result and 'quantitative_metrics' in result:
            metrics = result['quantitative_metrics']
            metrics['report_period'] = result.get('report_period')
            all_metrics.append(metrics)
            
    # Convert to DataFrame and save CSV
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        # Reorder columns to put report_period first
        cols = ['report_period'] + [c for c in df.columns if c != 'report_period']
        df = df[cols]
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nSaved all metrics to {OUTPUT_CSV_PATH}")
        print(df.head())
    else:
        print("\nNo metrics extracted.")

if __name__ == "__main__":
    main()
