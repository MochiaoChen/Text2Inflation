"""
Prompt-sensitivity robustness check.

For a 20-report stratified subset, re-extract the 10 narrative dimensions
TWICE — once with the original V1 system prompt and once with a V2 variant
(different expert-role framing, identical schema) — using the same model
(gemini-2.5-flash). This isolates *prompt sensitivity* from any V1-vs-V2
model differences. Computes Pearson & Spearman correlations per dimension.

Outputs:
    data/nlp_results_robust_v1/{period}.json
    data/nlp_results_robust_v2/{period}.json
    outputs/prompt_sensitivity/v1_features.csv
    outputs/prompt_sensitivity/v2_features.csv
    outputs/prompt_sensitivity/correlation.csv
"""
import os
import json
import time
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv

from nlp.extract_inflation_narrative import (
    extract_json_from_response,
    SYSTEM_PROMPT as SYSTEM_PROMPT_V1,
    REPORTS_DIR,
    DATA_DIR,
    BASE_DIR,
)

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash"

OUTPUT_JSON_DIR_V1 = os.path.join(DATA_DIR, "nlp_results_robust_v1")
OUTPUT_JSON_DIR_V2 = os.path.join(DATA_DIR, "nlp_results_robust_v2")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "prompt_sensitivity")
os.makedirs(OUTPUT_JSON_DIR_V1, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR_V2, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 20-report stratified sample across macro regimes
SAMPLE_PERIODS = [
    "2003-Q2", "2005-Q3", "2007-Q4", "2008-Q3",       # early 2000s + GFC
    "2010-Q2", "2012-Q1", "2013-Q4", "2014-Q3",       # post-GFC normalization
    "2015-Q4", "2016-Q3", "2018-Q2", "2019-Q1",       # communication reform
    "2020-Q1", "2020-Q4", "2021-Q3", "2022-Q2",       # COVID + commodity shock
    "2023-Q1", "2023-Q4", "2024-Q3", "2025-Q1",       # deflation regime
]

# Variant prompt: same 10 dimensions, different expert role and framing.
SYSTEM_PROMPT_V2 = """
# Role Definition
你是一名在国际投行（如高盛、摩根大通）担任中国利率与宏观研究主管的资深经济学家。你长期跟踪中国人民银行（PBoC）的政策传导路径，习惯通过央行《货币政策执行报告》中的措辞细节来评估通胀走势与政策方向，并据此为客户撰写投资建议。

# Task Context
我将上传一份完整的 PBoC《货币政策执行报告》（PDF）。请通读全文，从买方/卖方研究分析师的视角解读其中的通胀叙事和政策信号。

# Objective
请将报告中的非结构化通胀与政策叙事，量化为 10 个标准化维度的结构化指标，供后续时间序列建模使用。请基于经济学推理而非关键词匹配。

# Extraction Dimensions

## Group A: Inflation Outlook
1.  **`INF_Sentiment` (-10 to 10):** 央行对未来物价走势的整体判断。-10 表示对通缩极度担忧；0 表示物价处于合意区间；10 表示对通胀过热极度担忧。
2.  **`INF_Duration` (0 to 1):** 央行认为当前价格波动的持续性。0 表示阶段性/暂时性；1 表示长期性/结构性。

## Group B: Attribution Analysis (0-10 weights)
3.  **`DRV_Demand`:** 总需求侧（消费、投资）作为价格变动来源的强度。
4.  **`DRV_Supply`:** 供给侧（食品、能源、产能瓶颈）作为来源的强度。
5.  **`DRV_External`:** 外部因素（国际油价、海外通胀输入）的强度。
6.  **`DRV_Monetary`:** 货币与信贷因素（M2、社融）的强度。

## Group C: Policy Stance
7.  **`POL_Tone` (-10 to 10):** 与价格稳定相关的货币政策基调。-10 高度宽松/鸽派；0 稳健中性；10 高度紧缩/鹰派。
8.  **`POL_Priority` (0 to 10):** "稳定物价"在当前政策目标排序中的优先级；10 表示首要任务。

## Group D: Narrative Features
9.  **`TXT_Ambiguity` (0 to 10):** 央行措辞的模糊与对冲程度。
10. **`TXT_Confidence` (0 to 10):** 央行对自身判断/预测能力的自信程度。

# Output Format (JSON only, no markdown wrapper)
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
    "INF_Sentiment_Reasoning": "...",
    "POL_Tone_Reasoning": "..."
  },
  "key_narrative_shifts": "..."
}

# Constraint
请遵循 PBoC 官方语境：准确区分"跨周期"与"逆周期"调节、"精准滴灌"与"大水漫灌"等措辞。无法推断的维度填 null，但应尽量避免。
"""

DIMENSIONS = [
    "INF_Sentiment", "INF_Duration",
    "DRV_Demand", "DRV_Supply", "DRV_External", "DRV_Monetary",
    "POL_Tone", "POL_Priority",
    "TXT_Ambiguity", "TXT_Confidence",
]


def extract(pdf_path, period, prompt, out_dir, label):
    out_path = os.path.join(out_dir, f"{period}.json")
    if os.path.exists(out_path):
        print(f"  [skip cached {label}] {period}")
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"  [extract {label}] {period}")
    last_err = None
    for attempt in range(6):
        try:
            file_upload = client.files.upload(file=pdf_path)
            while file_upload.state.name == "PROCESSING":
                time.sleep(2)
                file_upload = client.files.get(name=file_upload.name)
            if file_upload.state.name != "ACTIVE":
                raise RuntimeError(f"upload state {file_upload.state.name}")
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[types.Content(role="user", parts=[
                    types.Part.from_uri(file_uri=file_upload.uri, mime_type=file_upload.mime_type),
                    types.Part.from_text(text=prompt),
                ])],
            )
            break
        except Exception as e:
            last_err = e
            wait = 5 * (2 ** attempt)
            print(f"    [retry {attempt+1}/6 after {wait}s] {type(e).__name__}: {str(e)[:120]}")
            time.sleep(wait)
    else:
        print(f"  [give up {label}] {period}: {last_err}")
        return None

    result = extract_json_from_response(response.text)
    if not result:
        print(f"  [parse fail {label}] {period}")
        return None
    result["report_period"] = period
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main():
    if not API_KEY:
        raise SystemExit("GEMINI_API_KEY not set")

    v1_rows, v2_rows = [], []
    for period in SAMPLE_PERIODS:
        pdf = os.path.join(REPORTS_DIR, f"{period}.pdf")
        if not os.path.exists(pdf):
            print(f"  [missing pdf] {period}")
            continue
        r1 = extract(pdf, period, SYSTEM_PROMPT_V1, OUTPUT_JSON_DIR_V1, "v1")
        r2 = extract(pdf, period, SYSTEM_PROMPT_V2, OUTPUT_JSON_DIR_V2, "v2")
        if r1 is None or r2 is None:
            continue
        m1 = r1.get("quantitative_metrics", {})
        m2 = r2.get("quantitative_metrics", {})
        v1_rows.append({"report_period": period, **{d: m1.get(d) for d in DIMENSIONS}})
        v2_rows.append({"report_period": period, **{d: m2.get(d) for d in DIMENSIONS}})

    v1 = pd.DataFrame(v1_rows).set_index("report_period")
    v2 = pd.DataFrame(v2_rows).set_index("report_period")
    v1.to_csv(os.path.join(OUTPUT_DIR, "v1_features.csv"))
    v2.to_csv(os.path.join(OUTPUT_DIR, "v2_features.csv"))

    common = v1.index.intersection(v2.index)
    v1c = v1.loc[common, DIMENSIONS].astype(float)
    v2c = v2.loc[common, DIMENSIONS].astype(float)

    rows = []
    for d in DIMENSIONS:
        s1, s2 = v1c[d], v2c[d]
        mask = s1.notna() & s2.notna()
        n = mask.sum()
        if n < 3:
            rows.append({"dimension": d, "n": n, "pearson": None, "spearman": None,
                         "mean_v1": s1[mask].mean(), "mean_v2": s2[mask].mean(),
                         "mean_abs_diff": None})
            continue
        rows.append({
            "dimension": d, "n": n,
            "pearson":  s1[mask].corr(s2[mask], method="pearson"),
            "spearman": s1[mask].corr(s2[mask], method="spearman"),
            "mean_v1":  s1[mask].mean(),
            "mean_v2":  s2[mask].mean(),
            "mean_abs_diff": (s1[mask] - s2[mask]).abs().mean(),
        })
    corr = pd.DataFrame(rows)
    corr.to_csv(os.path.join(OUTPUT_DIR, "correlation.csv"), index=False)
    print("\n=== Prompt-sensitivity correlation (V1 baseline vs V2 variant prompt) ===")
    print(corr.to_string(index=False))
    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'correlation.csv')}")


if __name__ == "__main__":
    main()
