# Text2Inflation: 基于央行沟通文本量化的通胀预测框架

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gemini API](https://img.shields.io/badge/LLM-Gemini_2.0_Flash-orange)](https://deepmind.google/technologies/gemini/)

**Text2Inflation** 是一个用于量化中国人民银行（PBoC）《货币政策执行报告》叙事信息的开源研究框架。本项目通过构建结构化 NLP 特征并将其融入经典宏观经济预测模型，旨在提升对中国 CPI 通胀率的预测精度。

本项目通过先进的大语言模型（LLM）技术，从非结构化的央行报告中提取关于通胀情绪、政策立场及归因分析的**高维结构化指标**，为货币政策传导机制研究和宏观预测提供了新的量化视角。

[**English Documentation**](./README.md)

---

## ✨ 核心特性

- **LLM 驱动的叙事提取**: 利用 **Google Gemini 2.0 Flash** 强大的理解能力，自动化解析 PDF 报告，提取包括通胀预期、政策基调、不确定性指数等在内的 10 维结构化数据。
- **增强型预测模型 (Enhanced Models)**: 提供了一整套融合了 NLP 特征的增强型计量模型（LASSO, Elastic Net, Random Forest 等），显著区别于传统的纯时间序列基准模型。
- **完善的基准测试 (Robust Baseline)**: 内置标准的时间序列预测模型作为对照组，确保模型评估的严谨性。
- **全流程自动化**: 从 PDF 报告的智能重命名、文本解析，到特征工程与模型评估的端到端流水线。

## 📂 项目结构

```text
Text2Inflation/
├── code/
│   ├── nlp/                     # LLM 提取管道
│   │   └── extract_inflation_narrative.py  # 核心提取脚本
│   ├── models/
│   │   ├── enhanced/            # NLP 增强模型
│   │   │   ├── lasso_enhanced.py
│   │   │   ├── elastic_net_enhanced.py
│   │   │   ├── random_forest_enhanced.py
│   │   │   ├── pca_enhanced.py
│   │   │   ├── pls_enhanced.py
│   │   │   └── comb_enhanced.py
│   │   └── baseline/            # 传统基准模型
│   │       ├── lasso.py
│   │       ├── elastic_net.py
│   │       ├── random_forest.py
│   │       ├── pca.py
│   │       ├── pls.py
│   │       └── comb.py
│   ├── utils/                   # 数据处理工具
│   │   ├── data_utils.py        # 数据加载与预处理
│   │   └── rename_reports.py    # 报告文件名标准化
│   ├── data/
│   │   ├── reports/             # 原始 PDF 报告目录
│   │   ├── CPI_Data.csv         # 宏观经济数据
│   │   └── nlp_features.csv     # 提取后的叙事特征
│   └── outputs/                 # 模型输出图表与指标
├── requirements.txt
└── README_CN.md
```

## 🚀 快速开始

### 1. 环境准备

请确保 Python 版本不低于 3.8。

```bash
git clone https://github.com/yourusername/Text2Inflation.git
cd Text2Inflation
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录 (`code/Text2Inflation/.env`) 创建环境配置文件，填入您的 Google Gemini API Key：

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. 数据准备

**标准化报告文件名:**
如果您的央行报告文件名是中文格式（如 `2023年第一季度...pdf`），请运行此脚本将其统一为 `YYYY-QN.pdf` 格式，以便程序识别：

```bash
python -m utils.rename_reports
```

### 4. 运行 NLP 提取管道

批量处理 `data/reports/` 下的所有 PDF 报告，生成结构化特征文件 `data/nlp_features.csv`。

```bash
python -m nlp.extract_inflation_narrative
```

### 5. 训练与评估

您可以分别运行 **基准模型 (Baseline)** 和 **增强模型 (Enhanced)** 进行对比分析。

**运行增强模型 (Enhanced Models):**
这些模型会自动加载 NLP 特征并与 CPI 数据合并。
```bash
python -m models.enhanced.lasso_enhanced        # LASSO 回归
python -m models.enhanced.elastic_net_enhanced  # Elastic Net
python -m models.enhanced.random_forest_enhanced # 随机森林
python -m models.enhanced.pca_enhanced          # PCA + OLS
python -m models.enhanced.pls_enhanced          # 偏最小二乘回归 (PLS)
python -m models.enhanced.comb_enhanced         # 组合预测模型
```

**运行基准模型 (Baseline Models):**
仅使用 CPI 历史数据和传统宏观变量。
```bash
python -m models.baseline.lasso
python -m models.baseline.elastic_net
python -m models.baseline.random_forest
python -m models.baseline.pca
python -m models.baseline.pls
python -m models.baseline.comb
```

## 📊 方法论

### 叙事维度 (Narrative Dimensions)
NLP 管道从每份报告中提取以下关键维度：
1.  **通胀情绪 (Inflation Sentiment)**: 央行对未来物价水平的担忧程度 (-10 to 10)。
2.  **政策立场 (Policy Stance)**: 货币政策的鸽派/鹰派倾向 (-10 to 10)。
3.  **归因分析 (Attribution)**: 通胀压力来源的权重分配（需求、供给、外部、货币因素）。
4.  **沟通特征 (Communication Features)**: 文本的模糊度与央行的自信程度。

### 建模策略
- **Baseline**: 基于历史 CPI 和宏观经济指标的自回归分布滞后模型。
- **Enhanced**: 将提取的叙事特征作为外生变量（经由频率转换处理），捕捉通胀动态中的前瞻性信息。

## 🤝 贡献指南

欢迎任何形式的贡献！如果您发现 Bug 或以有新的想法，请提交 Pull Request 或建立 Issue。

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](./LICENSE) 文件。
