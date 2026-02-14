# Text2Inflation

Quantifying central bank monetary policy reports through NLP to forecast CPI inflation. A collection of research papers and implementation code.

## 📁 项目结构

```
Text2Inflation/
├── README.md                    # 项目说明
├── requirements.txt             # Python 依赖
├── code/
│   ├── utils/                   # 公共模块
│   │   └── data_utils.py        # 数据加载、清洗、特征工程、评估等
│   ├── models/                  # 预测模型
│   │   ├── lasso.py             # LASSO 回归
│   │   ├── elastic_net.py       # Elastic Net 回归
│   │   ├── random_forest.py     # 随机森林回归
│   │   ├── pca.py               # PCA 降维 + OLS
│   │   ├── pls.py               # PLS 偏最小二乘回归
│   │   └── comb.py              # 组合预测（多模型平均）
│   ├── data/                    # 数据文件
│   │   ├── CPI_Data.csv         # CPI 月度数据
│   │   ├── GDP_Data.csv         # GDP 数据
│   │   └── raw_data.xlsx        # 原始 Excel 数据
│   ├── outputs/                 # 模型输出图表
│   └── xlsx_to_csv.py           # Excel → CSV 转换工具
```

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行单个模型

```bash
cd code
python -m models.lasso           # LASSO 回归
python -m models.elastic_net     # Elastic Net
python -m models.random_forest   # 随机森林
python -m models.pca             # PCA + OLS
python -m models.pls             # PLS 回归
python -m models.comb            # 组合预测
```

### 数据预处理

```bash
cd code
python xlsx_to_csv.py            # 将 Excel 原始数据转为 CSV
```

## 📊 模型说明

| 模型 | 方法 | 特点 |
|------|------|------|
| LASSO | L1 正则化线性回归 | 自动特征选择，稀疏系数 |
| Elastic Net | L1+L2 混合正则化 | 兼顾特征选择与共线性处理 |
| Random Forest | 集成决策树 | 捕捉非线性关系 |
| PCA + OLS | 主成分降维 + 线性回归 | 高维数据降维 |
| PLS | 偏最小二乘回归 | 同时考虑 X 和 Y 的协方差 |
| Comb | 多模型简单平均 | 降低单一模型偏差 |

## 🔧 技术栈

- Python 3.8+
- pandas, numpy — 数据处理
- scikit-learn — 机器学习建模
- matplotlib, seaborn — 数据可视化
