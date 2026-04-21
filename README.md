# Text2Inflation: Quantifying Central Bank Communication for Inflation Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gemini API](https://img.shields.io/badge/LLM-Gemini_3.0_Flash-orange)](https://deepmind.google/technologies/gemini/)

**Text2Inflation** is a research framework designed to quantify narrative information from the People's Bank of China (PBoC) *Monetary Policy Reports* and integrate these high-dimensional textual features into macroeconomic forecasting models.

By leveraging Large Language Models (LLM) to extract structured sentiment and policy stance indicators, this project enhances traditional time-series models (e.g., LASSO, Random Forest) to improve **CPI growth rate** forecasting accuracy.

[**中文文档 (Chinese Documentation)**](./README_CN.md)

---

## ✨ Key Features

- **LLM-Powered Narrative Extraction**: Utilizes **Google Gemini 2.0 Flash** to parse unstructured PDF reports into 10-dimensional structured data (e.g., Inflation Sentiment, Policy Stance, Attribution Weights).
- **Enhanced Predictive Models**: Implements "Enhanced" versions of classic econometric and ML models that fuse CPI time-series data with NLP-derived features to predict CPI growth rate.
- **Robust Baseline**: Includes standard forecasting models (LASSO, Elastic Net, PCA-OLS, PLS, Random Forest) for rigorous performance benchmarking.
- **Deep Learning & Gradient Boosting**: Adds **LSTM** (PyTorch, 2-layer hidden=64, sequence input) and **XGBoost** (GridSearch-tuned) enhanced models.
- **Expanding-Window Evaluation**: Every model is scored with one-step-ahead expanding-window forecasts, so baseline vs enhanced comparisons are directly Diebold-Mariano-testable.
- **Explainability & Statistical Testing**: Built-in SHAP explainability (bar, beeswarm, NLP-feature dependence, early/late time-segment comparison) and automated Diebold-Mariano tests across every baseline/enhanced pair.
- **Automated Pipeline**: End-to-end workflow from raw PDF renaming and parsing to feature engineering, model evaluation, and post-hoc analysis — all via `run_all.py`.

## 📂 Repository Structure

```text
Text2Inflation/
├── nlp/                     # LLM Extraction Pipeline
│   └── extract_inflation_narrative.py
├── models/
│   ├── enhanced/            # NLP-Enhanced Models
│   │   ├── lasso_enhanced.py
│   │   ├── elastic_net_enhanced.py
│   │   ├── random_forest_enhanced.py
│   │   ├── pca_enhanced.py
│   │   ├── pls_enhanced.py
│   │   ├── comb_enhanced.py
│   │   ├── lstm_enhanced.py        # PyTorch 2-layer LSTM, seq_len=12
│   │   └── xgboost_enhanced.py     # XGBoost + GridSearchCV
│   └── baseline/            # Standard Time-Series Models
│       ├── lasso.py
│       ├── elastic_net.py
│       ├── random_forest.py
│       ├── pca.py
│       ├── pls.py
│       └── comb.py
├── utils/                   # Data Processing & Analysis
│   ├── data_utils.py        # Shared loaders + expanding_window_predict helper
│   ├── rename_reports.py
│   ├── shap_analysis.py     # SHAP bar/beeswarm/dependence/segmented plots
│   └── dm_test.py           # Diebold-Mariano test across baseline/enhanced pairs
├── data/
│   ├── reports/             # Raw PDF Reports
│   ├── CPI_Data.csv         # Macroeconomic Data
│   └── nlp_features.csv     # Extracted Narrative Features
├── outputs/                 # Forecast Plots & Metrics
│   ├── cpi_growth_baseline/   # Baseline model forecast plots
│   ├── cpi_growth_enhanced/   # Enhanced model forecast plots
│   ├── predictions/           # y_true/y_pred CSV per model (DM + SHAP input)
│   ├── shap/                  # SHAP plots per explained model
│   ├── Outputs.csv            # Aggregated RMSE/MAE/R² table
│   └── dm_test_results.csv    # Diebold-Mariano summary
├── run_all.py               # One-click: all models + SHAP + DM test
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.8+ installed.

```bash
git clone https://github.com/yourusername/Text2Inflation.git
cd Text2Inflation
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root (`.env`) to configure your LLM provider:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Data Preparation

**Standardize Report Filenames:**
If your PBoC reports are named in Chinese (e.g., `2023年第一季度...pdf`), run the normalization script to convert them to `YYYY-QN.pdf` format:

```bash
python -m utils.rename_reports
```

### 4. Running the NLP Pipeline

Extract narrative features from all PDFs in `data/reports/`. This process generates `data/nlp_features.csv`.

```bash
python -m nlp.extract_inflation_narrative
```

### 5. Training & Evaluation

You can run both **Baseline** (CPI only) and **Enhanced** (CPI + NLP) models.

**One-Click Full Pipeline (recommended):**
```bash
python run_all.py                       # all baselines + all enhanced + SHAP + DM
python run_all.py --skip lstm_enhanced  # skip specific models
python run_all.py --no-shap --no-dm     # models only
```

**Run Enhanced Models Individually:**
```bash
python -m models.enhanced.lasso_enhanced
python -m models.enhanced.elastic_net_enhanced
python -m models.enhanced.random_forest_enhanced
python -m models.enhanced.pca_enhanced
python -m models.enhanced.pls_enhanced
python -m models.enhanced.comb_enhanced
python -m models.enhanced.lstm_enhanced     # PyTorch LSTM
python -m models.enhanced.xgboost_enhanced  # XGBoost + GridSearchCV
```

**Run Baseline Models:**
```bash
python -m models.baseline.lasso
python -m models.baseline.elastic_net
python -m models.baseline.random_forest
python -m models.baseline.pca
python -m models.baseline.pls
python -m models.baseline.comb
```

**Run Post-hoc Analysis Only:**
```bash
python -m utils.shap_analysis            # SHAP for XGBoost + Random Forest enhanced
python -m utils.shap_analysis --model xgboost
python -m utils.dm_test                  # DM test across every baseline/enhanced pair
```

## 📊 Methodology

### Narrative Dimensions
The NLP pipeline extracts the following dimensions from each report:
1.  **Inflation Sentiment**: Central bank's concern level regarding future price levels (-10 to 10).
2.  **Policy Stance**: Dovish vs. Hawkish tone (-10 to 10).
3.  **Attribution**: Weights assigned to Demand, Supply, External, and Monetary factors.
4.  **Communication Features**: Ambiguity and Confidence scores.

### Modeling Approach
- **Baseline**: Autoregressive distributed lag models using historical CPI and macroeconomic indicators (LASSO, Elastic Net, Random Forest, PCA+OLS, PLS, equal-weight Combination).
- **Enhanced**: Adds the 10 extracted narrative features as exogenous variables (forward-filled quarterly → monthly) on top of the baseline feature set. Includes the six classical models plus **LSTM** (PyTorch, 2-layer hidden=64, seq_len=12 sequence window, retrain every 6 steps) and **XGBoost** (GridSearchCV over `n_estimators`/`max_depth`/`learning_rate`/`subsample`, fixed after initial tune).
- **Evaluation**: All models use one-step-ahead **expanding-window** forecasts starting at 80% of the sample. Each model writes `y_true` / `y_pred` to `outputs/predictions/{model}.csv`, enabling direct DM testing without re-running.

### Explainability (SHAP)
`utils/shap_analysis.py` fits `TreeExplainer` on XGBoost Enhanced and Random Forest Enhanced, producing:
1. Global importance bar + beeswarm summary plots.
2. A SHAP dependence plot for each of the 10 NLP narrative features (picking the lag with the highest mean-|SHAP| per dimension).
3. Early vs late test-segment summary plots to expose regime-dependent drivers.

### Statistical Significance (Diebold-Mariano)
`utils/dm_test.py` implements the Harvey-Leybourne-Newbold small-sample-corrected DM test (squared-error loss). It pairs every baseline with its enhanced twin, plus the new LSTM/XGBoost against Random Forest Baseline, and writes `outputs/dm_test_results.csv` with DM statistic, p-value, and the winning model per pair.

## 🤝 Contributing

Contributions are welcome! Please submit a Pull Request or open an Issue for any bugs or feature suggestions.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
