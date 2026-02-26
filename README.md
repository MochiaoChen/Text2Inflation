# Text2Inflation: Quantifying Central Bank Communication for Inflation Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gemini API](https://img.shields.io/badge/LLM-Gemini_2.0_Flash-orange)](https://deepmind.google/technologies/gemini/)

**Text2Inflation** is a research framework designed to quantify narrative information from the People's Bank of China (PBoC) *Monetary Policy Reports* and integrate these high-dimensional textual features into macroeconomic forecasting models.

By leveraging Large Language Models (LLM) to extract structured sentiment and policy stance indicators, this project enhances traditional time-series models (e.g., LASSO, Random Forest) to improve **CPI growth rate** forecasting accuracy.

[**中文文档 (Chinese Documentation)**](./README_CN.md)

---

## ✨ Key Features

- **LLM-Powered Narrative Extraction**: Utilizes **Google Gemini 2.0 Flash** to parse unstructured PDF reports into 10-dimensional structured data (e.g., Inflation Sentiment, Policy Stance, Attribution Weights).
- **Enhanced Predictive Models**: Implements "Enhanced" versions of classic econometric models that fuse CPI time-series data with NLP-derived features to predict CPI growth rate.
- **Robust Baseline**: Includes standard forecasting models (LASSO, Elastic Net, PCA-OLS, PLS, Random Forest) for rigorous performance benchmarking.
- **Automated Pipeline**: End-to-end workflow from raw PDF renaming and parsing to feature engineering and model evaluation.

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
│   │   └── comb_enhanced.py
│   └── baseline/            # Standard Time-Series Models
│       ├── lasso.py
│       ├── elastic_net.py
│       ├── random_forest.py
│       ├── pca.py
│       ├── pls.py
│       └── comb.py
├── utils/                   # Data Processing Utilities
│   ├── data_utils.py
│   └── rename_reports.py
├── data/
│   ├── reports/             # Raw PDF Reports
│   ├── CPI_Data.csv         # Macroeconomic Data
│   └── nlp_features.csv     # Extracted Narrative Features
├── outputs/                 # Forecast Plots & Metrics
│   ├── cpi_growth_baseline/  # Baseline model results
│   └── cpi_growth_enhanced/  # NLP-enhanced model results
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

**Run Enhanced Models:**
```bash
python -m models.enhanced.lasso_enhanced        # LASSO
python -m models.enhanced.elastic_net_enhanced  # Elastic Net
python -m models.enhanced.random_forest_enhanced # Random Forest
python -m models.enhanced.pca_enhanced          # PCA + OLS
python -m models.enhanced.pls_enhanced          # Partial Least Squares
python -m models.enhanced.comb_enhanced         # Ensemble Model
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

## 📊 Methodology

### Narrative Dimensions
The NLP pipeline extracts the following dimensions from each report:
1.  **Inflation Sentiment**: Central bank's concern level regarding future price levels (-10 to 10).
2.  **Policy Stance**: Dovish vs. Hawkish tone (-10 to 10).
3.  **Attribution**: Weights assigned to Demand, Supply, External, and Monetary factors.
4.  **Communication Features**: Ambiguity and Confidence scores.

### Modeling Approach
- **Baseline**: Autoregressive distributed lag models using historical CPI and macroeconomic indicators.
- **Enhanced**: Incorporates the extracted narrative features as exogenous variables (forward-filled from quarterly to monthly frequency) to capture forward-looking component of inflation dynamics.

## 🤝 Contributing

Contributions are welcome! Please submit a Pull Request or open an Issue for any bugs or feature suggestions.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
