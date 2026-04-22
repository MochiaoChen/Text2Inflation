# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Research framework that quantifies narrative information from PBoC *Monetary Policy Reports* (PDFs) via LLM extraction, then feeds those features into CPI inflation forecasting models. Every pipeline decision is driven by that goal: extract structured indicators from text → merge with macro time series → forecast CPI growth.

## Environment

- Python 3.8+ (dev venv uses 3.12 at `/home/mochiao/code/.venv`).
- Install deps: `pip install -r requirements.txt`. Note `requirements.txt` is incomplete — the NLP pipeline also needs `google-genai` and `python-dotenv`.
- LLM config: `GEMINI_API_KEY` in a project-root `.env` file. Without it, `nlp/extract_inflation_narrative.py` only writes an empty stub CSV.

## Common Commands

All scripts are run as modules from the repo root (they rely on package-relative imports or `sys.path` manipulation):

```bash
# Normalize PDF filenames (Chinese → YYYY-QN.pdf)
python -m utils.rename_reports

# LLM extraction: data/reports/*.pdf → data/nlp_results/*.json → data/nlp_features.csv
python -m nlp.extract_inflation_narrative

# Baseline models (CPI-only)
python -m models.baseline.{lasso,elastic_net,random_forest,pca,pls,comb}

# Enhanced models (CPI + NLP features)
python -m models.enhanced.{lasso_enhanced,elastic_net_enhanced,random_forest_enhanced,pca_enhanced,pls_enhanced,comb_enhanced}
```

There is no test suite, linter config, or build step.

## Architecture

The pipeline has three layers that future edits must keep consistent:

1. **NLP extraction (`nlp/extract_inflation_narrative.py`)** — Uploads each PDF to Gemini (`gemini-2.0-flash`) with a strict system prompt defining 10 narrative dimensions (see Group A–D in the prompt: `INF_Sentiment`, `INF_Duration`, `DRV_{Demand,Supply,External,Monetary}`, `POL_{Tone,Priority}`, `TXT_{Ambiguity,Confidence}`). Per-report JSON is cached in `data/nlp_results/{YYYY-QN}.json` and the aggregate CSV is `data/nlp_features.csv`. If you change the dimension schema, update the prompt, the CSV column order in `main()`, and every downstream model.

2. **Shared data layer (`utils/data_utils.py`)** — All models import from here. Key contract:
   - `load_and_clean_data(stationarize=True)` returns monthly CPI data; when `stationarize=True` the target is `Inflation` (pct_change of CPI), otherwise `CPI` (raw index).
   - `load_enhanced_data(...)` merges `nlp_features.csv` into the monthly frame by mapping `YYYY-QN` to the quarter-end date, then `reindex(..., method='ffill')`. This is deliberately a backtest-style join; it does **not** model PBoC's real ~1–2 month publication lag. Respect that if you change the merge logic.
   - `create_lag_features` defaults to lags `[1,2,3,6,12]` on every non-target column.
   - `split_and_scale` does a chronological 80/20 split (no shuffling) and fits `StandardScaler` on train only.
   - `evaluate_and_plot` writes PNGs to `outputs/` and uses SimHei Chinese fonts.

3. **Model scripts (`models/baseline/`, `models/enhanced/`)** — Each file is a standalone runnable that wires `load_*_data → create_lag_features → split_and_scale → fit → evaluate_and_plot`. Enhanced variants differ from baselines only by calling `load_enhanced_data` instead of `load_and_clean_data`. `comb*.py` is an equal-weight average of LASSO, ElasticNet, and PCA+OLS predictions.

### Import quirks to watch for

- Model scripts do `sys.path.insert(0, <repo_root>)` then `from data_utils import ...`. This only works because `data_utils.py` is also (or is expected to be) importable from the root — in practice the canonical copy lives at `utils/data_utils.py`. When adding a new model, copy the exact import pattern used by sibling files rather than inventing a new one, and verify it actually resolves.
- `models/baseline/` contains duplicate/legacy files with spaces or Chinese names (`Elastic Net.py`, `随机森林.py`, `utils.py`). The canonical entry points are the snake_case filenames referenced in the README; treat the others as stale unless explicitly asked.

## Data Conventions

- Raw PDFs: `data/reports/YYYY-QN.pdf` (run `utils.rename_reports` if they arrive in Chinese).
- `data/CPI_Data.csv` must have a `date` column and either a `CPI(2001-01=100)` or `CPI` column; all other columns are treated as features and pct_change-transformed when `stationarize=True`.
- `report_period` in `nlp_features.csv` uses the format `YYYY-QN` (e.g. `2023-Q1`); the enhanced loader maps these to quarter-end dates.
