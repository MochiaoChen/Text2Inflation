"""SHAP explainability for the tree-based enhanced models.

Generates:
  1. Global feature importance — bar + beeswarm summary plots.
  2. Dependence plot for each of the 10 NLP narrative features.
  3. Before/after time-segment comparison (split at the test-set midpoint).

Outputs land in outputs/shap/{xgboost,random_forest}/. All chart labels are in English.

Usage:
    python -m utils.shap_analysis              # runs both RF and XGB
    python -m utils.shap_analysis --model xgb  # or rf
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from utils.data_utils import (
    load_enhanced_data, create_lag_features, NLP_FEATURE_COLS,
    SHAP_OUTPUT_DIR,
)


MODEL_BUILDERS = {
    'xgboost': lambda: XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                                    subsample=0.9, random_state=42, n_jobs=-1,
                                    objective='reg:squarederror', tree_method='hist'),
    'random_forest': lambda: RandomForestRegressor(n_estimators=400, min_samples_split=3,
                                                   n_jobs=-1, random_state=42),
}


def _prepare_data(start_ratio=0.8):
    """Returns (X_train, y_train, X_test, y_test) as DataFrames with feature names."""
    df, target_col = load_enhanced_data(stationarize=True)
    data = create_lag_features(df, target_col)
    split = int(len(data) * start_ratio)
    X = data.drop(columns=[target_col])
    y = data[target_col]
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


def _nlp_lag_feature_columns(feature_names):
    """All lag columns generated from NLP narrative features."""
    return [c for c in feature_names
            if any(c.startswith(f"{nlp}_lag") for nlp in NLP_FEATURE_COLS)]


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    saved: {path}")


def _summary_plots(shap_values, X, out_dir, tag=""):
    for plot_type, suffix in [('bar', 'summary_bar'), ('dot', 'summary_beeswarm')]:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, plot_type=plot_type, show=False,
                          max_display=20)
        title = f"Global Feature Importance ({tag})" if tag else "Global Feature Importance"
        plt.title(title)
        _save(None, os.path.join(out_dir, f"{suffix}{('_' + tag) if tag else ''}.png"))


def _dependence_plots(shap_values, X, out_dir):
    nlp_cols = _nlp_lag_feature_columns(X.columns)
    if not nlp_cols:
        print("    [warn] no NLP lag columns found in features; skipping dependence plots")
        return

    # For each NLP narrative dim, pick the single lag column with highest mean |SHAP|
    abs_mean = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(abs_mean, index=X.columns)
    per_dim = {}
    for nlp in NLP_FEATURE_COLS:
        candidates = [c for c in nlp_cols if c.startswith(f"{nlp}_lag")]
        if not candidates:
            continue
        per_dim[nlp] = importance[candidates].idxmax()

    dep_dir = os.path.join(out_dir, "dependence")
    os.makedirs(dep_dir, exist_ok=True)
    for nlp, col in per_dim.items():
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(col, shap_values, X, show=False, interaction_index=None)
        plt.title(f"SHAP Dependence — {nlp} ({col})")
        _save(None, os.path.join(dep_dir, f"{nlp}.png"))


def _segmented_summary(model_builder, X_train, y_train, X_test, y_test, out_dir):
    """Split the test window in half, retrain on data up to each split, compare SHAP."""
    if len(X_test) < 4:
        print("    [warn] test set too small for time-segment comparison; skipping")
        return

    mid = len(X_test) // 2
    segments = [
        ('early', X_test.iloc[:mid], y_test.iloc[:mid]),
        ('late',  X_test.iloc[mid:], y_test.iloc[mid:]),
    ]

    for tag, X_seg, _ in segments:
        # Train on everything up to the start of this segment (expanding).
        cutoff = X_seg.index[0]
        X_tr_all = pd.concat([X_train, X_test.loc[:cutoff].iloc[:-1]])
        y_tr_all = pd.concat([y_train, y_test.loc[:cutoff].iloc[:-1]])

        model = model_builder()
        model.fit(X_tr_all.values, y_tr_all.values)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_seg.values)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_seg, plot_type='bar', show=False, max_display=20)
        plt.title(f"Feature Importance — {tag} test segment")
        _save(None, os.path.join(out_dir, f"segment_{tag}_bar.png"))


def run_shap_for(model_key):
    if model_key not in MODEL_BUILDERS:
        raise ValueError(f"unknown model: {model_key}")

    print(f"\n=== SHAP analysis: {model_key} ===")
    X_train, y_train, X_test, y_test = _prepare_data()
    builder = MODEL_BUILDERS[model_key]

    model = builder()
    model.fit(X_train.values, y_train.values)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.values)

    out_dir = os.path.join(SHAP_OUTPUT_DIR, model_key)
    os.makedirs(out_dir, exist_ok=True)

    _summary_plots(shap_values, X_test, out_dir)
    _dependence_plots(shap_values, X_test, out_dir)
    _segmented_summary(builder, X_train, y_train, X_test, y_test, out_dir)

    # Save raw |SHAP| ranking for downstream analysis.
    ranking = pd.Series(np.abs(shap_values).mean(axis=0),
                        index=X_test.columns).sort_values(ascending=False)
    ranking.to_csv(os.path.join(out_dir, 'feature_importance.csv'),
                   header=['mean_abs_shap'])
    print(f"    feature importance table: {out_dir}/feature_importance.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['xgboost', 'random_forest', 'all'], default='all')
    args = ap.parse_args()

    keys = ['xgboost', 'random_forest'] if args.model == 'all' else [args.model]
    for k in keys:
        run_shap_for(k)


if __name__ == "__main__":
    main()
