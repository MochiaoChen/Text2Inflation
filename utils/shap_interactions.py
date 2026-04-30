"""SHAP interaction values for the tree-based enhanced models.

Reviewer Critique III (mandatory): the paper claims narrative features enter
inflation through "interaction effects" but provides no direct evidence.
This module computes 2nd-order SHAP interaction values, ranks the strongest
NLP × macro pairs, and saves 2-D dependence plots for the top interactions.

Outputs land in outputs/shap/{rf,xgb}/interactions/:
- top_interactions.csv     — top-K interaction pairs by |SHAP_int|
- nlp_macro_top.csv        — top NLP×macro pairs only
- interaction_<f1>__<f2>.png — 2-D SHAP dependence plot for each top pair

Usage:
    python -m utils.shap_interactions
    python -m utils.shap_interactions --model xgboost --topk 8
"""
from __future__ import annotations

import argparse
import os
from itertools import combinations

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


def _is_nlp_lag(col: str) -> bool:
    return any(col.startswith(f"{n}_lag") for n in NLP_FEATURE_COLS)


def _prepare_data(start_ratio=0.8, publication_lag_months=2):
    df, target = load_enhanced_data(stationarize=True,
                                    publication_lag_months=publication_lag_months)
    data = create_lag_features(df, target)
    split = int(len(data) * start_ratio)
    X = data.drop(columns=[target])
    y = data[target]
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


def _fit_model(model_key, X_train, y_train):
    m = MODEL_BUILDERS[model_key]()
    m.fit(X_train.values, y_train.values)
    return m


def _interaction_values(model, X_explain) -> np.ndarray:
    """Returns (n_samples, n_features, n_features) SHAP interaction tensor."""
    explainer = shap.TreeExplainer(model)
    return explainer.shap_interaction_values(X_explain.values)


def _rank_pairs(siv: np.ndarray, feature_names) -> pd.DataFrame:
    """Mean |off-diagonal| interaction per feature pair."""
    abs_int = np.abs(siv).mean(axis=0)  # (F, F)
    np.fill_diagonal(abs_int, 0.0)  # exclude self-interaction (main effect)
    rows = []
    for i, j in combinations(range(len(feature_names)), 2):
        rows.append({
            'feature_1': feature_names[i],
            'feature_2': feature_names[j],
            # off-diagonal entries are split symmetrically; sum = total contribution.
            'mean_abs_interaction': float(abs_int[i, j] + abs_int[j, i]),
        })
    return pd.DataFrame(rows).sort_values('mean_abs_interaction', ascending=False)


def _plot_interaction(siv: np.ndarray, X_explain: pd.DataFrame,
                      f1: str, f2: str, save_path: str):
    """2-D SHAP dependence: x = f1 value, y = SHAP_int(f1, f2), color = f2."""
    feat_idx = {c: i for i, c in enumerate(X_explain.columns)}
    i, j = feat_idx[f1], feat_idx[f2]
    inter = siv[:, i, j] + siv[:, j, i]  # symmetric pair contribution
    f1_vals = X_explain[f1].values
    f2_vals = X_explain[f2].values

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sc = ax.scatter(f1_vals, inter, c=f2_vals, cmap='coolwarm', s=42,
                    edgecolors='k', linewidths=0.3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(f2)
    ax.axhline(0, color='gray', linewidth=0.7, alpha=0.7)
    ax.set_xlabel(f1)
    ax.set_ylabel(f"SHAP interaction value ({f1} × {f2})")
    ax.set_title(f"Interaction: {f1} × {f2}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def run_for(model_key: str, topk: int = 10):
    print(f"\n=== SHAP interaction values: {model_key} ===")
    X_train, y_train, X_test, _ = _prepare_data()
    model = _fit_model(model_key, X_train, y_train)

    print(f"  computing interaction values on test set "
          f"(n={len(X_test)}, p={X_test.shape[1]})...")
    siv = _interaction_values(model, X_test)
    print(f"  interaction tensor shape = {siv.shape}")

    feature_names = list(X_test.columns)
    pair_df = _rank_pairs(siv, feature_names)

    # Mark NLP × macro pairs
    pair_df['type'] = pair_df.apply(
        lambda r: 'nlp_x_nlp' if _is_nlp_lag(r['feature_1']) and _is_nlp_lag(r['feature_2'])
        else 'nlp_x_macro' if _is_nlp_lag(r['feature_1']) ^ _is_nlp_lag(r['feature_2'])
        else 'macro_x_macro', axis=1)

    out_dir = os.path.join(SHAP_OUTPUT_DIR,
                          'random_forest' if model_key == 'random_forest' else 'xgboost',
                          'interactions')
    os.makedirs(out_dir, exist_ok=True)

    pair_df.to_csv(os.path.join(out_dir, 'all_interactions.csv'), index=False)
    pair_df.head(topk).to_csv(os.path.join(out_dir, 'top_interactions.csv'), index=False)
    nlp_macro = pair_df[pair_df['type'] == 'nlp_x_macro'].head(topk)
    nlp_macro.to_csv(os.path.join(out_dir, 'nlp_macro_top.csv'), index=False)

    print(f"\n  Top {topk} interaction pairs (any type):")
    print(pair_df.head(topk).to_string(index=False))
    print(f"\n  Top {topk} NLP × macro pairs:")
    print(nlp_macro.to_string(index=False))

    # 2-D plots for the top NLP × macro pairs (the reviewer's specific ask)
    print(f"\n  rendering 2-D interaction plots → {out_dir}/")
    for _, row in nlp_macro.iterrows():
        f1, f2 = row['feature_1'], row['feature_2']
        save = os.path.join(out_dir, f"interaction_{f1}__{f2}.png")
        _plot_interaction(siv, X_test, f1, f2, save)

    return pair_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['xgboost', 'random_forest', 'all'], default='all')
    ap.add_argument('--topk', type=int, default=10)
    args = ap.parse_args()
    keys = ['xgboost', 'random_forest'] if args.model == 'all' else [args.model]
    for k in keys:
        run_for(k, topk=args.topk)


if __name__ == "__main__":
    main()
