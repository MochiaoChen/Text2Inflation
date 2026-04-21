"""Combination Enhanced — 等权平均 LASSO / Elastic Net / PCA+OLS enhanced 预测。"""
import os
import pandas as pd

from utils.data_utils import (
    setup_plot_style, evaluate_and_plot, save_predictions,
    save_metrics_to_csv, ENHANCED_OUTPUT_DIR, PREDICTIONS_DIR,
)

MODEL_NAME = 'comb_enhanced'
MEMBERS = ['lasso_enhanced', 'elastic_net_enhanced', 'pca_enhanced']


def _load_or_run():
    from models.enhanced import lasso_enhanced, elastic_net_enhanced, pca_enhanced
    runners = {'lasso_enhanced': lasso_enhanced.run_lasso_enhanced,
               'elastic_net_enhanced': elastic_net_enhanced.run_elastic_net_enhanced,
               'pca_enhanced': pca_enhanced.run_pca_enhanced}
    preds = {}
    for name in MEMBERS:
        path = os.path.join(PREDICTIONS_DIR, f"{name}.csv")
        if not os.path.exists(path):
            print(f">>> Running missing member: {name}")
            runners[name]()
        preds[name] = pd.read_csv(path, index_col='date', parse_dates=True)
    return preds


def run_comb_enhanced(file_path=None):
    setup_plot_style()
    preds = _load_or_run()

    idx = preds[MEMBERS[0]].index
    y_true = preds[MEMBERS[0]]['y_true']
    y_pred_avg = pd.concat([preds[m]['y_pred'].reindex(idx) for m in MEMBERS], axis=1).mean(axis=1)

    metrics = evaluate_and_plot(y_true, y_pred_avg, 'Combination (Enhanced)',
                                os.path.join(ENHANCED_OUTPUT_DIR, '6_combination_enhanced.png'))
    save_predictions(MODEL_NAME, y_true, y_pred_avg)
    save_metrics_to_csv(MODEL_NAME, **metrics)
    return metrics


if __name__ == "__main__":
    run_comb_enhanced()
