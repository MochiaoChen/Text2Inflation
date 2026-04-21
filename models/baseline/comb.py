"""Combination baseline — 等权平均 LASSO / Elastic Net / PCA+OLS 的 expanding-window 预测。"""
import os
import pandas as pd

from utils.data_utils import (
    setup_plot_style, evaluate_and_plot, save_predictions,
    save_metrics_to_csv, BASELINE_OUTPUT_DIR, PREDICTIONS_DIR,
)

MODEL_NAME = 'comb_baseline'
MEMBERS = ['lasso_baseline', 'elastic_net_baseline', 'pca_baseline']


def _load_or_run():
    from models.baseline import lasso, elastic_net, pca
    runners = {'lasso_baseline': lasso.run_lasso_baseline,
               'elastic_net_baseline': elastic_net.run_elastic_net_baseline,
               'pca_baseline': pca.run_pca_baseline}
    preds = {}
    for name in MEMBERS:
        path = os.path.join(PREDICTIONS_DIR, f"{name}.csv")
        if not os.path.exists(path):
            print(f">>> Running missing member: {name}")
            runners[name]()
        preds[name] = pd.read_csv(path, index_col='date', parse_dates=True)
    return preds


def run_comb_baseline(file_path=None):
    setup_plot_style()
    preds = _load_or_run()

    idx = preds[MEMBERS[0]].index
    y_true = preds[MEMBERS[0]]['y_true']
    y_pred_avg = pd.concat([preds[m]['y_pred'].reindex(idx) for m in MEMBERS], axis=1).mean(axis=1)

    metrics = evaluate_and_plot(y_true, y_pred_avg, 'Combination (Baseline)',
                                os.path.join(BASELINE_OUTPUT_DIR, '6_combination.png'))
    save_predictions(MODEL_NAME, y_true, y_pred_avg)
    save_metrics_to_csv(MODEL_NAME, **metrics)
    return metrics


if __name__ == "__main__":
    run_comb_baseline()
