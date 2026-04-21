"""一键运行全部 baseline + enhanced 模型，再执行 SHAP 和 DM 检验。

Usage:
    python run_all.py                     # 全部
    python run_all.py --skip lstm xgboost # 跳过指定模型
    python run_all.py --only lasso_baseline lasso_enhanced
    python run_all.py --no-shap --no-dm
"""
import argparse
import time
import traceback

from models.baseline import (
    lasso as b_lasso, elastic_net as b_en, random_forest as b_rf,
    pca as b_pca, pls as b_pls, comb as b_comb,
)
from models.enhanced import (
    lasso_enhanced as e_lasso, elastic_net_enhanced as e_en,
    random_forest_enhanced as e_rf, pca_enhanced as e_pca,
    pls_enhanced as e_pls, comb_enhanced as e_comb,
    lstm_enhanced as e_lstm, xgboost_enhanced as e_xgb,
)
from utils import shap_analysis, dm_test


MODEL_RUNNERS = [
    # (model_name, runner)  — order matters: combination models read sibling predictions.
    ('lasso_baseline',         b_lasso.run_lasso_baseline),
    ('elastic_net_baseline',   b_en.run_elastic_net_baseline),
    ('random_forest_baseline', b_rf.run_random_forest_baseline),
    ('pca_baseline',           b_pca.run_pca_baseline),
    ('pls_baseline',           b_pls.run_pls_baseline),
    ('comb_baseline',          b_comb.run_comb_baseline),

    ('lasso_enhanced',         e_lasso.run_lasso_enhanced),
    ('elastic_net_enhanced',   e_en.run_elastic_net_enhanced),
    ('random_forest_enhanced', e_rf.run_random_forest_enhanced),
    ('pca_enhanced',           e_pca.run_pca_enhanced),
    ('pls_enhanced',           e_pls.run_pls_enhanced),
    ('comb_enhanced',          e_comb.run_comb_enhanced),

    ('lstm_enhanced',          e_lstm.run_lstm_enhanced),
    ('xgboost_enhanced',       e_xgb.run_xgboost_enhanced),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', help='whitelist of model names to run')
    ap.add_argument('--skip', nargs='*', default=[], help='names to skip')
    ap.add_argument('--no-shap', action='store_true')
    ap.add_argument('--no-dm', action='store_true')
    args = ap.parse_args()

    summary = []
    for name, runner in MODEL_RUNNERS:
        if args.only and name not in args.only:
            continue
        if name in args.skip:
            continue

        print(f"\n================  {name}  ================")
        t0 = time.time()
        try:
            metrics = runner()
            elapsed = time.time() - t0
            print(f"[done] {name} in {elapsed:.1f}s — RMSE={metrics['rmse']:.4f}")
            summary.append((name, 'ok', elapsed, metrics))
        except Exception:
            elapsed = time.time() - t0
            traceback.print_exc()
            summary.append((name, 'FAILED', elapsed, None))

    if not args.no_shap:
        print("\n================  SHAP analysis  ================")
        try:
            for k in ('xgboost', 'random_forest'):
                shap_analysis.run_shap_for(k)
        except Exception:
            traceback.print_exc()

    if not args.no_dm:
        print("\n================  Diebold-Mariano  ================")
        try:
            dm_test.run_dm_all()
        except Exception:
            traceback.print_exc()

    print("\n================  Summary  ================")
    for name, status, elapsed, metrics in summary:
        if metrics:
            print(f"  {name:30s} {status:8s} {elapsed:6.1f}s  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}")
        else:
            print(f"  {name:30s} {status:8s} {elapsed:6.1f}s")


if __name__ == "__main__":
    main()
