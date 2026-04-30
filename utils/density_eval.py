"""Density / interval forecast evaluation utilities.

Self-contained — no extra deps beyond numpy / scipy.

Metrics:
- CRPS (sample form, Hersbach 2000): for an ensemble {x_1..x_N} and observation y,
      CRPS = (1/N) Σ |x_i - y| - 1/(2 N^2) ΣΣ |x_i - x_j|.
- Pinball loss at quantile τ: ρ_τ(y - q) = (τ - 1{y < q})(y - q).
- Coverage rate at level α: mean indicator that y_t ∈ [q_{(1-α)/2}, q_{(1+α)/2}].
- Christoffersen (1998): LR_uc (unconditional coverage), LR_ind (independence),
  LR_cc = LR_uc + LR_ind, all χ² with df 1, 1, 2 respectively.
- Mincer-Zarnowitz style: |e_t| ~ ambiguity + controls with Newey-West SE.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


# ---------------- CRPS ----------------

def crps_sample(samples: np.ndarray, y: float) -> float:
    """Hersbach sample CRPS for one observation given an ensemble.

    samples : 1-D array of N draws from the predictive distribution
    y       : scalar observation
    """
    s = np.sort(np.asarray(samples, dtype=float).ravel())
    n = s.size
    if n == 0:
        return float('nan')
    term1 = float(np.mean(np.abs(s - y)))
    # E|X - X'| = (2/n^2) Σ_{i<j} (s_j - s_i) using sorted s
    # closed form: 2/n^2 * Σ_i s_i * (2i - n - 1) (1-indexed) — but simpler:
    diff = s[None, :] - s[:, None]
    term2 = float(np.mean(np.abs(diff))) / 2.0
    return term1 - term2


def crps_series(samples_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector of per-period CRPS values.

    samples_matrix : (T, N) — T forecasts, each with N ensemble members.
    y              : (T,)
    """
    return np.array([crps_sample(samples_matrix[t], y[t]) for t in range(len(y))])


# ---------------- Quantile metrics ----------------

def pinball_loss(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    e = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1) * e)))


def coverage_rate(y: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> float:
    y, q_low, q_high = map(np.asarray, (y, q_low, q_high))
    return float(np.mean((y >= q_low) & (y <= q_high)))


def average_interval_width(q_low: np.ndarray, q_high: np.ndarray) -> float:
    return float(np.mean(np.asarray(q_high) - np.asarray(q_low)))


# ---------------- Christoffersen tests ----------------

def christoffersen_test(hits: np.ndarray, alpha_target: float
                        ) -> dict:
    """Christoffersen (1998) coverage tests.

    hits : 0/1 array — 1 if y_t fell OUTSIDE the (1-α) PI (a 'violation').
    alpha_target : nominal violation rate (e.g. 0.10 for an 80% PI ≡ 0.10 each tail
                   summed = 0.20, OR 0.20 for the full miss rate; pass the rate
                   you actually want to test against).

    Returns dict of (LR_uc, LR_ind, LR_cc) with p-values.
    """
    h = np.asarray(hits, dtype=int).ravel()
    T = h.size
    n1 = int(h.sum())
    n0 = T - n1
    pi_hat = n1 / T if T else float('nan')

    # LR_uc: unconditional coverage
    if 0 < pi_hat < 1 and 0 < alpha_target < 1:
        ll0 = n0 * np.log(1 - alpha_target) + n1 * np.log(alpha_target)
        ll1 = n0 * np.log(1 - pi_hat) + n1 * np.log(pi_hat)
        lr_uc = -2 * (ll0 - ll1)
    else:
        lr_uc = float('nan')
    p_uc = 1 - chi2.cdf(lr_uc, df=1) if np.isfinite(lr_uc) else float('nan')

    # LR_ind: independence (1st-order Markov on hits)
    n00 = n01 = n10 = n11 = 0
    for i in range(T - 1):
        a, b = h[i], h[i + 1]
        if   a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        elif a == 1 and b == 1: n11 += 1
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi_pooled = (n01 + n11) / max(1, (n00 + n01 + n10 + n11))

    def _safe_log(p):
        return np.log(p) if 0 < p < 1 else 0.0

    ll_pooled = ((n00 + n10) * _safe_log(1 - pi_pooled)
                 + (n01 + n11) * _safe_log(pi_pooled))
    ll_markov = (n00 * _safe_log(1 - pi01) + n01 * _safe_log(pi01)
                 + n10 * _safe_log(1 - pi11) + n11 * _safe_log(pi11))
    lr_ind = -2 * (ll_pooled - ll_markov)
    p_ind = 1 - chi2.cdf(lr_ind, df=1) if np.isfinite(lr_ind) else float('nan')

    lr_cc = lr_uc + lr_ind if (np.isfinite(lr_uc) and np.isfinite(lr_ind)) else float('nan')
    p_cc = 1 - chi2.cdf(lr_cc, df=2) if np.isfinite(lr_cc) else float('nan')

    return {
        'n': T, 'n_violations': n1, 'violation_rate': pi_hat,
        'target_rate': alpha_target,
        'LR_uc': float(lr_uc), 'p_uc': float(p_uc),
        'LR_ind': float(lr_ind), 'p_ind': float(p_ind),
        'LR_cc': float(lr_cc), 'p_cc': float(p_cc),
    }


# ---------------- Diebold-Mariano on CRPS ----------------

def dm_crps(crps_a: np.ndarray, crps_b: np.ndarray, h: int = 1
            ) -> Tuple[float, float]:
    """DM test on CRPS loss differential (HLN small-sample correction)."""
    from utils.dm_test import dm_test
    # repurpose dm_test by feeding 'errors' such that |e|^2 reduces to crps.
    # Easier: implement directly.
    d = np.asarray(crps_a, float) - np.asarray(crps_b, float)
    T = len(d)
    mean_d = d.mean()
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for k in range(1, h):
        cov = np.cov(d[k:], d[:-k], ddof=0)[0, 1]
        var_d += 2 * cov
    var_d /= T
    if var_d <= 0:
        return float('nan'), float('nan')
    dm = mean_d / np.sqrt(var_d)
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat = dm * correction
    from scipy.stats import t as student_t
    p_value = 2 * (1 - student_t.cdf(abs(dm_stat), df=T - 1))
    return float(dm_stat), float(p_value)


# ---------------- Newey-West OLS ----------------

def ols_newey_west(y: np.ndarray, X: np.ndarray, lag: int = None
                   ) -> dict:
    """OLS with Newey-West HAC standard errors.

    X : (T, K) design matrix (caller adds intercept column if desired).
    lag : truncation lag — default floor(4 (T/100)^(2/9)) (Andrews-Newey-West rule).
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    T, K = X.shape
    if lag is None:
        lag = max(1, int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0))))

    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta

    # NW S = Γ_0 + Σ_{l=1..lag} (1 - l/(lag+1)) (Γ_l + Γ_l')
    u = X * resid[:, None]  # (T, K)
    S = u.T @ u
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        gamma_l = u[l:].T @ u[:-l]
        S += w * (gamma_l + gamma_l.T)

    cov_beta = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov_beta))
    tstat = beta / se
    pval = 2 * (1 - norm.cdf(np.abs(tstat)))
    # in-sample R²
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    return {
        'beta': beta, 'se': se, 't': tstat, 'p': pval,
        'cov': cov_beta, 'r2': r2, 'n': T, 'k': K, 'lag': lag,
        'resid': resid,
    }


def format_nw_table(res: dict, names: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame({
        'coef': res['beta'], 'se': res['se'],
        't': res['t'], 'p': res['p'],
    }, index=list(names))
