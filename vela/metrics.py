"""Performance, signal-quality and statistical-significance metrics.

Sharpe ratios use a zero risk-free rate (docs/DECISIONS.md, D7).

References
----------
* Bailey, D. & López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier
  (Probabilistic Sharpe Ratio). Journal of Risk.
* Bailey, D. & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality. J. Portfolio Management.
* Politis, D. & Romano, J. (1994). The Stationary Bootstrap. JASA.
* Newey, W. & West, K. (1987). HAC covariance estimation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


# ----------------------------------------------------------------- basic stats
def equity_curve(r: np.ndarray | pd.Series) -> np.ndarray:
    """Cumulative growth of 1 unit given simple daily returns (NaN treated as 0)."""
    return np.cumprod(1.0 + np.nan_to_num(np.asarray(r, dtype=float)))


def drawdown(r: np.ndarray | pd.Series) -> np.ndarray:
    """Percentage drop of the equity curve from its running peak, per day."""
    eq = equity_curve(r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return eq / peak - 1.0


def max_drawdown(r) -> float:
    """Worst peak-to-trough drawdown (a negative number)."""
    return float(drawdown(r).min()) if len(r) else 0.0


def sharpe(r, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio with a zero risk-free rate."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0


def sortino(r, periods: int = TRADING_DAYS) -> float:
    """Annualised Sortino ratio (downside deviation below zero)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    return float(r.mean() / downside * np.sqrt(periods)) if downside > 0 else 0.0


def cagr(r, periods: int = TRADING_DAYS) -> float:
    """Compound annual growth rate implied by daily returns."""
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return 0.0
    growth = equity_curve(r)[-1]
    if growth <= 0:
        return -1.0
    return float(growth ** (periods / len(r)) - 1.0)


def perf_stats(r, periods: int = TRADING_DAYS) -> dict:
    """Standard performance summary of a daily return series."""
    r = np.asarray(r, dtype=float)
    r = np.nan_to_num(r)
    mdd = max_drawdown(r)
    g = cagr(r, periods)
    return {
        "cagr": g,
        "ann_return": float(r.mean() * periods),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(periods)),
        "sharpe": sharpe(r, periods),
        "sortino": sortino(r, periods),
        "max_drawdown": mdd,
        "calmar": float(g / abs(mdd)) if mdd < 0 else np.nan,
        "hit_rate": float(np.mean(r[r != 0] > 0)) if np.any(r != 0) else np.nan,
        "skew": float(stats.skew(r)),
        "kurtosis": float(stats.kurtosis(r, fisher=False)),
        "n_days": int(len(r)),
    }


# -------------------------------------------------------- relative to benchmark
def _newey_west_se(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    n = X.shape[0]
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * resid[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.diag(cov)) * np.sqrt(n / max(n - X.shape[1], 1))


def relative_stats(r, bench, periods: int = TRADING_DAYS, nw_lags: int = 5) -> dict:
    """CAPM regression r = a + b * bench + e with Newey-West t-stats; IR on active returns."""
    r = np.asarray(r, dtype=float)
    b = np.asarray(bench, dtype=float)
    ok = np.isfinite(r) & np.isfinite(b)
    r, b = r[ok], b[ok]
    X = np.column_stack([np.ones_like(b), b])
    coef, *_ = np.linalg.lstsq(X, r, rcond=None)
    resid = r - X @ coef
    se = _newey_west_se(X, resid, nw_lags)
    active = r - b
    te = active.std(ddof=1) * np.sqrt(periods)
    return {
        "alpha_ann": float(coef[0] * periods),
        "alpha_t": float(coef[0] / se[0]) if se[0] > 0 else np.nan,
        "beta": float(coef[1]),
        "correlation": float(np.corrcoef(r, b)[0, 1]),
        "tracking_error": float(te),
        "information_ratio": float(active.mean() * periods / te) if te > 0 else np.nan,
        "excess_cagr": cagr(r, periods) - cagr(b, periods),
    }


# --------------------------------------------------- Sharpe-ratio significance
def probabilistic_sharpe(r, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_benchmark), adjusting for skew and kurtosis.

    ``sr_benchmark`` is in per-period (daily) units.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    sd = r.std(ddof=1)
    if n < 3 or sd == 0:
        return np.nan
    sr = r.mean() / sd
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, var_trial_sharpes: float) -> float:
    """Expected maximum of ``n_trials`` per-period Sharpe ratios under the null of no skill."""
    if n_trials < 2:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_trial_sharpes) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(r, n_trials: int, trial_sharpes_daily: np.ndarray | None = None,
                    var_trial_sharpes: float | None = None) -> dict:
    """Deflated Sharpe Ratio: PSR against the expected best Sharpe of ``n_trials`` skill-less trials.

    The variance of trial Sharpe ratios (per-period units) is taken from the actual trial
    log when given; otherwise the selected strategy's own Sharpe standard error is used,
    which is the textbook fallback.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if var_trial_sharpes is None:
        if trial_sharpes_daily is not None and len(trial_sharpes_daily) > 1:
            var_trial_sharpes = float(np.var(trial_sharpes_daily, ddof=1))
        else:
            var_trial_sharpes = 1.0 / n
    sr0 = expected_max_sharpe(n_trials, var_trial_sharpes)
    return {
        "dsr": probabilistic_sharpe(r, sr0),
        "sr0_annual": sr0 * np.sqrt(TRADING_DAYS),
        "n_trials": int(n_trials),
    }


def stationary_bootstrap_sharpe(r, n_boot: int = 2000, mean_block: int = 20, seed: int = 0,
                                periods: int = TRADING_DAYS) -> dict:
    """Politis-Romano stationary bootstrap confidence interval for the annualised Sharpe ratio."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    new_block = rng.random((n_boot, n)) < p
    jumps = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):  # loop over time, vectorised over resamples
        idx[:, t] = np.where(new_block[:, t], jumps[:, t], (idx[:, t - 1] + 1) % n)
    samples = r[idx]
    sr = samples.mean(1) / samples.std(1, ddof=1) * np.sqrt(periods)
    return {
        "sharpe_ci_low": float(np.quantile(sr, 0.025)),
        "sharpe_ci_high": float(np.quantile(sr, 0.975)),
        "p_sharpe_le_0": float(np.mean(sr <= 0)),
    }


# ------------------------------------------------------------- signal quality
def rank_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise ranks scaled to [0, 1], NaN-aware (average ranks for ties)."""
    out = np.full(x.shape, np.nan)
    for t in range(x.shape[0]):
        row = x[t]
        ok = np.isfinite(row)
        k = ok.sum()
        if k > 1:
            out[t, ok] = (stats.rankdata(row[ok]) - 1) / (k - 1)
    return out


def daily_rank_ic(pred: np.ndarray, realised: np.ndarray, min_names: int = 20) -> np.ndarray:
    """Spearman correlation between predictions and realised returns, per date."""
    ic = np.full(pred.shape[0], np.nan)
    for t in range(pred.shape[0]):
        ok = np.isfinite(pred[t]) & np.isfinite(realised[t])
        if ok.sum() >= min_names:
            a = stats.rankdata(pred[t, ok])
            b = stats.rankdata(realised[t, ok])
            ic[t] = np.corrcoef(a, b)[0, 1]
    return ic


def ic_summary(ic: np.ndarray, horizon: int = 1) -> dict:
    """Mean IC, IC information ratio, and a t-stat corrected for overlapping horizons."""
    ic = ic[np.isfinite(ic)]
    if len(ic) < 10:
        return {"ic_mean": np.nan, "ic_ir": np.nan, "ic_t": np.nan, "ic_hit": np.nan}
    m, sd = ic.mean(), ic.std(ddof=1)
    n_eff = len(ic) / max(horizon, 1)
    return {
        "ic_mean": float(m),
        "ic_ir": float(m / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
        "ic_t": float(m / sd * np.sqrt(n_eff)) if sd > 0 else np.nan,
        "ic_hit": float(np.mean(ic > 0)),
    }
