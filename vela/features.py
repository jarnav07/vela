"""Feature engineering on the (dates x tickers) panel.

Every feature on row *t* uses only prices and volumes up to and including the
close of day *t*. Features are computed for all tickers on all dates with
vectorised pandas/NumPy rolling operations, then cross-sectionally normalised
over the tradable universe of each day.

Feature families (motivation in docs/DECISIONS.md, D8):

* **Reversal / momentum**: returns over 1..252 days, 12-1 momentum, lagged daily
  returns, overnight vs intraday decomposition (short-term reversal concentrates in
  the intraday leg), distance from moving averages, 52-week-high proximity.
* **Risk**: realised, Parkinson (high-low) and idiosyncratic volatility, beta,
  maximum daily return (lottery demand), downside volatility.
* **Liquidity / attention**: dollar volume level, volume shocks, Amihud illiquidity.
* **Technical**: RSI(14), Bollinger z-score, close location within the day's range.
* **Market context** (same value for every stock on a date; lets tree and neural
  models condition on the regime): market returns, market volatility, cross-sectional
  dispersion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data.panel import Panel


@dataclass
class FeatureSet:
    """Feature tensor (T x N x K) with its feature names; market-level names are listed separately."""
    names: list[str]
    values: np.ndarray        # (T, N, K) float32, cross-sectionally normalised
    market_names: list[str]   # subset of names that are market-level (not ranked)

    def index(self, name: str) -> int:
        return self.names.index(name)


def _df(a: np.ndarray, panel: Panel) -> pd.DataFrame:
    return pd.DataFrame(a, index=panel.dates, columns=panel.tickers)


def _log_ret(df: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    return np.log(df / df.shift(k))


def raw_features(panel: Panel, tradable: np.ndarray) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Unnormalised stock-level features (DataFrames) and market-level features (Series)."""
    c = _df(panel.close, panel)
    o = _df(panel.open, panel)
    h = _df(panel.high, panel)
    l = _df(panel.low, panel)
    v = _df(panel.volume, panel)
    dv = c * v

    lr = _log_ret(c)                         # close-to-close log return
    on = np.log(o / c.shift(1))              # overnight
    intra = np.log(c / o)                    # intraday
    f: dict[str, pd.DataFrame] = {}

    # --- reversal / momentum
    for k in (1, 2, 3, 5, 10, 21, 63, 126, 252):
        f[f"ret_{k}"] = _log_ret(c, k)
    for lag in (1, 2, 3, 4):
        f[f"ret_lag{lag}"] = lr.shift(lag)
    f["mom_12_1"] = np.log(c.shift(21) / c.shift(252))
    f["mom_6_1"] = np.log(c.shift(21) / c.shift(126))
    f["overnight_1"] = on
    f["intraday_1"] = intra
    f["overnight_5"] = on.rolling(5).sum()
    f["intraday_5"] = intra.rolling(5).sum()
    for w in (10, 50, 200):
        f[f"dist_ma{w}"] = c / c.rolling(w, min_periods=int(w * 0.8)).mean() - 1
    f["high52_prox"] = c / c.rolling(252, min_periods=200).max()
    f["low52_prox"] = c / c.rolling(252, min_periods=200).min()

    # --- risk
    vol5 = lr.rolling(5).std()
    vol21 = lr.rolling(21, min_periods=15).std()
    vol63 = lr.rolling(63, min_periods=40).std()
    f["vol_5"], f["vol_21"], f["vol_63"] = vol5, vol21, vol63
    f["vol_ratio_5_63"] = np.log(vol5 / vol63)
    hl = np.log(h / l)
    f["hl_range_1"] = hl
    f["parkinson_21"] = np.sqrt((hl**2).rolling(21, min_periods=15).mean() / (4 * np.log(2)))
    f["max_ret_21"] = lr.rolling(21, min_periods=15).max()
    f["min_ret_21"] = lr.rolling(21, min_periods=15).min()
    f["downside_vol_63"] = np.sqrt((np.minimum(lr, 0) ** 2).rolling(63, min_periods=40).mean())
    f["skew_63"] = lr.rolling(63, min_periods=40).skew()

    # market = equal-weighted average of tradable names (known at close t)
    lr_trad = lr.where(tradable)
    mkt = lr_trad.mean(axis=1)
    mkt_df = pd.DataFrame(np.repeat(mkt.values[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns)
    cov = (lr * mkt_df).rolling(252, min_periods=126).mean() - lr.rolling(252, min_periods=126).mean() * mkt_df.rolling(252, min_periods=126).mean()
    var_m = mkt.rolling(252, min_periods=126).var(ddof=0)
    beta = cov.div(var_m, axis=0)
    f["beta_252"] = beta
    resid = lr - beta.shift(1) * mkt_df
    f["idio_vol_63"] = resid.rolling(63, min_periods=40).std()
    f["resid_ret_5"] = resid.rolling(5).sum()
    f["resid_ret_21"] = resid.rolling(21, min_periods=15).sum()

    # --- liquidity / attention
    dv21 = dv.rolling(21, min_periods=15).mean()
    f["log_dv_21"] = np.log(dv21)
    f["volume_shock_5_63"] = np.log(dv.rolling(5).mean() / dv.rolling(63, min_periods=40).mean())
    f["volume_shock_1_21"] = np.log(dv / dv21)
    f["amihud_21"] = np.log((lr.abs() / dv).rolling(21, min_periods=15).mean() * 1e9 + 1e-12)

    # --- technical
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["rsi_14"] = 100 - 100 / (1 + up / down)
    ma20 = c.rolling(20).mean()
    f["bollinger_z"] = (c - ma20) / c.rolling(20).std()
    rng = (h - l).replace(0, np.nan)
    f["close_loc_1"] = (c - l) / rng
    f["close_loc_5"] = ((c - l) / rng).rolling(5).mean()

    # --- market context
    m: dict[str, pd.Series] = {
        "mkt_ret_1": mkt,
        "mkt_ret_5": mkt.rolling(5).sum(),
        "mkt_ret_21": mkt.rolling(21).sum(),
        "mkt_vol_21": mkt.rolling(21).std(),
        "mkt_vol_ratio": np.log(mkt.rolling(5).std() / mkt.rolling(63).std()),
        "cs_dispersion_1": lr_trad.std(axis=1),
        "cs_dispersion_21": lr_trad.std(axis=1).rolling(21).mean(),
    }
    return f, m


def cross_sectional_rank(df: pd.DataFrame, tradable: np.ndarray) -> np.ndarray:
    """Rank within the tradable universe of each date, scaled to [-0.5, 0.5]; NaN -> 0 (median)."""
    x = df.where(tradable)
    r = x.rank(axis=1, pct=True, method="average")
    n = x.notna().sum(axis=1)
    # map pct ranks (1/n .. 1) to [-0.5, 0.5]
    scaled = (r.mul(n, axis=0) - 1).div((n - 1).clip(lower=1), axis=0) - 0.5
    return scaled.fillna(0.0).values.astype(np.float32)


def _standardise_market(s: pd.Series, window: int = 252) -> np.ndarray:
    """Time-series z-score using only the trailing window (no look-ahead)."""
    mu = s.rolling(window, min_periods=60).mean()
    sd = s.rolling(window, min_periods=60).std()
    return ((s - mu) / sd).clip(-5, 5).fillna(0.0).values.astype(np.float32)


def build_features(panel: Panel, tradable: np.ndarray, include: list[str] | None = None) -> FeatureSet:
    """Compute all (or ``include``-listed) features and normalise them: ranks for stock-level, trailing z-scores for market-level."""
    stock, market = raw_features(panel, tradable)
    if include is not None:
        stock = {k: v for k, v in stock.items() if k in include}
        market = {k: v for k, v in market.items() if k in include}
    T, N = panel.shape
    names = list(stock) + list(market)
    out = np.zeros((T, N, len(names)), dtype=np.float32)
    for k, name in enumerate(stock):
        out[:, :, k] = cross_sectional_rank(stock[name], tradable)
    for k, name in enumerate(market, start=len(stock)):
        out[:, :, k] = _standardise_market(market[name])[:, None]
    return FeatureSet(names=names, values=out, market_names=list(market))
