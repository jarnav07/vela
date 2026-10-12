"""Robustness analyses on top of the backtester: costs, capacity, sub-periods, signal decay."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import metrics as M
from .backtest.engine import CostModel
from .research import ResearchData, StrategySpec, backtest_scores


def cost_sensitivity(scores, data: ResearchData, spec: StrategySpec, start, end,
                     multipliers=(0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0), base: CostModel = CostModel()) -> pd.DataFrame:
    """Net performance when every cost component is scaled by each multiplier (0 = frictionless)."""
    rows = []
    for k in multipliers:
        res = backtest_scores(scores, data, spec, base.scaled(k)).window(start, end)
        s = M.perf_stats(res.net)
        rows.append({"cost_multiplier": k, "half_spread_bps": base.half_spread_bps * k,
                     "sharpe": s["sharpe"], "ann_return": s["ann_return"], "max_drawdown": s["max_drawdown"],
                     "cost_bps_day": float((res.cost + res.borrow).mean() * 1e4)})
    return pd.DataFrame(rows)


def break_even_multiplier(sens: pd.DataFrame) -> float:
    """Cost multiplier at which net Sharpe crosses zero (linear interpolation); inf if never."""
    s = sens.sort_values("cost_multiplier")
    x, y = s["cost_multiplier"].values, s["sharpe"].values
    for i in range(1, len(x)):
        if y[i - 1] > 0 >= y[i]:
            return float(x[i - 1] + (x[i] - x[i - 1]) * y[i - 1] / (y[i - 1] - y[i]))
    return float("inf") if y[-1] > 0 else 0.0


def capacity(scores, data: ResearchData, spec: StrategySpec, start, end,
             aums=(1e5, 1e6, 1e7, 5e7, 1e8, 5e8), base: CostModel = CostModel()) -> pd.DataFrame:
    """Net performance as strategy AUM grows: square-root impact scales with trade size relative to ADV."""
    rows = []
    for aum in aums:
        res = backtest_scores(scores, data, spec, replace(base, aum=aum)).window(start, end)
        s = M.perf_stats(res.net)
        rows.append({"aum": aum, "sharpe": s["sharpe"], "ann_return": s["ann_return"],
                     "cost_bps_day": float((res.cost + res.borrow).mean() * 1e4)})
    return pd.DataFrame(rows)


def yearly_table(net: pd.Series, bench: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Calendar-year return, Sharpe and max drawdown of ``net``, with benchmark returns alongside."""
    g = net.groupby(net.index.year)
    out = pd.DataFrame({
        "return": g.apply(lambda r: float(np.prod(1 + r) - 1)),
        "sharpe": g.apply(lambda r: M.sharpe(r.values)),
        "max_dd": g.apply(lambda r: M.max_drawdown(r.values)),
    })
    for name, b in (bench or {}).items():
        out[f"{name}_return"] = b.groupby(b.index.year).apply(lambda r: float(np.prod(1 + r.fillna(0)) - 1))
    return out


def ic_by_year(scores, data: ResearchData, horizon: int, start, end) -> pd.Series:
    """Mean daily rank IC of ``scores`` against the ``horizon``-day forward return, per calendar year."""
    m = (data.dates >= pd.Timestamp(start)) & (data.dates <= pd.Timestamp(end))
    ic = M.daily_rank_ic(scores[m], np.where(data.tradable, data.fwd(horizon), np.nan)[m])
    return pd.Series(ic, index=data.dates[m]).groupby(data.dates[m].year).mean()


def signal_decay(scores, data: ResearchData, start, end, horizons=(1, 2, 3, 5, 10, 21)) -> pd.Series:
    """Mean rank IC of the score against the *k*-th day's return after execution (not cumulative)."""
    m = (data.dates >= pd.Timestamp(start)) & (data.dates <= pd.Timestamp(end))
    out = {}
    R = data.R
    for k in horizons:
        # return over [open t+k, open t+k+1]: day k of the holding period
        fut = np.full_like(R, np.nan)
        fut[:-k] = R[k:]
        ic = M.daily_rank_ic(scores[m], np.where(data.tradable, fut, np.nan)[m])
        out[k] = float(np.nanmean(ic))
    return pd.Series(out, name="ic_day_k")


FEATURE_FAMILIES = {
    "short-term reversal": ["ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_lag1", "ret_lag2", "ret_lag3",
                            "ret_lag4", "overnight_1", "intraday_1", "overnight_5", "intraday_5", "dist_ma10",
                            "rsi_14", "bollinger_z", "close_loc_1", "close_loc_5", "resid_ret_5"],
    "momentum / trend": ["ret_21", "ret_63", "ret_126", "ret_252", "mom_12_1", "mom_6_1", "dist_ma50",
                         "dist_ma200", "high52_prox", "low52_prox", "resid_ret_21"],
    "volatility / risk": ["vol_5", "vol_21", "vol_63", "vol_ratio_5_63", "hl_range_1", "parkinson_21",
                          "max_ret_21", "min_ret_21", "downside_vol_63", "skew_63", "beta_252", "idio_vol_63"],
    "liquidity / volume": ["log_dv_21", "volume_shock_5_63", "volume_shock_1_21", "amihud_21"],
    "market regime": ["mkt_ret_1", "mkt_ret_5", "mkt_ret_21", "mkt_vol_21", "mkt_vol_ratio",
                      "cs_dispersion_1", "cs_dispersion_21"],
    "statistical industries": ["ind_ret_1", "ind_rel_ret_1", "ind_ret_5", "ind_rel_ret_5", "ind_ret_21",
                               "ind_rel_ret_21", "ind_ret_63", "ind_rel_ret_63", "ind_mom_6_1"],
}


def grouped_permutation_importance(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray, names: list[str],
                                   n_repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    """Drop in mean daily IC when a whole feature family is shuffled across stocks *within each date*.

    Shuffling within dates (not across the whole sample) keeps market-level structure intact and
    asks the cross-sectional question the model is used for. Market-level features are identical
    for every stock on a date, so for them whole dates are swapped instead (each date receives
    another random date's market state).
    """
    from .models.base import grouped_ic
    rng = np.random.default_rng(seed)
    base = grouped_ic(model.predict(X), y, groups)
    order = np.argsort(groups, kind="stable")
    bounds = np.flatnonzero(np.diff(groups[order])) + 1
    blocks = np.split(order, bounds)
    rows = []
    for fam, feats in FEATURE_FAMILIES.items():
        cols = [names.index(f) for f in feats if f in names]
        if not cols:
            continue
        drops = []
        market_level = all(np.ptp(X[blocks[0]][:, c]) == 0 for c in cols)
        for _ in range(n_repeats):
            Xp = X.copy()
            if market_level:
                donors = rng.permutation(len(blocks))
                for b, dnr in zip(blocks, donors):
                    Xp[np.ix_(b, cols)] = X[blocks[dnr][0], cols][None, :]
            else:
                for b in blocks:
                    perm = rng.permutation(b)
                    Xp[np.ix_(b, cols)] = X[np.ix_(perm, cols)]
            drops.append(base - grouped_ic(model.predict(Xp), y, groups))
        rows.append({"family": fam, "n_features": len(cols), "ic_drop": float(np.mean(drops)),
                     "ic_drop_std": float(np.std(drops)), "base_ic": base})
    return pd.DataFrame(rows).sort_values("ic_drop", ascending=False)
