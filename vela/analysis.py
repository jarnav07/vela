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
    rows = []
    for aum in aums:
        res = backtest_scores(scores, data, spec, replace(base, aum=aum)).window(start, end)
        s = M.perf_stats(res.net)
        rows.append({"aum": aum, "sharpe": s["sharpe"], "ann_return": s["ann_return"],
                     "cost_bps_day": float((res.cost + res.borrow).mean() * 1e4)})
    return pd.DataFrame(rows)


def yearly_table(net: pd.Series, bench: dict[str, pd.Series] | None = None) -> pd.DataFrame:
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
