"""Execution-aligned returns and prediction targets.

A signal formed at the close of day *t* is executed at the open of *t+1*, so the
return it can capture over an *h*-day holding period is::

    fwd_h[t] = open[t+1+h] / open[t+1] - 1

This label is only known at the open of day ``t+1+h``; the walk-forward splitter
purges training rows whose labels would not yet be known (vela/walkforward.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.panel import Panel


def open_to_open_returns(panel: Panel) -> np.ndarray:
    """R[s] = open[s+1] / open[s] - 1: the engine's execution-price return for bar s."""
    o = panel.open
    R = np.full_like(o, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        R[:-1] = o[1:] / o[:-1] - 1.0
    return R


def forward_returns(panel: Panel, horizon: int) -> np.ndarray:
    o = panel.open
    y = np.full_like(o, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        y[: -(horizon + 1)] = o[horizon + 1:] / o[1:-horizon] - 1.0
    return y


def make_target(fwd: np.ndarray, tradable: np.ndarray, kind: str = "rank",
                vol: np.ndarray | None = None) -> np.ndarray:
    """Cross-sectional target transforms (computed only over the tradable universe).

    rank       : percentile rank in [-0.5, 0.5]: robust to outliers, focuses on ordering
    demeaned   : return minus the cross-sectional mean, winsorised at the 1st/99th pct
    vol_scaled : demeaned return divided by trailing 21-day volatility, then ranked
    """
    y = np.where(tradable, fwd, np.nan)
    df = pd.DataFrame(y)
    if kind == "rank":
        r = df.rank(axis=1, pct=True)
        n = df.notna().sum(axis=1)
        return ((r.mul(n, axis=0) - 1).div((n - 1).clip(lower=1), axis=0) - 0.5).values
    if kind == "demeaned":
        lo = df.quantile(0.01, axis=1)
        hi = df.quantile(0.99, axis=1)
        clipped = df.clip(lower=lo, upper=hi, axis=0)
        return clipped.sub(clipped.mean(axis=1), axis=0).values
    if kind == "vol_scaled":
        if vol is None:
            raise ValueError("vol_scaled target needs vol")
        z = df.sub(df.mean(axis=1), axis=0) / pd.DataFrame(vol)
        return make_target(z.values, tradable & np.isfinite(z.values), "rank")
    raise ValueError(kind)
