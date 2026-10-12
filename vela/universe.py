"""Tradable-universe definition (point in time).

A stock is tradable at the close of day *t* if, using only information up to *t*:

* it is an S&P 500 constituent on day *t* (point-in-time membership),
* it has a valid open and close on day *t*,
* its 21-day median dollar volume is at least ``min_dollar_volume``
  (removes ticker-reuse junk series that trade a few hundred shares a day), and
* it has at least ``min_history`` days of price history (features need look-back).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.panel import Panel


def dollar_volume(panel: Panel) -> np.ndarray:
    """Close price times volume (true dollars traded, even with rebased prices)."""
    return panel.close * panel.volume


def rolling_median_dollar_volume(panel: Panel, window: int = 21) -> np.ndarray:
    """Trailing median dollar volume including day *t* (known at the close)."""
    dv = pd.DataFrame(dollar_volume(panel))
    return dv.rolling(window, min_periods=max(5, window * 2 // 3)).median().values


def tradable_mask(
    panel: Panel,
    min_dollar_volume: float = 5e6,
    min_history: int = 126,
    window: int = 21,
) -> np.ndarray:
    """Boolean (T x N) tradable universe per the rules in the module docstring."""
    med = rolling_median_dollar_volume(panel, window)
    history = np.cumsum(np.isfinite(panel.close), axis=0)
    with np.errstate(invalid="ignore"):
        ok = (
            panel.member
            & np.isfinite(panel.close)
            & np.isfinite(panel.open)
            & (np.nan_to_num(med) >= min_dollar_volume)
            & (history >= min_history)
        )
    return ok
