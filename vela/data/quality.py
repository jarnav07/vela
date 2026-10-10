"""Data-quality cleaning for the Yahoo-sourced Qlib panel.

Problems found when the raw panel was inspected (see docs/DEVLOG.md, "Data quality"):

* Ticker reuse and broken split/dividend adjustments produce series that jump
  between two price scales (CBE, TIE, HNZ: -98% then +4000% on alternate days)
  or compound fake quarterly jumps (CHK). These are not tradable prices.
* A handful of isolated one-day jumps far beyond anything a real S&P 500 stock has
  done (BMC +347% that never reverses: an adjustment level shift).
* Some bars have open/close outside the reported high/low.

Policy (deliberately conservative: we must not delete *real* crashes, which would
flatter long-only results):

1. OHLC consistency: ``high = max(o,h,l,c)``, ``low = min(o,h,l,c)``.
2. A *glitch* is a close-to-close or open-to-open move beyond ``x3`` or ``/3``
   (|log return| > log 3: +200% / -66.7%). No genuine S&P 500 constituent day in
   this sample needs that range once corrupt tickers are removed.
3. Tickers with ``>= max_glitches`` glitch days *while they are index members* are
   systematically corrupt and are dropped from the universe entirely. Glitches outside
   the membership window do not matter (the stock is not tradable then).
4. A single glitch-sized move inside membership is **kept**: on inspection these are
   real events (AIG on 2008-09-16, open-to-open -67% during the bailout). Neutralising
   them would delete genuine crash returns. ``panel.bad`` records every glitch day for
   diagnostics only; it is not used to alter returns.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .panel import Panel


def _log_ret(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = np.log(x[1:] / x[:-1])
    return out


def glitch_mask(panel: Panel, threshold: float = np.log(3.0)) -> np.ndarray:
    """True where the close-to-close or open-to-open log move exceeds ``threshold``."""
    lc = _log_ret(panel.close)
    lo = _log_ret(panel.open)
    with np.errstate(invalid="ignore"):
        return (np.abs(np.nan_to_num(lc)) > threshold) | (np.abs(np.nan_to_num(lo)) > threshold)


def clean_panel(panel: Panel, threshold: float = np.log(3.0), max_glitches: int = 2) -> tuple[Panel, dict]:
    o, h, l, c = (panel.open.copy(), panel.high.copy(), panel.low.copy(), panel.close.copy())
    v = panel.volume.copy()

    # non-positive prices / volumes are missing data
    for arr in (o, h, l, c):
        arr[~(arr > 0)] = np.nan
    v[~(v >= 0)] = np.nan
    # if close exists but open is missing, fall back to close (keeps open-to-open returns defined)
    o = np.where(np.isfinite(o), o, c)

    stack = np.stack([o, h, l, c])
    with warnings.catch_warnings():  # all-NaN columns are expected before listing / after delisting
        warnings.simplefilter("ignore", RuntimeWarning)
        smax, smin = np.nanmax(stack, 0), np.nanmin(stack, 0)
    n_ohlc_fixed = int(np.nansum((h < smax * (1 - 1e-6)) | (l > smin * (1 + 1e-6))))
    with np.errstate(invalid="ignore"):
        h = np.fmax(h, smax)
        l = np.fmin(l, smin)

    cleaned = Panel(
        dates=panel.dates, tickers=panel.tickers, open=o, high=h, low=l, close=c, volume=v,
        member=panel.member.copy(), benchmarks=panel.benchmarks, benchmark_opens=panel.benchmark_opens,
        meta=dict(panel.meta),
    )
    bad = glitch_mask(cleaned, threshold)
    counts = pd.Series((bad & panel.member).sum(0), index=panel.tickers)
    dropped = sorted(counts[counts >= max_glitches].index)
    drop_idx = panel.tickers.get_indexer(dropped)
    cleaned.member[:, drop_idx] = False

    isolated = [
        (str(panel.dates[t].date()), str(panel.tickers[j]))
        for t, j in np.argwhere(bad & cleaned.member)
    ]
    cleaned.meta.update(
        {
            "dropped_corrupt_tickers": dropped,
            "single_glitch_events_kept": isolated,
            "glitch_threshold_log": float(threshold),
        }
    )
    cleaned.bad = bad
    report = {
        "ohlc_bars_fixed": n_ohlc_fixed,
        "dropped_corrupt_tickers": dropped,
        "glitch_counts_of_dropped": {t: int(counts[t]) for t in dropped},
        "single_glitch_events_kept": isolated,
    }
    return cleaned, report
