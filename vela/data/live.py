"""Live data adapter: build a :class:`Panel` from Yahoo Finance via ``yfinance``.

Used for running the frozen model on current data (``scripts/live_signals.py``).
The research sandbox could not reach Yahoo, so this path is covered by a test that
mocks ``yfinance.download``'s output format, not by a live download.

Current S&P 500 constituents come from the public ``datasets/s-and-p-500-companies``
CSV on GitHub. For *live* trading, today's constituents are the right universe (no
survivorship issue arises when trading forward).
"""
from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd

from .panel import Panel

CONSTITUENTS_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"


def current_sp500_tickers() -> list[str]:
    """Today's S&P 500 symbols (Yahoo format, e.g. ``BRK-B``) from the public constituents CSV."""
    with urllib.request.urlopen(CONSTITUENTS_URL, timeout=30) as resp:
        df = pd.read_csv(io.StringIO(resp.read().decode()))
    return sorted(df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist())


def panel_from_yahoo_frame(raw: pd.DataFrame, tickers: list[str], benchmark: str = "SPY") -> Panel:
    """Convert ``yf.download(..., auto_adjust=True, group_by='column')`` output into a Panel."""
    raw = raw.sort_index()
    dates = pd.DatetimeIndex(raw.index.tz_localize(None) if raw.index.tz is not None else raw.index, name="date")
    names = [t for t in tickers if t in raw["Close"].columns]
    get = lambda f: raw[f].reindex(columns=names).to_numpy(dtype=float)  # noqa: E731
    close = get("Close")
    bench = raw["Close"][benchmark] if benchmark in raw["Close"].columns else pd.Series(np.nan, index=raw.index)
    bopen = raw["Open"][benchmark] if benchmark in raw["Open"].columns else pd.Series(np.nan, index=raw.index)
    return Panel(
        dates=dates, tickers=pd.Index(names, name="ticker"),
        open=get("Open"), high=get("High"), low=get("Low"), close=close, volume=get("Volume"),
        member=np.isfinite(close),
        benchmarks=pd.DataFrame({benchmark: bench.values}, index=dates),
        benchmark_opens=pd.DataFrame({benchmark: bopen.values}, index=dates),
        meta={"source": "yfinance"},
    )


def fetch_panel(tickers: list[str], start: str, end: str | None = None, benchmark: str = "SPY") -> Panel:
    """Download daily OHLCV from Yahoo Finance for ``tickers`` (+ benchmark) and build a Panel."""
    import yfinance as yf  # optional dependency: pip install -e ".[live]"

    raw = yf.download(sorted(set(tickers) | {benchmark}), start=start, end=end, auto_adjust=True,
                      group_by="column", progress=False, threads=True)
    return panel_from_yahoo_frame(raw, tickers, benchmark)
