"""Aligned (dates x tickers) market-data panel with point-in-time index membership.

Everything downstream (features, targets, backtests) works on dense ``T x N``
NumPy arrays that share one date index and one ticker index. Missing data is NaN.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .qlib_source import FIELDS, QlibSource

BENCHMARKS = ("SPY", "^GSPC")


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    tickers: pd.Index
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    member: np.ndarray  # bool: ticker was an index constituent on that date
    bad: np.ndarray | None = None  # bool: return on this day is a data glitch (see quality.py)
    benchmarks: pd.DataFrame = field(default_factory=pd.DataFrame)  # adjusted closes of SPY, ^GSPC
    benchmark_opens: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.close.shape

    def frame(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(getattr(self, name), index=self.dates, columns=self.tickers)

    def slice_dates(self, start=None, end=None) -> "Panel":
        mask = np.ones(len(self.dates), dtype=bool)
        if start is not None:
            mask &= self.dates >= pd.Timestamp(start)
        if end is not None:
            mask &= self.dates <= pd.Timestamp(end)
        return Panel(
            dates=self.dates[mask],
            tickers=self.tickers,
            **{f: getattr(self, f)[mask] for f in (*FIELDS, "member")},
            bad=None if self.bad is None else self.bad[mask],
            benchmarks=self.benchmarks.loc[self.dates[mask]],
            benchmark_opens=self.benchmark_opens.loc[self.dates[mask]],
            meta=dict(self.meta),
        )

    # ---------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            dates=self.dates.values.astype("datetime64[ns]").astype(np.int64),
            tickers=np.asarray(self.tickers, dtype=str),
            **{f: getattr(self, f).astype(np.float32) for f in FIELDS},
            member=self.member,
            bad=self.bad if self.bad is not None else np.zeros_like(self.member),
            bench_close=self.benchmarks.values.astype(np.float64),
            bench_open=self.benchmark_opens.values.astype(np.float64),
            bench_names=np.asarray(self.benchmarks.columns, dtype=str),
            meta=np.asarray(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Panel":
        z = np.load(path, allow_pickle=False)
        dates = pd.DatetimeIndex(z["dates"].astype("datetime64[ns]"), name="date")
        names = list(z["bench_names"])
        return cls(
            dates=dates,
            tickers=pd.Index(z["tickers"], name="ticker"),
            **{f: z[f].astype(np.float64) for f in FIELDS},
            member=z["member"].astype(bool),
            bad=z["bad"].astype(bool) if "bad" in z.files else None,
            benchmarks=pd.DataFrame(z["bench_close"], index=dates, columns=names),
            benchmark_opens=pd.DataFrame(z["bench_open"], index=dates, columns=names),
            meta=json.loads(str(z["meta"])),
        )


def membership_matrix(intervals: pd.DataFrame, dates: pd.DatetimeIndex, tickers: pd.Index) -> np.ndarray:
    """Boolean T x N matrix: True where the ticker is inside any of its membership intervals."""
    member = np.zeros((len(dates), len(tickers)), dtype=bool)
    col = {t: i for i, t in enumerate(tickers)}
    d = dates.values
    for row in intervals.itertuples(index=False):
        j = col.get(row.symbol)
        if j is None:
            continue
        member[:, j] |= (d >= np.datetime64(row.start)) & (d <= np.datetime64(row.end))
    return member


def build_panel(source: QlibSource, market: str = "sp500") -> Panel:
    """Load every symbol that was ever a member of ``market`` into an aligned panel."""
    cal = source.calendar()
    intervals = source.membership(market)
    symbols = sorted(intervals["symbol"].unique())
    available = [s for s in symbols if source.has_symbol(s)]
    missing = sorted(set(symbols) - set(available))

    arrays = {f: np.full((len(cal), len(available)), np.nan) for f in FIELDS}
    pos = pd.Series(np.arange(len(cal)), index=cal)
    for j, sym in enumerate(available):
        df = source.read_symbol(sym, cal)
        if df is None:
            continue
        rows = pos.loc[df.index].values
        for f in FIELDS:
            arrays[f][rows, j] = df[f].values

    tickers = pd.Index(available, name="ticker")
    member = membership_matrix(intervals, cal, tickers)

    bench_close, bench_open = {}, {}
    for b in BENCHMARKS:
        df = source.read_symbol(b, cal)
        if df is not None:
            bench_close[b] = df["close"].reindex(cal)
            bench_open[b] = df["open"].reindex(cal)

    meta = {
        "source": str(source.root),
        "market": market,
        "n_membership_symbols": len(symbols),
        "n_with_data": len(available),
        "missing_symbols": missing,
    }
    return Panel(
        dates=cal,
        tickers=tickers,
        **arrays,
        member=member,
        benchmarks=pd.DataFrame(bench_close, index=cal),
        benchmark_opens=pd.DataFrame(bench_open, index=cal),
        meta=meta,
    )
