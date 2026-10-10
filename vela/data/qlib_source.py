"""Reader for Microsoft Qlib's binary daily data format.

Layout of a Qlib data directory::

    calendars/day.txt                 one trading date per line
    instruments/<market>.txt          SYMBOL<TAB>start<TAB>end  (membership intervals)
    features/<prefix><symbol>/<field>.day.bin

Each ``.bin`` file is a little-endian float32 array whose first element is the
index of the series' first date in the calendar, followed by one value per
calendar day. The US dataset prefixes symbol folders with ``_`` and lower-cases
them; share classes use ``-`` (``BRK-B``) where the instrument files use ``.``.

Prices in the Qlib US dataset are split/dividend adjusted and rebased so the
first close of each series is 1.0; volume is scaled inversely, so
``close * volume`` is still the dollar volume traded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class QlibSource:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root))
        if not (self.root / "calendars" / "day.txt").exists():
            raise FileNotFoundError(f"No Qlib calendar under {self.root}")

    def calendar(self) -> pd.DatetimeIndex:
        dates = (self.root / "calendars" / "day.txt").read_text().split()
        return pd.DatetimeIndex(pd.to_datetime(dates), name="date")

    @staticmethod
    def folder_name(symbol: str) -> str:
        return "_" + symbol.lower().replace(".", "-")

    def has_symbol(self, symbol: str) -> bool:
        return (self.root / "features" / self.folder_name(symbol)).is_dir()

    def read_field(self, symbol: str, field: str, calendar: pd.DatetimeIndex | None = None) -> pd.Series | None:
        path = self.root / "features" / self.folder_name(symbol) / f"{field}.day.bin"
        if not path.exists():
            return None
        raw = np.fromfile(path, dtype="<f4")
        if raw.size < 2:
            return None
        cal = self.calendar() if calendar is None else calendar
        start = int(raw[0])
        values = raw[1:].astype(np.float64)
        end = min(start + values.size, len(cal))
        return pd.Series(values[: end - start], index=cal[start:end], name=field)

    def read_symbol(self, symbol: str, calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame | None:
        cal = self.calendar() if calendar is None else calendar
        cols = {f: self.read_field(symbol, f, cal) for f in FIELDS}
        if any(v is None for v in cols.values()):
            return None
        return pd.DataFrame(cols)

    def membership(self, market: str = "sp500") -> pd.DataFrame:
        """Membership intervals as a DataFrame[symbol, start, end]. A symbol may appear more than once."""
        df = pd.read_csv(
            self.root / "instruments" / f"{market}.txt",
            sep="\t",
            header=None,
            names=["symbol", "start", "end"],
            usecols=[0, 1, 2],
        )
        df["start"] = pd.to_datetime(df["start"])
        df["end"] = pd.to_datetime(df["end"])
        return df

    def list_symbols(self) -> list[str]:
        return sorted(p[1:].upper() for p in os.listdir(self.root / "features"))
