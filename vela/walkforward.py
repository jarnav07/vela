"""Annual walk-forward splits with label purging.

For test year *Y*:

* **test**  = trading days in *Y*;
* **train** = days from ``train_start`` up to the first test day, *minus* a purge gap
  of ``horizon + 2`` days. A label formed at the close of day *t* is realised at the
  open of ``t + 1 + horizon``, so without the purge the last training labels would
  overlap the test period (López de Prado 2018, ch. 7);
* **val**   = the final ``val_days`` of the training window (after its own purge gap),
  used only for early stopping and model selection *inside* the fold.

``train_years=None`` gives an expanding window; an integer gives a rolling window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    test_year: int
    train_idx: np.ndarray   # date indices
    val_idx: np.ndarray
    test_idx: np.ndarray

    def describe(self, dates: pd.DatetimeIndex) -> str:
        def span(ix):
            return f"{dates[ix[0]].date()}..{dates[ix[-1]].date()}" if len(ix) else "-"
        return f"{self.test_year}: train {span(self.train_idx)} | val {span(self.val_idx)} | test {span(self.test_idx)}"


def annual_folds(
    dates: pd.DatetimeIndex,
    first_test_year: int,
    last_test_year: int,
    horizon: int,
    train_start: str = "2001-01-01",
    train_years: int | None = None,
    val_days: int = 252,
    last_test_date: str | None = None,
) -> list[Fold]:
    purge = horizon + 2
    folds = []
    years = dates.year
    for Y in range(first_test_year, last_test_year + 1):
        test = np.flatnonzero(years == Y)
        if last_test_date is not None:
            test = test[dates[test] <= pd.Timestamp(last_test_date)]
        if len(test) == 0:
            continue
        t0 = test[0]
        start = pd.Timestamp(train_start)
        if train_years is not None:
            start = max(start, pd.Timestamp(f"{Y - train_years}-01-01"))
        lo = int(np.searchsorted(dates.values, np.datetime64(start)))
        hi = t0 - purge                       # exclusive
        window = np.arange(lo, max(hi, lo))
        if val_days and len(window) > val_days + purge + 252:
            val = window[-val_days:]
            train = window[: -(val_days + purge)]
        else:
            val = np.array([], dtype=int)
            train = window
        folds.append(Fold(Y, train, val, test))
    return folds
