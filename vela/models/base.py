"""Common model interface and helpers.

Every model maps a (rows x features) matrix to a score; only the cross-sectional
*ordering* of scores matters downstream. ``fit`` receives a time-ordered validation
set (the last year of the training window) for early stopping. It is never drawn
at random, which would leak information across time.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy import stats


class Model(Protocol):
    name: str

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None, groups_val: np.ndarray | None = None) -> "Model": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


def grouped_ic(pred: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Mean per-date Spearman IC (groups = date index of each row). Used for early stopping."""
    order = np.argsort(groups, kind="stable")
    g, p, t = groups[order], pred[order], y[order]
    bounds = np.flatnonzero(np.diff(g)) + 1
    ics = []
    for ps, ts in zip(np.split(p, bounds), np.split(t, bounds)):
        if len(ps) > 10:
            ics.append(stats.spearmanr(ps, ts).statistic)
    return float(np.nanmean(ics)) if ics else np.nan
