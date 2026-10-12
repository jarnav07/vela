"""Rank-average ensembles of model scores.

Models are combined on cross-sectional *ranks*, not raw outputs, because their scales
differ (GBM leaf values, MLP outputs, ridge projections) and only the ordering
matters for portfolio construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_average(score_list: list[np.ndarray], tradable: np.ndarray, weights: list[float] | None = None) -> np.ndarray:
    """Weighted average of per-date cross-sectional ranks (centred on 0); NaN where no model scores a name."""
    weights = weights or [1.0] * len(score_list)
    total = None
    wsum = None
    for s, w in zip(score_list, weights):
        r = pd.DataFrame(np.where(tradable, s, np.nan)).rank(axis=1, pct=True).values - 0.5
        contrib = np.nan_to_num(r) * w
        have = np.isfinite(r) * w
        total = contrib if total is None else total + contrib
        wsum = have if wsum is None else wsum + have
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(wsum > 0, total / wsum, np.nan)
    return out
