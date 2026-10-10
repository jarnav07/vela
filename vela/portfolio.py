"""Portfolio construction: daily model scores -> target weights.

Every function returns a (T x N) weight matrix in which row *t* uses only the scores
and tradability known at the close of day *t*. The engine handles the execution lag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _row_rank(scores: np.ndarray, tradable: np.ndarray) -> np.ndarray:
    """Percentile rank in (0, 1] among tradable names with a finite score; NaN elsewhere."""
    s = pd.DataFrame(np.where(tradable, scores, np.nan))
    return s.rank(axis=1, pct=True, method="first").values


def smooth_scores(scores: np.ndarray, tradable: np.ndarray, halflife: float | None) -> np.ndarray:
    """EMA of daily cross-sectional ranks: trades less by not reacting fully to daily noise."""
    ranks = _row_rank(scores, tradable)
    if not halflife:
        return ranks
    sm = pd.DataFrame(ranks).ewm(halflife=halflife, ignore_na=True).mean().values
    return np.where(tradable & np.isfinite(ranks), sm, np.nan)


def _leg_weights(mask: np.ndarray, gross: float, inv_vol: np.ndarray | None) -> np.ndarray:
    if inv_vol is None:
        base = mask.astype(np.float64)
    else:
        base = np.where(mask, np.nan_to_num(inv_vol), 0.0)
    tot = base.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, base / tot * gross, 0.0)


def long_short_quantile(scores: np.ndarray, tradable: np.ndarray, q: float = 0.1,
                        long_gross: float = 1.0, short_gross: float = 1.0,
                        vol: np.ndarray | None = None, min_names: int = 50) -> np.ndarray:
    """Dollar-neutral: long the top ``q`` fraction, short the bottom ``q`` fraction.

    Equal-weighted within each leg, or inverse-volatility weighted if ``vol`` is given.
    """
    r = _row_rank(scores, tradable)
    n = np.isfinite(r).sum(1, keepdims=True)
    longs = r > 1 - q
    shorts = r <= q
    inv_vol = None if vol is None else 1.0 / np.where(vol > 0, vol, np.nan)
    W = _leg_weights(longs, long_gross, inv_vol) - _leg_weights(shorts, short_gross, inv_vol)
    W[(n < min_names).ravel()] = 0.0
    return W


def long_only_top(scores: np.ndarray, tradable: np.ndarray, q: float = 0.2,
                  vol: np.ndarray | None = None, min_names: int = 50) -> np.ndarray:
    r = _row_rank(scores, tradable)
    n = np.isfinite(r).sum(1, keepdims=True)
    inv_vol = None if vol is None else 1.0 / np.where(vol > 0, vol, np.nan)
    W = _leg_weights(r > 1 - q, 1.0, inv_vol)
    W[(n < min_names).ravel()] = 0.0
    return W


def equal_weight_universe(tradable: np.ndarray) -> np.ndarray:
    return _leg_weights(tradable, 1.0, None)


def rebalance_mask(T: int, k: int, offset: int = 0) -> np.ndarray:
    """True on every ``k``-th bar: the days on which a new target is traded.

    The engine holds *shares* between these days (weights drift with prices).
    """
    return (np.arange(T) - offset) % max(k, 1) == 0


def rebalance_every(W: np.ndarray, tradable: np.ndarray, k: int, offset: int = 0) -> np.ndarray:
    """Constant-mix alternative: hold the previous *target weights* between rebalance days.

    Superseded by ``rebalance_mask`` + buy-and-hold in the engine (see DEVLOG, Phase 4),
    which is what a real account does; kept for comparison.
    """
    if k <= 1:
        return W
    T = W.shape[0]
    is_reb = (np.arange(T) - offset) % k == 0
    held = pd.DataFrame(np.where(is_reb[:, None], W, np.nan)).ffill().fillna(0.0).values
    return np.where(tradable, held, 0.0)


def hysteresis_long_short(scores: np.ndarray, tradable: np.ndarray, enter_q: float = 0.1, exit_q: float = 0.2,
                          long_gross: float = 1.0, short_gross: float = 1.0, min_names: int = 50) -> np.ndarray:
    """Buffered long-short book: enter at the top/bottom ``enter_q``, exit only beyond ``exit_q``.

    Holding a position until its rank deteriorates past a wider band cuts turnover
    a lot for a small loss of signal. This is a state machine over time (one loop over
    dates, vectorised across stocks).
    """
    r = _row_rank(scores, tradable)
    T, N = r.shape
    long_pos = np.zeros(N, dtype=bool)
    short_pos = np.zeros(N, dtype=bool)
    W = np.zeros((T, N))
    for t in range(T):
        rt = r[t]
        ok = np.isfinite(rt)
        if ok.sum() < min_names:
            long_pos[:] = False
            short_pos[:] = False
            continue
        long_pos = ok & ((rt > 1 - enter_q) | (long_pos & (rt > 1 - exit_q)))
        short_pos = ok & ((rt <= enter_q) | (short_pos & (rt <= exit_q)))
        nl, ns = long_pos.sum(), short_pos.sum()
        if nl:
            W[t, long_pos] = long_gross / nl
        if ns:
            W[t, short_pos] = -short_gross / ns
    return W


def hysteresis_long_only(scores: np.ndarray, tradable: np.ndarray, enter_q: float = 0.2, exit_q: float = 0.4,
                         min_names: int = 50) -> np.ndarray:
    r = _row_rank(scores, tradable)
    T, N = r.shape
    pos = np.zeros(N, dtype=bool)
    W = np.zeros((T, N))
    for t in range(T):
        rt = r[t]
        ok = np.isfinite(rt)
        if ok.sum() < min_names:
            pos[:] = False
            continue
        pos = ok & ((rt > 1 - enter_q) | (pos & (rt > 1 - exit_q)))
        if pos.any():
            W[t, pos] = 1.0 / pos.sum()
    return W
