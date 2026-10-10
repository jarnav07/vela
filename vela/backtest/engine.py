"""Vectorised NumPy backtesting engine.

Timing convention
-----------------
* ``weights[t]`` is the target portfolio decided at the **close of day t**, using
  only information available at that close. Weights are fractions of NAV; long
  positive, short negative.
* It is executed ``lag`` bars later at the execution price (default: next day's
  open), and held until the next execution.
* ``returns[s]`` is the return of the execution price from bar *s* to bar *s+1*
  (for open execution: ``open[s+1] / open[s] - 1``).

So the weights held over interval *s* are ``H[s] = weights[s - lag]`` and the
gross portfolio return for that interval is ``sum_i H[s, i] * returns[s, i]``.
Results are labelled by the *start* of the holding interval.

Trading costs
-------------
Between executions positions drift with prices, so the weights just before the
trade at bar *s* are::

    D[s] = H[s-1] * (1 + r[s-1]) / (1 + R_p[s-1])

and the traded notional (as a fraction of NAV) is ``|H[s] - D[s]|``. This charges
for passive-drift rebalancing as well as signal changes. Every step is a whole-array
operation, with no Python loop over dates or assets.

Per-trade cost rate = commission + half-spread + square-root market impact::

    impact_i = Y * sigma_i * sqrt(|trade_i| * AUM / ADV_i)

with ``sigma`` the daily volatility and ``ADV`` the average daily dollar volume,
both known at the signal date. Shorts pay an annual borrow fee on their notional.

Missing returns (no price on the next bar, e.g. after a delisting) count as 0;
the strategy layer forces such positions to zero, and the exit is costed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.5
    half_spread_bps: float = 5.0
    impact_coef: float = 1.0          # Y in the square-root law; 0 disables impact
    aum: float = 10e6                 # strategy capital in dollars (drives impact)
    borrow_bps_annual: float = 50.0   # stock-loan fee on short notional

    @classmethod
    def zero(cls) -> "CostModel":
        return cls(commission_bps=0.0, half_spread_bps=0.0, impact_coef=0.0, borrow_bps_annual=0.0)

    def scaled(self, k: float) -> "CostModel":
        """Multiply every cost component by ``k`` (for sensitivity analysis)."""
        return CostModel(
            commission_bps=self.commission_bps * k,
            half_spread_bps=self.half_spread_bps * k,
            impact_coef=self.impact_coef * k,
            aum=self.aum,
            borrow_bps_annual=self.borrow_bps_annual * k,
        )


@dataclass
class BacktestResult:
    dates: pd.DatetimeIndex
    gross: np.ndarray
    cost: np.ndarray          # trading costs (commission + spread + impact), fraction of NAV
    borrow: np.ndarray        # borrow fees, fraction of NAV
    turnover: np.ndarray      # one-way traded notional / NAV
    long_exposure: np.ndarray
    short_exposure: np.ndarray
    n_long: np.ndarray
    n_short: np.ndarray
    held: np.ndarray | None = field(default=None, repr=False)

    @property
    def net(self) -> np.ndarray:
        return self.gross - self.cost - self.borrow

    def series(self, which: str = "net") -> pd.Series:
        return pd.Series(getattr(self, which), index=self.dates, name=which)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "gross": self.gross,
                "net": self.net,
                "cost": self.cost,
                "borrow": self.borrow,
                "turnover": self.turnover,
                "long_exposure": self.long_exposure,
                "short_exposure": self.short_exposure,
                "n_long": self.n_long,
                "n_short": self.n_short,
            },
            index=self.dates,
        )

    def window(self, start=None, end=None) -> "BacktestResult":
        m = np.ones(len(self.dates), dtype=bool)
        if start is not None:
            m &= self.dates >= pd.Timestamp(start)
        if end is not None:
            m &= self.dates <= pd.Timestamp(end)
        return BacktestResult(
            dates=self.dates[m],
            **{k: getattr(self, k)[m] for k in (
                "gross", "cost", "borrow", "turnover", "long_exposure", "short_exposure", "n_long", "n_short")},
            held=None if self.held is None else self.held[m],
        )


def _shift(a: np.ndarray, lag: int, fill: float = 0.0) -> np.ndarray:
    if lag == 0:
        return a.copy()
    out = np.full_like(a, fill, dtype=np.float64)
    out[lag:] = a[:-lag]
    return out


def buy_and_hold_weights(targets: np.ndarray, R: np.ndarray, rebalance: np.ndarray, lag: int) -> np.ndarray:
    """Held weights when targets are traded only on rebalance bars and *shares* are held in between.

    Within a segment that starts with an execution at bar s0 with target weights w,
    position values grow with each asset's cumulative return, so (relative to NAV at s0)::

        P[s] = w * Gc[s] / Gc[s0],   V[s] = 1 + sum(w * (Gc[s] / Gc[s0] - 1)),   H[s] = P[s] / V[s]

    where ``Gc[s] = prod_{m<s} (1 + R[m])``. Fully vectorised: segment starts are found with a
    running maximum, cumulative growth with a cumulative sum of log returns.
    """
    T, N = targets.shape
    exec_bar = np.zeros(T, dtype=bool)
    exec_bar[lag:] = np.asarray(rebalance, dtype=bool)[:-lag]
    tgt = _shift(targets, lag)
    seg = np.maximum.accumulate(np.where(exec_bar, np.arange(T), -1))
    log_g = np.zeros((T, N))
    log_g[1:] = np.cumsum(np.log1p(np.clip(R[:-1], -0.9999, None)), axis=0)
    valid = seg >= 0
    s0 = np.where(valid, seg, 0)
    w = np.where(valid[:, None], tgt[s0], 0.0)
    ratio = np.exp(log_g - log_g[s0])
    V = 1.0 + (w * (ratio - 1.0)).sum(1)
    return w * ratio / np.maximum(V, 1e-9)[:, None]


def run_backtest(
    weights: np.ndarray,
    returns: np.ndarray,
    dates: pd.DatetimeIndex,
    costs: CostModel = CostModel(),
    lag: int = 1,
    sigma: np.ndarray | None = None,
    adv: np.ndarray | None = None,
    keep_holdings: bool = False,
    rebalance: np.ndarray | None = None,
) -> BacktestResult:
    """Run a vectorised backtest.

    Parameters
    ----------
    weights : (T, N) target weights decided at the close of each bar (NaN = 0).
    returns : (T, N) execution-price returns from bar s to s+1 (NaN = 0).
    sigma, adv : (T, N) daily volatility and dollar ADV known at each signal date;
        required only when ``costs.impact_coef > 0``.
    rebalance : optional (T,) bool. If given, targets are traded only on bars where it is
        True and shares are held (weights drift) in between. If None, the book is
        rebalanced back to ``weights`` every bar.
    """
    W = np.nan_to_num(np.asarray(weights, dtype=np.float64))
    R = np.nan_to_num(np.asarray(returns, dtype=np.float64))
    if W.shape != R.shape:
        raise ValueError(f"weights {W.shape} and returns {R.shape} differ")
    if lag < 1:
        raise ValueError("lag must be >= 1: a signal formed at the close cannot trade at that close")

    if rebalance is None:
        H = _shift(W, lag)                   # weights held over interval s
    else:
        H = buy_and_hold_weights(W, R, rebalance, lag)
    gross = np.einsum("ij,ij->i", H, R)      # row-wise dot product

    D = np.zeros_like(H)                     # pre-trade (drifted) weights at bar s
    D[1:] = H[:-1] * (1.0 + R[:-1]) / (1.0 + gross[:-1])[:, None]
    trades = np.abs(H - D)
    trades[trades < 1e-12] = 0.0             # floating-point dust between rebalances
    turnover = trades.sum(1)

    rate = (costs.commission_bps + costs.half_spread_bps) / 1e4
    cost = rate * turnover
    if costs.impact_coef > 0:
        if sigma is None or adv is None:
            raise ValueError("impact_coef > 0 requires sigma and adv")
        sig = _shift(np.nan_to_num(sigma, nan=0.02), lag)  # 2% daily vol if unknown
        dv = _shift(np.nan_to_num(adv, nan=0.0), lag)
        with np.errstate(divide="ignore", invalid="ignore"):
            participation = np.where(dv > 0, trades * costs.aum / dv, 0.0)
        impact = costs.impact_coef * sig * np.sqrt(participation)
        cost = cost + np.einsum("ij,ij->i", trades, impact)

    short_notional = np.clip(-H, 0, None).sum(1)
    borrow = short_notional * costs.borrow_bps_annual / 1e4 / TRADING_DAYS

    return BacktestResult(
        dates=pd.DatetimeIndex(dates),
        gross=gross,
        cost=cost,
        borrow=borrow,
        turnover=turnover,
        long_exposure=np.clip(H, 0, None).sum(1),
        short_exposure=short_notional,
        n_long=(H > 0).sum(1),
        n_short=(H < 0).sum(1),
        held=H if keep_holdings else None,
    )
