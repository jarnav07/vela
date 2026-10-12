"""Event-driven paper-trading simulator.

Replays a strategy day by day the way it would be run on a real account:
on each rebalance day, after the close, compute target weights from the model's
scores; send orders; fill them at the next day's open plus half the spread and a
commission; record every fill in a :class:`Ledger`. The ledger is then valued and
analysed with the same tracker used for a real account.

Because it shares no code with the vectorised engine apart from the target weights,
it doubles as an **independent check of the engine**: the two NAV paths should agree
apart from discrete effects (rounding, minimum order size, per-order fees).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ledger import Ledger
from .orders import generate_orders


@dataclass
class PaperConfig:
    """Account size, execution costs and order rules for a paper-trading replay."""
    initial_capital: float = 10_000.0
    half_spread_bps: float = 5.0
    commission_bps: float = 0.5
    commission_per_order: float = 0.0
    fractional: bool = True
    min_trade_value: float = 0.0


def simulate(
    target_weights: np.ndarray,       # (T, N) decided at each close; only rebalance rows are used
    rebalance: np.ndarray,            # (T,) bool
    opens: np.ndarray,                # (T, N) execution prices
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    start: str,
    end: str,
    cfg: PaperConfig = PaperConfig(),
) -> Ledger:
    """Replay target weights day by day through orders and fills; returns the resulting trade ledger."""
    ledger = Ledger()
    t_idx = np.flatnonzero((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))
    first = t_idx[0]
    ledger.deposit(dates[first], cfg.initial_capital)
    positions = pd.Series(dtype=float)
    cash = cfg.initial_capital
    last_px = pd.Series(np.nan, index=tickers)
    slip = cfg.half_spread_bps / 1e4
    for t in t_idx[:-1]:
        px_open_next = pd.Series(opens[t + 1], index=tickers)
        last_px = px_open_next.fillna(last_px)
        if not rebalance[t]:
            continue
        # value the book at the price we will trade at (next open), as the engine does
        nav = cash + float((positions * last_px.reindex(positions.index)).sum())
        w = pd.Series(target_weights[t], index=tickers)
        w = w[w != 0]
        orders = generate_orders(w, positions, px_open_next, nav, cfg.fractional, cfg.min_trade_value)
        fill_date = dates[t + 1]
        for tic, o in orders.iterrows():
            q = o["order_qty"]
            fill = o["price"] * (1 + slip * np.sign(q))
            fee = abs(q) * fill * cfg.commission_bps / 1e4 + cfg.commission_per_order
            ledger.trade(fill_date, tic, q, fill, fee)
            cash -= q * fill + fee
            positions.loc[tic] = positions.get(tic, 0.0) + q
        positions = positions[positions.abs() > 1e-12]
    return ledger
