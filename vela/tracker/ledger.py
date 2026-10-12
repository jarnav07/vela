"""Portfolio tracker: trade/cash-flow ledger -> NAV, P&L, risk and benchmark-relative metrics.

Designed for a real (or paper) self-directed account:

* load trades and cash flows from CSV (e.g. exported from a broker),
* mark positions to market with a daily price table,
* compute NAV, realised/unrealised P&L (average-cost method), time-weighted
  returns that strip out deposits and withdrawals, drawdowns, and
  Sharpe / Sortino / max-DD / beta / alpha / information ratio against a benchmark.

CSV formats
-----------
trades.csv     : date,ticker,quantity,price,fees      (quantity > 0 buy, < 0 sell)
cashflows.csv  : date,amount                          (deposit > 0, withdrawal < 0)

Conventions: trades and flows are booked at the date given; positions are marked at
that day's close. Deposits are treated as arriving at the start of the day, so
time-weighted return ``r_t = NAV_t / (NAV_{t-1} + flow_t) - 1``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import metrics as M


TRADE_COLS = ["date", "ticker", "quantity", "price", "fees"]
FLOW_COLS = ["date", "amount"]


class Ledger:
    """Trades and external cash flows. New entries are buffered in lists and turned into
    DataFrames lazily, so recording tens of thousands of fills stays O(n)."""

    def __init__(self, trades: pd.DataFrame | None = None, cashflows: pd.DataFrame | None = None):
        self._trades = trades.copy() if trades is not None else pd.DataFrame(columns=TRADE_COLS)
        self._flows = cashflows.copy() if cashflows is not None else pd.DataFrame(columns=FLOW_COLS)
        self._new_trades: list[list] = []
        self._new_flows: list[list] = []

    @staticmethod
    def _merge(df: pd.DataFrame, rows: list[list], cols: list[str]) -> pd.DataFrame:
        add = pd.DataFrame(rows, columns=cols)
        out = add if len(df) == 0 else pd.concat([df, add], ignore_index=True)
        out["date"] = pd.to_datetime(out["date"])
        return out

    @property
    def trades(self) -> pd.DataFrame:
        if self._new_trades:
            self._trades = self._merge(self._trades, self._new_trades, TRADE_COLS)
            self._new_trades = []
        return self._trades

    @property
    def cashflows(self) -> pd.DataFrame:
        if self._new_flows:
            self._flows = self._merge(self._flows, self._new_flows, FLOW_COLS)
            self._new_flows = []
        return self._flows

    # ------------------------------------------------------------- building
    def deposit(self, date, amount: float) -> None:
        self._new_flows.append([pd.Timestamp(date), float(amount)])

    def trade(self, date, ticker: str, quantity: float, price: float, fees: float = 0.0) -> None:
        self._new_trades.append([pd.Timestamp(date), ticker, float(quantity), float(price), float(fees)])

    def add_trades(self, df: pd.DataFrame) -> None:
        self._new_trades.extend(df[TRADE_COLS].values.tolist())

    @classmethod
    def from_csv(cls, trades_csv: str | Path, cashflows_csv: str | Path) -> "Ledger":
        t = pd.read_csv(trades_csv, parse_dates=["date"])
        if "fees" not in t:
            t["fees"] = 0.0
        c = pd.read_csv(cashflows_csv, parse_dates=["date"])
        return cls(t[TRADE_COLS], c[FLOW_COLS])

    def to_csv(self, folder: str | Path, compress: bool = False) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.trades.to_csv(folder / ("trades.csv.gz" if compress else "trades.csv"), index=False)
        self.cashflows.to_csv(folder / "cashflows.csv", index=False)

    # ------------------------------------------------------------ valuation
    def positions(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        if len(self.trades) == 0:
            return pd.DataFrame(index=dates)
        q = self.trades.pivot_table(index="date", columns="ticker", values="quantity", aggfunc="sum")
        return q.reindex(dates.union(q.index)).fillna(0.0).cumsum().reindex(dates).fillna(0.0)

    def cash(self, dates: pd.DatetimeIndex) -> pd.Series:
        flows = self.cashflows.groupby("date")["amount"].sum() if len(self.cashflows) else pd.Series(dtype=float)
        if len(self.trades):
            t = self.trades
            spend = (t["quantity"] * t["price"] + t["fees"]).groupby(t["date"]).sum()
        else:
            spend = pd.Series(dtype=float)
        idx = dates.union(flows.index).union(spend.index)
        net = flows.reindex(idx).fillna(0.0) - spend.reindex(idx).fillna(0.0)
        return net.cumsum().reindex(dates).ffill().fillna(0.0)

    def valuation(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Daily NAV, cash, market value and external flows, given a (date x ticker) close-price table."""
        dates = prices.index
        pos = self.positions(dates)
        px = prices.reindex(columns=pos.columns).ffill()
        mv = (pos * px).sum(axis=1, min_count=1).fillna(0.0)
        cash = self.cash(dates)
        flows = (self.cashflows.groupby("date")["amount"].sum().reindex(dates).fillna(0.0)
                 if len(self.cashflows) else pd.Series(0.0, index=dates))
        nav = cash + mv
        return pd.DataFrame({"nav": nav, "cash": cash, "market_value": mv, "flow": flows})

    def pnl_by_position(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Realised and unrealised P&L per ticker (average-cost method), as of the last price date."""
        last = prices.ffill().iloc[-1]
        rows = []
        for tic, g in self.trades.sort_values("date").groupby("ticker"):
            qty, cost, realised, fees = 0.0, 0.0, 0.0, 0.0
            for r in g.itertuples(index=False):
                fees += r.fees
                if qty == 0 or np.sign(r.quantity) == np.sign(qty):     # opening / adding
                    cost += r.quantity * r.price
                    qty += r.quantity
                else:                                                  # reducing / closing / flipping
                    avg = cost / qty
                    closing = -r.quantity if abs(r.quantity) <= abs(qty) else qty
                    realised += closing * (r.price - avg)
                    qty_new = qty + r.quantity
                    if np.sign(qty_new) == np.sign(qty) or qty_new == 0:
                        cost = avg * qty_new
                    else:                                              # flipped through zero
                        cost = qty_new * r.price
                    qty = qty_new
            px = last.get(tic, np.nan)
            unreal = qty * px - cost if qty != 0 else 0.0
            rows.append({"ticker": tic, "quantity": qty, "avg_cost": cost / qty if qty else np.nan,
                         "last_price": px, "market_value": qty * px if qty else 0.0,
                         "realised_pnl": realised, "unrealised_pnl": unreal, "fees": fees,
                         "total_pnl": realised + unreal - fees})
        return pd.DataFrame(rows).set_index("ticker").sort_values("total_pnl", ascending=False)


def time_weighted_returns(val: pd.DataFrame) -> pd.Series:
    """Daily returns of the account with deposits/withdrawals removed (flows assumed at the start of day)."""
    nav, flow = val["nav"], val["flow"]
    base = nav.shift(1) + flow
    r = nav / base - 1.0
    r[(base <= 0) | base.isna()] = 0.0
    return r.fillna(0.0)


def performance_report(val: pd.DataFrame, benchmark_prices: pd.Series | None = None) -> dict:
    """NAV, P&L, risk and (optionally) benchmark-relative statistics for a valued ledger."""
    r = time_weighted_returns(val)
    start = val.index[(val["nav"] > 0).argmax()] if (val["nav"] > 0).any() else val.index[0]
    r = r.loc[start:]
    rep = {
        "start": str(r.index[0].date()),
        "end": str(r.index[-1].date()),
        "final_nav": float(val["nav"].iloc[-1]),
        "net_deposits": float(val["flow"].sum()),
        "total_pnl": float(val["nav"].iloc[-1] - val["flow"].sum()),
        "twr_total": float(np.prod(1 + r) - 1),
        **M.perf_stats(r.values),
        "current_drawdown": float(M.drawdown(r.values)[-1]),
    }
    if benchmark_prices is not None:
        b = benchmark_prices.reindex(r.index).ffill().pct_change().fillna(0.0)
        rep["benchmark_twr_total"] = float(np.prod(1 + b) - 1)
        rep["benchmark_sharpe"] = M.sharpe(b.values)
        rep["benchmark_max_drawdown"] = M.max_drawdown(b.values)
        rep.update({f"vs_bench_{k}": v for k, v in M.relative_stats(r.values, b.values).items()})
    return rep
