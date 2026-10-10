"""Turn target weights into broker orders for a small account."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_orders(
    target_weights: pd.Series,
    positions: pd.Series,
    prices: pd.Series,
    nav: float,
    fractional: bool = True,
    min_trade_value: float = 0.0,
    cash_buffer: float = 0.0,
) -> pd.DataFrame:
    """Orders that move ``positions`` (shares) to ``target_weights`` (fraction of NAV).

    * ``fractional=False`` rounds to whole shares (towards zero for buys, so the
      account never needs more cash than it has).
    * Orders smaller than ``min_trade_value`` are skipped: not worth a commission.
    * ``cash_buffer`` keeps a fraction of NAV in cash for fees and rounding.
    """
    tickers = target_weights.index.union(positions.index)
    tw = target_weights.reindex(tickers).fillna(0.0) * (1.0 - cash_buffer)
    cur = positions.reindex(tickers).fillna(0.0)
    px = prices.reindex(tickers)
    target_qty = tw * nav / px
    if not fractional:
        target_qty = np.trunc(target_qty)
    target_qty = target_qty.where(px.notna(), cur)        # cannot trade without a price
    order = target_qty - cur
    value = order * px
    keep = (order.abs() > 1e-9) & (value.abs() >= min_trade_value)
    out = pd.DataFrame({
        "current_qty": cur, "target_qty": target_qty, "order_qty": order.where(keep, 0.0),
        "price": px, "order_value": value.where(keep, 0.0), "target_weight": tw,
    })
    return out[out["order_qty"] != 0].sort_values("order_value")
