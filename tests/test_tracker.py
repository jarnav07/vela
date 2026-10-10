import numpy as np
import pandas as pd
import pytest

from vela.backtest.engine import CostModel, run_backtest
from vela.tracker.ledger import Ledger, performance_report, time_weighted_returns
from vela.tracker.orders import generate_orders
from vela.tracker.paper import PaperConfig, simulate

D = pd.bdate_range("2021-01-04", periods=5)


def test_nav_cash_and_twr_with_deposit():
    prices = pd.DataFrame({"AAA": [10.0, 11.0, 11.0, 12.1, 12.1]}, index=D)
    L = Ledger()
    L.deposit(D[0], 1000)
    L.trade(D[0], "AAA", 100, 10.0)
    L.deposit(D[2], 1100)            # new money must not count as return
    val = L.valuation(prices)
    np.testing.assert_allclose(val["nav"], [1000, 1100, 2200, 2310, 2310])
    r = time_weighted_returns(val)
    np.testing.assert_allclose(r.values, [0, 0.10, 0.0, 0.05, 0.0], atol=1e-12)
    rep = performance_report(val)
    assert rep["twr_total"] == pytest.approx(1.1 * 1.05 - 1)
    assert rep["total_pnl"] == pytest.approx(2310 - 2100)


def test_average_cost_pnl():
    prices = pd.DataFrame({"AAA": [10.0, 12.0, 15.0]}, index=D[:3])
    L = Ledger()
    L.deposit(D[0], 10_000)
    L.trade(D[0], "AAA", 100, 10.0, fees=1.0)
    L.trade(D[1], "AAA", 100, 12.0, fees=1.0)    # avg cost 11
    L.trade(D[2], "AAA", -50, 15.0, fees=1.0)    # realise 50 * (15 - 11) = 200
    pnl = L.pnl_by_position(prices).loc["AAA"]
    assert pnl["realised_pnl"] == pytest.approx(200)
    assert pnl["unrealised_pnl"] == pytest.approx(150 * (15 - 11))
    assert pnl["total_pnl"] == pytest.approx(200 + 600 - 3)


def test_generate_orders_whole_shares_never_overspend():
    w = pd.Series({"A": 0.5, "B": 0.5})
    o = generate_orders(w, pd.Series(dtype=float), pd.Series({"A": 33.0, "B": 70.0}), nav=1000, fractional=False)
    assert set(o.index) == {"A", "B"}
    assert o["order_qty"].tolist() == [15.0, 7.0] or sorted(o["order_qty"]) == [7.0, 15.0]
    assert (o["order_qty"] * o["price"]).sum() <= 1000


def test_paper_simulation_matches_vectorised_engine():
    """Independent event-driven ledger vs vectorised engine: same NAV path with fractional shares."""
    rng = np.random.default_rng(11)
    T, N = 120, 8
    dates = pd.bdate_range("2019-01-01", periods=T)
    tickers = pd.Index([f"T{i}" for i in range(N)])
    opens = 50 * np.exp(np.cumsum(rng.normal(0, 0.015, (T, N)), 0))
    R = np.full_like(opens, np.nan)
    R[:-1] = opens[1:] / opens[:-1] - 1
    W = np.abs(rng.normal(size=(T, N)))
    W = W / W.sum(1, keepdims=True)       # long-only, fully invested
    reb = np.arange(T) % 5 == 0
    hs = 5.0
    eng = run_backtest(W, R, dates, CostModel(commission_bps=0.5, half_spread_bps=hs, impact_coef=0, borrow_bps_annual=0),
                       rebalance=reb)
    ledger = simulate(W, reb, opens, dates, tickers, str(dates[0].date()), str(dates[-1].date()),
                      PaperConfig(initial_capital=1e5, half_spread_bps=hs, commission_bps=0.5))
    val = ledger.valuation(pd.DataFrame(opens, index=dates, columns=tickers))
    # ledger NAV at open of bar s+1 vs engine compounded net return through bar s
    first_exec = 1
    eng_nav = 1e5 * np.cumprod(1 + eng.net[first_exec:-1])
    led_nav = val["nav"].values[first_exec + 1:]
    np.testing.assert_allclose(led_nav, eng_nav, rtol=2e-3)
