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


def test_track_portfolio_cli_end_to_end(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("tp", "scripts/track_portfolio.py")
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)
    dates = pd.bdate_range("2021-01-04", periods=30)
    rng = np.random.default_rng(0)
    px = pd.DataFrame({"AAA": 100 * np.cumprod(1 + rng.normal(0, 0.01, 30)),
                       "BBB": 50 * np.cumprod(1 + rng.normal(0, 0.01, 30)),
                       "SPY": 300 * np.cumprod(1 + rng.normal(0, 0.008, 30))}, index=dates)
    px.index.name = "date"
    pd.DataFrame({"date": [dates[0], dates[10]], "ticker": ["AAA", "BBB"], "quantity": [50, 40],
                  "price": [px["AAA"].iloc[0], px["BBB"].iloc[10]], "fees": [1.0, 1.0]}).to_csv(tmp_path / "t.csv", index=False)
    pd.DataFrame({"date": [dates[0], dates[10]], "amount": [6000, 2000]}).to_csv(tmp_path / "c.csv", index=False)
    px.to_csv(tmp_path / "p.csv")
    L = Ledger.from_csv(tmp_path / "t.csv", tmp_path / "c.csv")
    rep = tp.run(L, pd.read_csv(tmp_path / "p.csv", parse_dates=["date"], index_col="date"), "SPY", tmp_path / "out")
    assert rep["net_deposits"] == pytest.approx(8000)
    expected_nav = 8000 - 2 + 50 * (px["AAA"].iloc[-1] - px["AAA"].iloc[0]) + 40 * (px["BBB"].iloc[-1] - px["BBB"].iloc[10])
    assert rep["final_nav"] == pytest.approx(expected_nav)
    assert (tmp_path / "out" / "equity.png").exists()
    assert "vs_bench_beta" in rep


def test_live_adapter_parses_yfinance_frame():
    """yfinance (auto_adjust, group_by='column') returns MultiIndex columns (field, ticker)."""
    from vela.data.live import panel_from_yahoo_frame
    dates = pd.bdate_range("2024-01-02", periods=5, tz="America/New_York")
    fields_ = ["Close", "High", "Low", "Open", "Volume"]
    cols = pd.MultiIndex.from_product([fields_, ["AAA", "SPY"]], names=["Price", "Ticker"])
    raw = pd.DataFrame(np.arange(50, dtype=float).reshape(5, 10) + 1, index=dates, columns=cols)
    p = panel_from_yahoo_frame(raw, ["AAA", "ZZZ"])
    assert list(p.tickers) == ["AAA"]
    np.testing.assert_allclose(p.close[:, 0], raw[("Close", "AAA")].values)
    np.testing.assert_allclose(p.benchmarks["SPY"].values, raw[("Close", "SPY")].values)
    assert p.dates.tz is None and p.member.all()


def test_live_hysteresis_step_matches_rule():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ls", "scripts/live_signals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ranks = pd.Series({"A": 0.95, "B": 0.70, "C": 0.30, "D": 0.75})
    held = pd.Series({"A": 0.0, "B": 3.0, "C": 2.0, "D": 0.0})
    w = mod.hysteresis_targets(ranks, held, enter_q=0.2, exit_q=0.4, short=False)
    # enter A (top 20%) and keep B (held, still above 60th pct); drop C; D (75th pct) is not held and below the entry band
    assert set(w.index) == {"A", "B"} and w.sum() == pytest.approx(1.0)
