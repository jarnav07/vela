import numpy as np
import pandas as pd
import pytest

from vela.backtest.engine import CostModel, run_backtest

DATES = pd.bdate_range("2020-01-01", periods=6)


def test_single_asset_buy_and_hold_no_costs():
    r = np.array([[0.01], [0.02], [-0.01], [0.03], [0.0], [0.01]])
    w = np.ones_like(r)
    res = run_backtest(w, r, DATES, costs=CostModel.zero(), lag=1)
    # weights decided at close of bar 0 are first held over bar 1
    np.testing.assert_allclose(res.gross, [0.0, 0.02, -0.01, 0.03, 0.0, 0.01])
    np.testing.assert_allclose(res.turnover, [0, 1, 0, 0, 0, 0])


def test_entry_cost_is_turnover_times_rate():
    r = np.zeros((4, 2))
    w = np.array([[0.5, -0.5]] * 4)
    costs = CostModel(commission_bps=1.0, half_spread_bps=4.0, impact_coef=0.0, borrow_bps_annual=0.0)
    res = run_backtest(w, r, DATES[:4], costs=costs, lag=1)
    # enter 0.5 long + 0.5 short = 1.0 of NAV traded at 5 bps
    np.testing.assert_allclose(res.cost, [0.0, 1.0 * 5e-4, 0.0, 0.0])


def test_drift_rebalancing_turnover():
    # 50/50 static target; asset 0 rises 10% on the first held bar
    r = np.array([[0.0, 0.0], [0.10, 0.0], [0.0, 0.0]])
    w = np.array([[0.5, 0.5]] * 3)
    res = run_backtest(w, r, DATES[:3], costs=CostModel.zero(), lag=1)
    port = 0.05
    drifted = np.array([0.5 * 1.1, 0.5]) / (1 + port)
    expected = np.abs(np.array([0.5, 0.5]) - drifted).sum()
    np.testing.assert_allclose(res.turnover[2], expected)
    assert res.gross[1] == pytest.approx(port)


def test_borrow_fee_on_shorts_only():
    r = np.zeros((3, 2))
    w = np.array([[1.0, -0.5]] * 3)
    costs = CostModel(commission_bps=0, half_spread_bps=0, impact_coef=0, borrow_bps_annual=252.0)
    res = run_backtest(w, r, DATES[:3], costs=costs, lag=1)
    np.testing.assert_allclose(res.borrow, [0.0, 0.5 * 0.0252 / 252, 0.5 * 0.0252 / 252])


def test_square_root_impact():
    r = np.zeros((2, 1))
    w = np.array([[0.1], [0.1]])
    sigma = np.full((2, 1), 0.02)
    adv = np.full((2, 1), 1e8)
    costs = CostModel(commission_bps=0, half_spread_bps=0, impact_coef=1.0, aum=1e7, borrow_bps_annual=0)
    res = run_backtest(w, r, DATES[:2], costs=costs, lag=1, sigma=sigma, adv=adv)
    trade = 0.1
    expected = trade * 1.0 * 0.02 * np.sqrt(trade * 1e7 / 1e8)
    assert res.cost[1] == pytest.approx(expected)


def test_nan_returns_and_weights_are_zero():
    r = np.array([[0.01, np.nan], [np.nan, 0.02], [0.01, 0.01]])
    w = np.array([[np.nan, 1.0], [1.0, np.nan], [0.0, 0.0]])
    res = run_backtest(w, r, DATES[:3], costs=CostModel.zero(), lag=1)
    assert np.all(np.isfinite(res.net))
    np.testing.assert_allclose(res.gross, [0.0, 0.02, 0.01])


def test_lag_zero_rejected():
    with pytest.raises(ValueError):
        run_backtest(np.zeros((2, 1)), np.zeros((2, 1)), DATES[:2], lag=0)


def _random_walk(T=3000, N=50, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0003, 0.02, (T, N)), pd.bdate_range("2000-01-03", periods=T)


def test_lookahead_canary():
    """A signal that peeks at the return it will earn must look spectacular;
    the same signal formed one bar earlier (i.e. honestly) must not."""
    R, dates = _random_walk()
    peek = np.zeros_like(R)
    peek[:-1] = np.sign(R[1:])            # at close t, "knows" the interval t+1 return
    honest = np.sign(R)                  # interval-t return: realised by the time interval t+1 is held
    honest_lagged = np.zeros_like(R)
    honest_lagged[1:] = np.sign(R[:-1])  # one bar older still
    zero = CostModel.zero()
    sr = lambda res: res.net.mean() / res.net.std() * np.sqrt(252)
    assert sr(run_backtest(peek / 50, R, dates, zero)) > 20
    assert abs(sr(run_backtest(honest_lagged / 50, R, dates, zero))) < 1.0
    assert abs(sr(run_backtest(honest / 50, R, dates, zero))) < 1.0


def test_random_signal_loses_roughly_its_costs():
    R, dates = _random_walk(seed=7)
    rng = np.random.default_rng(3)
    w = rng.choice([-1.0, 1.0], size=R.shape) / R.shape[1]
    costs = CostModel(commission_bps=0, half_spread_bps=5.0, impact_coef=0, borrow_bps_annual=0)
    res = run_backtest(w, R, dates, costs)
    assert res.gross.mean() == pytest.approx(0.0, abs=3 * res.gross.std() / np.sqrt(len(R)))
    np.testing.assert_allclose(res.net.mean(), res.gross.mean() - 5e-4 * res.turnover.mean(), rtol=1e-9)


def test_buy_and_hold_between_rebalances_matches_loop():
    rng = np.random.default_rng(5)
    T, N = 40, 4
    R = rng.normal(0.001, 0.02, (T, N))
    W = rng.normal(0, 0.3, (T, N))
    reb = np.zeros(T, dtype=bool)
    reb[::7] = True
    res = run_backtest(W, R, pd.bdate_range("2020-01-01", periods=T), CostModel.zero(), rebalance=reb)
    # reference: explicit share-holding simulation
    nav, pos, navs = 1.0, np.zeros(N), []
    for s in range(T):
        if s >= 1 and reb[s - 1]:
            pos = W[s - 1] * nav            # dollar positions set at execution bar s
        pnl = (pos * R[s]).sum()
        navs.append(pnl / nav)
        pos = pos * (1 + R[s])
        nav += pnl
    np.testing.assert_allclose(res.gross, navs, atol=1e-12)
    # no trading between rebalances
    exec_bars = np.flatnonzero(np.r_[False, reb[:-1]])
    assert np.all(res.turnover[np.setdiff1d(np.arange(T), exec_bars)] == 0)


def test_daily_rebalance_mask_equals_constant_mix():
    rng = np.random.default_rng(6)
    R = rng.normal(0, 0.02, (30, 3))
    W = rng.normal(0, 0.3, (30, 3))
    a = run_backtest(W, R, pd.bdate_range("2020-01-01", periods=30), CostModel.zero())
    b = run_backtest(W, R, pd.bdate_range("2020-01-01", periods=30), CostModel.zero(), rebalance=np.ones(30, bool))
    np.testing.assert_allclose(a.gross, b.gross, atol=1e-12)
    np.testing.assert_allclose(a.turnover, b.turnover, atol=1e-12)
