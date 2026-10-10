import numpy as np
import pytest

from vela import metrics as M


def test_max_drawdown_known_path():
    r = np.array([0.10, -0.50, 0.20, 0.0])
    # equity 1.1 -> 0.55 -> 0.66; peak 1.1 => dd = 0.55/1.1 - 1 = -0.5
    assert M.max_drawdown(r) == pytest.approx(-0.5)


def test_sharpe_and_cagr():
    r = np.array([0.01, -0.005] * 126)
    assert M.sharpe(r) == pytest.approx(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252))
    assert M.cagr(np.full(252, 0.001)) == pytest.approx(1.001**252 - 1)


def test_psr_monotone_and_dsr_penalises_trials():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0008, 0.01, 2520)  # ~1.27 annual Sharpe, 10y
    psr0 = M.probabilistic_sharpe(r, 0.0)
    assert psr0 > 0.99
    d1 = M.deflated_sharpe(r, n_trials=1)["dsr"]
    d100 = M.deflated_sharpe(r, n_trials=100, var_trial_sharpes=(0.5 / np.sqrt(252)) ** 2)["dsr"]
    assert d100 < d1


def test_expected_max_sharpe_grows_with_trials():
    v = (1 / np.sqrt(252)) ** 2
    assert M.expected_max_sharpe(10, v) < M.expected_max_sharpe(1000, v)


def test_relative_stats_recovers_beta():
    rng = np.random.default_rng(1)
    b = rng.normal(0, 0.01, 5000)
    r = 0.0002 + 0.5 * b + rng.normal(0, 0.002, 5000)
    s = M.relative_stats(r, b)
    assert s["beta"] == pytest.approx(0.5, abs=0.02)
    assert s["alpha_ann"] == pytest.approx(0.0002 * 252, abs=0.02)


def test_rank_ic_perfect_and_null():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 100))
    assert np.nanmean(M.daily_rank_ic(x, x)) == pytest.approx(1.0)
    assert abs(np.nanmean(M.daily_rank_ic(x, rng.normal(size=x.shape)))) < 0.05


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0005, 0.01, 1500)
    ci = M.stationary_bootstrap_sharpe(r, n_boot=500)
    assert ci["sharpe_ci_low"] < M.sharpe(r) < ci["sharpe_ci_high"]
