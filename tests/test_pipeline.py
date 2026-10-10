"""Causality tests for features, targets and walk-forward splits (synthetic panel, no data needed)."""
import numpy as np
import pandas as pd
import pytest

from vela.data.panel import Panel
from vela.features import build_features
from vela.targets import forward_returns, open_to_open_returns
from vela.universe import tradable_mask
from vela.walkforward import annual_folds


def synthetic_panel(T=700, N=60, seed=0) -> Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2001-01-01", periods=T)
    lr = rng.normal(0.0003, 0.02, (T, N))
    close = 50 * np.exp(np.cumsum(lr, 0))
    open_ = close * np.exp(rng.normal(0, 0.005, (T, N)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, (T, N))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, (T, N))))
    vol = rng.lognormal(14, 0.5, (T, N))
    member = np.ones((T, N), dtype=bool)
    member[:300, :5] = False  # some names join later
    bench = pd.DataFrame({"SPY": close.mean(1)}, index=dates)
    return Panel(dates, pd.Index([f"S{i}" for i in range(N)]), open_, high, low, close, vol, member,
                 benchmarks=bench, benchmark_opens=bench)


def test_features_are_causal():
    """Features at date D must not change when data after D is removed."""
    p = synthetic_panel()
    cut = 500
    full = build_features(p, tradable_mask(p)).values
    trunc_panel = p.slice_dates(end=p.dates[cut])
    trunc = build_features(trunc_panel, tradable_mask(trunc_panel)).values
    np.testing.assert_allclose(full[cut], trunc[cut], atol=1e-6)
    np.testing.assert_allclose(full[cut - 37], trunc[cut - 37], atol=1e-6)


def test_forward_return_alignment():
    p = synthetic_panel(T=50, N=3)
    h = 5
    y = forward_returns(p, h)
    t = 10
    np.testing.assert_allclose(y[t], p.open[t + 1 + h] / p.open[t + 1] - 1)
    assert np.all(np.isnan(y[-(h + 1):]))
    R = open_to_open_returns(p)
    # compounding h daily execution returns from bar t+1 reproduces the h-day label
    np.testing.assert_allclose(np.prod(1 + R[t + 1:t + 1 + h], axis=0) - 1, y[t])


@pytest.mark.parametrize("h", [1, 5, 21])
def test_walkforward_purge(h):
    dates = pd.bdate_range("2000-01-03", "2012-12-31")
    folds = annual_folds(dates, 2006, 2012, horizon=h)
    for f in folds:
        # the label of the last training/validation row is realised at index t + 1 + h,
        # which must be strictly before the first test decision
        last_label_realised = max(f.train_idx.max(), f.val_idx.max() if len(f.val_idx) else 0) + 1 + h
        assert last_label_realised < f.test_idx[0]
        if len(f.val_idx):
            assert f.train_idx.max() + 1 + h < f.val_idx[0]
        assert dates[f.test_idx[0]].year == f.test_year


def test_tradable_mask_requires_membership_and_history():
    p = synthetic_panel()
    U = tradable_mask(p, min_dollar_volume=0, min_history=126)
    assert not U[:125].any()
    assert not U[:300, :5].any()
    assert U[400:, 5:].all()
