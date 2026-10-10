"""Smoke tests: each model learns an obvious synthetic signal."""
import numpy as np
import pandas as pd
import pytest

from vela.models.base import grouped_ic
from vela.models.sklearn_models import GBMModel, RidgeModel


def _data(n_dates=60, n_names=80, k=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-0.5, 0.5, (n_dates * n_names, k)).astype(np.float32)
    y = (0.6 * X[:, 0] - 0.4 * X[:, 1] * (X[:, 2] > 0) + rng.normal(0, 0.3, len(X))).astype(np.float32)
    g = np.repeat(np.arange(n_dates), n_names)
    return X, y, g


@pytest.mark.parametrize("model", [RidgeModel(alpha=1.0), GBMModel(max_iter=100, check_every=25, min_samples_leaf=50)])
def test_sklearn_models_learn(model):
    X, y, g = _data()
    Xv, yv, gv = _data(seed=1)
    model.fit(X, y, Xv, yv, gv)
    assert grouped_ic(model.predict(Xv), yv, gv) > 0.3


def test_mlp_learns():
    from vela.models.torch_models import MLPModel
    X, y, g = _data()
    Xv, yv, gv = _data(seed=1)
    m = MLPModel(hidden=(32,), max_epochs=8, n_seeds=1, batch_size=512).fit(X, y, Xv, yv, gv)
    assert grouped_ic(m.predict(Xv), yv, gv) > 0.3


def test_lstm_learns_reversal_from_sequences():
    """The sequence model must discover that yesterday's return predicts tomorrow's (negatively)."""
    from vela.models.sequence import LSTMModel
    rng = np.random.default_rng(0)
    T, N, C = 400, 60, 5
    X = rng.normal(size=(T, N, C)).astype(np.float32)
    y_full = -X[:, :, 0] + 0.5 * rng.normal(size=(T, N))   # target = minus today's channel-0 value
    m = LSTMModel(seq_len=10, hidden=8, max_epochs=6, n_seeds=1, batch_size=256)
    m._X = X
    t_tr, j_tr = np.meshgrid(np.arange(20, 300), np.arange(N), indexing="ij")
    t_va, j_va = np.meshgrid(np.arange(310, 390), np.arange(N), indexing="ij")
    t_tr, j_tr, t_va, j_va = t_tr.ravel(), j_tr.ravel(), t_va.ravel(), j_va.ravel()
    m.fit_rows(t_tr, j_tr, y_full[t_tr, j_tr], t_va, j_va, y_full[t_va, j_va])
    pred = m.predict_rows(t_va, j_va)
    assert grouped_ic(pred, y_full[t_va, j_va], t_va) > 0.5
