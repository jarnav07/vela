"""LSTM over raw daily sequences (in the spirit of Fischer & Krauss, 2018).

Instead of hand-engineered features, the network sees the last ``seq_len`` days of
per-day inputs for one stock, each standardised causally by that stock's trailing
volatility as known on that day:

* close-to-close log return / trailing vol
* overnight log return / trailing vol
* intraday log return / trailing vol
* log dollar-volume shock (today vs trailing 21-day mean)
* the equal-weighted market return / trailing market vol

Sequences are gathered on the fly from (T x N x C) arrays by fancy indexing,
so the (rows x seq_len x C) tensor never has to be materialised for all rows.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from .base import grouped_ic

torch.set_num_threads(4)


def build_sequence_inputs(panel, tradable) -> np.ndarray:
    """(T x N x 5) per-day inputs for the LSTM, each standardised causally by trailing volatility."""
    c = pd.DataFrame(panel.close)
    o = pd.DataFrame(panel.open)
    dv = pd.DataFrame(panel.close * panel.volume)
    lr = np.log(c / c.shift(1))
    on = np.log(o / c.shift(1))
    intra = np.log(c / o)
    vol = lr.rolling(63, min_periods=20).std().shift(1)  # known before day t's return
    mkt = lr.where(tradable).mean(axis=1)
    mvol = mkt.rolling(63, min_periods=20).std().shift(1)
    vshock = np.log(dv / dv.rolling(21, min_periods=10).mean().shift(1))
    chans = [lr / vol, on / vol, intra / vol, vshock,
             pd.DataFrame(np.repeat((mkt / mvol).values[:, None], c.shape[1], axis=1))]
    X = np.stack([ch.clip(-8, 8).fillna(0.0).values for ch in chans], axis=-1).astype(np.float32)
    return X


class _LSTMNet(nn.Module):
    def __init__(self, c_in: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(c_in, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class LSTMModel:
    """Seed-averaged LSTM that scores (date, stock) rows from their trailing input sequences."""
    name = "lstm"

    def __init__(self, seq_len: int = 60, hidden: int = 32, layers: int = 1, dropout: float = 0.1,
                 lr: float = 1e-3, batch_size: int = 2048, max_epochs: int = 15, patience: int = 3,
                 n_seeds: int = 2, seed: int = 0):
        self.seq_len, self.hidden, self.layers, self.dropout = seq_len, hidden, layers, dropout
        self.lr, self.batch_size, self.max_epochs, self.patience = lr, batch_size, max_epochs, patience
        self.n_seeds, self.seed = n_seeds, seed
        self.nets: list[nn.Module] = []
        self.best_epochs_: list[int] = []
        self._X = None

    def attach(self, data) -> None:
        cache = getattr(data, "_seq_inputs", None)
        if cache is None:
            cache = build_sequence_inputs(data.panel, data.tradable)
            data._seq_inputs = cache
        self._X = cache

    def _gather(self, t: np.ndarray, j: np.ndarray) -> torch.Tensor:
        offs = np.arange(-self.seq_len + 1, 1)
        ti = np.clip(t[:, None] + offs[None, :], 0, None)
        return torch.from_numpy(self._X[ti, j[:, None]])

    def _predict(self, net, t, j, batch=16384):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(t), batch):
                out.append(net(self._gather(t[i:i + batch], j[i:i + batch])).numpy())
        return np.concatenate(out) if out else np.array([])

    def _train_one(self, t, j, y, vt, vj, vy, seed):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = _LSTMNet(self._X.shape[-1], self.hidden, self.layers, self.dropout)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-5)
        yt = torch.from_numpy(y.astype(np.float32))
        best, best_state, best_ep, bad = -np.inf, None, 0, 0
        for ep in range(self.max_epochs):
            net.train()
            perm = rng.permutation(len(t))
            for i in range(0, len(t), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = nn.functional.mse_loss(net(self._gather(t[idx], j[idx])), yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            if vt is None or len(vt) == 0:
                best_state, best_ep = copy.deepcopy(net.state_dict()), ep
                continue
            score = grouped_ic(self._predict(net, vt, vj), vy, vt)
            if score > best + 1e-5:
                best, best_state, best_ep, bad = score, copy.deepcopy(net.state_dict()), ep, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.best_epochs_.append(best_ep + 1)
        return net

    def fit_rows(self, t, j, y, vt=None, vj=None, vy=None):
        self.nets = [self._train_one(t, j, y, vt, vj, vy, self.seed + s) for s in range(self.n_seeds)]
        return self

    def predict_rows(self, t, j):
        return np.mean([self._predict(n, t, j) for n in self.nets], axis=0)
