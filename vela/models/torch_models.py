"""PyTorch models: a tabular MLP and an LSTM over raw daily return sequences.

Both are trained to regress the cross-sectional rank target with early stopping on
the time-ordered validation IC, and both average several random seeds (deep models
on noisy financial targets are high-variance; seed-averaging is a cheap variance cut).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .base import grouped_ic

torch.set_num_threads(4)


class _MLP(nn.Module):
    def __init__(self, k: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = k
        for hdim in hidden:
            layers += [nn.Linear(d, hdim), nn.BatchNorm1d(hdim), nn.SiLU(), nn.Dropout(dropout)]
            d = hdim
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _predict_batched(net: nn.Module, X: np.ndarray, batch: int = 65536) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(net(torch.from_numpy(np.ascontiguousarray(X[i:i + batch], dtype=np.float32))).numpy())
    return np.concatenate(out) if out else np.array([])


class MLPModel:
    name = "mlp"

    def __init__(self, hidden=(128, 64), dropout: float = 0.2, lr: float = 1e-3, weight_decay: float = 1e-5,
                 batch_size: int = 4096, max_epochs: int = 30, patience: int = 4, n_seeds: int = 3, seed: int = 0):
        self.hidden, self.dropout, self.lr, self.wd = tuple(hidden), dropout, lr, weight_decay
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.n_seeds, self.seed = n_seeds, seed
        self.nets: list[nn.Module] = []
        self.best_epochs_: list[int] = []

    def _train_one(self, X, y, X_val, y_val, groups_val, seed):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = _MLP(X.shape[1], self.hidden, self.dropout)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        best, best_state, best_ep, bad = -np.inf, None, 0, 0
        for ep in range(self.max_epochs):
            net.train()
            perm = torch.from_numpy(rng.permutation(len(X)))
            for i in range(0, len(X), self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss = nn.functional.mse_loss(net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            if X_val is None or len(X_val) == 0:
                best_state, best_ep = copy.deepcopy(net.state_dict()), ep
                continue
            score = grouped_ic(_predict_batched(net, X_val), y_val, groups_val)
            if score > best + 1e-5:
                best, best_state, best_ep, bad = score, copy.deepcopy(net.state_dict()), ep, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.best_epochs_.append(best_ep + 1)
        return net

    def fit(self, X, y, X_val=None, y_val=None, groups_val=None):
        self.nets = [self._train_one(X, y, X_val, y_val, groups_val, self.seed + s) for s in range(self.n_seeds)]
        return self

    def predict(self, X):
        return np.mean([_predict_batched(n, X) for n in self.nets], axis=0)
