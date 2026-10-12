"""scikit-learn models: Ridge, histogram gradient boosting, random forest."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from .base import grouped_ic


class RidgeModel:
    """L2-regularised linear regression on the ranked features."""
    name = "ridge"

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y, X_val=None, y_val=None, groups_val=None):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    @property
    def coef_(self):
        return self.model.coef_


class GBMModel:
    """LightGBM-style histogram GBM with early stopping on the time-ordered validation IC."""

    name = "gbm"

    def __init__(self, learning_rate: float = 0.05, max_leaf_nodes: int = 31, min_samples_leaf: int = 2000,
                 l2_regularization: float = 1.0, max_features: float = 0.5, max_iter: int = 600,
                 check_every: int = 50, patience: int = 3, seed: int = 0):
        self.params = dict(
            learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization, max_features=max_features, random_state=seed,
            early_stopping=False, max_bins=127,
        )
        self.max_iter, self.check_every, self.patience = max_iter, check_every, patience
        self.best_iter_ = None
        self.val_curve_ = []

    def fit(self, X, y, X_val=None, y_val=None, groups_val=None):
        if X_val is None or len(X_val) == 0:
            self.model = HistGradientBoostingRegressor(max_iter=self.max_iter // 2, **self.params).fit(X, y)
            self.best_iter_ = self.max_iter // 2
            return self
        model = HistGradientBoostingRegressor(max_iter=self.check_every, warm_start=True, **self.params)
        best, best_it, bad = -np.inf, self.check_every, 0
        for it in range(self.check_every, self.max_iter + 1, self.check_every):
            model.max_iter = it
            model.fit(X, y)
            score = grouped_ic(model.predict(X_val), y_val, groups_val)
            self.val_curve_.append((it, score))
            if score > best + 1e-5:
                best, best_it, bad = score, it, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        # keep the over-trained model; predict() truncates to the best stage via staged_predict
        self.model = model
        self.best_iter_ = best_it
        return self

    def predict(self, X):
        if self.best_iter_ is None or self.best_iter_ >= self.model.n_iter_:
            return self.model.predict(X)
        for i, pred in enumerate(self.model.staged_predict(X), start=1):
            if i == self.best_iter_:
                return pred
        return pred


class RandomForestModel:
    """Bagged regression trees with large leaves (noisy target) and row subsampling."""
    name = "rf"

    def __init__(self, n_estimators: int = 200, min_samples_leaf: int = 2000, max_features: float = 0.33,
                 max_samples: float = 0.2, seed: int = 0):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features,
            max_samples=max_samples, n_jobs=-1, random_state=seed,
        )

    def fit(self, X, y, X_val=None, y_val=None, groups_val=None):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
