"""Statistical industries: point-in-time clusters of co-moving stocks.

The dataset has no sector codes, so industries are learned from returns. Each January,
using only the previous calendar year of data:

1. take daily log returns of stocks tradable at the end of that year, minus the
   equal-weighted market return (removes the common factor that dominates correlations);
2. standardise each stock's residual series and embed the stocks with PCA;
3. cluster the normalised loadings with k-means.

Labels are then fixed for the whole year. Stocks that enter the universe mid-year
are assigned to the nearest centroid using their own trailing returns at the start
of the year (or stay unassigned, label -1, if they lack history).

Motivation: short-term reversal is concentrated in the *industry-relative* return
(Da, Liu & Schaumburg 2014), while industry-level returns show momentum
(Moskowitz & Grinblatt 1999). Separating the two lets a model trade the
persistent part more cheaply.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .data.panel import Panel


def annual_clusters(panel: Panel, tradable: np.ndarray, n_clusters: int = 20, n_components: int = 15,
                    min_obs: int = 200, seed: int = 0) -> np.ndarray:
    """(T x N) int labels, -1 where unassigned. Labels for year Y use data from year Y-1 only."""
    T, N = panel.shape
    labels = np.full((T, N), -1, dtype=np.int16)
    lr = np.log(panel.close[1:] / panel.close[:-1])
    lr = np.vstack([np.full((1, N), np.nan), lr])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mkt = np.nanmean(np.where(tradable, lr, np.nan), axis=1)
    resid = lr - mkt[:, None]
    years = panel.dates.year
    for Y in range(years.min() + 1, years.max() + 1):
        prev = np.flatnonzero(years == Y - 1)
        cur = np.flatnonzero(years == Y)
        if len(prev) < 200 or len(cur) == 0:
            continue
        block = resid[prev]
        obs = np.isfinite(block).sum(0)
        eligible = (obs >= min_obs) & tradable[prev[-1]]
        cols = np.flatnonzero(eligible)
        if len(cols) < 5 * n_clusters:
            continue
        X = block[:, cols]
        X = (X - np.nanmean(X, 0)) / np.nanstd(X, 0)
        X = np.nan_to_num(X)
        pca = PCA(n_components=n_components, random_state=seed).fit(X.T)  # stocks as samples
        emb = pca.transform(X.T)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(emb)
        lab = np.full(N, -1, dtype=np.int16)
        lab[cols] = km.labels_
        # late entrants with partial history: nearest centroid on whatever history they have
        partial = np.flatnonzero((obs >= 60) & ~eligible)
        if len(partial):
            Xp = block[:, partial]
            Xp = (Xp - np.nanmean(Xp, 0)) / np.nanstd(Xp, 0)
            ep = pca.transform(np.nan_to_num(Xp).T)
            ep /= np.linalg.norm(ep, axis=1, keepdims=True) + 1e-12
            lab[partial] = km.predict(ep)
        labels[cur] = lab[None, :]
    return labels


def group_mean(values: np.ndarray, labels: np.ndarray, mask: np.ndarray, n_groups: int) -> np.ndarray:
    """Per-date mean of ``values`` within each label (over ``mask``), broadcast back to (T x N)."""
    T, N = values.shape
    out = np.full((T, N), np.nan)
    v = np.where(mask & np.isfinite(values) & (labels >= 0), values, 0.0)
    w = (mask & np.isfinite(values) & (labels >= 0)).astype(float)
    lab = np.where(labels >= 0, labels, 0).astype(np.int64)
    rows = np.repeat(np.arange(T), N)
    flat = rows * n_groups + lab.ravel()
    sums = np.bincount(flat, weights=v.ravel(), minlength=T * n_groups).reshape(T, n_groups)
    cnts = np.bincount(flat, weights=w.ravel(), minlength=T * n_groups).reshape(T, n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / cnts
    gathered = np.take_along_axis(means, lab, axis=1)
    out = np.where(labels >= 0, gathered, np.nan)
    return out


def industry_features(panel: Panel, tradable: np.ndarray, n_clusters: int = 20) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    """Industry and industry-relative returns from point-in-time clusters; returns (features, labels)."""
    labels = annual_clusters(panel, tradable, n_clusters=n_clusters)
    c = pd.DataFrame(panel.close)
    feats: dict[str, pd.DataFrame] = {}
    for k in (1, 5, 21, 63):
        r = np.log(c / c.shift(k)).values
        ind = group_mean(r, labels, tradable, n_clusters)
        feats[f"ind_ret_{k}"] = pd.DataFrame(ind)
        feats[f"ind_rel_ret_{k}"] = pd.DataFrame(r - ind)
    r126 = np.log(c.shift(21) / c.shift(126)).values
    feats["ind_mom_6_1"] = pd.DataFrame(group_mean(r126, labels, tradable, n_clusters))
    return feats, labels
