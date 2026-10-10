"""Research pipeline: data bundle -> walk-forward scores -> evaluation -> experiment log."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import portfolio as P
from .backtest.engine import BacktestResult, CostModel, run_backtest
from .config import DATA_DIR, DEV_END, FIRST_TEST_YEAR, PANEL_PATH, RESULTS_DIR
from .data.panel import Panel
from .features import FeatureSet, build_features
from .targets import forward_returns, make_target, open_to_open_returns
from .universe import rolling_median_dollar_volume, tradable_mask
from .walkforward import annual_folds

FEATURE_VERSIONS = ("v1", "v2")


def _feature_paths(version: str) -> tuple[Path, Path]:
    return DATA_DIR / f"features_{version}.npy", DATA_DIR / f"features_{version}.json"


def _build_feature_version(panel: Panel, U: np.ndarray, version: str) -> FeatureSet:
    """v1: 53 price/volume features. v2: v1 + statistical-industry features (vela/industries.py)."""
    if version == "v1":
        return build_features(panel, U)
    if version == "v2":
        from .features import cross_sectional_rank
        from .industries import industry_features
        base = _load_or_build(panel, U, "v1")
        ind, _ = industry_features(panel, U)
        extra = np.stack([cross_sectional_rank(df.set_axis(panel.dates).set_axis(panel.tickers, axis=1), U)
                          for df in ind.values()], axis=-1)
        vals = np.concatenate([np.asarray(base.values), extra], axis=-1)
        return FeatureSet(base.names + list(ind), vals, base.market_names)
    raise ValueError(version)


def _load_or_build(panel: Panel, U: np.ndarray, version: str, rebuild: bool = False) -> FeatureSet:
    cache, meta_path = _feature_paths(version)
    if cache.exists() and not rebuild:
        meta = json.loads(meta_path.read_text())
        return FeatureSet(meta["names"], np.load(cache, mmap_mode="r"), meta["market_names"])
    t0 = time.time()
    feats = _build_feature_version(panel, U, version)
    np.save(cache, np.asarray(feats.values, dtype=np.float32))
    meta_path.write_text(json.dumps({"names": feats.names, "market_names": feats.market_names}))
    print(f"features {version} built in {time.time() - t0:.0f}s: {len(feats.names)} features")
    return FeatureSet(feats.names, np.load(cache, mmap_mode="r"), feats.market_names)


# ------------------------------------------------------------------ data bundle
@dataclass
class ResearchData:
    panel: Panel
    tradable: np.ndarray
    features: FeatureSet
    R: np.ndarray            # open-to-open execution returns
    sigma: np.ndarray        # trailing 21d daily vol (close-to-close), for impact and vol weighting
    adv: np.ndarray          # trailing 21d median dollar volume
    bench_R: pd.DataFrame    # benchmark open-to-open returns (SPY, ^GSPC, EW universe)
    _fwd: dict = field(default_factory=dict)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.panel.dates

    def fwd(self, h: int) -> np.ndarray:
        if h not in self._fwd:
            self._fwd[h] = forward_returns(self.panel, h)
        return self._fwd[h]

    @classmethod
    def load(cls, features: str = "v2", rebuild_features: bool = False) -> "ResearchData":
        panel = Panel.load(PANEL_PATH)
        U = tradable_mask(panel)
        feats = _load_or_build(panel, U, features, rebuild_features)
        R = open_to_open_returns(panel)
        lr = np.log(panel.close[1:] / panel.close[:-1])
        sigma = np.full_like(panel.close, np.nan)
        sigma[1:] = pd.DataFrame(lr).rolling(21, min_periods=15).std().values
        adv = rolling_median_dollar_volume(panel)
        bo = panel.benchmark_opens
        bench = pd.DataFrame(index=panel.dates)
        for c in bo.columns:
            bench[c] = (bo[c].shift(-1) / bo[c] - 1).values
        # equal-weighted tradable universe, timed exactly like a strategy (zero cost)
        bench["EW"] = run_backtest(P.equal_weight_universe(U), R, panel.dates, CostModel.zero()).gross
        return cls(panel, U, feats, R, sigma, adv, bench)


# --------------------------------------------------------------- configuration
@dataclass
class ExperimentConfig:
    name: str
    model: str = "gbm"                  # ridge | gbm | rf | mlp | lstm | baseline
    model_params: dict = field(default_factory=dict)
    horizon: int = 5                    # label horizon in days
    target: str = "rank"                # rank | demeaned | vol_scaled
    features: list[str] | None = None   # None = all features of feature_set
    feature_set: str = "v1"             # v1: 53 price/volume features; v2: + statistical industries
    baseline_feature: str | None = None # for model == "baseline": score = sign * feature
    baseline_sign: float = 1.0
    train_years: int | None = None      # None = expanding window
    sample_every: int = 2               # use every k-th training day (adjacent days are near-duplicates)
    first_test_year: int = FIRST_TEST_YEAR
    last_test_year: int = 2015
    last_test_date: str | None = DEV_END
    seed: int = 0


def make_model(cfg: ExperimentConfig):
    from .models.sklearn_models import GBMModel, RandomForestModel, RidgeModel
    if cfg.model == "ridge":
        return RidgeModel(**cfg.model_params)
    if cfg.model == "gbm":
        return GBMModel(seed=cfg.seed, **cfg.model_params)
    if cfg.model == "rf":
        return RandomForestModel(seed=cfg.seed, **cfg.model_params)
    if cfg.model == "mlp":
        from .models.torch_models import MLPModel
        return MLPModel(seed=cfg.seed, **cfg.model_params)
    if cfg.model == "lstm":
        from .models.sequence import LSTMModel
        return LSTMModel(seed=cfg.seed, **cfg.model_params)
    raise ValueError(cfg.model)


def _rows(mask: np.ndarray, date_idx: np.ndarray, every: int = 1) -> tuple[np.ndarray, np.ndarray]:
    d = date_idx[::every] if every > 1 else date_idx
    sub = mask[d]
    t, j = np.nonzero(sub)
    return d[t], j


def run_walkforward(cfg: ExperimentConfig, data: ResearchData, verbose: bool = True) -> tuple[np.ndarray, list[dict]]:
    """Out-of-sample scores (T x N, NaN outside test years) and per-fold diagnostics."""
    T, N = data.panel.shape
    scores = np.full((T, N), np.nan)
    feat_names = cfg.features or data.features.names
    fidx = [data.features.index(f) for f in feat_names]
    U = data.tradable

    folds = annual_folds(data.dates, cfg.first_test_year, cfg.last_test_year, cfg.horizon,
                         train_years=cfg.train_years, last_test_date=cfg.last_test_date)

    if cfg.model == "baseline":
        k = data.features.index(cfg.baseline_feature)
        for fold in folds:
            ti = fold.test_idx
            scores[ti] = np.where(U[ti], cfg.baseline_sign * np.asarray(data.features.values[ti, :, k]), np.nan)
        return scores, []

    vol = data.sigma if cfg.target == "vol_scaled" else None
    y_all = make_target(data.fwd(cfg.horizon), U, cfg.target, vol=vol)
    label_ok = U & np.isfinite(y_all)
    diags = []
    F = data.features.values
    for fold in folds:
        t0 = time.time()
        tr_t, tr_j = _rows(label_ok, fold.train_idx, cfg.sample_every)
        va_t, va_j = _rows(label_ok, fold.val_idx, cfg.sample_every)
        X_tr = np.asarray(F[tr_t, tr_j][:, fidx], dtype=np.float32)
        y_tr = y_all[tr_t, tr_j].astype(np.float32)
        X_va = np.asarray(F[va_t, va_j][:, fidx], dtype=np.float32) if len(va_t) else None
        y_va = y_all[va_t, va_j].astype(np.float32) if len(va_t) else None
        model = make_model(cfg)
        if cfg.model == "lstm":
            model.attach(data)  # sequence model reads raw price history itself
            model.fit_rows(tr_t, tr_j, y_tr, va_t, va_j, y_va)
            te_t, te_j = _rows(U, fold.test_idx)
            pred = model.predict_rows(te_t, te_j)
        elif cfg.model in ("ridge", "rf") and X_va is not None:
            # no early stopping to do: train on train + validation so the model sees the latest year
            model.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
            te_t, te_j = _rows(U, fold.test_idx)
            pred = model.predict(np.asarray(F[te_t, te_j][:, fidx], dtype=np.float32))
        else:
            model.fit(X_tr, y_tr, X_va, y_va, va_t)
            te_t, te_j = _rows(U, fold.test_idx)
            pred = model.predict(np.asarray(F[te_t, te_j][:, fidx], dtype=np.float32))
        scores[te_t, te_j] = pred
        d = {"year": fold.test_year, "n_train": int(len(y_tr)), "n_test": int(len(pred)),
             "secs": round(time.time() - t0, 1)}
        for attr in ("best_iter_", "best_epochs_"):
            if hasattr(model, attr):
                d[attr.strip("_")] = getattr(model, attr)
        diags.append(d)
        if verbose:
            print(f"  [{cfg.name}] {fold.describe(data.dates)} n_train={len(y_tr):,} {d}", flush=True)
    return scores, diags


# ------------------------------------------------------------------ evaluation
@dataclass
class StrategySpec:
    kind: str = "ls"            # ls | lo | ls_hyst | lo_hyst
    q: float = 0.1              # quantile (enter quantile for hysteresis variants)
    exit_q: float | None = None
    rebalance: int = 1          # days between rebalances
    smooth_halflife: float | None = None
    vol_weight: bool = False


def build_weights(scores: np.ndarray, data: ResearchData, spec: StrategySpec) -> tuple[np.ndarray, np.ndarray | None]:
    """Target weights and the rebalance mask (None = rebalance every day)."""
    U = data.tradable & np.isfinite(scores)
    s = P.smooth_scores(scores, U, spec.smooth_halflife) if spec.smooth_halflife else scores
    vol = data.sigma if spec.vol_weight else None
    if spec.kind == "ls":
        W = P.long_short_quantile(s, U, spec.q, vol=vol)
    elif spec.kind == "lo":
        W = P.long_only_top(s, U, spec.q, vol=vol)
    elif spec.kind == "ls_hyst":
        W = P.hysteresis_long_short(s, U, spec.q, spec.exit_q or 2 * spec.q)
    elif spec.kind == "lo_hyst":
        W = P.hysteresis_long_only(s, U, spec.q, spec.exit_q or 2 * spec.q)
    else:
        raise ValueError(spec.kind)
    if spec.rebalance <= 1:
        return W, None
    return W, P.rebalance_mask(len(W), spec.rebalance)


def backtest_scores(scores: np.ndarray, data: ResearchData, spec: StrategySpec,
                    costs: CostModel = CostModel(), keep_holdings: bool = False) -> BacktestResult:
    W, reb = build_weights(scores, data, spec)
    return run_backtest(W, data.R, data.dates, costs=costs, lag=1, sigma=data.sigma, adv=data.adv,
                        keep_holdings=keep_holdings, rebalance=reb)


def decile_returns(scores: np.ndarray, fwd: np.ndarray, tradable: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Average forward return of each score decile (equal-weighted, averaged over dates)."""
    r = P._row_rank(scores, tradable & np.isfinite(scores) & np.isfinite(fwd))
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = (r > b / n_bins) & (r <= (b + 1) / n_bins)
        vals = np.where(m, fwd, np.nan)
        with np.errstate(invalid="ignore"):
            out[b] = np.nanmean(np.nanmean(vals, axis=1))
    return out


def evaluate(scores: np.ndarray, data: ResearchData, horizon: int, start: str, end: str,
             specs: dict[str, StrategySpec] | None = None, costs: CostModel = CostModel()) -> dict:
    dates = data.dates
    m = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    U = data.tradable
    s = np.where(m[:, None], scores, np.nan)
    out: dict = {}
    for h in sorted({1, horizon}):
        ic = M.daily_rank_ic(s[m], np.where(U, data.fwd(h), np.nan)[m])
        out.update({f"{k}_h{h}": v for k, v in M.ic_summary(ic, h).items()})
    dec = decile_returns(s[m], data.fwd(horizon)[m], U[m])
    out["decile_spread_bps"] = float((dec[-1] - dec[0]) * 1e4)
    out["decile_monotonicity"] = float(pd.Series(dec).corr(pd.Series(np.arange(10)), method="spearman"))
    specs = specs or default_specs(horizon)
    for label, spec in specs.items():
        res = backtest_scores(s, data, spec, costs).window(start, end)
        g, n = M.perf_stats(res.gross), M.perf_stats(res.net)
        out[f"{label}_sharpe_gross"] = g["sharpe"]
        out[f"{label}_sharpe_net"] = n["sharpe"]
        out[f"{label}_ann_ret_net"] = n["ann_return"]
        out[f"{label}_maxdd_net"] = n["max_drawdown"]
        out[f"{label}_turnover"] = float(res.turnover.mean())
        out[f"{label}_cost_bps_day"] = float((res.cost + res.borrow).mean() * 1e4)
        if spec.kind.startswith("lo"):
            rel = M.relative_stats(res.net, data.bench_R["SPY"].values[m])
            out[f"{label}_alpha_spy"] = rel["alpha_ann"]
            out[f"{label}_ir_spy"] = rel["information_ratio"]
            rel_ew = M.relative_stats(res.net, data.bench_R["EW"].values[m])
            out[f"{label}_ir_ew"] = rel_ew["information_ratio"]
    return out


def default_specs(horizon: int) -> dict[str, StrategySpec]:
    return {
        "ls": StrategySpec("ls", q=0.1, rebalance=max(1, horizon)),
        "lo": StrategySpec("lo", q=0.2, rebalance=max(1, horizon)),
    }


# ----------------------------------------------------------------- logging
EXPERIMENT_LOG = RESULTS_DIR / "experiments.csv"


def log_experiment(cfg: ExperimentConfig, result: dict, diags: list[dict] | None = None,
                   period: str = "dev", path: Path = EXPERIMENT_LOG) -> None:
    row = {"timestamp": pd.Timestamp.now().isoformat(timespec="seconds"), "period": period,
           **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in asdict(cfg).items()},
           **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in result.items()}}
    if diags:
        row["fold_secs"] = round(sum(d["secs"] for d in diags), 1)
    df = pd.DataFrame([row])
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def save_scores(name: str, scores: np.ndarray) -> Path:
    p = DATA_DIR / "scores" / f"{name}.npy"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, scores.astype(np.float32))
    return p


def load_scores(name: str) -> np.ndarray:
    return np.load(DATA_DIR / "scores" / f"{name}.npy").astype(np.float64)
