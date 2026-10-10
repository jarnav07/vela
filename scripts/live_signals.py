"""Run the frozen strategy on current data and print orders for a real or paper account.

    pip install -e ".[live]"
    python scripts/live_signals.py train                      # fit the frozen model(s) on all history
    python scripts/live_signals.py signal --holdings holdings.csv --cash 2500 --book lo

holdings.csv: ticker,quantity  (current positions; empty file for a new account)

``signal`` downloads ~2 years of daily data for today's S&P 500 constituents from Yahoo
Finance, computes the same features as in research, scores them with the saved model,
applies the frozen hysteresis rule *relative to your current holdings*, and prints
orders sized to your account (fractional shares by default).

The research environment could not reach Yahoo Finance, so this script was exercised
only against mocked data. Check the first orders by hand before trading.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela.config import RESULTS_DIR, ROOT  # noqa: E402
from vela.research import ExperimentConfig, ResearchData, StrategySpec, make_model  # noqa: E402

MODEL_DIR = ROOT / "models"


def _frozen():
    return json.loads((RESULTS_DIR / "frozen_config.json").read_text())


def train() -> None:
    from vela.targets import make_target
    frozen = _frozen()
    data = ResearchData.load(frozen.get("feature_set", "v2"))
    MODEL_DIR.mkdir(exist_ok=True)
    for d in frozen["models"]:
        cfg = ExperimentConfig(**{k: v for k, v in d.items() if k in {f.name for f in fields(ExperimentConfig)}})
        y = make_target(data.fwd(cfg.horizon), data.tradable, cfg.target, vol=data.sigma)
        ok = data.tradable & np.isfinite(y)
        idx = np.arange(len(data.dates))
        val_start = len(idx) - 252
        names = cfg.features or data.features.names
        fidx = [data.features.index(f) for f in names]
        tr_t, tr_j = np.nonzero(ok[: val_start - cfg.horizon - 2][:: cfg.sample_every])
        tr_t = tr_t * cfg.sample_every
        va_t, va_j = np.nonzero(ok[val_start:])
        va_t = va_t + val_start
        F = data.features.values
        model = make_model(cfg)
        model.fit(np.asarray(F[tr_t, tr_j][:, fidx]), y[tr_t, tr_j], np.asarray(F[va_t, va_j][:, fidx]),
                  y[va_t, va_j], va_t)
        with open(MODEL_DIR / f"{cfg.name}.pkl", "wb") as fh:
            pickle.dump({"model": model, "features": names, "feature_set": cfg.feature_set}, fh)
        print(f"saved {MODEL_DIR / (cfg.name + '.pkl')} trained on {len(tr_t):,} rows")


def hysteresis_targets(ranks: pd.Series, held: pd.Series, enter_q: float, exit_q: float, short: bool) -> pd.Series:
    """One step of the research hysteresis rule given current holdings (sign of quantity)."""
    longs = (ranks > 1 - enter_q) | ((held > 0) & (ranks > 1 - exit_q))
    w = longs.astype(float) / max(longs.sum(), 1)
    if short:
        shorts = (ranks <= enter_q) | ((held < 0) & (ranks <= exit_q))
        w = w - shorts.astype(float) / max(shorts.sum(), 1)
    return w[w != 0]


def signal(holdings_csv: str, cash: float, book: str) -> None:
    from vela.data.live import current_sp500_tickers, fetch_panel
    from vela.ensemble import rank_average
    from vela.features import build_features
    from vela.tracker.orders import generate_orders
    from vela.universe import tradable_mask

    frozen = _frozen()
    spec = StrategySpec(**frozen["strategies"][book])
    tickers = current_sp500_tickers()
    start = (pd.Timestamp.today() - pd.Timedelta(days=760)).strftime("%Y-%m-%d")
    panel = fetch_panel(tickers, start)
    panel.member[:] = np.isfinite(panel.close)
    U = tradable_mask(panel)
    feats = build_features(panel, U)
    if any(d.get("feature_set") == "v2" for d in frozen["models"]):
        from vela.features import cross_sectional_rank
        from vela.industries import industry_features
        ind, _ = industry_features(panel, U)
        extra = np.stack([cross_sectional_rank(df.set_axis(panel.dates).set_axis(panel.tickers, axis=1), U)
                          for df in ind.values()], axis=-1)
        feats.values = np.concatenate([feats.values, extra], axis=-1)
        feats.names = feats.names + list(ind)
    t = len(panel.dates) - 1
    comp = []
    for d in frozen["models"]:
        with open(MODEL_DIR / f"{d['name']}.pkl", "rb") as fh:
            saved = pickle.load(fh)
        fidx = [feats.names.index(f) for f in saved["features"]]
        s = np.full((1, panel.shape[1]), np.nan)
        live = U[t]
        s[0, live] = saved["model"].predict(feats.values[t][live][:, fidx])
        comp.append(s)
    score = rank_average(comp, U[t:t + 1]) if len(comp) > 1 else comp[0]
    ranks = pd.Series(score[0], index=panel.tickers).rank(pct=True).dropna()

    held = pd.read_csv(holdings_csv).set_index("ticker")["quantity"] if Path(holdings_csv).stat().st_size else pd.Series(dtype=float)
    prices = pd.Series(panel.close[t], index=panel.tickers)
    nav = cash + float((held * prices.reindex(held.index)).sum())
    w = hysteresis_targets(ranks, held.reindex(ranks.index).fillna(0.0), spec.q, spec.exit_q or 2 * spec.q,
                           short=book.startswith("ls"))
    orders = generate_orders(w, held, prices, nav, fractional=True, min_trade_value=1.0)
    print(f"As of close {panel.dates[t].date()}: NAV {nav:,.2f}, {len(w)} target positions")
    print(orders.round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "signal"])
    ap.add_argument("--holdings", default="holdings.csv")
    ap.add_argument("--cash", type=float, default=0.0)
    ap.add_argument("--book", default="lo")
    a = ap.parse_args()
    train() if a.cmd == "train" else signal(a.holdings, a.cash, a.book)


if __name__ == "__main__":
    main()
