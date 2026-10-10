"""Re-evaluate cached out-of-sample scores, optionally sweeping portfolio construction.

    python scripts/evaluate_scores.py ridge_h5 gbm_h5               # default specs
    python scripts/evaluate_scores.py gbm_h5 --sweep                # portfolio-construction sweep

Sweep rows are appended to results/portfolio_trials.csv. They count towards the
number of trials used to deflate the final Sharpe ratio.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela import metrics as M  # noqa: E402
from vela.config import DEV_END, RESULTS_DIR  # noqa: E402
from vela.research import ResearchData, StrategySpec, backtest_scores, load_scores  # noqa: E402

SWEEP = {
    # long-short
    "ls_q10_r1": StrategySpec("ls", q=0.1, rebalance=1),
    "ls_q10_r5": StrategySpec("ls", q=0.1, rebalance=5),
    "ls_q20_r5": StrategySpec("ls", q=0.2, rebalance=5),
    "ls_q10_r5_sm3": StrategySpec("ls", q=0.1, rebalance=5, smooth_halflife=3),
    "ls_hyst_10_30_r1": StrategySpec("ls_hyst", q=0.1, exit_q=0.3, rebalance=1),
    "ls_hyst_10_30_r5": StrategySpec("ls_hyst", q=0.1, exit_q=0.3, rebalance=5),
    "ls_hyst_10_30_r5_bn": StrategySpec("ls_hyst", q=0.1, exit_q=0.3, rebalance=5, beta_neutral=True),
    "ls_hyst_10_30_r1_bn": StrategySpec("ls_hyst", q=0.1, exit_q=0.3, rebalance=1, beta_neutral=True),
    # long-only
    "lo_q20_r5": StrategySpec("lo", q=0.2, rebalance=5),
    "lo_q10_r5": StrategySpec("lo", q=0.1, rebalance=5),
    "lo_hyst_20_40_r5": StrategySpec("lo_hyst", q=0.2, exit_q=0.4, rebalance=5),
    "lo_hyst_20_40_r1": StrategySpec("lo_hyst", q=0.2, exit_q=0.4, rebalance=1),
}


def evaluate_spec(scores, data, spec, start, end) -> dict:
    res = backtest_scores(scores, data, spec).window(start, end)
    m = (data.dates >= pd.Timestamp(start)) & (data.dates <= pd.Timestamp(end))
    g, n = M.perf_stats(res.gross), M.perf_stats(res.net)
    out = {"sharpe_gross": g["sharpe"], "sharpe_net": n["sharpe"], "ann_ret_gross": g["ann_return"],
           "ann_ret_net": n["ann_return"], "ann_vol": n["ann_vol"], "maxdd_net": n["max_drawdown"],
           "turnover": float(res.turnover.mean()), "cost_bps_day": float((res.cost + res.borrow).mean() * 1e4),
           "n_long": float(res.n_long.mean()), "n_short": float(res.n_short.mean())}
    if spec.kind.startswith("lo"):
        for b in ("SPY", "EW"):
            rel = M.relative_stats(res.net, data.bench_R[b].values[m])
            out[f"alpha_{b}"] = rel["alpha_ann"]
            out[f"alpha_t_{b}"] = rel["alpha_t"]
            out[f"ir_{b}"] = rel["information_ratio"]
            out[f"beta_{b}"] = rel["beta"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--start", default="2006-01-01")
    ap.add_argument("--end", default=DEV_END)
    args = ap.parse_args()
    data = ResearchData.load("v1")  # evaluation needs prices only; any feature cache works
    rows = []
    for name in args.names:
        scores = load_scores(name)
        specs = SWEEP if args.sweep else {"ls_q10_r5": SWEEP["ls_q10_r5"], "lo_q20_r5": SWEEP["lo_q20_r5"]}
        for label, spec in specs.items():
            r = {"scores": name, "spec": label, **evaluate_spec(scores, data, spec, args.start, args.end)}
            rows.append(r)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)
    df = pd.DataFrame(rows)
    df.insert(0, "timestamp", pd.Timestamp.now().isoformat(timespec="seconds"))
    df.insert(1, "period", f"{args.start}..{args.end}")
    path = RESULTS_DIR / "portfolio_trials.csv"
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
