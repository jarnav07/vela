"""Model ladder on the development period, every model under the same portfolio construction.

Re-scores each saved development-period score file with the frozen L/S and long-only books
and the base-case cost model, so models are compared like for like. Writes
results/model_ladder.csv and reports/figures/model_ladder.png; appends to the trial log.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela import plots  # noqa: E402
from vela.config import DEV_END, FIGURES_DIR, RESULTS_DIR  # noqa: E402
from vela.ensemble import rank_average  # noqa: E402
from vela.research import ResearchData, StrategySpec, evaluate, load_scores  # noqa: E402

LADDER = [
    ("12-1 momentum", ["base_mom12_h5"]),
    ("5-day reversal", ["base_rev5_h5"]),
    ("1-day reversal", ["base_rev1_h5"]),
    ("Ridge", ["ridge_h5"]),
    ("PyTorch MLP", ["mlp_h5"]),
    ("PyTorch LSTM", ["lstm_h5"]),
    ("GBM + industry feats", ["gbm_v2_h5", "gbm_v2_h5_s1"]),
    ("GBM (seed 0)", ["gbm_h5"]),
    ("GBM (seed 1)", ["gbm_h5_s1"]),
    ("GBM (seed bag)", ["gbm_h5", "gbm_h5_s1"]),
]


def main() -> None:
    spec_ = importlib.util.spec_from_file_location("ev", Path(__file__).parent / "evaluate_scores.py")
    ev = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(ev)
    frozen = json.loads((RESULTS_DIR / "frozen_config.json").read_text())
    ls_spec = StrategySpec(**frozen["strategies"]["ls"])
    lo_spec = StrategySpec(**frozen["strategies"]["lo"])
    d = ResearchData.load("v1")
    rows, trials = [], []
    for label, names in LADDER:
        try:
            comps = [load_scores(n) for n in names]
        except FileNotFoundError:
            print("skip", label)
            continue
        s = rank_average(comps, d.tradable) if len(comps) > 1 else comps[0]
        r = evaluate(s, d, 5, "2006-01-01", DEV_END, specs={})
        ls = ev.evaluate_spec(s, d, ls_spec, "2006-01-01", DEV_END)
        lo = ev.evaluate_spec(s, d, lo_spec, "2006-01-01", DEV_END)
        rows.append({"model": label, "scores": "+".join(names), "ic_1d": r["ic_mean_h1"], "ic_5d": r["ic_mean_h5"],
                     "ic_t_5d": r["ic_t_h5"], "decile_spread_5d_bps": r["decile_spread_bps"],
                     "ls_sharpe_gross": ls["sharpe_gross"], "ls_sharpe_net": ls["sharpe_net"],
                     "lo_sharpe_net": lo["sharpe_net"], "lo_ir_spy": lo["ir_SPY"], "lo_ir_ew": lo["ir_EW"]})
        trials += [{"scores": "+".join(names), "spec": "frozen_ls", **ls}, {"scores": "+".join(names), "spec": "frozen_lo", **lo}]
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "model_ladder.csv", index=False)
    t = pd.DataFrame(trials)
    t.insert(0, "timestamp", pd.Timestamp.now().isoformat(timespec="seconds"))
    t.insert(1, "period", f"2006-01-01..{DEV_END}")
    p = RESULTS_DIR / "portfolio_trials.csv"
    pd.concat([pd.read_csv(p), t], ignore_index=True).to_csv(p, index=False)

    plots.bars(df["model"].tolist(), (df["ic_5d"] * 100).values, FIGURES_DIR / "model_ladder_ic.png",
               "Model ladder, dev 2006-2015: mean daily rank IC vs 5-day forward return", "IC x 100",
               horizontal=True)
    plots.bars(df["model"].tolist(), df["ls_sharpe_net"].values, FIGURES_DIR / "model_ladder_sharpe.png",
               "Model ladder, dev 2006-2015: market-neutral book, net Sharpe after costs", "Net Sharpe",
               horizontal=True)


if __name__ == "__main__":
    main()
