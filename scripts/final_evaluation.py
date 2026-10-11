"""Final evaluation of the frozen configuration: development period + locked holdout.

    python scripts/final_evaluation.py            # reads results/frozen_config.json

1. Re-runs each frozen model's walk-forward over 2006-01 .. 2020-11 and checks that
   the development-period scores reproduce the ones used for model selection.
2. Builds the ensemble and the frozen strategies, backtests them, and reports
   development (2006-2015) and holdout (2016-2020) separately.
3. Statistical tests: Probabilistic and Deflated Sharpe Ratio (trial count taken from
   the experiment logs), stationary-bootstrap Sharpe CIs, Newey-West alpha t-stats.
4. Robustness: cost sensitivity, capacity, return clipping (data-glitch sensitivity),
   execution at the next close instead of the next open, per-year results.
5. A paper-trading replay of the holdout through the trade ledger.
6. Figures in reports/figures/, numbers in results/final/.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela import analysis as A  # noqa: E402
from vela import metrics as M  # noqa: E402
from vela import plots  # noqa: E402
from vela.backtest.engine import CostModel, run_backtest  # noqa: E402
from vela.config import DATA_END, DEV_END, FIGURES_DIR, HOLDOUT_START, RESULTS_DIR  # noqa: E402
from vela.ensemble import rank_average  # noqa: E402
from vela.research import (ExperimentConfig, ResearchData, StrategySpec, build_weights,  # noqa: E402
                           load_scores, run_walkforward, save_scores)
from vela.tracker.ledger import performance_report  # noqa: E402
from vela.tracker.paper import PaperConfig, simulate  # noqa: E402

OUT = RESULTS_DIR / "final"
DEV = ("2006-01-01", DEV_END)
HOLD = (HOLDOUT_START, DATA_END)
FULL = ("2006-01-01", DATA_END)


def _cfg(d: dict) -> ExperimentConfig:
    names = {f.name for f in fields(ExperimentConfig)}
    return ExperimentConfig(**{k: v for k, v in d.items() if k in names})


def count_trials() -> tuple[int, np.ndarray]:
    """Number of development trials and their daily Sharpe ratios (for the DSR variance term)."""
    exp = pd.read_csv(RESULTS_DIR / "experiments.csv")
    exp = exp[exp["period"] == "dev"]
    port = pd.read_csv(RESULTS_DIR / "portfolio_trials.csv") if (RESULTS_DIR / "portfolio_trials.csv").exists() else pd.DataFrame()
    sh = []
    for col in ("ls_sharpe_net", "lo_sharpe_net"):
        if col in exp:
            sh += exp[col].dropna().tolist()
    if len(port):
        sh += port["sharpe_net"].dropna().tolist()
    sh = np.array(sh) / np.sqrt(252)
    return len(sh), sh


def strategy_report(name, res_full, data, horizon, scores, n_trials, trial_sr) -> dict:
    out = {}
    for label, (a, b) in {"dev": DEV, "holdout": HOLD, "full": FULL}.items():
        r = res_full.window(a, b)
        m = (data.dates >= pd.Timestamp(a)) & (data.dates <= pd.Timestamp(b))
        g, n = M.perf_stats(r.gross), M.perf_stats(r.net)
        row = {"gross_" + k: v for k, v in g.items() if k in ("sharpe", "ann_return", "cagr")}
        row.update({"net_" + k: v for k, v in n.items()})
        row["turnover_daily"] = float(r.turnover.mean())
        row["cost_bps_day"] = float((r.cost + r.borrow).mean() * 1e4)
        row["avg_long"], row["avg_short"] = float(r.n_long.mean()), float(r.n_short.mean())
        row["psr_vs_0"] = M.probabilistic_sharpe(r.net)
        row.update(M.stationary_bootstrap_sharpe(r.net, n_boot=1000))
        for bname in ("SPY", "EW"):
            rel = M.relative_stats(r.net, data.bench_R[bname].values[m])
            row.update({f"{bname}_{k}": v for k, v in rel.items()})
            row[f"{bname}_sharpe"] = M.sharpe(data.bench_R[bname].values[m])
            row[f"{bname}_cagr"] = M.cagr(np.nan_to_num(data.bench_R[bname].values[m]))
            row[f"{bname}_max_drawdown"] = M.max_drawdown(np.nan_to_num(data.bench_R[bname].values[m]))
        ic = M.daily_rank_ic(scores[m], np.where(data.tradable, data.fwd(horizon), np.nan)[m])
        row.update(M.ic_summary(ic, horizon))
        if label == "dev":
            # (a) cross-trial Sharpe variance from the trial log: harsh, because the log contains
            #     deliberately extreme trials (e.g. daily 1-day reversal at Sharpe -6);
            # (b) textbook null variance 1/T of a skill-less Sharpe estimate.
            row.update({f"dsr_{k}": v for k, v in M.deflated_sharpe(r.net, n_trials, trial_sr).items()})
            row["dsr_nullvar"] = M.deflated_sharpe(r.net, n_trials)["dsr"]
            row["dsr_nullvar_sr0_annual"] = M.deflated_sharpe(r.net, n_trials)["sr0_annual"]
        out[label] = row
    return out


def main() -> None:
    frozen = json.loads((RESULTS_DIR / "frozen_config.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    data = ResearchData.load(frozen.get("feature_set", "v2"))
    t0 = time.time()

    # ---- 1. models over the full period, with a reproducibility check on the dev years
    comp_scores, repro = [], {}
    for d in frozen["models"]:
        cfg = _cfg(d)
        dev_path = RESULTS_DIR.parent / "data" / "scores" / f"{cfg.name}.npy"
        dev_scores = load_scores(cfg.name) if dev_path.exists() else None
        cfg.last_test_year, cfg.last_test_date = 2020, DATA_END
        cached = RESULTS_DIR.parent / "data" / "scores" / f"final_{cfg.name}.npy"
        if cached.exists():
            s = load_scores(f"final_{cfg.name}")
        else:
            s, _ = run_walkforward(cfg, data)
            save_scores(f"final_{cfg.name}", s)
        m = (data.dates >= pd.Timestamp(DEV[0])) & (data.dates <= pd.Timestamp(DEV[1]))
        if dev_scores is not None:
            a, b = s[m], dev_scores[m]
            ok = np.isfinite(a) & np.isfinite(b)
            repro[cfg.name] = {"max_abs_diff_dev": float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else None,
                               "rank_corr_dev": float(pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman"))}
        else:
            repro[cfg.name] = "no development-period run to compare (seed added at freeze)"
        comp_scores.append(s)
    scores = rank_average(comp_scores, data.tradable, frozen.get("ensemble_weights")) if len(comp_scores) > 1 else comp_scores[0]
    horizon = frozen["models"][0]["horizon"]
    n_trials, trial_sr = count_trials()
    costs = CostModel(**frozen.get("cost_model", {}))
    summary = {"frozen": frozen, "reproducibility": repro, "n_trials": n_trials, "strategies": {}}

    # ---- 2-4. strategies
    bench = {k: pd.Series(data.bench_R[k].values, index=data.dates) for k in ("SPY", "EW")}
    for sname, sd in frozen["strategies"].items():
        spec = StrategySpec(**sd)
        W, reb = build_weights(scores, data, spec)
        res = run_backtest(W, data.R, data.dates, costs, sigma=data.sigma, adv=data.adv, rebalance=reb)
        rep = strategy_report(sname, res, data, horizon, scores, n_trials, trial_sr)

        # robustness
        sens = A.cost_sensitivity(scores, data, spec, *FULL, base=costs)
        cap = A.capacity(scores, data, spec, *FULL, base=costs)
        clipped = run_backtest(W, np.clip(data.R, -0.25, 0.25), data.dates, costs, sigma=data.sigma, adv=data.adv,
                               rebalance=reb).window(*FULL)
        Rc = np.full_like(data.R, np.nan)
        Rc[:-1] = data.panel.close[1:] / data.panel.close[:-1] - 1          # next-close execution
        at_close = run_backtest(W, Rc, data.dates, costs, sigma=data.sigma, adv=data.adv, rebalance=reb).window(*FULL)
        rep["robustness"] = {
            "break_even_cost_multiplier": A.break_even_multiplier(sens),
            "break_even_half_spread_bps": A.break_even_multiplier(sens) * costs.half_spread_bps,
            "sharpe_returns_clipped_25pct": M.sharpe(clipped.net),
            "sharpe_execute_next_close": M.sharpe(at_close.net),
        }
        sens.to_csv(OUT / f"{sname}_cost_sensitivity.csv", index=False)
        cap.to_csv(OUT / f"{sname}_capacity.csv", index=False)
        net = res.window(*FULL).series("net")
        yt = A.yearly_table(net, {k: v.loc[net.index] for k, v in bench.items()})
        yt.to_csv(OUT / f"{sname}_yearly.csv")
        res.window(*FULL).frame().to_csv(OUT / f"{sname}_daily.csv")
        summary["strategies"][sname] = rep

        # figures
        title = frozen.get("titles", {}).get(sname, sname)
        series = {title: net, "SPY (total return)": bench["SPY"].loc[net.index], "EW universe": bench["EW"].loc[net.index]}
        colors = {title: plots.ROLE["strategy"], "SPY (total return)": plots.ROLE["SPY"], "EW universe": plots.ROLE["EW"]}
        if sname.startswith("ls"):
            series = {title: net, "Same book, gross of costs": res.window(*FULL).series("gross")}
            colors = {title: plots.ROLE["strategy"], "Same book, gross of costs": plots.ROLE["alt"]}
        plots.equity_curves(series, FIGURES_DIR / f"{sname}_equity.png",
                            f"{title}: net of costs, 2006-2020 (shaded: locked holdout)", colors, shade=HOLD)
        plots.drawdowns(series, FIGURES_DIR / f"{sname}_drawdown.png", f"{title}: drawdowns", colors)
        plots.annual_returns(series, FIGURES_DIR / f"{sname}_annual.png", f"{title}: calendar-year returns", colors)
        plots.line_xy(sens["half_spread_bps"], {"net Sharpe": sens["sharpe"].values}, FIGURES_DIR / f"{sname}_costs.png",
                      f"{title}: cost sensitivity (2006-2020)", "half-spread (bps); commission, impact and borrow scaled alike",
                      "Net Sharpe", vline=costs.half_spread_bps)
        plots.line_xy(np.log10(cap["aum"]), {"net Sharpe": cap["sharpe"].values}, FIGURES_DIR / f"{sname}_capacity.png",
                      f"{title}: capacity (square-root impact)", "log10 AUM ($)", "Net Sharpe", vline=np.log10(costs.aum))

        # paper-trading replay of the holdout
        if frozen.get("paper", {}).get(sname):
            pc = PaperConfig(**frozen["paper"][sname])
            ledger = simulate(W, reb if reb is not None else np.ones(len(W), bool), data.panel.open, data.dates,
                              data.panel.tickers, HOLD[0], HOLD[1], pc)
            px = pd.DataFrame(data.panel.open, index=data.dates, columns=data.panel.tickers).loc[HOLD[0]:HOLD[1]]
            val = ledger.valuation(px)
            spy_px = data.panel.benchmark_opens["SPY"].loc[HOLD[0]:HOLD[1]]
            prep = performance_report(val, spy_px)
            eng = res.window(*HOLD)
            prep["engine_twr_same_period"] = float(np.prod(1 + eng.net) - 1)
            prep["n_trades"] = int(len(ledger.trades))
            summary["strategies"][sname]["paper_holdout"] = prep
            ledger.to_csv(OUT / f"{sname}_paper_ledger")
            ledger.pnl_by_position(px).head(15).to_csv(OUT / f"{sname}_paper_top_positions.csv")

    # signal diagnostics
    dec = {}
    from vela.research import decile_returns
    for label, (a, b) in {"dev": DEV, "holdout": HOLD}.items():
        m = (data.dates >= pd.Timestamp(a)) & (data.dates <= pd.Timestamp(b))
        dec[label] = (decile_returns(scores[m], data.fwd(horizon)[m], data.tradable[m]) * 1e4).tolist()
    summary["decile_fwd_return_bps"] = dec
    summary["signal_decay_ic"] = A.signal_decay(scores, data, *FULL).to_dict()
    icy = A.ic_by_year(scores, data, horizon, *FULL)
    summary["ic_by_year"] = icy.to_dict()
    plots.bars([f"D{i}" for i in range(1, 11)], dec["holdout"], FIGURES_DIR / "deciles_holdout.png",
               f"Mean {horizon}-day forward return by score decile, holdout 2016-2020", "bps", fmt="{:.0f}")
    plots.bars([str(y) for y in icy.index], icy.values * 100, FIGURES_DIR / "ic_by_year.png",
               f"Mean daily rank IC of the ensemble score ({horizon}-day forward return), by year", "IC x 100",
               fmt="{:.1f}")
    summary["runtime_secs"] = round(time.time() - t0, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "strategies"}, indent=1, default=float)[:3000])
    for sname, rep in summary["strategies"].items():
        for label in ("dev", "holdout"):
            r = rep[label]
            print(f"{sname:>6} {label:<8} net Sharpe {r['net_sharpe']:.2f} (gross {r['gross_sharpe']:.2f}) "
                  f"CAGR {r['net_cagr']:.1%} maxDD {r['net_max_drawdown']:.1%} "
                  f"alpha vs SPY {r['SPY_alpha_ann']:.1%} (t {r['SPY_alpha_t']:.1f}) IR(EW) {r['EW_information_ratio']:.2f} "
                  f"IC {r['ic_mean']:.3f}")
        print("   robustness:", rep["robustness"])
        if "dsr_dsr" in rep["dev"]:
            print("   DSR (dev): trial-variance", round(rep["dev"]["dsr_dsr"], 3), "| null-variance",
                  round(rep["dev"]["dsr_nullvar"], 3), "| n_trials", n_trials)


if __name__ == "__main__":
    main()
