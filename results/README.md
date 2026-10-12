# results/: what each file is

Everything here is small, text-based and committed. Large regenerable artefacts (the cleaned panel, feature
caches, per-model score matrices, trained models) live in the git-ignored `data/` and `models/` folders and
are rebuilt by the scripts named below.

## Development-period trial logs (2006–2015)

| File | Written by | Contents |
|---|---|---|
| `experiments.csv` | `scripts/run_experiment.py` | One row per **model trial**: the full `ExperimentConfig` (model, params, horizon, target, feature set, seed, …) plus signal and default-backtest metrics: `ic_mean_h{1,h}`, `ic_t_h*`, `decile_spread_bps`, `ls_*` / `lo_*` Sharpe (gross/net), turnover, cost, IR vs SPY/EW, and `fold_secs` (training time). Rows logged before DEVLOG Phase 4 used the constant-mix engine for their `ls_*`/`lo_*` columns. Their scores were re-evaluated later; see `portfolio_trials.csv` and `model_ladder.csv`. The `feature_set` column is empty for rows logged before it existed (all v1). |
| `portfolio_trials.csv` | `scripts/evaluate_scores.py`, `scripts/model_ladder.py`, ad-hoc sweeps recorded in DEVLOG | One row per **(scores, portfolio construction)** evaluation: Sharpe gross/net, return, vol, drawdown, turnover, cost (bps/day), average names long/short, and for long-only books alpha/t/IR/beta vs SPY and EW. |
| `model_ladder.csv` | `scripts/model_ladder.py` | Every model under the **frozen** construction and costs: the like-for-like comparison table in RESULTS §2. |
| `feature_importance.csv` | `scripts/feature_importance.py` | IC lost when each feature family is shuffled (2014–15). |
| `data_quality.json` | `scripts/build_dataset.py` | Date range, universe size per year, membership symbols without price data, dropped corrupt tickers and their glitch counts, single glitch events kept. |

The Deflated Sharpe Ratio's trial count is the number of net-Sharpe values across both logs: two per
`experiments.csv` row (L/S and long-only) plus one per `portfolio_trials.csv` row. It was 136 when the final
evaluation ran (`scripts/final_evaluation.py::count_trials`).

## The frozen configuration and the holdout

| File | Contents |
|---|---|
| `frozen_config.json` | The configuration fixed before the holdout: model configs (three GBM seeds), both strategy specs, cost model, paper-account settings. Committed in "Freeze strategy configuration before holdout evaluation". |
| `final/summary.json` | Everything `scripts/final_evaluation.py` computed. Per book (`ls`, `lo`) and period (`dev`, `holdout`, `full`): gross/net performance, turnover, cost, PSR, bootstrap Sharpe CI, CAPM stats vs SPY and EW (alpha, Newey-West t, beta, IR), IC summary, DSR (dev); robustness (break-even cost, clipped returns, next-close execution); paper-account report. Also reproducibility check, trial count, decile returns, signal decay, IC by year. |
| `final/{ls,lo}_daily.csv` | Daily gross/net return, cost, borrow, turnover, long/short exposure and name counts, 2006–2020. Returns are labelled by the *start* of the holding interval. |
| `final/{ls,lo}_yearly.csv` | Calendar-year return, Sharpe and max DD, with SPY and EW returns. |
| `final/{ls,lo}_cost_sensitivity.csv` | Net results with all costs scaled ×0 … ×3. |
| `final/{ls,lo}_capacity.csv` | Net results at AUM $0.1M … $500M (square-root impact). |
| `final/ls_regime_analysis.csv` | Post-holdout diagnosis: L/S returns by ex-ante market-volatility tercile (RESULTS §8). |
| `final/lo_paper_ledger/` | The holdout paper-trading account: `trades.csv.gz` (134,103 fills: date, ticker, quantity, price incl. spread, fees) and `cashflows.csv`. Replay it with `python scripts/track_portfolio.py --demo`. |
| `final/lo_paper_top_positions.csv` | Average-cost P&L of the 15 best positions in the paper account. |

## Logs

`logs/` holds the console output of every long run: experiment batches, final scoring, final evaluation,
and the figure-only rerun whose `summary.json` matched the first run exactly.
