# VELA: ML equity signals, vectorised backtesting and portfolio tracking

VELA is a from-scratch research stack for **short-horizon cross-sectional equity prediction** on the
S&P 500. It covers:

1. **Data.** 20.9 years of daily OHLCV (2000-01 → 2020-11) with **point-in-time S&P 500 membership**
   and a documented data-cleaning pass.
2. **Features.** 53 causal features (reversal, momentum, volatility, liquidity, technicals, market regime),
   plus 9 point-in-time *statistical-industry* features learned from returns, which were tested and rejected.
3. **Models.** A ladder from heuristics to Ridge, gradient-boosted trees, a PyTorch MLP and a PyTorch LSTM,
   trained in a **purged annual walk-forward** with time-ordered early stopping.
4. **Backtesting.** A **vectorised NumPy engine** with next-open execution, drift-aware turnover,
   commission + spread + square-root market impact + borrow fees, and buy-and-hold between rebalances.
5. **Evaluation.** Information coefficients, decile spreads, cost and capacity curves, bootstrap
   confidence intervals, and the **Deflated Sharpe Ratio** using the actual number of trials run.
6. **Portfolio tracking.** A trade ledger with NAV, P&L, time-weighted return, drawdown, Sharpe and
   alpha/IR against a benchmark, an order generator for a small account, and an event-driven
   paper-trading simulator that independently cross-checks the vectorised engine.

Results are in [`docs/RESULTS.md`](docs/RESULTS.md). The process is documented in
[`docs/PLAN.md`](docs/PLAN.md) (the original plan), [`docs/DECISIONS.md`](docs/DECISIONS.md)
(design decisions and the alternatives rejected) and [`docs/DEVLOG.md`](docs/DEVLOG.md)
(what went wrong and how it was fixed).

## Results at a glance

Out-of-sample, net of costs (5 bps half-spread + 0.5 bps commission + square-root impact at $10M + borrow).
Model selection used 2006–2015 only. 2016-01 → 2020-11 was a **locked holdout**, run once after the
configuration was committed.

| | Development 2006–15 | **Holdout 2016–20** | Full 2006–20 |
|---|---|---|---|
| Signal: daily rank IC, 5-day (t) | 0.032 (4.2) | **0.016 (1.3)** | 0.027 (4.2) |
| **Long-only** ML book: net Sharpe / CAGR | 0.59 / 13.0% | **0.75 / 14.3%** | 0.63 / 13.4% |
| SPY (total return): Sharpe / CAGR | 0.44 / 7.0% | 0.87 / 14.5% | 0.56 / 9.4% |
| Equal-weight universe: Sharpe / CAGR | 0.58 / 11.3% | 0.72 / 12.8% | 0.62 / 11.8% |
| Long-only information ratio vs EW | +0.37 | **+0.27** | +0.34 |
| **Dollar-neutral** L/S book: net (gross) Sharpe | 0.61 (1.44) | **0.13 (0.86)** | 0.45 (1.25) |

- The GBM signal is 1.8x as strong as the best single-factor baseline, and stays positive (if halved)
  in the holdout.
- **$100k paper-trading replay of the holdout:** +108% vs +93% for SPY.
- The dollar-neutral long/short book's gross edge is real (Sharpe 1.25), but its profits concentrate in volatile markets,
  and at institutional size costs absorb most of them. Capacity: net Sharpe 0.78 at $100k → 0.45 at $10M → 0 at $50M.

![Long-only vs benchmarks](reports/figures/lo_equity.png)

Full write-up: [docs/RESULTS.md](docs/RESULTS.md) · CV wording: [docs/CV.md](docs/CV.md)


## Repository layout

```
vela/
  data/          Qlib binary reader, aligned panel, data-quality cleaning
  universe.py    point-in-time tradable universe (membership + liquidity + history)
  features.py    53 price/volume features, cross-sectionally ranked
  industries.py  point-in-time statistical industries (PCA + k-means on residual returns)
  targets.py     execution-aligned forward returns and target transforms
  walkforward.py purged annual walk-forward splits
  models/        Ridge, HistGBM, random forest (sklearn); MLP, LSTM (PyTorch)
  ensemble.py    rank-average ensembles
  portfolio.py   long/short, long-only, hysteresis bands, beta-neutralising, sizing
  backtest/      vectorised engine + cost model
  metrics.py     performance, relative, PSR/DSR, bootstrap, IC
  analysis.py    cost sensitivity, capacity, decay, per-year tables
  research.py    pipeline: data bundle → walk-forward → evaluation → trial log
  tracker/       ledger, order generation, paper-trading simulator
  plots.py       report figures
scripts/         fetch_data.sh, build_dataset.py, run_experiment.py, evaluate_scores.py,
                 final_evaluation.py, track_portfolio.py, sync_gcs.sh
tests/           35 tests: look-ahead canary, feature causality, purge, cost maths, engine vs ledger
results/         experiment and portfolio trial logs, final metrics
reports/figures  charts used in docs/RESULTS.md
```

## Reproducing

```bash
pip install -e ".[dev]"
scripts/fetch_data.sh                       # 450 MB Qlib US archive from GitHub releases
python scripts/build_dataset.py             # cleaned panel -> data/panel.npz
python scripts/run_experiment.py baselines ml_v1 ml_v2 seeds lstm   # development trials (2006-2015)
python scripts/evaluate_scores.py gbm_h5 --sweep                    # portfolio-construction sweep
python scripts/final_evaluation.py          # frozen config: dev + locked holdout 2016-2020
pytest                                      # test suite
```

Tracking a real or paper account:

```bash
python scripts/track_portfolio.py --trades my_trades.csv --cashflows my_cash.csv --prices prices.csv
```

## Key design choices (details in docs/DECISIONS.md)

- **Cross-sectional ranking, not market timing.** Predict which stocks beat other stocks over the next
  5 days. This removes most market beta and gives ~400 labelled samples per day.
- **No look-ahead by construction.** Signals at close *t* trade at the open of *t+1*. Labels start at
  that open. Training windows are purged by the label horizon. Tests delete all data after a date and
  check that features do not change.
- **A locked holdout.** Every modelling choice was made on 2006–2015. 2016-01 → 2020-11 was run once,
  with a frozen configuration.
- **Costs from day one.** 5 bps half-spread + 0.5 bps commission + square-root impact (σ·√participation)
  at $10M AUM + 50 bps/yr borrow. Results are also shown as cost and capacity curves.

## Limitations

- Data ends 2020-11 (the only full-history source reachable from the build environment); live use needs
  a current price feed, for which the tracker is source-agnostic.
- Residual survivorship bias: 86 of 755 historical index members (mostly acquired or bankrupt companies)
  have no price data. The equal-weighted-universe benchmark shares this bias, so strategy-vs-EW
  comparisons are fair; SPY comparisons flatter long-only results somewhat.
- Costs are modelled, not measured; no intraday data was available to calibrate the spread.
- Sharpe ratios use a zero risk-free rate (no T-bill series was reachable).
