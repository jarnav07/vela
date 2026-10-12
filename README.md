# VELA - ML equity signals, vectorised backtesting and portfolio tracking

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


## Documentation

| Document | What it covers |
|---|---|
| [docs/PLAN.md](docs/PLAN.md) | The plan written before any modelling, plus what changed by the end |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Design decisions D1–D12: context, alternatives rejected, evidence |
| [docs/DEVLOG.md](docs/DEVLOG.md) | Chronological log of every phase, problem, bug and fix (Phases 0–13) |
| [docs/RESULTS.md](docs/RESULTS.md) | Full evaluation: model ladder, both books, significance, robustness, diagnosis, limitations |
| [docs/FEATURES.md](docs/FEATURES.md) | Definition of every feature |
| [docs/CV.md](docs/CV.md) | CV wording and interview talking points, every number traceable to `results/` |
| [results/README.md](results/README.md) | What each results file and column contains |

## Repository layout

```
vela/
  config.py      paths and the research calendar (dev 2006-2015, holdout 2016-2020, fixed up front)
  data/          qlib_source.py (Qlib binary reader), panel.py (aligned T x N panel),
                 quality.py (cleaning), live.py (Yahoo Finance adapter for live use)
  universe.py    point-in-time tradable universe (membership + liquidity + history)
  features.py    53 price/volume features, cross-sectionally ranked (docs/FEATURES.md)
  industries.py  point-in-time statistical industries (PCA + k-means on residual returns)
  targets.py     execution-aligned forward returns and target transforms
  walkforward.py purged annual walk-forward splits
  models/        base.py (interface), sklearn_models.py (Ridge, HistGBM, random forest),
                 torch_models.py (MLP), sequence.py (LSTM)
  experiments.py named experiment presets, so every trial is reproducible and countable
  ensemble.py    rank-average ensembles
  portfolio.py   long/short, long-only, hysteresis bands, beta-neutralising, sizing
  backtest/      engine.py: vectorised engine + cost model
  metrics.py     performance, relative (Newey-West), PSR/DSR, bootstrap, IC
  analysis.py    cost sensitivity, capacity, decay, per-year tables, permutation importance
  research.py    pipeline: data bundle -> walk-forward -> evaluation -> trial log
  tracker/       ledger.py (NAV, P&L, TWR, report), orders.py (order generation),
                 paper.py (event-driven paper-trading simulator)
  plots.py       report figures
scripts/         see the table below
tests/           35 tests: look-ahead canary, feature causality, purge, cost maths, engine vs ledger
results/         trial logs, frozen config, final metrics (results/README.md)
reports/         figures/ (charts used in the docs), paper_account/ (tracker demo output)
```

| Script | Purpose |
|---|---|
| `scripts/fetch_data.sh` | Download and unpack the Qlib US dataset (450 MB) into `data/` |
| `scripts/build_dataset.py` | Build and clean the research panel -> `data/panel.npz`, `results/data_quality.json` |
| `scripts/run_experiment.py <preset> …` | Run walk-forward trials on the development period and log them |
| `scripts/evaluate_scores.py <scores> [--sweep]` | Re-score cached model outputs under different portfolio constructions |
| `scripts/model_ladder.py` | Compare every model under the frozen construction (RESULTS §2) |
| `scripts/feature_importance.py` | Grouped permutation importance by feature family |
| `scripts/final_scores.py` | Train the frozen models over 2006–2020 (scores only, no evaluation) |
| `scripts/final_evaluation.py` | The one-time dev + holdout evaluation, robustness checks and figures |
| `scripts/live_signals.py train / signal` | Fit the frozen model on all history; generate orders for your account from current data |
| `scripts/track_portfolio.py` | Track a real or paper account (broker CSVs) against a benchmark |
| `scripts/sync_gcs.sh push / pull` | Mirror `data/` to a Google Cloud Storage bucket (optional) |

## Reproducing the research

```bash
pip install -e ".[dev]"            # on a CPU-only machine, install torch from download.pytorch.org/whl/cpu first
scripts/fetch_data.sh              # 450 MB Qlib US archive from GitHub releases
python scripts/build_dataset.py    # cleaned panel -> data/panel.npz (feature caches build on first use)

# development trials, 2006-2015 (presets in vela/experiments.py; ~3 h on 4 CPU cores)
python scripts/run_experiment.py baselines ml_v1 ml_v2 seeds horizons lstm
python scripts/evaluate_scores.py base_rev5_h5 ridge_h5 gbm_h5 --sweep
python scripts/model_ladder.py
python scripts/feature_importance.py

# the frozen configuration (results/frozen_config.json) on dev + the locked 2016-2020 holdout
python scripts/final_scores.py     # ~45 min
python scripts/final_evaluation.py # tables -> results/final/, figures -> reports/figures/
pytest                             # 35 tests, synthetic data only
```

Some sweeps recorded in the DEVLOG (hysteresis bands × rebalance frequency, sizing schemes, seed bags) were
run as short ad-hoc scripts that call the same `evaluate_spec` function. Their rows are in
`results/portfolio_trials.csv`.

## Using it on a real or paper account

```bash
pip install -e ".[live]"                               # adds yfinance
python scripts/live_signals.py train                   # fit the 3 frozen GBM seeds on all history -> models/
python scripts/live_signals.py signal --holdings holdings.csv --cash 2500 --book lo
#   holdings.csv: ticker,quantity  -> prints orders for the long-only book, relative to what you hold
python scripts/track_portfolio.py --trades trades.csv --cashflows cashflows.csv --prices prices.csv --benchmark SPY
#   -> NAV, P&L by position, time-weighted return, Sharpe, max drawdown, alpha/IR vs SPY + charts
python scripts/track_portfolio.py --demo               # same report for the 2016-2020 paper-trading replay
```

`signal` downloads today's S&P 500 constituents and two years of prices from Yahoo Finance. It was tested
offline (DEVLOG Phase 13) because the build environment could not reach Yahoo. Check the first orders by hand,
and set a minimum order size in your broker: the book holds ~110–210 names (≈180 in 2016–2020) and re-equalises them every two days.

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

- Data ends 2020-11 (the only full-history source reachable from the build environment). Live use goes
  through `scripts/live_signals.py` with a current Yahoo Finance feed.
- Residual survivorship bias: 86 of 755 historical index members (mostly acquired or bankrupt companies)
  have no price data. The equal-weighted-universe benchmark shares this bias, so strategy-vs-EW
  comparisons are fair; SPY comparisons flatter long-only results somewhat.
- Costs are modelled, not measured; no intraday data was available to calibrate the spread.
- Sharpe ratios use a zero risk-free rate (no T-bill series was reachable).
- No licence file has been added; all rights are reserved by default until one is chosen.
