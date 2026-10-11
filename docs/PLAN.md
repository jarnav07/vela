# VELA — Project Plan

*Written at the start of the project (2026-10-10), before any modelling code existed.
Later deviations from this plan are recorded in [DEVLOG.md](DEVLOG.md) and [DECISIONS.md](DECISIONS.md);
this file is kept as the original plan, with a short "what changed" note at the bottom.*

## 1. Goal

Build, from scratch, a machine-learning equity trading strategy and the infrastructure to evaluate it honestly:

1. **Signal model.** Python ML models (scikit-learn and PyTorch) that forecast short-term equity price
   movements from historical market data. Iterate on model architecture and feature engineering.
2. **Backtester.** A vectorised NumPy backtesting engine with slippage and transaction-cost modelling,
   used to evaluate the strategy over 10+ years of history.
3. **Portfolio tracking.** Tooling to run the strategy on a real or paper account and track P&L,
   drawdown and risk-adjusted returns (Sharpe, max drawdown) against benchmark indices.

"Works well" means a strategy whose out-of-sample, **after-cost** performance holds up under
statistical scrutiny, not the best-looking in-sample equity curve. The project is set up so that
it is hard to fool ourselves.

## 2. Constraints discovered during planning

| Constraint | Impact | Response |
|---|---|---|
| The sandbox's network policy blocks Yahoo Finance, Stooq, Kaggle, FRED, Wikipedia, HuggingFace | No `yfinance` download | Use Microsoft Qlib's public US daily dataset (Yahoo-sourced), distributed as a GitHub release asset, which *is* reachable. Keep the loader pluggable so `yfinance` can be used where it is reachable. |
| Qlib US data ends **2020-11-10** | No 2021–2026 data for research | 2000-01 → 2020-11 is ~20.9 years, comfortably "10+ years". Live/paper-trading tooling is written against a generic price source. |
| No GPU; 4 CPU cores, 15 GB RAM | Deep models must be small | Prefer compact PyTorch models (MLP, small LSTM). Use sklearn's histogram GBM for trees. |
| No risk-free rate series reachable | Sharpe uses rf = 0 | Stated wherever Sharpe is reported. Long-only results are also judged *relative* to SPY (alpha, information ratio), where rf largely cancels. |

## 3. Research design

### 3.1 Universe and data
- **Point-in-time S&P 500 membership** (Qlib's `sp500.txt` has entry and exit dates per constituent).
  A stock can be traded on day *t* only if it was an index member on day *t*. This removes most of the
  survivorship bias that comes from using today's constituents.
- Daily adjusted OHLCV from Qlib. Data-quality pass: bad prints, zero volume, stale prices, gaps.
- Benchmarks: **SPY** (total return, includes dividends), the **equal-weighted universe** (same
  survivorship profile as the strategy, so the comparison is fair), and `^GSPC` (price only, for reference).

### 3.2 Problem framing
- **Cross-sectional ranking.** Predict which stocks will outperform *other stocks* over the next *h*
  days, not whether the market will go up. This removes most market beta from the problem,
  gives ~500 labelled samples per day, and is the standard framing in the literature
  (Krauss, Do & Huck 2017; Gu, Kelly & Xiu 2020).
- Target: forward *h*-day open-to-open return, cross-sectionally demeaned and rank-transformed.
  Candidate horizons are 1 and 5 days (and 21 as a check); the horizon is chosen on the development period.

### 3.3 Features (all computable at the close of day *t* from data up to *t*)
- Returns over 1, 2, 3, 5, 10, 21, 63, 126, 252 days; 12-1 momentum; individual lagged daily returns.
- Overnight vs intraday return decomposition; close location in the daily range.
- Volatility (5/21/63-day), Parkinson high–low volatility, volatility ratios, idiosyncratic volatility, beta.
- Liquidity: log dollar volume, volume shock (5d/63d), Amihud illiquidity.
- Technical: RSI(14), distance from 10/50/200-day moving averages, Bollinger z-score, MACD.
- Market context: index return and volatility, cross-sectional dispersion.
- Stock-level features are **cross-sectionally ranked** each day (robust to outliers and regime shifts).

### 3.4 Models (the ladder, simplest first)
1. Heuristic baselines: short-term reversal, 12-1 momentum, random signal (sanity check).
2. Linear: Ridge regression.
3. Trees: scikit-learn `HistGradientBoostingRegressor` (LightGBM-style).
4. PyTorch MLP (tabular features).
5. PyTorch LSTM on raw return sequences (Fischer & Krauss 2018 style).
6. Rank-average ensemble of the models that earn their place.

A model only moves up the ladder if it beats the simpler one **out of sample, after costs**.

### 3.5 Validation: walk-forward with purging, plus a locked holdout
- Retrain once a year on an expanding (or rolling) window and predict the following year.
- **Purge** the last *h* days before each test year, so training labels never overlap the test
  period (López de Prado 2018). Early stopping uses a time-ordered validation slice inside the training window.
- **Development period (model selection):** walk-forward out-of-sample 2006-01 → 2015-12 (10 years).
- **Locked holdout:** 2016-01 → 2020-11 (~5 years). Touched **once**, at the end, with a frozen
  configuration. Every development experiment is logged so the number of trials is known for the
  Deflated Sharpe Ratio.

### 3.6 Portfolio construction
- Long-short, dollar-neutral: long the top quantile, short the bottom quantile, equal weighted.
- Long-only top-quantile variant (what a personal account can actually trade), compared with SPY and the EW universe.
- Turnover control: rebalance every *k* days and/or hold positions until they leave a wider rank band (hysteresis).

### 3.7 Backtesting engine (vectorised NumPy)
- Input: a target-weights matrix **W** (*T* × *N*) decided at the close of each day.
- Execution: at the **next day's open** (realistic; avoids trading on the close used to compute the signal).
  Holding returns are open-to-open.
- Turnover is measured against **drifted** weights, so passive drift between rebalances is costed correctly.
- Cost model per trade: commission + half bid-ask spread (estimated from daily high/low/close data,
  with a floor) + square-root market impact `k·σ·sqrt(trade$/ADV$)`, plus an annual borrow fee on shorts.
- Delistings: a position whose price disappears is closed (and costed) at the last available price.
- Output: daily gross and net returns, costs, turnover, exposures, holdings count.
- **Tests:** hand-computed cost examples, a look-ahead canary (a signal built from future returns must
  look spectacular, and the same signal lagged correctly must not), random signals ≈ zero edge minus costs.

### 3.8 Evaluation
- Performance: CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, hit rate, turnover, cost drag.
- Signal quality: daily rank IC (Spearman), IC t-stat, decile return spreads (monotonicity).
- Robustness: per-year results, cost-sensitivity curve (break-even cost), bootstrap Sharpe CIs,
  Probabilistic and **Deflated** Sharpe Ratio (Bailey & López de Prado 2014), market beta.

### 3.9 Portfolio tracker
- Ledger of trades and cash flows → daily NAV, realised and unrealised P&L, time-weighted returns,
  drawdown, Sharpe, Sortino, beta, alpha, tracking error and information ratio vs a benchmark.
- An order generator that turns the latest model scores into share orders for a small account.
- Demonstrated on a simulated paper account over the holdout period. Real-capital use needs the
  user's broker exports.

## 4. Deliverables
- `vela/` package, `scripts/` entry points, `tests/` (pytest).
- `docs/DECISIONS.md` (decision log), `docs/DEVLOG.md` (chronological journal of problems and fixes),
  `docs/RESULTS.md` (final evaluation), `docs/CV.md` (CV wording backed by the numbers).
- `results/experiments.csv`: every development trial with its configuration and metrics.

## 5. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Look-ahead bias | Execution lag in the engine; look-ahead canary tests; features use only data ≤ *t* |
| Overfitting through many trials | Locked holdout, trial log, Deflated Sharpe Ratio |
| Survivorship bias | Point-in-time membership; residual bias documented; EW-universe benchmark shares it |
| Costs eat short-horizon alpha (the main known failure mode of daily ML strategies) | Realistic costs from day one; turnover controls; break-even cost analysis |
| Data errors (bad prints) | Quality filters; winsorised targets; rank features |

---

## What changed from this plan (written at the end)

- **Data:** as planned (Qlib, PIT S&P 500), plus a stricter cleaning pass than anticipated: 8 corrupt tickers (D3).
- **Horizon:** 5 days. The 21-day label had no significant IC; the 3-day label churned more.
- **Models:** the full ladder was run (baselines → Ridge → GBM → MLP → LSTM). The ensemble across model
  families was dropped because it hurt; the final model is a three-seed GBM bag.
- **Features:** a planned extension, statistical industries, was built and rejected on evidence (D8).
- **Portfolio construction:** hysteresis bands with a 2-day cadence and √ADV sizing, chosen after measuring
  the signal's ~3-day decay. Beta-neutralisation and score smoothing were tried and rejected.
- **Engine:** gained a buy-and-hold-between-rebalances mode after a semantic bug was found (DEVLOG Phase 4),
  and an independent event-driven ledger for validation.
- **Evaluation:** as planned, plus seed-noise measurement, feature-family importance, a regime diagnosis
  and capacity curves.
- **Live/real-capital tooling:** a live-signal script using yfinance (not testable from the sandbox,
  mock-tested) and a ledger-based tracker CLI.
