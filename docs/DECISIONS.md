# Decision Log

Each entry records a design decision: the context, the options considered, what was chosen and why.
Entries are added as decisions are made; a superseded decision is marked as such, not deleted.

---

## D1 — Data source: Qlib's public US daily dataset (Yahoo-sourced), 2000-01 → 2020-11
**Context.** The development sandbox's egress policy blocks Yahoo Finance, Stooq, Kaggle, FRED,
Wikipedia and HuggingFace. GitHub and PyPI are reachable.
**Options.** (a) `yfinance`: blocked. (b) Kaggle S&P 500 dumps: blocked, and current-constituent only,
so survivorship-biased. (c) GitHub-hosted CSV repos: the ones found were notebooks without data,
or 38-ticker toy sets. (d) Microsoft Qlib's US dataset, published as a GitHub release asset.
**Decision.** (d). It provides 20.9 years of adjusted daily OHLCV for ~9,000 US symbols, a
**point-in-time S&P 500 membership file** (entry and exit dates), and SPY / `^GSPC` for benchmarks.
**Consequences.** Data stops at 2020-11-10, so the research period is 2000–2020. The loader is
pluggable (`vela/data/`), and the live/paper-trading tools take any price source.
Prices are rebased so each series starts at 1.0; `close × volume` is still true dollar volume
(checked: AAPL on 2020-11-10 gives ≈ $16bn, which matches reality).

## D2 — Universe: point-in-time S&P 500 constituents with a liquidity floor
**Context.** Training on today's constituents builds in survivorship bias: it trains only on companies
that went on to succeed.
**Decision.** A stock is tradable on day *t* only if it was an S&P 500 member on *t*, has a valid
open and close, has a 21-day median dollar volume ≥ $5M, and has ≥ 126 days of history.
**Residual bias (documented, not fixed).** 86 of the 755 membership records have no price data in
the source, mostly companies that were acquired or went bankrupt (e.g. YHOO, CELG, LEH). These are
missing from both the strategy and the equal-weighted universe benchmark, so the strategy-vs-EW
comparison shares the bias. The SPY comparison does not, because SPY's returns are real.

## D3 — Data cleaning: drop systematically corrupt tickers, keep real crashes
**Context.** Inspection found ticker-reuse and adjustment corruption: CBE alternating −98% / +4,000%
moves, TIE ×1,900 jumps, HNZ alternating −66% / +194%, CHK compounding fake quarterly jumps.
**Options.** (1) Volatility-scaled outlier filter: rejected. It cannot separate genuine shocks
(MRK −27% on the Vioxx recall, KHC −27%) from adjustment glitches (DHR +52%) because both are ~30–60σ.
(2) Neutralise every move with |log r| > log 3: rejected after it flagged AIG's genuine −67% bailout gap.
(3) Drop a ticker only when it has **≥ 2 moves beyond ×3 or ÷3 inside its membership window**.
**Decision.** (3), plus OHLC consistency repair. 8 tickers dropped (CBE, TIE, CHK, HNZ, BMC, MEE, GR, TRB).
Mid-sized glitches that cannot be told apart from real events are left in. A robustness test
(returns clipped at ±25%) checks that results do not depend on extreme days.

## D4 — Framing: cross-sectional ranking, not market timing
**Context.** "Forecast short-term price movements" can mean predicting each stock's direction or
predicting relative performance.
**Decision.** Predict the **cross-sectional rank** of forward returns. Per-stock direction is dominated
by the market factor, which is close to unpredictable at daily horizons and would make the strategy a
leveraged market bet. Ranking turns ~400 stocks a day into ~400 comparable labelled samples, and
the long-short portfolio that follows has little market beta. This matches the framing of
Krauss, Do & Huck (2017) and Gu, Kelly & Xiu (2020).

## D5 — Validation: annual walk-forward, purged, with a locked holdout
**Decision.** Fixed in `vela/config.py` before any model was trained:
- Walk-forward: retrain each January on all data before it (minus a purge gap equal to the label horizon),
  predict that calendar year.
- Development out-of-sample period 2006–2015 (10 years) is used for every modelling choice.
- **Holdout 2016-01-01 → 2020-11-10** stays untouched until the configuration is frozen, then is run once.
- Every development run is appended to `results/experiments.csv`, so the number of trials feeding the
  Deflated Sharpe Ratio is known.

## D6 — PyTorch installed from PyPI's CUDA wheel, run on CPU
The CPU-only wheel index (download.pytorch.org) is blocked; PyPI's Linux wheel needs the NVIDIA runtime
libraries to import even with no GPU. Installed with its CUDA dependencies (~3 GB) and run on CPU.
Models are sized for 4 cores.

## D7 — Risk-free rate = 0 in Sharpe ratios
No T-bill series is reachable from the sandbox (FRED is blocked). Sharpe ratios use rf = 0 and are labelled
as such. The dollar-neutral long-short book is self-financing, so its return is already an excess return.
Long-only variants are judged by alpha and information ratio against SPY, where rf largely cancels.
