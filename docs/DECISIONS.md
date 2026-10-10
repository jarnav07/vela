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

## D8 — Feature set: v1 (53 price/volume features); statistical industries rejected
**Context.** v2 adds nine point-in-time statistical-industry features. The clusters are economically sensible
(energy, banks, utilities and pharma each emerge as one cluster).
**Evidence.** GBM IC (5d) on two seeds each: v1 0.032 / 0.030, v2 0.027 / 0.027. The drop repeats across
seeds, so it is not noise.
**Decision.** Keep v1. The industry information appears to be already available to the trees through
interactions of return and volatility features, and nine extra correlated columns dilute the GBM's random
feature subsampling.

## D9 — Model: histogram GBM, 5-day rank target, bagged over three seeds
**Evidence (dev 2006–2015).** Ridge IC 0.020, MLP 0.024, LSTM (see DEVLOG), **GBM 0.031** (mean of two seeds).
Ensembling the GBM with weaker model families *lowered* IC. The 21-day label has no significant IC.
The 3-day label has a sharper 1-day IC but less persistent scores and more turnover, and blending it in
improved net Sharpe by less than the seed-to-seed spread.
Re-running the same GBM with a different seed moved weekly L/S net Sharpe from 0.27 to 0.04. **Single-seed
comparisons are unreliable**, and bagging seeds is pure variance reduction (no selection), so the frozen
model is the rank-average of seeds {0, 1, 2}.

## D10 — Portfolio construction: hysteresis bands, 2-day rebalance
**Evidence.** The signal decays in ≈ 3 days (IC by holding day: 0.018, 0.016, 0.011, 0.003). Daily rebalancing
pays costs larger than the edge; weekly rebalancing gives up alpha. A sweep of exit bands × rebalance
frequency (32 trials, all logged) gives a smooth surface. Wider exit bands and 2-day rebalancing win in both
books, which suggests a real cost/decay trade-off rather than a lucky cell.
**Decision.**
- **Market-neutral L/S:** long the top 10% / short the bottom 10%; hold until the rank crosses the median
  (exit band 50%); rebalance every 2 days; positions sized ∝ √ADV within each leg; 100% long / 100% short.
- **Long-only:** buy the top 20%; hold until the rank falls below the 40th percentile (exit 60%); rebalance
  every 2 days; equal weight. This is the book a personal account can actually run.
Rejected: beta-neutral scaling (raises gross Sharpe but adds turnover, so net is lower), inverse-vol sizing,
score EMA smoothing (kills the fast signal).

## D11 — Base-case costs: 5 bps half-spread + 0.5 bps commission + square-root impact at $10M AUM + 50 bps/yr borrow
Deliberately on the conservative side for S&P 500 names after 2005, where quoted half-spreads were mostly
1–3 bps. Impact uses `Y = 1`, daily σ and 21-day median dollar volume. Results are also reported as cost
curves (break-even spread) and capacity curves (AUM $0.1M → $500M), so a reader can apply their own assumptions.

## D12 — Freeze
Configuration frozen in `results/frozen_config.json` before the holdout was run. The holdout
(2016-01-04 → 2020-11-10) is evaluated once by `scripts/final_evaluation.py`; any change after that point
would be reported as post-holdout.
