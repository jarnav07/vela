# Development Log

A chronological record of what was done, what went wrong and how it was fixed.
Design decisions are summarised in [DECISIONS.md](DECISIONS.md) (referenced as D*n*).

---

## Phase 0 — Environment and data access

**Problem: no market data access.** Yahoo Finance, Stooq, Kaggle, FRED, Wikipedia and HuggingFace all return
`403` from the sandbox's egress proxy. GitHub (git, raw content, release assets) and PyPI work.

- Searched for GitHub-hosted equity datasets. `AlanWangyl/NASDAQ_Historical_Daily_Prices_Dataset`
  turned out to hold only the download notebook (its data lives on Kaggle).
- Microsoft Qlib's US dataset is distributed as a GitHub **release asset** (`qlib_data_us_1d_latest.zip`,
  450 MB), which downloads fine. It covers 2000-01-03 → 2020-11-10 for ~9,000 symbols, plus
  `instruments/sp500.txt` with point-in-time membership intervals. **Adopted (D1).**
- All release tags (v0–v3) carry the same archive, so there is no newer US data from this source.

**Problem: PyTorch would not import.** PyPI's Linux wheel is a CUDA build and fails with
`libcudart.so.13: cannot open shared object file` when installed `--no-deps`. The CPU wheel index is blocked.
Fix: install with its NVIDIA runtime dependencies. It then runs on CPU (`torch.cuda.is_available() == False`) (D6).

**Note: GCS bucket.** The user pointed to the AERIS Google Cloud Storage bucket for large files. The bucket
(`gs://aeris-datasets-ml-aerofoil`) was found via the AERIS repo's `CLAUDE.md`, but the sandbox's GCS
credentials are rejected (`401 Invalid Credentials`). The cleaned dataset is ~150 MB and fully
reproducible from the public archive, so nothing depends on the bucket. `scripts/sync_gcs.sh` is
provided for when credentials are available.

## Phase 1 — Data layer and data quality

Wrote `vela/data/qlib_source.py`, a reader for Qlib's binary format: a float32 array whose first element is
the calendar offset. Symbol folders are lower-cased with an `_` prefix, and `BRK.B` becomes `_brk-b`.
`vela/data/panel.py` builds aligned `T × N` arrays and the membership matrix.

**Checks on the raw data**
- AAPL's adjusted closes and dollar volumes reconcile with known values (2020-11-10: ≈ $16bn traded).
- `^GSPC` close × the 1999-12-31 level (1469.25) gives 3545 on 2020-11-10, which matches the real index.
- SPY total-return CAGR 2000–2020 is 6.3% against 4.3% for the price index. The ~2% gap is dividends,
  so SPY is the benchmark used.
- 86 of 755 membership records have no price data (acquired or bankrupt names such as YHOO, CELG, LEH).
  This is residual survivorship bias, documented in D2.

**Problem: corrupt series.** Scanning member-days for |return| > 50% found 582 events. Many were garbage:
- CBE alternated between two price scales (−98%, +4,000%, −98% …) for years: ticker reuse.
- TIE jumped ×1,900 and back; HNZ alternated −66% / +194%; CHK compounded ×2–×25 jumps every quarter
  (broken dividend adjustment); BMC +347% with no reversal (an adjustment level shift).
- CEG's pre-2012 history is a different, illiquid security (57k shares/day for an S&P 500 company).

Others were real and must stay: AAPL −52% (2000-09-29), WMB −61% (2002), AIG −61% (2008-09-15),
financials in late 2008.

**Iterations on the cleaning rule (D3)**
1. *Volatility-scaled outliers*: abandoned. MRK's real −27% (Vioxx) is 36σ while DHR's +52% adjustment
   glitch is 53σ, so no threshold separates them.
2. *Neutralise any |log r| > log 3*: rejected. It removed AIG's real −67% open-to-open gap (2008-09-16) and
   dropped ABK (Ambac), whose glitches all fall *after* it left the index, which would have removed a
   genuine collapse from the sample.
3. **Final rule**: count ×3 / ÷3 moves **only inside the membership window**, and drop tickers with ≥ 2.
   8 dropped (CBE 352 glitches, TIE 31, CHK 14, HNZ 4, BMC 3, MEE 3, GR 3, TRB 2). AIG and PWR keep their
   single real events.

A 21-day median dollar-volume floor of $5M (`vela/universe.py`) also removes junk series like CEG's
without hindsight. Tradable names per day: ~290 (2000) → ~440 (2015) → ~500 (2020).

## Phase 2 — Backtesting engine, metrics and tests

- `vela/backtest/engine.py`: fully vectorised (no Python loop over dates or assets). Weights decided at
  close *t* are executed at the open of *t+1* (`lag ≥ 1` is enforced) and earn open-to-open returns.
  Turnover is measured against **drifted** weights `H[s-1]·(1+r)/(1+R_p)`, so passive drift is costed.
- Cost model: commission + half-spread + square-root impact `Y·σ·√(trade$/ADV$)` + short borrow fee.
- **Spread estimation experiment (abandoned).** The Abdi–Ranaldo (2017) high/low/close estimator was
  computed for every stock-day. For S&P 500 names it is mostly noise: 36–52% of estimates are exactly zero
  and the interquartile range runs from 0 to 50–140 bps. Fixed instead: 5 bps half-spread + 0.5 bps
  commission. That is conservative for large caps after decimalisation and in line with the 5 bps
  per half-turn in Krauss et al. (2017). Square-root impact on top adds time variation: it widens when
  volatility spikes and for less liquid names.
- `vela/metrics.py`: CAGR, Sharpe, Sortino, max drawdown, Calmar, CAPM alpha/beta with Newey-West t-stats,
  information ratio, Probabilistic and Deflated Sharpe Ratios, stationary-bootstrap Sharpe CIs, rank IC.
- Tests (`tests/`): hand-computed costs, drift turnover, borrow and impact; a **look-ahead canary**
  (a peeking signal gives Sharpe > 20, an honest one gives ≈ 0); random signals lose about their costs;
  feature causality (features at date *D* are identical when all data after *D* is deleted);
  forward-label alignment; walk-forward purge for h = 1, 5, 21.

## Phase 3 — Features, targets, first baselines

53 features (`vela/features.py`): 46 stock-level, cross-sectionally ranked each day, and 7 market-level,
z-scored on trailing windows. The target is the forward open-to-open return from the next open
(the actual execution price), ranked cross-sectionally.

**Benchmarks, dev period 2006–2015:** SPY CAGR 7.0%, Sharpe 0.44, max DD −55%. EW universe CAGR 11.3%,
Sharpe 0.58. The EW universe beats the real equal-weight S&P ETF by ~2–3%/yr, which is residual
survivorship bias plus daily rebalancing. **Long-only strategies must therefore be judged against EW (same
bias), not only against SPY.**

**Heuristic baselines (dev 2006–2015, 5.5 bps + impact):**

| Signal | IC (1d) | IC t-stat | L/S gross Sharpe | L/S net Sharpe | Daily turnover |
|---|---|---|---|---|---|
| 1-day reversal, daily rebalance | 0.011 | 3.4 | 0.96 | **−6.08** | 342% |
| 5-day reversal, daily rebalance | 0.008 | 2.6 | 0.83 | −2.16 | 162% |
| 5-day reversal, 5-day rebalance | — | — | 0.71 | −0.75 | 72% |
| 12-1 momentum, 5-day rebalance | 0.012 | 1.9 | −0.13 | −0.37 | 16% |

The short-term reversal premium is real (t = 3.4) but **costs are larger than the edge**. Daily
reversal turns over ~3.4× NAV a day. This sets the bar for the ML work: a stronger signal *and*
turnover control are both needed. Momentum has a positive IC but a negative extreme-decile spread
in 2006–2015, driven by the 2009 momentum crash.

## Phase 4 — Engine correction: hold shares between rebalances

**Problem found while writing the paper-trading simulator.** Rebalancing every *k* days was implemented by
forward-filling the target *weights*. The engine then rebalanced back to those weights every day,
trading small amounts to undo price drift, and charged for it. A real account holds *shares* between
rebalances. The old behaviour overstated turnover on non-rebalance days and was not what the strategy
description said.

**Fix.** `run_backtest(..., rebalance=mask)`: targets are traded only on rebalance bars, and positions
drift with prices in between. It stays vectorised. Within a segment starting at execution bar s₀ with
targets *w*, `H[s] = w·G[s]/G[s₀] / V[s]` with `V[s] = 1 + Σ w·(G[s]/G[s₀] − 1)`, where *G* is cumulative
growth (a cumulative sum of `log1p` returns) and s₀ comes from a running maximum of rebalance indices.
Tests: it matches an explicit share-by-share loop to 1e-12, there are no trades between rebalances,
and an all-True mask reproduces daily constant-mix exactly.

The ML runs already in flight were scored with the old engine. Their saved out-of-sample scores were
re-evaluated with `scripts/evaluate_scores.py`, so nothing had to be retrained.

**Independent cross-check.** `vela/tracker/paper.py` is an event-driven simulator: a day-by-day loop,
orders → fills at the next open ± half-spread → a trade ledger → NAV. It shares no code with the
vectorised engine apart from the target weights, and its NAV path matches the engine's to within 0.2%
on a synthetic 120-day test (`tests/test_tracker.py`).

## Phase 5 — Portfolio-construction sweep on the first signals (dev 2006–2015)

| Scores | Construction | Gross Sharpe | Net Sharpe | Turnover/day | Cost (bps/day) |
|---|---|---|---|---|---|
| Ridge h=5 | L/S decile, daily | **1.44** | −2.27 | 168% | 26.5 |
| Ridge h=5 | L/S decile, weekly | 0.76 | −0.49 | 55% | 8.8 |
| Ridge h=5 | L/S hysteresis 10%/30%, daily | 1.35 | −0.30 | 75% | 10.0 |
| Ridge h=5 | L/S hysteresis 10%/30%, weekly | 0.84 | −0.16 | 42% | 5.9 |
| Ridge h=5 | Long-only top 20%, weekly | 0.65 | 0.34 | 23% | 3.3 |

Findings:
1. The ML signal is real and better than any single heuristic (gross Sharpe 1.44 with daily rebalancing),
   but most of it decays within 1–2 days. That is the short-term reversal component, and it is the
   most expensive part to trade.
2. Hysteresis bands are the most cost-efficient construction: most of the gross Sharpe at less than
   half the turnover.
3. Long-only books built on reversal-heavy scores load on **beta ≈ 1.2–1.3**, because recent losers are
   high-volatility names. Their alpha against the EW benchmark is negative. Next: risk control.

## Phase 6 — Statistical industries (feature set v2)

No sector codes are available, so industries are learned from returns (`vela/industries.py`). Each
January, the previous year's market-residual returns are embedded with PCA (15 components) and
clustered with k-means (20 clusters). Labels are then frozen for the year, so the clustering is point in time.
Sanity check: energy (XOM, CVX, COP, SLB, HAL), large banks (JPM, BAC, C, WFC), utilities and pharma
each come out as one cluster, and home-improvement retail (HD, LOW) groups together.
New features: industry return and industry-relative return over 1/5/21/63 days, plus industry 6-1 momentum.
Motivation: reversal is concentrated in the industry-relative component (Da, Liu & Schaumburg 2014),
while industry returns trend (Moskowitz & Grinblatt 1999).

## Phase 7 — Gradient boosting beats linear; signal decay; capacity

**GBM vs Ridge (v1 features, h = 5, dev 2006–2015).** Nonlinear interactions matter, consistent with Gu, Kelly & Xiu (2020):

| Model | IC (5d) | IC t-stat | Decile spread (5d) | L/S weekly gross / net Sharpe | Long-only IR vs SPY / vs EW |
|---|---|---|---|---|---|
| Ridge | 0.020 | 3.1 | 30 bps | 0.79 / −0.47 | 0.07 / −0.49 |
| **HistGBM** | **0.032** | **4.2** | **48 bps** | **1.29 / +0.27** | **0.48 / +0.15** |

GBM portfolio sweep (corrected engine): the beta-neutral, daily-rebalanced hysteresis book reaches **gross Sharpe 1.98**,
but net −0.56 at 111% daily turnover. The best net results are hysteresis 10%/30% weekly (L/S 0.30) and
long-only hysteresis 20%/40% weekly (net Sharpe 0.55, alpha vs SPY +3.8%/yr, t = 1.3).
Beta-neutralising cut L/S volatility from 19% to 14%, so the unhedged book had been carrying
market noise.

**Signal decay.** Rank IC of the GBM score against the return on day *k* after execution:
day 1 0.018, day 2 0.016, day 3 0.011, day 5 0.003, day 10 −0.001. The edge lasts about three days.
So weekly rebalancing leaves alpha on the table, and daily rebalancing pays too much for it.
Rebalancing every 2–3 days is the natural compromise.

**Capacity (L/S hysteresis, weekly).** Net Sharpe 0.75 at $0.1M, 0.65 at $1M, 0.30 at $10M, −0.31 at $50M.
Square-root impact dominates beyond ~$20M. Long-only: 0.66 at $0.1M, 0.55 at $10M, 0.29 at $100M.
The L/S book is a small-capital strategy; the long-only book scales.

## Phase 8 — What did *not* work (dev 2006–2015)

| Experiment | IC (h) | t-stat | L/S gross / net Sharpe | Verdict |
|---|---|---|---|---|
| PyTorch MLP (128-64, 3 seeds), h = 5 | 0.024 | 3.2 | 0.68 / −0.39 | Weaker than GBM: the usual result on tabular data |
| GBM + MLP rank ensemble | 0.030 | 3.9 | 0.92 / 0.13 | **Worse than GBM alone.** Score correlation 0.67; the weaker model adds noise |
| GBM + MLP + Ridge ensemble | 0.029 | 3.9 | 0.95 / 0.14 | Same |
| GBM with industry features (v2), h = 5 | 0.027 | 3.8 | 1.04 / 0.03 | Lower than v1 (0.032); seed-noise check queued |
| GBM v2, h = 21 (monthly) | 0.013 | 0.9 | 0.24 / −0.03 | **No significant monthly predictability** from price/volume features |
| GBM v2, volatility-scaled target | 0.025 | 3.1 | 0.94 / −0.04 | No better than the rank target |
| Inverse-volatility sizing | — | — | net 0.32 vs 0.44 equal-weight | Hurts: down-weights the volatile names where the reversal edge lives |

Kept: **square-root-of-ADV sizing** in the L/S book (net Sharpe 0.44 → 0.54; cost 6.6 → 6.1 bps/day).
It shifts capital toward names that are cheap to trade, so the same signal costs less to harvest.

The h = 21 failure is consistent with the measured signal decay (≈ 3 days). The predictive content of daily
price/volume features is short-horizon, and the project's edge has to be harvested at a 2–5 day cadence.

## Phase 9 — Seed noise, horizon matching, what the model learned, freezing

**Seed noise.** The same GBM with seed 1 instead of 0: IC 0.030 vs 0.032, but weekly L/S net Sharpe 0.04 vs 0.27.
The randomness of column subsampling alone moves portfolio results as much as many design changes did.
With the hysteresis constructions the gap shrinks (L/S net 0.48 vs 0.54, long-only 0.58 vs 0.58): they
do not react to marginal rank changes. Lesson recorded in D9: compare configurations on seed bags, and
bag seeds in the final model.

**v1 vs v2 features, two seeds each:** IC 0.032/0.030 vs 0.027/0.027. The industry features really do hurt (D8).

**3-day label.** The sharpest short-term signal (1-day IC 0.021, 3-day IC t = 5.2), but its scores churn more
(turnover 66% vs 52%/day), so L/S net Sharpe falls to 0.22. Blending it with the 5-day bag gained +0.02
Sharpe, inside the seed-noise band, so it was not adopted (Occam).

**What the GBM uses** (`scripts/feature_importance.py`: train 2001–12, validate 2013, measure on 2014–15;
each family shuffled across stocks *within* dates; market-level features swapped *across* dates):

| Feature family | IC lost when shuffled (base IC 0.023) |
|---|---|
| Momentum / trend | 0.0102 |
| Market regime | 0.0088 |
| Volatility / risk | 0.0078 |
| Short-term reversal | 0.0070 |
| Liquidity / volume | −0.0004 (unused) |

The model is not a repackaged reversal factor. It combines trend, risk and reversal, and conditions them
on market state. The market-regime features are constant across stocks on a date, so they can only
matter through interactions, which is what a linear model cannot capture.
*Bug found and fixed on the way:* the first version shuffled market features within dates, where they are
constant, and reported zero importance. They are now swapped across dates.

**LSTM sizing.** The first LSTM configuration (40-day sequences, 32 hidden units, two seeds, expanding
window) took ~9 min on the *smallest* fold and early-stopped at epoch 2 for both seeds. Projected runtime
was 3–4 h on 4 CPU cores. It was resized (30-day sequences, 24 hidden units, one seed, 5-year rolling window,
every 5th training day) to fit the compute budget.

**Freeze.** Configuration written to `results/frozen_config.json` (D12): GBM, v1 features, 5-day rank target,
seeds {0, 1, 2}; L/S hysteresis 10%/50%, 2-day rebalance, √ADV sizing; long-only hysteresis 20%/60%, 2-day
rebalance; base-case costs at $10M AUM.

**LSTM result (resized).** Mean IC 0.008 at 5 days (t = 1.5); 1-day IC 0.010, about the same as the plain
1-day-reversal baseline. Given only raw return/volume sequences and the CPU budget, the network mostly
re-learned short-term reversal. The hand-built cross-sectional features (trend, risk, regime) carry
information that 30 days of a single stock's own history does not. This is not evidence that sequence
models cannot work here; a larger model with cross-sectional inputs and a GPU would be the next experiment.

**Interruption.** The session's worker process restarted mid-run, killing the full-period scoring of the
frozen models (it had reached fold 2011 of seed 0). It was restarted from scratch. Nothing had been
evaluated on the holdout.

## Phase 10 — Model ladder under one construction (dev 2006–2015)

Every saved model re-scored with the *frozen* L/S and long-only books and base-case costs
(`scripts/model_ladder.py`, `results/model_ladder.csv`):

| Model | IC 5d | IC t | Decile spread (bps/5d) | L/S gross Sharpe | L/S net Sharpe | Long-only net Sharpe | IR vs SPY | IR vs EW |
|---|---|---|---|---|---|---|---|---|
| 12-1 momentum | 0.017 | 1.9 | −7.9 | −0.13 | −0.22 | 0.49 | 0.29 | −0.38 |
| 1-day reversal | 0.012 | 1.8 | 25.1 | 0.94 | −1.58 | 0.09 | −0.64 | −1.54 |
| 5-day reversal | 0.013 | 2.0 | 26.3 | 1.10 | −0.02 | 0.47 | 0.35 | −0.15 |
| Ridge | 0.020 | 3.1 | 30.2 | 1.11 | 0.23 | 0.54 | 0.57 | 0.06 |
| PyTorch MLP | 0.024 | 3.2 | 31.5 | 0.99 | 0.21 | 0.52 | 0.51 | 0.06 |
| PyTorch LSTM | 0.008 | 1.5 | 16.2 | 0.91 | −1.23 | 0.28 | −0.30 | −1.44 |
| GBM + industry features (2 seeds) | 0.028 | 3.8 | 40.9 | 1.12 | 0.30 | 0.56 | 0.62 | 0.26 |
| **GBM (2-seed bag)** | **0.031** | **4.2** | **48.7** | **1.42** | **0.58** | **0.59** | **0.69** | **0.33** |

Two separate contributions show up. **Portfolio construction** (hysteresis + 2-day cadence) alone
takes 5-day reversal from deeply negative net to break-even. **The GBM signal** then adds ≈ +0.6
net Sharpe on top of the same construction.
