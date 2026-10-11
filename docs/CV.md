# CV entry and talking points

Every number below comes from `results/final/summary.json` (out-of-sample, net of costs unless stated).
They are deliberately the *pre-registered* figures, including the weaker holdout results.

## Suggested CV entry

**Algorithmic Trading Strategy (VELA) | Personal Project** *09/2026 – Present*

- Built an ML stock-ranking model in Python (scikit-learn gradient boosting; benchmarked against Ridge,
  PyTorch MLP and LSTM) on 53 engineered features across point-in-time S&P 500 constituents; achieved an
  out-of-sample rank IC of 0.027 (t = 4.2) over 15 years of purged walk-forward testing, 1.8x the best
  single-factor baseline (5-day reversal, IC 0.015)
- Wrote a vectorised NumPy backtesting engine with next-open execution, drift-aware turnover and
  spread + square-root market-impact + borrow costs, validated against an independent event-driven ledger
  (0.9995 return correlation); the long-only strategy compounded 13.4% p.a. net vs 9.4% for the S&P 500
  (SPY) over 2006–2020 and beat the equal-weighted index in a locked 2016–20 holdout
- Built a portfolio tracker and order generator (trade ledger → NAV, time-weighted P&L, Sharpe, max
  drawdown, alpha/IR vs SPY); a $100k paper-trading replay of the holdout returned +108% vs +93% for SPY

### Shorter two-bullet version

- Built a gradient-boosted stock-ranking model (scikit-learn; benchmarked vs Ridge, PyTorch MLP/LSTM) on
  20 years of S&P 500 data; out-of-sample IC 0.027 (t = 4.2) in a purged walk-forward with a locked holdout
- Wrote a vectorised NumPy backtester with spread, square-root impact and borrow costs, validated against an
  event-driven ledger; long-only strategy returned 13.4% p.a. net vs 9.4% for SPY over 2006–2020

### About the original third bullet ("self-directed portfolio … with real capital")

The repository provides the tooling for this (`scripts/live_signals.py` to generate orders from current
data, `scripts/track_portfolio.py` to track a broker export against SPY). Whether real capital has been
deployed is a fact only you can state. Keep "real capital" on the CV only if you are actually trading it.
If not, the paper-trading bullet above is accurate.

## Interview talking points (with numbers)

1. **Why cross-sectional ranking.** Predicting *relative* 5-day returns removes most market beta and turns
   ~400 stocks a day into comparable labelled samples. Predicting each stock's direction mostly predicts the
   market, which is close to unpredictable at daily horizons.
2. **Costs were the hardest part.** 1-day reversal has a statistically real edge (t = 3.4), but trading it
   daily turns over 340% of NAV a day and turns gross Sharpe 0.96 into net −6. The fix was a
   cost-aware design: a 2-day cadence matched to the measured ~3-day signal decay, plus hysteresis bands
   (enter at the top 10%, exit at the median). Turnover fell to about a third.
3. **Why gradient boosting won.** IC 0.031 vs 0.020 (Ridge) and 0.024 (MLP) on the same features.
   Permutation importance shows market-regime features matter as much as stock features. Those features are
   identical across stocks on a given day, so they can only help through interactions, which trees capture
   and linear models cannot.
4. **Negative results I kept.** Statistical-industry features, a 21-day horizon, a volatility-scaled target,
   a GBM+MLP ensemble and an LSTM on raw sequences were all worse. All are documented in `docs/DEVLOG.md`.
5. **Seed noise.** Changing only the random seed moved one backtest's net Sharpe from 0.27 to 0.04. Lesson:
   compare configurations on seed bags; the final model bags three seeds.
6. **Overfitting controls.** A purged walk-forward; a holdout fixed in code before modelling and run once,
   after the configuration was committed; every trial logged (136). The Deflated Sharpe Ratio says the
   development search alone does not prove skill, which is why the holdout matters.
7. **What happened out of sample.** The signal held up but halved (IC 0.032 → 0.016). The long-only book
   still beat the equal-weighted index (IR 0.27) but not SPY's mega-cap-led 2016–20. The market-neutral
   book's gross edge scales with volatility while its costs do not: it lost money in calm 2016–18 and made
   +57% in 2020. That is the "reversal is liquidity provision" result of Nagel (2012), rediscovered on my
   own data.
8. **Engineering.** Look-ahead canary test, feature-causality test, bit-exact reproducibility of
   development scores, an independent ledger simulation matching the vectorised engine to 0.12 pp over
   4 years, and 35 tests in CI.
