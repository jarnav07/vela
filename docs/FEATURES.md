# Feature reference

Every feature on day *t* uses only data up to and including the close of day *t*
(`tests/test_pipeline.py::test_features_are_causal` deletes all later data and checks nothing changes).
Code: [`vela/features.py`](../vela/features.py) (v1) and [`vela/industries.py`](../vela/industries.py) (v2 additions).

**Notation.** *c, o, h, l* = adjusted close, open, high, low; *dv* = close × volume (dollar volume);
*r* = daily close-to-close log return; *mkt* = equal-weighted mean of *r* over the tradable universe;
windows are in trading days.

**Normalisation.** Stock-level features are converted to cross-sectional percentile ranks within the day's
tradable universe and scaled to [−0.5, 0.5] (missing → 0, the median). Market-level features are identical for
every stock on a day and are z-scored against their own trailing 252-day mean and s.d. (clipped to ±5). Only the
ordering of stocks matters to the model, so ranking makes features robust to outliers and to regime shifts in scale.

The **final model uses the 53 v1 features**. The 9 v2 industry features were tested and rejected (DECISIONS D8).

## Short-term reversal (19)

| Name | Definition |
|---|---|
| `ret_1`, `ret_2`, `ret_3`, `ret_5`, `ret_10` | log(c<sub>t</sub> / c<sub>t−k</sub>) for k = 1, 2, 3, 5, 10 |
| `ret_lag1` … `ret_lag4` | daily log return *r* lagged 1–4 days |
| `overnight_1` | log(o<sub>t</sub> / c<sub>t−1</sub>): the overnight gap |
| `intraday_1` | log(c<sub>t</sub> / o<sub>t</sub>): the open-to-close move |
| `overnight_5`, `intraday_5` | 5-day sums of the two legs above |
| `dist_ma10` | c / 10-day moving average − 1 |
| `rsi_14` | Wilder's 14-day relative strength index |
| `bollinger_z` | (c − 20-day MA) / 20-day s.d. of price |
| `close_loc_1` | (c − l) / (h − l): where the close sits in the day's range |
| `close_loc_5` | 5-day mean of `close_loc_1` |
| `resid_ret_5` | 5-day sum of market-residual returns, r − β<sub>t−1</sub> · mkt |

## Momentum / trend (11)

| Name | Definition |
|---|---|
| `ret_21`, `ret_63`, `ret_126`, `ret_252` | log(c<sub>t</sub> / c<sub>t−k</sub>) |
| `mom_12_1` | log(c<sub>t−21</sub> / c<sub>t−252</sub>): 12-month momentum skipping the latest month |
| `mom_6_1` | log(c<sub>t−21</sub> / c<sub>t−126</sub>) |
| `dist_ma50`, `dist_ma200` | c / moving average − 1 |
| `high52_prox` | c / 252-day max of c (George & Hwang 52-week-high proximity) |
| `low52_prox` | c / 252-day min of c |
| `resid_ret_21` | 21-day sum of market-residual returns |

## Volatility / risk (12)

| Name | Definition |
|---|---|
| `vol_5`, `vol_21`, `vol_63` | rolling s.d. of *r* |
| `vol_ratio_5_63` | log(vol_5 / vol_63): volatility regime of the stock |
| `hl_range_1` | log(h / l) |
| `parkinson_21` | Parkinson high–low volatility, √(mean(log(h/l)²) / 4 ln 2) over 21 days |
| `max_ret_21`, `min_ret_21` | largest / smallest daily return over 21 days (lottery demand, Bali et al. 2011) |
| `downside_vol_63` | √ mean(min(r, 0)²) over 63 days |
| `skew_63` | 63-day skewness of *r* |
| `beta_252` | 252-day beta of *r* to *mkt* |
| `idio_vol_63` | 63-day s.d. of market-residual returns |

## Liquidity / volume (4)

| Name | Definition |
|---|---|
| `log_dv_21` | log of 21-day mean dollar volume (size/liquidity proxy) |
| `volume_shock_5_63` | log(5-day mean dv / 63-day mean dv) |
| `volume_shock_1_21` | log(today's dv / 21-day mean dv) |
| `amihud_21` | log of 21-day mean of \|r\| / dv (Amihud illiquidity) |

## Market regime (7, same value for every stock on a day)

| Name | Definition |
|---|---|
| `mkt_ret_1`, `mkt_ret_5`, `mkt_ret_21` | equal-weighted market return over 1, 5, 21 days |
| `mkt_vol_21` | 21-day s.d. of the market return |
| `mkt_vol_ratio` | log(5-day / 63-day market volatility) |
| `cs_dispersion_1` | cross-sectional s.d. of today's stock returns |
| `cs_dispersion_21` | 21-day mean of `cs_dispersion_1` |

## Statistical industries (9, feature set v2, rejected)

Industries are re-learned each January from the previous year's market-residual returns (PCA with 15 components,
then k-means with 20 clusters) and frozen for the year (`vela/industries.py`).

| Name | Definition |
|---|---|
| `ind_ret_1`, `ind_ret_5`, `ind_ret_21`, `ind_ret_63` | mean log return of the stock's cluster over k days |
| `ind_rel_ret_1` … `ind_rel_ret_63` | stock's k-day log return minus its cluster's |
| `ind_mom_6_1` | cluster mean of log(c<sub>t−21</sub> / c<sub>t−126</sub>) |

## Which families the model relies on

Grouped permutation importance (`scripts/feature_importance.py`, development data only):
momentum/trend > market regime > volatility/risk > short-term reversal ≫ liquidity (≈ 0).
See [RESULTS.md §6](RESULTS.md#6-what-the-model-learned).
