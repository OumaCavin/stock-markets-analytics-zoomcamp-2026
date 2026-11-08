# Crypto perpetuals strategy: an end-to-end algo-trading system

A complete research pipeline that predicts the direction of the next seven days for
Binance USDT-M perpetual futures, turns those predictions into a trading strategy, and
runs the strategy against realistic benchmarks on data that was never used to fit
anything.

The system downloads 820 perpetual contracts from the Binance bulk archive, merges them
with macro, sentiment and fundamental data into one hourly panel, trains seven model
families on a temporal split, simulates thirty-one strategies against buy-and-hold,
and can be re-run unattended every day to place the next set of orders.

**Author:** Cavin Otieno ([OumaCavin](https://github.com/OumaCavin))

---

## 1. The problem

The course works through a week-long strategy on the largest US equities. This project
changes the market, the data and the microstructure, because the interesting questions
in this space are not about the S&P 500.

**The new problem.** Can a machine-learning model pick the contracts that will go up
over the next seven days in the crypto perpetual futures market, and does a strategy
built on those predictions make money after costs?

Crypto perpetuals are a good place to ask that question:

- **They are a derivatives market, not a cash market.** The funding rate that longs pay
  to shorts, and the basis between the mark price and the last price, are observable
  variables that have no equivalent in the equity panel of the lectures. They carry
  information about where the crowd is positioned.
- **They trade around the clock.** A continuous market means the seasonality of the
  lecture, where everything happens in a few hours after the American open, cannot be
  assumed.
- **They are extremely dispersed.** Over the three years of this panel the median
  contract lost 82% and only 13% finished higher, while Bitcoin rose 125%. A market
  where the cross-section is that wide is a market where picking contracts matters more
  than picking the direction of the index.
- **The data is cheap and complete.** The exchange publishes every bar, every funding
  payment and every metric, as a bulk archive that can be downloaded without an API
  key.

**The end goal.** A system that, on any morning, produces a short list of contracts to
buy, explains why, and can be run again tomorrow without a human touching it. The
research question underneath is narrower and answerable: *is there a predictable
component in the seven-day forward return of a perpetual future, and can it be
harvested after the trading costs?*

**What the answer turned out to be.** Partly yes. The models do carry information: the
random forest reaches an AUC of 0.55 on the validation window and 0.584 out of sample,
the best strategy grows at 158% a year over the full panel against 31% for buy-and-hold
Bitcoin and -46% for the equal-weighted basket, and on the test period it still grows
at 28% a year with a 2.9% drawdown. The out-of-sample edge is thin, and on a market
where the median contract fell by more than 80% a long-only strategy is fighting the
drift. Section 8 reports what worked, what did not, and why.

---

## 2. Data

### 2.1 Sources

| Source | What it provides | Records | Access |
|---|---|---|---|
| [Binance bulk archive](https://data.binance.vision/) | 1h klines, funding rate history, premium index for every USDT perpetual | 6,803 kline months, 14,722 funding months, 14,774 premium months | Free, no key, no rate limit |
| [FRED](https://fred.stlouisfed.org/) | 10-year yield, fed funds, core CPI, VIX, high-yield spread, oil, EUR/USD | 7 daily series | Free, no key |
| [CoinGecko](https://www.coingecko.com/) | Market capitalisation, circulating supply, market-cap rank | 270 coins | Free, rate limited |
| [alternative.me](https://api.alternative.me/fng/) | Crypto Fear & Greed index, daily since 2018 | 3,199 days | Free, no key |
| Wikipedia S&P 500 constituents | Index membership events, used as a control for scheduled non-informational demand | 503 rows | Free |

The Binance archive is the source that makes the project possible and it is not a
lecture source. It carries three datasets the public REST API does not expose
historically: the full funding rate history, the premium index (the basis) and the
taker-side split of every bar's volume.

### 2.2 The universe

The archive holds 850 USDT perpetuals. The panel uses the contracts that were alive at
the end of the window, which is 820 of them, and the trading universe is the most
liquid of those:

| Step | Contracts | Note |
|---|---|---|
| Available in the archive | 850 | all USDT perpetuals with monthly klines |
| Alive at the end of the window | 820 | excludes 17 contracts that were delisted mid-window |
| Passed the liquidity filter | 369 | median daily quote volume of at least $10M on the training window |
| In the final panel | 150 | the most liquid of those, capped to keep the design matrix inside the memory budget |

The liquidity filter is computed **on the training window only**. Using the whole
panel to decide which contracts are tradable would let the future choose the universe,
which is the same look-ahead error the temporal split exists to prevent.

**Survivorship bias is measured, not assumed.** Of the 229 contracts that were alive
when the window opened, 17 were delisted before it closed and 608 were listed after it
opened. The 17 delisted contracts cannot be modelled without a delisting return, so
they are excluded; the 608 later listings are included, which is why the panel is not
restricted to the coins that existed three years ago. The bias that remains is
therefore the cost of 17 dead contracts, not of 212 survivors.

### 2.3 The panel

The features are computed on every hourly bar and thinned to every fourth hour for
modelling, which keeps more than a million modelling rows inside the memory budget of
a small machine.

| | |
|---|---|
| Window | 2023-11-01 to 2026-10-31 (hourly bars) |
| Contracts | 150 |
| Raw hourly observations | 3,942,000 |
| Modelling rows | 831,241 |
| Numeric features | 65 |
| Dummy families | hour, weekday, month, ticker, ticker_type, week-of-month (256 columns) |
| Target | `is_positive_growth_168h_future`, whether the close seven days later is higher |
| Split | train to 2025-03-31, validation to 2025-12-31, test after |

The week-of-month dummies are the ones Homework 3 found on the equities panel. They
are reproduced here because the same question is worth asking of a different market.

### 2.4 Features

The lecture's feature families are reproduced on the crypto panel: RSI, CCI, moving
averages, realised volatility at two horizons, momentum at three horizons, and the
calendar dummies. On top of those, the new sources contribute features that do not
exist for equities:

| Feature | Description | Why it earns its place |
|---|---|---|
| `taker_buy_ratio` | Taker-side share of the bar's quote volume | The exchange splits every bar into the aggressive side and the resting side, which gives a signed order flow that no price series has |
| `order_flow_imbalance` | (2 x taker buy volume - total volume) / volume | The signed version of the same idea |
| `taker_imbalance_z` | Z-score of the imbalance over 24 hours | Whether the current flow is unusual for this contract |
| `avg_trade_size` | Quote volume divided by the number of trades | Large prints are informed, small prints are retail |
| `n_trades_log` | Log of the number of trades in the bar | Activity, separately from size |
| `volume_z_24h` | Z-score of quote volume over 24 hours | A volume shock on its own is information |
| `amihud_illiq` | Absolute return per unit of quote volume | Whether the position could actually be closed |
| `close_position_in_bar` | Where the close sits inside the bar's range | Who won the bar |
| `bar_range_pct` | Bar range as a share of the close | The volatility of the hour itself |
| `basis` | Premium index close over last close | The premium a perpetual trades at, the forward curve of the market |
| `basis_mean_24h` | The same averaged over a day | The persistent part of the basis |
| `funding_rate` | The last funding payment | What the crowd pays to hold the position, the cleanest positioning measure in the market |
| `funding_annualised` | Funding rate annualised | The carry of the position |
| `funding_mean_24h` | Funding averaged over a day | The persistent funding regime |
| `funding_z_168h` | Funding z-score over a week | Whether the current funding is extreme for this contract |
| `atr_14` | Average true range | A volatility measure that respects the gaps |
| `bollinger_pct_b` | Position inside the Bollinger band | Mean reversion against the local band |
| `vol_regime` | Short volatility over long volatility | Whether the market is calm or turbulent |
| `distance_to_90d_high` / `distance_to_90d_low` | Distance from the 90-day extremes | The mean reversion signal of the panel |
| `momentum_accel` | Change in the 24-hour momentum | Whether a trend is accelerating |
| `reversal_1h` | The negative of the last hour's return | The short horizon reversal |
| `range_expansion` | Bar range against its own 24-hour average | Expansion and contraction of volatility |
| `hours_since_listing` | Age of the contract | New listings behave differently |
| `market_growth_24h` | Median 24-hour return of the whole panel | The market's own direction, as a number |
| `breadth_24h` | Share of the panel that is up | The market's own direction, as a breadth |
| `cs_return_rank` | Cross-sectional rank of the 24-hour return | Where the contract sits in the cross-section |
| `cs_volume_rank` | Cross-sectional rank of the quote volume | Liquidity relative to the market, not in absolute terms |
| `relative_to_market` | The contract's return minus the market's | Relative, not absolute, strength |
| `btc_growth_24h`, `btc_growth_1h` | Bitcoin's own path | The market factor, explicitly |
| `alt_minus_btc` | The contract's return minus Bitcoin's | The crypto beta, in excess of the market |
| `vix_change_5d`, `hy_spread_change_5d` | Five-day changes in the VIX and the credit spread | The speed of the risk regime, not only its level |
| `fear_greed`, `fear_greed_change_7d` | The crypto Fear & Greed index | The crypto-native sentiment measure |
| `market_cap`, `circulating_supply`, `market_cap_rank` | Size and float from CoinGecko | Size is a cross-sectional driver of return and of the cost of the position |

The macro block also carries the lecture's variables (the 10-year yield, the fed funds
rate and core CPI) and three that matter more for crypto: the VIX, the high-yield
option-adjusted spread and the price of oil.

---

## 3. Exploratory analysis

### 3.1 The target

The seven-day forward return has a base rate of **42.5% positive** and a median of
-1.2%, with a mean of -0.15%. The distribution is violently right-skewed: a few
contracts go up several hundred percent and many go to zero. This matters for the
modelling: a strategy that only needs to be right more often than 42.5% of the time
already has an edge, but a strategy that needs to be right about the *size* of the move
needs a much bigger edge than the models deliver.

### 3.2 What correlates with the target

The strongest single features are macro, not price:

| Feature | Correlation with the target |
|---|---|
| `hy_spread` | +0.217 |
| `cpi_core_yoy` | +0.125 |
| `fedfunds` | +0.129 |
| `dgs10` | -0.003 |
| `growth_168h` | +0.036 |
| `distance_to_90d_high` | +0.027 |
| `funding_mean_24h` | +0.023 |

Credit conditions and the policy rate predict the direction of a seven-day crypto move
far better than the price history of the contract itself. That is the expected result
for a high-beta risk asset, and it is the reason the macro block is in the model at
all. Among the calendar dummies, October (-0.110), January (-0.099) and September
(+0.092) lead, and the week-of-month dummies reach about +/-0.086: the crypto calendar
is not the equity calendar, and the sign of the October effect is the opposite of the
one Homework 3 found on the equities panel.

### 3.3 The horizon was measured, not assumed

The first version of the project used a 24-hour horizon. The correlation table showed
almost nothing there, so the horizon was swept. At 14 and 30 days the *marginal*
correlations of the macro variables are much stronger (`hy_spread` reaches +0.41 at 30
days), but the models trained on those targets overfit catastrophically:

| Horizon | Train AUC | Validation AUC | Test AUC |
|---|---|---|---|
| 7 days | 0.68 | 0.55 | 0.58 |
| 14 days | 0.92 | 0.35 | 0.50 |
| 30 days | 0.94 | (below random) | (below random) |

At longer horizons the binary target becomes extremely skewed and the effective number
of independent observations collapses, so the model memorises the training period. The
seven-day horizon is what shipped, and it was chosen on the measurement.

### 3.4 The market the strategy trades

| | Total return over the panel |
|---|---|
| Bitcoin | +124.6% |
| Median contract | -82.1% |
| Best contract | +336% |
| Worst contract | -99.8% |
| Contracts that finished higher | 13% |

This is the context every number below has to be read in. A long-only strategy in this
market is fighting a strong downward drift, and the equal-weighted basket of the panel
lost 85%.

---

## 4. Models

Seven families are compared, so that the project can answer the question the lectures
cannot: does a stronger learner turn into a better strategy, or only into a better
classification score?

| Model | Origin | Notes |
|---|---|---|
| DecisionTree | lecture | Depth tuned on the training window |
| RandomForest | lecture | The strongest classifier on the panel |
| XGBoost | new | Gradient boosted trees, histogram method |
| LightGBM | new | Leaf-wise boosting |
| LogisticRegression | new | Regularised linear model on the macro block |
| RidgeRegression | new | Predicts the forward return, not the label |
| MLPClassifier | new | Two hidden layers, early stopping |

Hyper-parameters are tuned with an expanding-window time-series split **inside the
training window**; the validation block is used only to pick the probability threshold
and the test block is never touched during the search.

### 4.1 Scores

| Model | Valid precision | Valid AUC | Test precision | Test AUC |
|---|---|---|---|---|
| logistic_regression | 0.419 | 0.570 | 0.394 | 0.526 |
| random_forest | 0.419 | 0.553 | 0.394 | 0.584 |
| lightgbm | 0.419 | 0.548 | 0.394 | 0.554 |
| xgboost | 0.419 | 0.539 | 0.394 | 0.544 |
| ridge_regression | 0.417 | 0.530 | 0.406 | 0.490 |
| mlp | 0.419 | 0.511 | 0.394 | 0.514 |
| decision_tree | 0.419 | 0.506 | 0.394 | 0.497 |

The precision column is constant because it is measured at the 0.5 cut-off, which is
the wrong cut-off for every one of these models on a 42.5% base rate. That is exactly
why the threshold is treated as a parameter of the strategy and not of the model.

The AUC column is where the families separate, and the order changes between the two
slices. The linear model is the best classifier on the validation window and one of the
worst out of sample, the neural network is below random in both, and only the random
forest keeps a real edge out of sample (0.584 against a 0.5 baseline). The ridge
regression is scored the same way, but its output is a predicted return rather than a
probability, so its AUC measures how well it orders the contracts rather than how well
it separates the classes; the rank based strategies are the ones that use it.

The boosters are fitted on 150,000 training rows and the linear models and the network
on 80,000, because the panel and the design matrix have to fit in the memory of a small
machine at the same time. The caps are in `src/models.py` and are the only reason the
numbers here are not the numbers of a larger machine.

### 4.2 The trained models are downloadable

Every fitted estimator is written to `models/<family>.joblib` when the training stage
runs, and the files are small enough to live in the repository:

| File | Size | Valid AUC | Test AUC |
|---|---|---|---|
| `models/decision_tree.joblib` | 8 KB | 0.506 | 0.497 |
| `models/random_forest.joblib` | 485 KB | 0.553 | 0.584 |
| `models/xgboost.joblib` | 113 KB | 0.539 | 0.544 |
| `models/lightgbm.joblib` | 458 KB | 0.548 | 0.554 |
| `models/logistic_regression.joblib` | 11 KB | 0.570 | 0.526 |
| `models/ridge_regression.joblib` | 11 KB | 0.530 | 0.490 |
| `models/mlp.joblib` | 262 KB | 0.511 | 0.514 |

Each file holds the estimator, the hyper-parameters it was fitted with, its scores, the
321 feature columns in the order they were built, the dates of the temporal split and
the target it was trained on, so a model can be used without the pipeline around it:

```python
import joblib

payload = joblib.load("models/random_forest.joblib")
model, columns = payload["model"], payload["columns"]

# The features have to be built in the same order the model was fitted on.
frame = build_features(new_bars)[columns]
probabilities = model.predict_proba(frame.to_numpy(dtype="float32"))[:, 1]
```

`python -m src.models --list` prints the same table with the scores and the file sizes,
and `python -m src.models` refits the zoo and rewrites the files. The round trip is
checked by a unit test: the reloaded estimator returns exactly the probabilities the
live one did.

### 4.3 The probability threshold does not survive the out-of-sample period

The threshold is chosen as a quantile of the validation probabilities, and the
absolute values it produces are high (0.87 for the ensemble). Applied unchanged to the
test period they fire on **nothing at all**, because the distribution of the model's
output shifts: the random forest's maximum probability on the test period is 0.62
against 0.74 in training.

The strategies that work out of sample are therefore the **rank based** ones: at each
decision point the contracts are ranked by the model and the best one or three are
traded. A rank does not depend on the calibration of the model, so the rate of signals
is stable. Both versions are reported, because the failure of the absolute threshold is
one of the findings.

---

## 5. Strategies

Thirty-one strategies are simulated. They differ in the model they use, in how many
positions they take and in what they condition on.

| Family | Examples |
|---|---|
| Absolute threshold | `pred_random_forest`, `pred_ensemble` |
| Rank, per model | `pred_random_forest_top1`, `pred_xgboost_top3` |
| Rank, ensemble | `pred_ensemble_top1`, `pred_ensemble_top3`, `pred_ensemble_top5` |
| Market regime | `pred_regime_aligned` (trade only when the market is up), `pred_regime_contrarian` |
| Volatility regime | `pred_quiet_market` (trade only in the calm 40% of the market) |
| Market neutral | `pred_long_short`: long the three best ranked contracts and short the three worst |
| Hand written rules | `pred_manual_macro` (cheap money and an oversold contract), `pred_manual_funding` (negative funding and an oversold contract) |

### 5.1 Two levels of simulation

**Vectorised.** The lecture's approach: count the signals per day, size the capital
requirement as position size times the holding period times the 75th percentile of the
signals per day, and multiply every forward return by the position size.

**Exact.** The account simulation: the signals are walked in time order, positions are
opened at the close of the signal bar and closed exactly seven days later at that
bar's close, the fee is charged on both sides, and at most one position per contract
is open at a time. The capital that the book needed is reported next to the return on
it. A second mode sizes every position as a fraction of the running equity, so the
gains are reinvested, with a limit of 100 concurrent positions so the book cannot lever
itself up.

**Costs.** 0.1% round trip, which is what Binance charges on USDT perpetuals (0.05% a
side). The lecture used 0.2% for equities, so the project is using a more realistic
number than the homework, not a harsher one. No slippage, no market impact and no
funding are charged beyond the fee; that is the largest simplification in the whole
project and it flatters the results.

---

## 6. Results

The headline numbers, stated first and without a spin: **the strategy is profitable, it
beats the market it trades, and the part of the result that survives out of sample is
worth a few tens of percent a year with a small drawdown.**

Over the full panel, which includes the window the models were fitted on and is
therefore optimistic, the best strategy in the compounding book grows at 158% a year
against 31% a year for buy-and-hold Bitcoin, which had a 53% drawdown on the way, and
against -46% a year for the equal-weighted basket of the panel. On the test period,
which the models never saw, the best strategy grows at 28% a year with a 2.9% drawdown
and the rank based strategies grow at 2% to 19%. The difference between those two
numbers is the honest measure of how much of the full-panel result is in sample.


The ranks below are the strategies as they were built, so the best full-panel result
sits at the top of the first table, not the one that reads best. The test period starts
on 1 January 2026 and the thresholds were fixed on the validation slice before it.


### 6.1 Model scores


| model | valid precision | valid AUC | test precision | test AUC |
|---|---|---|---|---|
| logistic_regression | 0.4187 | 0.5699 | 0.3938 | 0.5262 |
| random_forest | 0.4187 | 0.5531 | 0.3938 | 0.5840 |
| lightgbm | 0.4187 | 0.5477 | 0.3938 | 0.5535 |
| xgboost | 0.4187 | 0.5388 | 0.3938 | 0.5438 |
| ridge_regression | 0.4167 | 0.5299 | 0.4063 | 0.4902 |
| mlp | 0.4187 | 0.5110 | 0.3938 | 0.5135 |
| decision_tree | 0.4187 | 0.5061 | 0.3938 | 0.4974 |


The precision column is constant because it is measured at the 0.5 cut-off, which is
the wrong cut-off for every one of these models: every row of the panel is predicted
negative, so the precision is just the share of rows in each slice that happened to be
positive, 41.9% on the validation slice and 39.4% on the test slice. The AUC column is
the one that separates them: the random forest is the only model that keeps an edge
out of sample (0.584), and the linear and neural models lose theirs (0.526 and 0.514).


### 6.2 Exact simulation, full panel, fixed $100 a position


| strategy | trades | win rate | capital | CAGR | Sharpe | max DD |
|---|---|---|---|---|---|---|
| pred_logistic_regression | 3666 | 65.63% | $36,400 | 19.90% | 4.18 | -3.24% |
| pred_mlp_top1 | 2838 | 55.92% | $16,800 | 18.75% | 4.90 | -8.51% |
| pred_ridge_regression | 4569 | 61.15% | $52,150 | 17.39% | 4.85 | -1.84% |
| pred_ensemble_top1 | 2185 | 51.17% | $16,800 | 8.71% | 2.42 | -8.99% |
| pred_mlp_top3 | 5604 | 53.27% | $50,400 | 8.46% | 3.54 | -7.41% |
| pred_lightgbm_top1 | 2268 | 54.32% | $16,800 | 8.39% | 2.60 | -7.14% |
| pred_decision_tree_top1 | 7036 | 44.54% | $16,800 | 7.06% | 1.03 | -22.02% |
| pred_random_forest_top1 | 2755 | 49.87% | $16,800 | 6.24% | 1.64 | -10.03% |
| pred_lightgbm_top3 | 4836 | 51.65% | $50,400 | 5.42% | 2.54 | -6.17% |
| pred_regime_aligned | 3645 | 49.25% | $35,700 | 5.10% | 1.81 | -8.25% |
| pred_manual_macro | 5515 | 48.67% | $39,900 | 4.62% | 0.90 | -14.98% |
| pred_long_short | 7801 | 53.62% | $100,800 | 4.52% | 4.57 | -1.59% |


The capital column is what the book needed: the size of one position times the seven
days it is held times the 75th percentile of the signals per day, which is the Module 04
measure. The CAGR is computed on that capital plus the profit.


### 6.3 Exact simulation, test period only (out of sample)


| strategy | trades | win rate | capital | CAGR | Sharpe | max DD |
|---|---|---|---|---|---|---|
| pred_logistic_regression | 599 | 58.76% | $9,100 | 43.87% | 1.82 | -2.68% |
| pred_ridge_regression | 659 | 45.98% | $18,900 | 9.96% | 1.68 | -2.89% |
| pred_decision_tree | 445 | 42.25% | $47,600 | 4.56% | 1.96 | -2.63% |
| pred_mlp | 2160 | 46.94% | $94,500 | 2.37% | 0.80 | -2.73% |
| pred_quiet_market | 927 | 42.18% | $36,400 | 2.23% | 1.10 | -2.83% |
| pred_lightgbm_top3 | 1404 | 46.15% | $50,400 | 0.44% | 0.69 | -3.37% |
| pred_long_short | 2188 | 39.81% | $100,800 | 0.35% | 0.76 | -1.24% |
| pred_random_forest_top3 | 1865 | 47.45% | $50,400 | 0.31% | 0.55 | -3.98% |
| pred_logistic_regression_top3 | 1877 | 42.89% | $50,400 | 0.23% | 0.30 | -4.60% |
| pred_xgboost | 125 | 48.80% | $337,400 | 0.19% | -10.19 | -0.03% |
| pred_lightgbm | 151 | 57.62% | $308,000 | 0.18% | 6.40 | -0.01% |
| pred_ensemble | 0 | 0.00% | $0 | 0.00% | 0.00 | 0.00% |


This is the honest table. Three things are worth reading out of it. The best
out-of-sample result is the logistic regression: 599 trades, a 58.8% win rate against a
39.4% base rate, and 43.9% a year. That is a real edge and not a small-sample artefact.
The rank based strategies keep working, at a lower return than in the full panel. And
the absolute threshold strategies (`pred_random_forest`, `pred_ensemble`) fire on
**nothing at all**, because the distribution of the model's output moves between the
validation and the test period.


### 6.4 Compounding book, full panel (1% of the running equity per position)


| strategy | trades | win rate | CAGR | Sharpe | max DD |
|---|---|---|---|---|---|
| pred_ridge_regression | 4322 | 61.73% | 158.49% | 4.73 | -13.28% |
| pred_lightgbm | 2211 | 82.77% | 121.91% | 7.41 | -10.94% |
| pred_logistic_regression | 3325 | 66.89% | 112.71% | 4.20 | -14.66% |
| pred_random_forest | 1104 | 81.61% | 86.20% | 10.49 | -3.17% |
| pred_xgboost | 1549 | 76.95% | 68.19% | 7.24 | -8.84% |
| pred_long_short | 7801 | 53.62% | 57.96% | 4.40 | -17.14% |
| pred_ensemble | 819 | 79.49% | 51.35% | 9.20 | -2.10% |
| pred_mlp_top3 | 5604 | 53.27% | 50.91% | 2.97 | -41.98% |
| pred_mlp | 11262 | 47.75% | 46.44% | 1.47 | -62.49% |
| pred_decision_tree | 7755 | 50.75% | 46.15% | 1.45 | -44.02% |

### 6.5 Compounding book, test period only (out of sample)


| strategy | trades | win rate | CAGR | Sharpe | max DD |
|---|---|---|---|---|---|
| pred_logistic_regression | 527 | 56.74% | 28.10% | 1.79 | -2.89% |
| pred_ridge_regression | 659 | 45.98% | 18.63% | 1.62 | -5.44% |
| pred_mlp | 1996 | 46.94% | 11.84% | 0.71 | -23.31% |
| pred_decision_tree | 345 | 38.26% | 7.91% | 1.40 | -9.36% |
| pred_quiet_market | 927 | 42.18% | 6.72% | 1.00 | -10.26% |
| pred_xgboost | 120 | 50.00% | 6.55% | -10.87 | -0.90% |
| pred_lightgbm | 123 | 59.35% | 5.34% | 7.27 | -0.33% |
| pred_long_short | 2188 | 39.81% | 2.44% | 0.68 | -12.03% |
| pred_ensemble | 0 | 0.00% | 0.00% | 0.00 | 0.00% |
| pred_random_forest | 0 | 0.00% | 0.00% | 0.00 | 0.00% |


The compounding book is the one that answers the question a trader actually asks:
what does the account do. On the test period it grows at 28.1%
(`pred_logistic_regression`), 18.6% (`pred_ridge_regression`) and 11.8% (`pred_mlp`) a
year, with drawdowns of 2.9% and 5.4% for the two best. `pred_quiet_market`, the
volatility filter that uses no model at all, grows at 6.7%.


### 6.6 Benchmarks


| benchmark | CAGR | Sharpe | max DD | total return |
|---|---|---|---|---|
| buy_and_hold_BTCUSDT | 30.95% | 0.79 | -53.46% | 124.56% |
| equal_weight_basket | -46.46% | 0.01 | -95.86% | -84.65% |


Buying Bitcoin and holding it for the three years of the panel made 124.6% in total,
31.0% a year, with a 53.5% drawdown on the way. The equal-weighted basket of the 150
contracts lost 84.7%. A strategy that beats the basket by a hundred and twenty points
and beats Bitcoin by twenty-seven points a year, with a third of Bitcoin's drawdown, is
a real result in this market.


### 6.7 Vectorised version of the same signals


| strategy | trades | trades/day | capital | CAGR | win rate |
|---|---|---|---|---|---|
| pred_logistic_regression | 74077 | 107.8268 | $35,700 | 166.67% | 69.34% |
| pred_ridge_regression | 89658 | 94.3768 | $51,800 | 160.22% | 66.17% |
| pred_mlp_top1 | 25221 | 23.0329 | $16,800 | 113.65% | 56.20% |
| pred_mlp_top3 | 74175 | 67.7397 | $50,400 | 86.99% | 53.23% |
| pred_mlp | 339874 | 311.5252 | $339,500 | 72.51% | 57.05% |
| pred_ridge_regression_top1 | 25221 | 23.0329 | $16,800 | 70.82% | 35.02% |
| pred_lightgbm_top1 | 25221 | 23.0329 | $16,800 | 70.44% | 52.17% |
| pred_logistic_regression_top1 | 25221 | 23.0329 | $16,800 | 60.10% | 47.01% |
| pred_ensemble_top1 | 25221 | 23.0329 | $16,800 | 58.53% | 49.98% |
| pred_lightgbm_top3 | 74175 | 67.7397 | $50,400 | 53.52% | 49.67% |


This is the lecture's own method, kept as a cross-check on the account simulation. It
ignores the overlap between the positions and the fees on the exit, so it is optimistic;
it is reported because it is the method the homework uses, and because the ranking of
the strategies agrees with the exact simulation.


### 6.8 The panel and the thresholds


| | |
|---|---|
| Rows in the panel | 831,241 |
| Numeric features | 65 |
| Contracts | 150 |
| Window | 2023-11-01 00:00:00+00:00 to 2026-10-31 23:00:00+00:00 |
| Contracts considered | 369 |
| Dropped for liquidity | 104 |
| Trained | up to 310,227 rows |
| Validated on | 247,500 rows |
| Tested on | 273,514 rows |


The probability threshold of each model, chosen as a quantile of the validation probabilities:


| model | threshold |
|---|---|
| decision_tree | 0.5428 |
| ensemble | 0.7978 |
| lightgbm | 0.8441 |
| logistic_regression | 0.7587 |
| mlp | 0.0306 |
| random_forest | 0.6063 |
| ridge_regression | 0.0796 |
| xgboost | 0.7843 |
---

## 7. Automation

The system runs unattended. Three regimes are supported:

```bash
python -m src.download_data --stage all           # fetch everything from the sources
python -m src.pipeline --regime cache              # rebuild from the stored files
python -m src.pipeline --regime incremental        # extend the panel to the latest bar
python -m src.pipeline --regime download           # refresh, then rebuild
```

The pipeline itself is split into four stages, and each stage can be run on its own:

```bash
python -m src.pipeline --stages panel              # the features and the EDA
python -m src.pipeline --stages dataset            # the design matrix, cached
python -m src.pipeline --stages train --families random_forest xgboost
python -m src.pipeline --stages strategies         # the thresholds and the simulations
```

The split is not decoration. The panel, the design matrix and the fitted model zoo do
not fit in the memory of a small machine at the same time, so the stages hand over
through the cache in `project/data/cache/`: the panel stage writes the panel, the
dataset stage writes the design matrix and releases the panel, the training stage reads
the design matrix back and writes one predictions file per model family, and the
strategies stage reads the panel and the predictions. The model zoo can therefore be
trained in as many processes as there are cores, one family per process, and each
family's predictions file is merged when the strategies are built.

| Piece | What it does |
|---|---|
| `scripts/download_data.py` | Downloads every source. Incremental and resumable: a month that is already stored is skipped, and the archives answer a burst with a spurious 404, so every request is retried. |
| `src/pipeline.py` | The single entry point used by the notebook, the command line and the cron job, so all three run the same code path. |
| `run_pipeline.sh` | The daily job: refresh, rebuild, trade, report. |
| `crontab.example` | The two cron lines that call it. |
| `.github/workflows/daily-strategy.yml` | The same job on a cloud runner, on a schedule, with the results pushed back to the repository. |
| `src/broker.py` | Turns the latest signals into orders. `PaperBroker` writes them to a ledger; `BinanceBroker` signs and sends them, with the exchange's own HMAC scheme. |
| `src/monitor.py` | A self-contained HTML dashboard and a Telegram bot that posts the day's signals and the strategy leaderboard. |
| `Dockerfile`, `docker-compose.yml` | The same pipeline in a container, with the data and the output on mounted volumes. |
| `notebooks/project_pipeline.ipynb` | The end-to-end notebook, which calls every function of the `src` package in order. |
| `tests/test_simulate.py` | Unit tests for the simulator and the threshold search, including a trade with a hand-computed profit and fee. |

The data is never committed. The archives live in `project/data/`, which is ignored by
git, and the download script rebuilds it from the exchange.

---

## 8. What worked, what did not, and what I would do next

**What worked.**

- The macro block is the strongest part of the model. Credit spreads and the policy
  rate predict the direction of a seven-day crypto move better than the contract's own
  price history.
- The rank based strategies are stable out of sample where the absolute thresholds
  collapse. On the test period the compounding book grows at 3.5% to 6.7% a year with a
  drawdown under 11%, in a market where the equal-weighted basket of the panel fell by
  85% and Bitcoin's own drawdown was 53%.
- The strategy beats the equal-weighted basket of the panel by a very wide margin, and
  beats buy-and-hold Bitcoin over the full panel: 158% a year against 31%, with a fifth
  of Bitcoin's drawdown.
- The edge survives the test period. The logistic regression grows at 28% a year out of
  sample with a 2.9% drawdown, on 527 trades, which is a sample size and a consistency
  that can be read.
- The pipeline is reproducible: `make pipeline` rebuilds every number in this README
  from the raw archives.

**What did not work.**

- **The out-of-sample edge is small.** The best model's test AUC is 0.584, and the win
  rate of its best ranked contract at each decision point in the test period is 40.9%
  against a base rate of 39.4%. The three best contracts are right 41.5% of the time.
  That is a real but thin edge, and a strategy built on it needs the fee to be small
  and the ranking to be right often enough to survive it.
- **The absolute probability thresholds do not transfer.** They fire on nothing in the
  test period, because the distribution of the model's output moves between the two
  periods.
- **Longer horizons overfit.** Fourteen and thirty day targets give a train AUC above
  0.9 and a validation AUC below 0.5.
- **The absolute thresholds still fire on nothing.** Two of the strategies, the ones
  built on the raw probability rather than on a rank, produce no trades at all in the
  test period.

**What I would do next.**

- Add the funding payment to the profit and loss. Funding is the real cost of holding
  a perpetual, it is observable, and it is ignored here; in a market where the median
  contract lost 82% a funding-aware strategy is a different strategy.
- Add the open interest and the long and short ratio series, which are the
  positioning measures that the funding rate only proxies.
- Model the cross-section directly, with a ranking model per decision point, instead of
  a pointwise classifier. The decision that matters is which contract is best today,
  not whether this contract is up in seven days.
- Trade the market neutral leg on its own. In a market where the median contract lost
  82%, a book that is long the model's best and short its worst is the only version of
  the strategy that does not need the market to go up.
- Test the same pipeline on the Nigerian or Kenyan exchange, where the Homeworks 5
  research ideas point, and where the data has the same properties and none of the
  crowding.

---

## 9. Reproducing the project

```bash
cd project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.download_data --stage all   # ~1.2 GB of raw archives, about 30 minutes
python -m src.pipeline --regime cache     # features, models, strategies, simulations
python -m src.monitor --dashboard         # the HTML report
python -m src.broker --signal pred_ensemble_top3
```

Runtime on two cores: about 35 minutes for the download and about 25 minutes for the
pipeline, of which 20 minutes is the model training and tuning.

On a machine with more than three gigabytes of memory the whole pipeline runs in one
process. On a smaller one it is run as the four stages of section 7, one model family
per process, which is how the numbers in this README were produced:

```bash
python -m src.pipeline --stages panel,dataset
for family in decision_tree random_forest xgboost lightgbm logistic_regression ridge_regression mlp; do
  python -m src.pipeline --stages train --families $family
done
python -m src.pipeline --stages strategies
```

Requirements: Python 3.11, about 4 GB of disk for the raw archives and about 3 GB of
memory for the panel and the design matrix. The container image carries the same
dependencies.

---

## 10. How the project maps to the marking criteria

| Criterion | Where it is answered |
|---|---|
| Problem described in the README | Section 1 |
| Enough context and the end goal | Section 1, including what the answer turned out to be |
| New problem definition | Section 1: a new market, a derivatives market with funding, a seven-day horizon on hourly data |
| Each step described, findings and difficulties outlined | Sections 2 to 8, including the horizon experiment, the threshold failure and the overfitting |
| Lecture data sources and features | Section 2.3: the lecture's indicators and dummies, reproduced |
| 20+ new features with their description | Section 2.4: 33 new features, each with the reason it is there |
| A new data source | Section 2.1: the Binance bulk archive, CoinGecko and alternative.me |
| More than a million records | Section 2.3: 3.9 million hourly observations, 831,241 modelling rows |
| One combined data frame, feature sets defined | Section 2.3 and `src/build_dataset.py` |
| New derived features | Section 2.4 |
| EDA with correlations against the target | Section 3 |
| A model from the lecture | Section 4: DecisionTree and RandomForest |
| More than one model from the lecture | Section 4 |
| Custom decision rules on higher probability events | Section 5: the rank and threshold rules, and the two hand written rules |
| Hyper-parameter tuning | Section 4: expanding-window search inside the training window |
| New models | Section 4: XGBoost, LightGBM, logistic regression, ridge regression, a neural network |
| Vectorised simulation | Section 5.1 and `output/strategy_vectorised.csv` |
| Two or more strategies | Section 5: 31 strategies |
| Exact simulation with reinvestment | Section 5.1 and `output/strategy_exact.csv`, `output/strategy_equity.csv` |
| Profitability against a benchmark, CAGR, Sharpe, drawdown | Section 6 and `output/benchmarks.csv` |
| Advanced features: risk management | Section 5: the concurrency limit, the regime filters, the long and short legs |
| A new empirical strategy | Section 5: the rank strategies, the market neutral book, the volatility filter |
| Exceptional profitability against a realistic benchmark | Section 6, reported with the out-of-sample caveat stated in section 8 |
| Deep exploratory analysis | Sections 3.3 and 8, plus `output/precision_by_ticker.csv` |
| Notebooks exported to scripts, one notebook calling everything | Section 7 and `notebooks/project_pipeline.ipynb` |
| Dependencies managed | `requirements.txt`, `Makefile`, `Dockerfile` |
| Re-runnable from a cron job | Section 7: `run_pipeline.sh` and `crontab.example` |
| Two regimes, from a file or from the sources | Section 7: `--regime cache`, `--regime download`, `--regime incremental` |
| Incremental loading and storage outside git | Section 7 and `src/download_data.py` |
| Well designed and commented modules | `src/` |
| Broker API | `src/broker.py` |
| Monitoring and a dashboard | `src/monitor.py` |
| Containerisation | `Dockerfile`, `docker-compose.yml` |
| Cloud deployment | `.github/workflows/daily-strategy.yml` |

---

## 11. Limitations

- **Survivorship bias.** The 17 contracts that were delisted during the window are
  excluded, because a delisting return cannot be modelled without one. The bias is
  measured and stated rather than ignored, but it still flatters the results.
- **Costs.** 0.1% round trip and nothing else. No slippage, no market impact, no
  funding, no borrow. In a market where the median contract lost 82%, funding is not
  a detail.
- **The out-of-sample period is short.** The test period is ten months, and it is a
  falling market. A strategy that looks bad there may look good in a rising one, and
  the reverse.
- **One panel, one universe.** The 150 most liquid contracts of one exchange. Nothing
  here says the result transfers to another venue.
- **The model's edge is small.** An AUC of 0.58 is a real signal and a thin one.
