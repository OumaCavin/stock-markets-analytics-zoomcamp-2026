# Homework 3 — The Model

Notebook: [`notebooks/hw3_research.ipynb`](notebooks/hw3_research.ipynb) · Scripts: [`src/hw3_questions.py`](src/hw3_questions.py) · Output: [`output/hw3_answers.json`](output/hw3_answers.json)

Run:

```bash
python scripts/download_data.py                 # once
python homework3/src/hw3_questions.py
```

The notebook is the research write-up of the assignment: the question, the method, the
code and the outcome, with the outputs stored in the file so the result can be read
without running anything. It calls the functions of the script above, so there is one
implementation of each question:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  homework3/notebooks/hw3_research.ipynb
```

Built on the Module 03 Colab notebook, with the month-of-week dummies added as the
assignment requires.

## Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Highest \|correlation\| between a `month_wom` dummy and `is_positive_growth_30d_future` | **0.025** (`October_w4`, +0.024584) |
| 2 | Precision of the best NEW hand rule (`pred3`/`pred4`) on TEST | **0.588** (`pred3_manual_dgs10_5`) |
| 3 | TEST records where `pred5_clf_10` is the only correct positive predictor | **1171** |
| 4 | Optimal `max_depth` (1–12) for a DecisionTreeClassifier | **4** (TEST precision 0.6292) |

---

### Pipeline notes

Starting from `stocks_df_combined_2026_09_18.parquet.brotli` (240,759 rows, 33 tickers,
1972-06-01 → 2026-09-18), truncated to 2000-01-03 and split temporally 70/15/15:

| Split | Date range | Rows |
|---|---|---|
| train | 2000-01-03 → 2018-09-13 | 136,842 |
| validation | 2018-09-14 → 2022-09-15 | 32,287 |
| test | 2022-09-16 → 2026-09-18 | 33,163 |

Feature counts match the assignment exactly: **115 dummy variables** (12 `Month`, 7
`Weekday`, 33 `Ticker`, 3 `ticker_type`, **60 `month_wom`**) and **299 features** in total
(NUMERICAL 184 + DUMMIES 115) for the Module 03 feature set.

### Question 1 — Most correlated month-of-week dummy

`month_wom` is built as `Month + '_w' + ((day - 1) // 7 + 1)`, e.g. `October_w4`.

| Dummy | Correlation |
|---|---|
| **month_wom_October_w4** | **+0.024584** |
| month_wom_November_w3 | +0.023251 |
| month_wom_February_w1 | −0.020861 |
| month_wom_January_w2 | −0.019846 |
| month_wom_January_w5 | −0.019792 |

**Answer: 0.025** (rounded from 0.024584).

> Implementation note: `DataFrame.corr()` on a frame that *contains* the target returns the
> target's self-correlation (1.0) as a row, and `growth_future_30d` (0.698) as another.
> Both must be dropped before ranking, otherwise the answer is trivially 1.0. The script
> filters to the `month_wom` columns only.

### Question 2 — Precision of the new hand rules

Both new rules are applied to the whole dataset and evaluated on the TEST set:

| Rule | Flagged on TEST | Precision |
|---|---|---|
| **pred3_manual_dgs10_5** = `(DGS10 ≤ 4.5) & (DGS5 ≤ 4)` | 15,910 | **0.587744 → 0.588** |
| pred4_manual_dgs10_fedfunds = `(DGS10 > 4) & (FEDFUNDS ≤ 4.795)` | 15,921 | 0.502544 → 0.503 |

**Answer: 0.588** — `pred3_manual_dgs10_5` is the better of the two and is the rule that
makes positive predictions on the test set.

### Question 3 — Unique correct positives from `pred5_clf_10`

`DecisionTreeClassifier(max_depth=10, random_state=42)` is fitted on TRAIN + VALIDATION and
used to predict on the **entire** dataset. A TEST row counts when the true label is 1,
`pred5_clf_10` is 1, and all five hand rules (pred0–pred4) are 0.

**Answer: 1171** of 33,163 TEST records.

The script computes this twice — once with explicit column conditions and once with a
generic function applied to the prediction columns — and both return 1171.

### Question 4 — Optimal tree depth

Each depth is fitted on TRAIN + VALIDATION and scored by precision on TEST
(`random_state=42` throughout):

| Depth | 1 | 2 | 3 | **4** | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TEST precision | 0.5701 | 0.6225 | 0.5720 | **0.6292** | 0.5991 | 0.5966 | 0.6229 | 0.6194 | 0.5956 | 0.5993 | 0.5980 | 0.5837 |

**Answer: 4** — `max_depth=4` gives the best TEST precision (0.6292).

The table is the one produced by the pinned `scikit-learn==1.7.2`. The last depth moves
by about one point on a newer scikit-learn (0.5837 against 0.5744), because the tree
breaks a tie between two features differently; the best depth and its precision do not
move.

Depths 1–4 reproduce the reference values published in the Module 04 notebook
(`{1: 0.5701, 2: 0.6225, 3: 0.572, 4: 0.6292, …}`) exactly, which confirms the feature
construction and the temporal split. Depths 5–12 differ slightly from that table because
the Module 04 notebook adds two extra features (`div_payout`, `stock_split`) that the
Module 03 feature set does not contain.

### Question 5 — [Exploratory] What data is missing?

The shared dataset is wide (203 columns) but conceptually narrow: **33 price series, US
macro variables, and technical indicators**. The gaps that actually limit the model, in
descending order of importance:

**1. Country-specific macro data is missing entirely.** All eight macro variables
(`gdppot_us_yoy`, `gdppot_us_qoq`, `cpi_core_yoy`, `cpi_core_mom`, `FEDFUNDS`, `DGS1`,
`DGS5`, `DGS10`) are **US-only**, yet a third of the tickers are European and Indian
(`ticker_type` splits the panel across US / EU / INDIA). The decision tree splits on
`cpi_core_yoy` and `DGS10` — a US inflation and Treasury series — and applies that split
to European and Indian stocks. Adding the ECB area rate and HICP (`ECBMRRFR`,
`CP0000EZ19M086NEST`) and India's repo rate and CPI (`IRSTCI01EZM156N`, `66980`) would
let the model condition on the macro regime of the market each stock actually trades in.
This is the single biggest structural gap.

**2. No currency adjustment.** Returns for `TICKER.NS` (INR) and `TICKER.DE`/`TICKER.PA`
(EUR) are local-currency price changes compared against a US-dollar target. FRED's
`DEXINUS` and `DEXUSEU` would allow a common-currency frame; without them the target
mixes three currencies.

**3. No analyst expectations.** The dataset contains only *realised* future growth and
*reported* EPS. There is no consensus estimate and no revision series, so the model
cannot distinguish "good news" from "better than expected". This is exactly what the
`pred3`/`pred4` rules try to approximate with `DGS10`/`FEDFUNDS` levels.

**4. No fundamentals, no valuation, no sector.** There is no P/E, P/B, free-cash-flow
yield, revenue, margin, leverage or GICS sector classification. A "value" or "quality"
model is impossible, and the tree cannot condition on sector leadership — which matters
enormously for a 33-stock panel that is effectively a mega-cap index.

**5. Dividends and splits survive only as flags.** `TO_DROP` removes the `Dividends` and
`Stock Splits` columns, keeping only `div_payout`/`stock_split` indicators. The *magnitude*
of dividends and the split ratio are lost, so a 10-for-1 split looks identical to a
2-for-1 split, and the total-return adjustment is not recoverable from the frame.

**6. No event calendar.** There is no marker for earnings announcement dates, macro
release dates, index-inclusion events or FOMC meetings. Since the model learns that
October and November matter (Question 1), and earnings and CPI releases cluster in
exactly those months, the current design cannot separate a genuine seasonal effect from
event-timing clustering. This is the gap that Homework 5, Question 3 is designed to close.

**7. No liquidity or microstructure.** Daily volume is present but there is no turnover
relative to float, no Amihud illiquidity, no bid-ask spread and no short interest. The
Homework 4 simulator assumes $100 positions are executable in any of the 33 names; that
assumption is untested.

**8. No implied volatility.** `^VIX` and a volatility term structure would provide a
cheap regime filter and would let us test whether the calendar correlations are
volatility artefacts rather than return artefacts.

**9. Survivorship and selection bias.** The universe is "the largest stocks of the US, EU
and India" *as of 2026* — companies that survived and grew into the panel. Any strategy
backtested on it inherits that bias, and it is the most likely reason the backtested
CAGR exceeds 1.1.

**10. The target is mechanically tied to the feature set.** `growth_future_30d` is
`Close.shift(-30) / Close`, and `growth_30d` (a feature) is the same quantity shifted
backwards. A model with enough depth can reconstruct the target from lagged information,
which is why the depth-10 tree needs 9,861 trades to reach only 0.59 precision — and why
the honest comparison is always against a *macro-only* baseline, not against nothing.
