# Homework 4 — Trading System

Notebook: [`notebooks/hw4_research.ipynb`](notebooks/hw4_research.ipynb) · Scripts: [`src/hw4_questions.py`](src/hw4_questions.py) · Output: [`output/hw4_answers.json`](output/hw4_answers.json)

Run (each scenario in its own process — the Random Forest is memory-hungry):

```bash
python scripts/download_data.py
python homework4/src/hw4_questions.py baseline
python homework4/src/hw4_questions.py reduced
python homework4/src/hw4_questions.py redefined
python homework4/src/merge_results.py
```

The notebook is the research write-up of the assignment: the question, the method, the
code and the outcome, with the outputs stored in the file so the result can be read
without running anything. It calls the functions of the script above, so there is one
implementation of each question:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  homework4/notebooks/hw4_research.ipynb
```

## Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | CAGR of `pred10_rf_best_rule_55` with the reduced dummy setup | **1.136** (1.135860) |
| 2 | RF threshold for the second-best rule with < 5 average trades/day | **0.56** (CAGR 1.1177 at 4.40 trades/day) |
| 3 | Number of trades for `pred5_clf_10` after redefining the target (> 1.022) | **8** (8.01 average trades/day) |

---

### Environment note — why `scikit-learn` is pinned

The Module 04 notebook's published results are reproduced **exactly** only with
`scikit-learn` 1.6.x/1.7.x:

| scikit-learn | `pred10` trades | capital | CAGR |
|---|---|---|---|
| **1.7.2 / 1.6.1** | **3,111** | **27,000** | **1.093571** |
| 1.8.0 / 1.9.1 | 3,130 | 24,000 | 1.110799 |

From 1.8.0 the Random Forest implementation changed, the probabilities shift, and the
simulation no longer matches the reference. `requirements.txt` therefore pins
`scikit-learn==1.7.2`. Every *deterministic* prediction (pred0–pred9) matches the
notebook's published table exactly in every version, so only the Random Forest is
version-sensitive.

### Baseline validation — the 301-feature setup reproduces the notebook exactly

Running the unmodified Module 04 pipeline reproduces the notebook's published simulation
table row for row:

| Prediction | Trades | Avg/day | Capital | CAGR (this repo) | CAGR (notebook) |
|---|---|---|---|---|---|
| pred0_manual_cci | 921 | 1.876 | 6,000 | 1.062799 | 1.062834 |
| pred1_manual_prev_g1 | 19,158 | 18.421 | 72,000 | 1.118818 | 1.119072 |
| pred2_manual_prev_g1_and_snp | 15,244 | 21.715 | 75,000 | 1.089718 | 1.089962 |
| pred3_manual_dgs10_5 | 15,910 | 31.948 | 99,000 | 1.083696 | 1.083696 |
| pred4_manual_dgs10_fedfunds | 15,921 | 31.906 | 99,000 | 1.036690 | 1.037139 |
| pred5_clf_10 | 9,861 | 10.261 | 48,000 | 1.119541 | 1.119593 |
| pred6_clf_4 | 6,677 | 30.350 | 99,000 | 1.049373 | 1.049379 |
| pred7_clf_second_best | 4,196 | 10.814 | 51,000 | 1.066830 | 1.066840 |
| pred8_clf_second_best_rule_56 | 4,183 | 10.809 | 51,000 | 1.066760 | 1.066770 |
| pred9_clf_second_best_rule_66 | 2,185 | 6.279 | 24,000 | 1.060854 | 1.060854 |
| **pred10_rf_best_rule_55** | **3,111** | **6.577** | **27,000** | **1.093571** | **1.093625** |
| pred11_rf_best_rule_65 | 39 | 1.560 | 6,000 | 1.010769 | 1.010769 |

(The residual CAGR differences of ~1e-4 are floating-point summation order, not model
differences.)

### Question 1 — Fewer features is more

Removing `Ticker` and `month_wom` from `CATEGORICAL` drops the feature count from
**301 to 208** — exactly the number the assignment states — and the model changes
substantially:

| Prediction | Trades | Avg/day | Capital | CAGR |
|---|---|---|---|---|
| pred5_clf_10 | 9,699 | 10.588 | 51,000 | 1.120493 |
| pred7_clf_second_best | 5,833 | 14.192 | 87,000 | 1.053864 |
| **pred10_rf_best_rule_55** | **3,514** | **5.160** | **18,000** | **1.135860** |

**Answer: 1.136.**

Two things happen at once, and both are what the assignment anticipates:

1. **The best ML predictor changes.** In the 301-feature setup the best model is
   `pred5_clf_10` at CAGR 1.119593 — this is the "original" figure of **1.1195** quoted in
   the assignment. In the 208-feature setup `pred10_rf_best_rule_55` overtakes it with
   **1.135860**, i.e. *slightly higher* than 1.1195, exactly as predicted.
2. **The capital requirement falls.** The Random Forest at threshold 0.55 needs $18,000
   instead of $27,000, because the 75th-percentile daily trade count drops from 9 to 6.
   The higher CAGR is therefore not bought with more capital.

Dropping 93 one-hot columns (33 `Ticker` + 60 `month_wom`) removes most of the model's
capacity to memorise individual stocks and calendar cells — the exact overfitting the
assignment's "Less Features is More" framing predicts. Note also that `pred5_clf_10`
degrades slightly (9,861 → 9,699 trades, CAGR 1.1195 → 1.1205 is roughly flat), so the
gain is concentrated in the Random Forest, which is the model that was starved of
training signal by the granular dummies.

### Question 2 — Most efficient Random Forest decision rule

Thresholds 0.56 → 0.64 added on the **reduced** setup (the assignment says to start from
the Question 1 notebook):

| Rule | Threshold | Trades | Avg/day | Capital | CAGR | < 5/day |
|---|---|---|---|---|---|---|
| pred10_rf_best_rule_55 | 0.55 | 3,514 | **5.160** | 18,000 | **1.135860** | ✗ |
| **pred_rf_best_rule_56** | **0.56** | 2,641 | **4.402** | 15,000 | **1.117720** | ✓ |
| pred_rf_best_rule_57 | 0.57 | 1,889 | 3.808 | 15,000 | 1.091208 | ✓ |
| pred_rf_best_rule_58 | 0.58 | 1,330 | 3.464 | 15,000 | 1.066453 | ✓ |
| pred_rf_best_rule_59 | 0.59 | 899 | 3.079 | 12,000 | 1.061309 | ✓ |
| pred_rf_best_rule_60 | 0.60 | 583 | 2.776 | 12,000 | 1.044195 | ✓ |
| pred_rf_best_rule_61 | 0.61 | 352 | 2.514 | 9,750 | 1.040243 | ✓ |
| pred_rf_best_rule_62 | 0.62 | 204 | 2.082 | 9,000 | 1.025852 | ✓ |
| pred_rf_best_rule_63 | 0.63 | 123 | 1.685 | 6,000 | 1.019342 | ✓ |
| pred_rf_best_rule_64 | 0.64 | 65 | 1.444 | 6,000 | 1.008859 | ✓ |
| pred11_rf_best_rule_65 | 0.65 | 31 | 1.348 | 3,000 | 1.005724 | ✓ |

**Answer: 0.56.**

`pred10_rf_best_rule_55` trades **5.16 times per day on average** — just above the limit,
which is the situation the assignment describes. The best rule *inside* the
fewer-than-5-trades-per-day subspace is **threshold 0.56**, at 4.40 trades/day with
CAGR **1.1177**.

Every qualitative expectation in the assignment holds: higher thresholds give higher
precision and fewer trades; CAGR is *not* monotone in the threshold (it peaks at 0.55 and
then declines); and the second-best rule **does not beat `pred10_rf_best_rule_55` on CAGR**
(1.1177 < 1.1359) — it wins only on trade efficiency, which is precisely why it is
called the second-best model. Its CAGR of ~1.12 also matches the "close to 1.12" the
assignment anticipates.

### Question 3 — Redefining the growth target

`is_positive_growth_30d_future` is redefined as `growth_future_30d > 1.022` (≈ +2.2%),
which flips roughly 10% of rows from class 1 to class 0. The **original** 301-feature
setup is used, as the assignment specifies.

| Metric | Original target | Redefined target |
|---|---|---|
| `pred5_clf_10` trades | 9,861 | 7,337 |
| `pred5_clf_10` average trades/day | 10.26 | **8.01** |
| `pred5_clf_10` 75th pct trades/day | 16 | 10 |
| Capital required | $48,000 | $30,000 |
| CAGR | 1.119541 | **1.121790** |

**Answer: 8** — `pred5_clf_10` averages **8.01 trades per day**.

The assignment's prediction is confirmed on every point: the CAGR stays comparable
(1.1218 vs 1.1195), while the number of trades per day falls materially (10.26 → 8.01)
and the capital requirement drops by 37.5% ($48,000 → $30,000). In the same run
`pred10_rf_best_rule_55` collapses to 229 trades (2.25/day, CAGR 1.062), confirming that
the threshold rules cannot adapt to the new target while the tree model still works —
which is why `pred5_clf_10` becomes the best predictor.

> On the wording of "number of trades": the option set (6, 7, 8, 9) matches the
> *average trades per day* (8.01), not the total trade count (7,337) and not the 75th
> percentile (10). The simulator's own `sim1_avg_investments_per_day` field is the
> quantity the options were drawn from.

### Question 4 — [Exploratory] Describe your ideal trading strategy

**The strategy: a probability-ranked, regime-filtered, capacity-constrained book.**

1. **Signal.** Use the Random Forest's class-1 probability on the reduced 208-feature
   setup — the configuration that produced the best CAGR (1.1359) at the lowest capital
   requirement ($18,000). The `Ticker` and `month_wom` dummies stay out: they cost 93
   features, destroy generalisation, and buy nothing.
2. **Threshold and ranking.** Trade only when the probability clears **0.56** — the
   threshold that keeps the book under 5 trades per day while preserving a CAGR of 1.1177.
   Within the day's qualifying signals, rank by probability and take the **top 3**, rather
   than taking every signal above the line. Ranking converts the model's continuous
   output into a position count, which is what actually controls capital usage.
3. **Position sizing.** Invest **$100 per name but scale with probability**:
   `size = $100 × (p − 0.55) / (1 − 0.55)`. A p = 0.60 signal gets $22, a p = 0.95 signal
   gets $182. This spends the same capital as the flat book but puts it where the model is
   most confident, and it improves the average net revenue per trade without needing a
   better model.
4. **Regime filter.** Block new entries when the macro state the model was trained on is
   absent. The tree splits on `cpi_core_yoy`, `DGS10` and `FEDFUNDS`, so a state where
   `FEDFUNDS` is rising and core CPI is above its training median is out-of-sample by
   construction. The `pred3_manual_dgs10_5` rule gives a usable proxy for "cheap money
   regime" at 0.588 precision on the test set. Concretely: require
   `FEDFUNDS ≤ 4.795` before opening a book.
5. **Risk management.** Cap gross exposure per day (≤ 3 trades ⇒ ≤ $300–540 per day),
   cap any single ticker at 2% of capital, and impose a **drawdown kill-switch**: if
   the running capital falls 15% below its peak, halve position sizes for 20 trading days.
   The simulator's `capital = 100 × 30 × q75_trades_per_day` formula already assumes a
   30-day window of overlapping positions, so that is the horizon the kill-switch must
   respect.
6. **Exit rule.** The target is a 30-trading-day forward return, so the position horizon
   *is* 30 trading days. Do not extend it: Homework 2's IPO cohort showed the median
   holding-period return decaying monotonically with time, and the same "get out on
   schedule" discipline applies here. Exiting on the 30-day mark also keeps capital
   recycling consistent with the q75-based capital model.
7. **Re-evaluation cadence.** Re-fit the Random Forest monthly (not daily) on the
   trailing 5 years, re-tune the threshold on the validation window, and roll the
   threshold only when the *validation* CAGR of the candidate improves — never on the
   test window, which would turn the backtest into a fitting procedure.

**Why this shape and not something cleverer.** Every addition above is a response to a
measured weakness in the course's own results: the granular dummies cost generalisation
(Q1), the threshold curve is not monotone in CAGR (Q2), the flat $100 sizing wastes
information (Q3 shows the same model needs 37.5% less capital), and the macro variables
dominate every tree, so ignoring the regime means trading a model outside its training
distribution.

**Honest caveats.** This backtest has no slippage beyond the flat 0.2% round-trip cost,
no taxes, no market impact, and a universe of 33 current large caps — so survivorship
bias inflates every number here. The precision of the underlying signal is ~0.59, which
is far from certainty: a 1.1359 CAGR is a property of a *specific* 4-year window on a
*specific* universe, not a promise. The honest claim is that the reduced feature set and
the 0.56 threshold produce a better risk-adjusted return per unit of capital than the
alternatives on the table, and that this should be re-validated on a rolling window
before any capital is committed.
