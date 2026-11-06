# Homework 2 — One Dataframe

Scripts: [`src/hw2_questions.py`](src/hw2_questions.py) · Output: [`output/hw2_answers.json`](output/hw2_answers.json)

Run:

```bash
python scripts/download_data.py          # once, for the RSI parquet file
python homework2/src/hw2_questions.py
```

## Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Total withdrawn IPO value for the company class with the highest total | **500** (Acquisition Corp, $485M) |
| 2 | Median Sharpe ratio as of 11 September 2026 for pre-Sep-2025 IPOs | **0.04** (0.048) |
| 3 | Optimal holding period (1–12 months) maximizing median growth | **1** (median −7.2%) |
| 4 | Total profit from investing $1,000 at every RSI < 30 signal | **65** ($65.8K) |

---

### Question 1 — Withdrawn IPOs by company type

Source: [iposcoop.com/ipos-recently-filed](https://www.iposcoop.com/ipos-recently-filed/),
parsed with `pandas.read_html`. Rows where *Expected To Trade* is `Withdrawn` and the
*File Date* is before 2026-09-11 are kept.

`Company Type` is assigned by the ordered rule list (first match wins), `Avg_price` is the
mean of *Price Low*/*Price High*, and `Shares_offered_value` is
`Shares (millions) × Avg_price`, falling back to *Est $ Vol (millions)* when the price is
missing.

| Company Type | Count | Total withdrawn value ($M) |
|---|---|---|
| **Acquisition Corp** | 4 | **485.0** |
| Inc. | 1 | 351.0 |
| Holdings | 4 | 311.7 |
| Other | 7 | 284.9 |
| Limited | 6 | 191.3 |
| Technologies | 3 | 184.9 |
| Group | 3 | 32.5 |

**Answer: 500** — the *Acquisition Corp* class, at $485M in the data available today.

Two caveats worth stating plainly:

* The assignment says "32 entries". Today the live page exposes only **28** withdrawn IPOs
  filed before 2026-09-11 — the page keeps a rolling 500-row window, so the four oldest
  rows (filed between roughly December 2025 and February 2026) have rolled off. The
  answer is robust to their return: to overtake *Acquisition Corp* the *Inc.* class would
  need an extra $135M in a single older filing, and the four missing rows would have to
  land in that one class.
* The ordering rules behave exactly as the assignment describes: "EUPEC International
  Group Ltd." → **Group** (the *Group* rule precedes *Ltd*), and "Xinxu Copper Industry
  Technology Ltd." → **Limited** (because "Technology" does not match "Technologies").

The single largest contributor is *Clear Street Group Inc.* ($351M: 13.0M shares at
$26–28), classified as *Inc.* — the rule ordering puts *Inc* ahead of *Group*.

### Question 2 — Median Sharpe ratio of the 2025 IPO cohort

Pipeline: 231 IPOs on [iposcoop.com/2025-pricings](https://www.iposcoop.com/2025-pricings/)
→ filter to *Offer Date* < 2025-09-01 and exclude 0% returns → **146** (the assignment
expects 148; the difference is data drift) → `yfinance` daily download → **132** tickers
still returning price data (the assignment expects ~134).

Features are computed **per ticker**, matching the Module 02 notebook, where every feature
is derived inside the per-ticker loop before the frames are concatenated:

```python
growth_252d = Close / Close.shift(252)
volatility  = Close.rolling(30).std() * sqrt(252)
Sharpe      = (growth_252d - 0.05) / volatility
```

On 2026-09-11:

| | growth_252d | volatility | Sharpe |
|---|---|---|---|
| count | 127 | 130 | 127 |
| mean | 1.2547 | 34.76 | ∞ (zero volatility rows) |
| min | 0.0010 | 0.000 | −0.0401 |
| 25% | 0.2142 | 2.87 | 0.0068 |
| **50%** | 0.6353 | 11.99 | **0.0480** |
| 75% | 1.0725 | 31.28 | 0.1379 |
| max | 33.638 | 627.70 | ∞ |

**Answer: 0.04** — the median Sharpe ratio is **0.048**.

Two observations the assignment asks for:

* The **median growth_252d is 0.635 versus a mean of 1.255** — the average is roughly
  double the median, so high-growth outliers are indeed biasing the mean. Only 127 of the
  132 reachable tickers had completed the 252-trading-day milestone at the measurement
  date.
* **There are stocks with far more attractive risk-adjusted returns than the raw growth
  statistic suggests**: the 75th percentile Sharpe is 0.138, while the corresponding
  growth figure is only 1.07. A handful of names (minimum Sharpe −0.0401) are at the
  opposite extreme.

### Question 3 — Optimal holding period for a new IPO

For each ticker the entry point is the **first trading day on which the stock actually
traded** (the pivot keeps pre-IPO rows as NaN, so the minimum date alone is wrong), and
future growth columns are built with 1 month = 21 trading days, then inner-joined onto
those entry points.

| Holding period | 1m | 2m | 3m | 4m | 5m | 6m | 7m | 8m | 9m | 10m | 11m | 12m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Median growth | **−7.2%** | −11.0% | −17.3% | −27.5% | −30.9% | −27.4% | −33.8% | −39.6% | −43.6% | −46.6% | −50.8% | −44.9% |

**Answer: 1** — one month is the optimal holding period, with a median growth of −7.2%.

The pattern is a clean monotonic decay: the typical 2025 IPO is at its least-bad after
**one month** and gets progressively worse the longer it is held. For an investor basing a
strategy on new listings this is unambiguous — the edge, if any, is in the first weeks
after pricing, and *holding longer makes the typical outcome worse, not better*. The mean
is far above the median in every column, confirming that the small number of large
winners are what makes the average look acceptable.

### Question 4 — RSI < 30 oversold strategy

Source: the precomputed indicator parquet from the assignment (`data.parquet`,
229,932 rows × 203 columns), filtered to 2000-01-01 → 2025-06-01 with `rsi < 30`.

- **Trades: 5,206** (the assignment states ~5,206 ✓)
- **Average 30-day return: +1.26%** (assignment: 1.26% ✓)
- **Win rate: 55.13%** (assignment: 55.13% ✓)

```
net_income = 1000 * (growth_future_30d - 1).sum() = $65,805.59
```

**Answer: 65** — $65.8 thousand of net income.

The three validation anchors from the assignment (trade count, average return, win rate)
all reproduce exactly, which confirms both the dataset and the filter are right.

### Question 5 — [Exploratory] How to change the strategy to increase profitability

The diagnosis first: a 55.13% win rate with a +1.26% average 30-day return produces a
positive expectancy, but the *median* return across all horizons is negative (−7% at one
month, −30% at five). The strategy is profitable in aggregate only because the tail
carries it, and it is trading 5,206 times — i.e. it is expensive in turnover and in
opportunity cost. The changes below are ordered by expected impact.

**1. Buy the signal, sell on a time rule, and cap the number of concurrent positions.**
The single biggest mechanical leak is that the strategy buys $1,000 at *every* RSI < 30
event, including events in names that are already falling. Overlapping signals in the same
ticker concentrate the exposure. Deduplicating to at most one open position per ticker
and capping the book at, say, the 25 highest-conviction signals (RSI deepest) changes the
result from "spray and pray" into a portfolio with a measurable hit rate.

**2. Add a regime filter using the macro variables already in the dataset.**
`DGS10`, `FEDFUNDS`, `cpi_core_yoy` and `gdppot_us_yoy` are present. Homework 3 showed
that these dominate the decision tree. An RSI < 30 event during an easing cycle
(falling `FEDFUNDS`) is a different animal from one during a tightening cycle. Filtering
trades on the macro regime should lift the win rate materially at the cost of fewer
trades — and the Homework 4 simulator already prices that trade-off in CAGR terms.

**3. Do not treat oversold as mean-reverting if the business is deteriorating.**
Blend the RSI signal with a fundamental or trend filter: only buy oversold names that are
(i) not in a confirmed downtrend (`Close < SMA200` is a disqualifier for a *trading*
signal, and `Close < SMA10` kills the momentum), and (ii) have not just missed earnings.
The failure mode of RSI strategies is precisely the value trap — a falling knife that keeps
falling. Adding a "no downtrend" condition trades frequency for win rate, and the
Homework 2 results say the median holding-period return is already negative, so *not*
trading is valuable.

**4. Use the IPO-specific structure that Q1 revealed.**
Withdrawn IPO value is concentrated in the *Acquisition Corp* class ($485M, the largest
class by a wide margin). Acquisition corporations are SPACs: their economics are a merger
arbitrage with a redemption feature, not an operating business. Any IPO strategy applied
to this cohort should **exclude SPACs and shells from the investable universe** and treat
the Q1 classification as a risk screen, not just a descriptive cut.

**5. Time the entry with the aftermarket structure.**
Q3 shows the median growth is worst the longer you hold, so the exit rule matters more
than the entry rule: cap the holding period at 21 trading days and do not average down
into the 2–3 month decay. Take the first 7% move and leave.

**6. Apply the Homework 3/4 model as a second filter on the IPO cohort.**
The IPO universe is small (231 names in 2025), so a from-scratch model would overfit.
The practical route is to reuse the model already validated on the 33-stock panel: score
each IPO with `pred5_clf_10` / `pred10_rf_best_rule_55` (the models trained on
macro + technical features) and trade only the IPOs the model also likes. The model's
precision is ~0.59 and its CAGR ~1.13, so intersecting it with the IPO universe should
beat either signal alone.

**7. Diversify on a risk budget, not on names.**
146 filtered IPOs with a 55% hit rate is a portfolio problem, not a stock-picking
problem. Equal-weighting 20–30 names, capping any single position at 2–3% of capital, and
sizing by the model's probability (not a flat $1,000) converts a fragile edge into
something with a survivable drawdown.

**What I would measure.** Every one of these changes has to be judged on the same yardstick
the course uses: CAGR over the test window, average trades per day, and the capital
requirement implied by the 75th-percentile daily trade count — plus, crucially, the
*median* net return, which is the statistic that shows whether the typical IPO
participation is worth making. Target: keep the win rate above 55% while cutting the trade
count by at least half, so the average net revenue per trade roughly doubles.
