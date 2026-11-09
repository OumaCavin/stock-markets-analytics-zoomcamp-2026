# Homework 1 — Intro and Data Sources

Notebook: [`notebooks/hw1_research.ipynb`](notebooks/hw1_research.ipynb) · Scripts: [`src/hw1_questions.py`](src/hw1_questions.py) · Output: [`output/hw1_answers.json`](output/hw1_answers.json)

Run:

```bash
python homework1/src/hw1_questions.py
```

The notebook is the research write-up of the assignment: the question, the method, the
code and the outcome, with the outputs stored in the file so the result can be read
without running anything. It calls the functions of the script above, so there is one
implementation of each question:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  homework1/notebooks/hw1_research.ipynb
```

## Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Which year had the highest number of additions (starting from 2020)? | **2025** (17 additions) |
| 2 | How many indexes (out of 10) have better YTD returns than the US (S&P 500) as of 21 August 2026? | **2** (Japan, Canada) |
| 3 | Median drawdown (%) of significant (≥ 5%) S&P 500 corrections | **8** (7.99%) |
| 4 | Median 2-day % change following positive AMZN earnings surprises | **0.35** (0.353%) |

---

### Question 1 — S&P 500 additions by year

Source: the [Wikipedia S&P 500 constituents list](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies),
scraped with `pandas.read_html`, year extracted from the *Date added* column.

| Year | Additions |
|------|-----------|
| 2020 | 10 |
| 2021 | 10 |
| 2022 | 14 |
| 2023 | 13 |
| 2024 | 16 |
| **2025** | **17** |
| 2026 *(partial, year in progress)* | 19 |

**Answer: 2025.** The assignment asks for the best ***full*** year; 2026 is still in progress
and has the highest raw count, but it is incomplete. Among completed years 2020–2025 the
maximum is 2025 with 17 additions. The ranking is independently corroborated by
third-party change trackers, which also put 2025 ahead of 2024 and 2022 for a full year
(e.g. [tickericons.com/changes](https://tickericons.com/changes/2025)).

*Additional:* **223 of the 503** current constituents have been in the index for more than
20 years.

### Question 2 — World index YTD returns as of 2026-08-21

Window `start_date='2026-01-01'`, `end_date='2026-08-21'`, close-to-close, currency effects
ignored. Note that `yfinance` treats `end` as **exclusive**, so the last observation is
2026-08-20.

| Index | YTD |
|-------|-----|
| Japan — Nikkei 225 | **+27.75%** |
| Canada — S&P/TSX Composite | **+14.06%** |
| United States — S&P 500 | +11.41% |
| United Kingdom — FTSE 100 | +8.01% |
| Germany — DAX | +5.88% |
| Brazil — Ibovespa | +4.60% |
| Australia — S&P/ASX 200 | +4.08% |
| Mexico — IPC Mexico | +0.32% |
| Hong Kong — Hang Seng | −2.43% |
| China — Shanghai Composite | −2.97% |
| India — Nifty 50 | −7.32% |

**Answer: 2** — only the **Nikkei 225** and the **S&P/TSX Composite** beat the S&P 500.

The answer is stable across window conventions: an inclusive 2026-08-21 endpoint (S&P
500 = +11.90%) and a 2025-12-31 base (S&P 500 = +11.62%) both give 2. The S&P 500 figure
is externally verified: the 2025-12-31 close of 6,845.50 and 2026-08-31 close of 7,686.14
imply +12.28% YTD, matching Bloomberg data as published by
[AM Invest](https://www.aminvest.com/docs/aminvestlibraries/knowledge/market-overview/generalmarketoutlook.pdf)
(12.28% YTD at 31 Aug 2026).

### Question 3 — Median drawdown of S&P 500 corrections

Method (as specified): find all-time-high closes; for each consecutive pair of all-time
highs take the minimum close in between; drawdown = (high − low) / high; duration = days
from the all-time high to the trough. Corrections below 5% are dropped.

**The method is confirmed by reproducing the assignment's own hint list exactly:**

| All-time high | Trough | Drawdown | Days |
|---|---|---|---|
| 2007-10-09 | 2009-03-09 | 56.78% | 517 |
| 2000-03-24 | 2002-10-09 | 49.15% | 929 |
| 1973-01-11 | 1974-10-03 | 48.20% | 630 |
| 1968-11-29 | 1970-05-26 | 36.06% | 543 |
| 2020-02-19 | 2020-03-23 | 33.92% | 33 |
| 1987-08-25 | 1987-12-04 | 33.51% | 101 |
| 1961-12-12 | 1962-06-26 | 27.97% | 196 |
| 1980-11-28 | 1982-08-12 | 27.11% | 622 |
| 2022-01-03 | 2022-10-12 | 25.43% | 282 |
| 1966-02-09 | 1966-10-07 | 22.18% | 240 |

74 corrections of at least 5% occurred between 1950 and 2026-11-06.

- 25th percentile drawdown: 6.24%
- **Median drawdown: 7.99% → answer 8**
- 75th percentile drawdown: 14.02%
- Median correction duration: 40.5 days

### Question 4 — AMZN earnings surprises

`yf.Ticker('AMZN').get_earnings_dates()` returns a rolling 25-row window, so the window
described by the assignment (25 entries starting **2020-10-29**) is reconstructed
explicitly — 24 reported events from 2020-10-29 to 2026-07-30 plus one future entry
without a surprise. Yahoo's index carries a wall-clock time with mixed UTC offsets, so
dates are normalised to US Eastern calendar dates before joining.

For each announcement the 2-day return is `Close[Day3] / Close[Day1] − 1` where Day 2 is
the announcement day (Day 1 = previous trading day, Day 3 = next trading day).

- 24 events with a reported surprise, **20 with a positive surprise**
- **Median 2-day return: 0.353% → answer 0.35**
- Mean 2-day return: +0.464%
- Correlation between surprise magnitude and 2-day return: **+0.41** across all events,
  **+0.48** across positive surprises only

The relationship is economically meaningful: a larger beat is associated with a larger
short-term reaction, and roughly half of that reaction is realised within two days.

---

### Question 5 — [Exploratory] Capstone project idea

**Is the October/November 30-day growth effect real, or an artifact of earnings and macro
release timing?**

Homework 3 produced a striking result: among 60 `month_wom` dummies, `October_w4`
(corr +0.0246) and `November_w3` (+0.0233) are the most correlated with
`is_positive_growth_30d_future`, while `February_w1` (−0.0209) is the most negative. A
2.5% correlation looks negligible, but it is the only *structural* feature that survives
next to macro variables.

My thesis is that this is a **measurement artifact rather than a seasonal premium**. Two
concrete mechanisms:

1. **Earnings clustering.** Q3 reporting concentrates in mid-to-late October for both US
   and European issuers, and stocks earn a premium in scheduled earnings-announcement
   months (Frazzini & Lamont, 2006). "October" may simply be a proxy for "the market is
   full of fresh fundamental information".
2. **Macro release clustering.** CPI prints in mid-month and other releases land late in
   the month. Research on the equity premium finds that macroeconomic announcement days
   account for roughly 60% of it once day-of-month effects are controlled for
   ([AEA, "More than 100% of the equity premium"](https://www.aeaweb.org/conference/2022/preliminary/paper/ktNibY2Y)),
   and the turn-of-the-month effect has been shown to be entirely attributable to
   announcement days (Fraser et al., Euronext).

**Scope.** The 33-stock US/EU/India panel, 2000-01-03 → 2026-09-18, target
`is_positive_growth_30d_future`, horizon 30 trading days forward.

**Test.** Build three event counters — earnings releases per day, macro release days
(weighted by release importance, from FRED/ALFRED release dates), index-event days — and
re-estimate the dummy coefficients with and without them; then re-estimate inside
high- vs low-announcement-intensity strata; then re-estimate on the test window alone
(2022-09-16 → 2026-09-18); then compare a fixed-calendar bucket (week 4 of October)
against an event-aligned bucket (the 10 trading days after a CPI release).

**Falsification.** If the dummy coefficients are unchanged after controlling for the event
counters, the artifact hypothesis is rejected. A **permutation placebo** (shuffling the
`month_wom` labels within each month) is the decisive check: with 60 dummies tested, a
spurious pick is close to guaranteed, and if the placebo matches the real dummies the
signal was always noise. This matters practically — the capstone model should not ship
high-dimensional calendar features if they are artifacts.

**Why this is a good capstone.** It uses the course's own stack end-to-end (feature
engineering → tree models → the Module 04 trading simulator for the economic test), it has
a clean benchmark (the macro-only tree from Homework 3) and a pre-registered rejection
criterion, and a rigorous *negative* result is equally publishable.

### Question 6 — [Exploratory] New metrics to investigate

Each of these is a metric I would download and test against the shared dataset, with the
retrieval path spelled out.

**1. Macroeconomic release calendar (ALFRED release dates)** — the single highest-value
addition. `https://api.stlouisfed.org/fred/releases` (or `fredgraph.csv`) gives the exact
publication date of every CPI, PCE, payroll and FOMC release from 1950 onwards, so macro
"level" variables can be decomposed into a *release effect* and a *news effect*.
`pandas_datareader`/`fredapi` for the values, ALFRED for the dates.

**2. Consensus earnings estimates and revisions** — the dataset contains only *realised*
future growth, so the model can never learn what the market *expected*. Analyst EPS
estimates and 1-month revisions (`Alpha Vantage` `EARNINGS`, `Financial Modeling Prep`
`analyst-estimates`) would allow a proper "surprise" variable and a revision-momentum
feature. This is also the missing ingredient for HW3's `pred3`/`pred4` rules.

**3. Implied volatility and option skew** — `yf.download('^VIX')` plus a term-structure
(`^VIX3M`, `^VIX9D`) gives a cheap regime filter and a direct test of whether the
calendar dummies are volatility artefacts rather than return artefacts.

**4. Credit-market conditions** — ICE BofA option-adjusted spreads on
[FRED](https://fred.stlouisfed.org/series/BAMLC0A0CM) (high yield) and investment grade
(`BAMLC0A4CBBB`), plus the St. Louis Fed Financial Stress Index. The current tree splits
on `DGS10` and `FEDFUNDS`; credit spreads are the missing risk-pricing variable that
would explain equity moves those rate variables cannot.

**5. Liquidity and microstructure** — Amihud illiquidity (`|return| / dollar volume`),
turnover-to-float, and the 2025 single-unit-trading change on the Nairobi exchange.
The Homework 4 simulator assumes $100 positions are executable; this metric tests whether
that assumption holds.

**6. Index-membership events** — the Wikipedia S&P 500 constituent list (already used in
Homework 1) gives addition/removal dates, and MSCI's public review files give the same
for frontier markets. Index inclusion is a scheduled, non-informational demand shock, so
these dates can be included as event controls for the seasonality test.

**7. Valuation and fundamentals** — price-to-earnings, price-to-book and free-cash-flow
yield per ticker. The dataset has raw prices and returns but no accounting data, so
"value" and "quality" are entirely absent from the feature set.

**8. FX series for the non-US tickers** — the panel mixes US, EU and Indian names but all
macro variables are US-only. Adding EUR/USD and USD/INR (from FRED `DEXUSEU`, `DEXINUS`)
would allow currency-adjusted returns, which materially changes the cross-country
comparison the model is implicitly making.
