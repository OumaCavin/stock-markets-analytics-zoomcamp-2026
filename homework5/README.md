# Homework 5 — Automation & New Ideas (2026 cohort)

Notebook: [`notebooks/hw5_research.ipynb`](notebooks/hw5_research.ipynb)

Three distinct stock research ideas. Each one follows the structure required by the
assignment: market and scope, investment thesis, potential edge, data and sources,
research approach, and validation and risks.

The notebook below is the write-up of the three ideas: the question, the hypothesis, the
data, the method, the test that would prove the idea wrong, and the collection code
that would build the panel.

```bash
jupyter nbconvert --to notebook --execute --inplace \
  homework5/notebooks/hw5_research.ipynb
```

The three ideas deliberately differ in **economic mechanism**, **market**, **data source**
and **investment horizon**:

| # | Idea | Mechanism | Market | Horizon |
|---|------|-----------|--------|---------|
| 1 | Foreign-flow exhaustion in a frontier market | Order-flow imbalance / retail absorption | Kenya (Nairobi) | Days → weeks |
| 2 | Index-review passive flows | Benchmarked passive demand | Global frontier markets | Announcement → effective date (~3 months) |
| 3 | Is the "October/November" seasonality real? | Measurement artifact vs. true seasonality | US / EU / India large caps (course dataset) | 30-day forward |

---

## Question 1 — Foreign-flow exhaustion and domestic absorption on the Nairobi Stock Exchange

### 1. Market and scope

Kenya's Nairobi Stock Exchange (NSE), focusing on the liquid large caps — Safaricom, KCB,
EABL, Equity, KPLC, ABSA — plus the NSE 20 / NASI indices as the benchmark. Horizon is
**5 to 20 trading days**, measured from the date a flow signal becomes observable. This is
a short-horizon, flow-driven study, not a fundamental one.

### 2. Investment thesis

Kenya's equity market is unusually two-sided in its ownership base. Foreign investors were
historically the marginal price-setter, but their participation has been falling while
domestic investors have repeatedly absorbed their selling:

- Average foreign participation fell to **25.58% in Q2 2026** from **32.27% in Q1 2026**
  (CMA Quarterly Statistical Bulletin, Q2 2026).
- Foreign investors were **net sellers for nine consecutive months** of 2026: cumulative net
  foreign outflows of about **KSh 15.65 billion since January 2026** (CMA Capital Markets
  Soundness Report, Q1 FY 2026/27). September alone accounted for ~64% of that quarter's
  outflow (KSh 3.611bn), and the Q3 bulletin shows foreign investors buying KSh 21.7bn while
  selling KSh 27.4bn.
- Despite that selling, the market rallied hard: **NASI +26.8% in H1 2026**, making it the
  fourth best-performing stock market in the world in that half-year (Business Daily), with
  NSE 20 at 3,755.44 at end-Q2 and market capitalisation crossing **KSh 4 trillion** in
  August 2026.

The thesis is that this divergence is *not* a contradiction but a **regime change in the
marginal buyer**. When foreign selling is persistent and disclosed with a lag (CMA publishes
monthly), while domestic absorption is now deep enough to absorb it — helped by the
**single-unit trading change introduced in August 2025**, which lets retail buy one share
instead of a full board lot — then a *large* foreign net outflow should predict *positive*
subsequent returns rather than negative ones. The classic "flow drives price" relationship
inverts once domestic depth replaces foreign marginality.

### 3. Potential edge

Three sources of edge, none of which require forecasting Kenyan GDP:

1. **Information lag.** The CMA publishes foreign participation and net flows with a
   one-to-three-month lag. A model that uses the *published* flow data is, by construction,
   trading on information that is already public but not yet in the price.
2. **Slow repricing of a structural change.** The single-unit-trading regime (Aug 2025)
   changed the shape of the demand curve; the market's own price discovery has not fully
   absorbed it. This is a *level* effect, not a *timing* effect — investors keep applying
   the old "foreign selling = bearish" heuristic.
3. **Flow concentration.** Foreign selling concentrates in exactly the two most liquid
   counters (KCB and Safaricom supplied ~54% of session turnover on 22 July 2026), while
   turnover overall fell 64.9% year-to-date to KES 347.3m. A market-level flow signal
   therefore has a specific, tradeable transmission channel rather than being diffuse noise.

### 4. Data and sources

| Source | What it gives | Access |
|---|---|---|
| [CMA Quarterly Statistical Bulletin (Q2 2026, Q3 2026)](https://www.cma.or.ke/download/85/2026/6890/capital-markets-authority-cma-quarterly-statistical-bulletin-q3-2026.pdf) | Monthly foreign participation %, net foreign purchases/sales, turnover, market cap, investor counts | PDF, quarterly, public |
| [CMA Capital Markets Soundness Report, Q1 FY 2026/27](https://www.cma.or.ke/download/9/cm-soundness-report/6892/capital-markets-soundness-report-cmsr-quarter-1-fy-2026-2027.pdf) | Monthly foreign net flows (July/Aug/Sept split), participation by month, concentration ratios | PDF, quarterly, public |
| [NSE (nse.co.ke)] | Daily OHLCV and turnover for every listed counter | Public site / paid feeds |
| yfinance | Daily OHLCV for `^NSEI`-style and NSE-listed symbols (`SAF.KA`, `KCB.KA`, …) | `yf.download()` — **verified reachable in this environment** |
| Central Bank of Kenya | Interbank rate, FX, T-bill yields — the macro regime the flow signal is conditioned on | Public statistics |

**Verified**: the CMA bulletins above were opened and the figures quoted here read directly
from them. **To be verified**: whether NSE publishes a *machine-readable* daily
investor-flow file (versus quarterly PDF only); if not, flows will have to be
interpolated/regressed at monthly frequency, which materially reduces the number of
independent observations and must be treated as a binding constraint on the test.

### 5. Research approach

1. Build a monthly panel: for each month from 2015 (or the earliest CMA bulletin) to the
   present, record the S&P-like market return (NASI), foreign participation %, net foreign
   flow in KSh bn, total turnover, and market capitalisation.
2. Form the signal: `FlowShock = −(net foreign flow / market cap)` bucketed into deciles,
   interacted with `Δforeign participation` (the *change* in foreign presence, which
   separates "foreign selling that domestic money absorbed" from "foreign selling that
   dominated").
3. Test: regress monthly/weekly index and stock-level returns on the signal with
   HAC (Newey-West) standard errors, and run a placebo on *pre-2025* data (before
   single-unit trading) to see whether the effect exists only in the new regime.
4. Convert to a trade: on days following a flow-shock decile, buy the NSE 20 equal-weight
   basket for 10 trading days, funded by the capital model from Homework 4
   (`capital = 100 × 30 × q75_investments_per_day`), and evaluate CAGR and max drawdown with
   the Module 04 simulator.
5. Also test the transmission channel directly: do the *liquid* counters (KCB, Safaricom)
   react differently from illiquid ones? If the thesis is about domestic absorption of
   foreign supply, the effect must be strongest where foreign flow is concentrated.

### 6. Validation and risks

- **Benchmark**: buy-and-hold NASI, and a zero-flow strategy that trades the same number of
  days (to separate "the signal works" from "it just avoids bad months").
- **Falsification**: if the return following large foreign outflows is *negative* in the
  post-August-2025 sample as well, the thesis is rejected. If the effect disappears once
  turnover is controlled for, it is a liquidity artefact, not a flow signal.
- **Risks**: publication lag means the signal may already be priced by the time it is
  observable — the honest benchmark is the *post-publication* return, not the post-flow
  return; flow data are quarterly/monthly, so the sample is small and statistical power is
  genuinely limited; Kenya's market is concentrated (up to 82.9% of market cap in ten
  names), so idiosyncratic single-stock risk dominates; currency and repatriation risk
  apply to any foreign investor; and CMA methodology changes can break the series.
- **Publication delays**: the signal is slow by construction (monthly), so it cannot
  compete on turnover, only on capital efficiency.

---

## Question 2 — MSCI index-review passive flows in frontier markets

### 1. Market and scope

Single stocks in markets newly entering or already inside the **MSCI Frontier Markets
Index** — 28 countries, 246 constituents, roughly 85% of free-float-adjusted market
capitalisation per country. Concretely testable 2026 cases already visible in the public
record: the May 2026 review's additions (Ho Chi Minh City Development Joint Stock
Commercial Bank and Binh Son Refining and Petrochemical in Vietnam, BICICI in Ivory Coast),
the August 2026 additions in Nigeria (Bua Foods, Dangote Sugar Refinery, HBM Nigeria), and
Pakistan's addition to Frontier Markets. Horizon: **announcement to effective date**
(typically 11–17 days) plus a 60-day post-effective window.

### 2. Investment thesis

Index inclusion is not an information event — it is a *scheduled demand shock*. When a
security is added to a benchmark that institutions track, the mechanical buying that follows
is large relative to the float in small markets. The academic record is specific: the
announcement-to-implementation median cumulative abnormal return for additions is around
**5.5%**, with foreign investors buying on the order of **2.7% of shares outstanding** and
roughly **74% of the trades occurring on the effective date itself**
(Williams, *The Anatomy of Index Rebalancings*). Earlier work found about **+3.4%** on the
day after announcement and a further **+4.5%** into the effective date, with the effect
markedly stronger in emerging markets (Chakrabarti, Huang, Jayaraman & Lee, *Journal of
Financial and Quantitative Analysis*, 2005).

The thesis is that **the effect scales inversely with free float and analyst coverage**, so
it should be *larger* in frontier markets than the developed-market estimates the literature
reports — and that frontier-market liquidity is too thin for arbitrageurs to fully absorb
the demand shock, leaving a persistent post-effective drift rather than a one-day pop.

### 3. Potential edge

1. **The dates are public and calendar-driven.** MSCI publishes the additions/deletions
   list and the announcement date well in advance; there is no forecasting required. The
   research question is about *magnitude and persistence*, not about prediction.
2. **Frontier markets are structurally under-covered.** MSCI rates Kenya, Bangladesh,
   Pakistan, Sri Lanka, Vietnam and others as Frontier or Standalone, meaning limited
   sell-side coverage, wide spreads and small free floats. The same demand shock that
   produces a 5.5% median CAR in larger markets should be larger in percentage terms and
   slower to unwind here.
3. **The reversion side is under-studied.** The NBER work on MSCI market *reclassifications*
   (Frontier → Emerging) shows overshoots of roughly **+15%** between announcement and
   effective date that fully revert within a year, and the opposite for downgrades
   (−9.2% in the announcement-to-effective window for markets moving to less-benchmarked
   status). Deletions are rarer and more impactful per name.

### 4. Data and sources

| Source | What it gives | Access |
|---|---|---|
| [MSCI Index Reviews](https://app2.msci.com/eqb/fm/index_review.html) — Feb/May/Aug/Nov review lists of additions & deletions | Exact constituent lists, announcement dates, effective dates | Public PDF/CSV per review |
| [MSCI Market Classification](https://www.msci.com/indexes/index-resources/market-classification) and the [MSCI 2026 Global Market Accessibility Review](https://www.msci.com/downloads/web/msci-com/indexes/index-resources/market-classification/MSCI%202026%20GLOBAL%20MARKET%20ACCESSIBILITY%20REVIEW%20REPORT.pdf) | Which markets are Frontier vs Standalone; free-float and foreign-ownership constraints | Public PDF |
| [MSCI Frontier Markets Index factsheet](https://www.msci.com/documents/10199/255599/msci-frontier-markets-index.pdf) | 28 countries, 246 constituents, ~85% free-float coverage | Public PDF |
| Local exchange price data (Yahoo Finance for the larger names; exchange feeds for the rest) | Daily OHLCV and turnover around the event | `yf.download()` — **verified reachable** |
| MSCI country index series | The benchmark for abnormal-return computation | Public index data |

**Verified**: the MSCI classification, review cadence, and the specific 2026 additions
listed above are all from MSCI's own published documents and review lists.
**To be verified**: (a) whether historical frontier-market constituent lists can be
reconstructed for a long enough sample (public review PDFs cover only recent years — a
deep backtest may require a commercial data licence); (b) actual free-float percentages
per name, which are not in the public review files; (c) whether Yahoo carries reliable
prices for the smaller frontier names, which it often does not.

### 5. Research approach

1. For each review between, say, 2019 and 2026, collect the additions and deletions with
   announcement and effective dates.
2. For each affected stock, compute daily abnormal returns against its local MSCI country
   index (or the local broad index where the country index is unavailable), then build
   cumulative abnormal returns over windows AD−5 → AD+1, AD+1 → CD, and CD+1 → CD+60.
3. The core test is a **cross-sectional comparison**: run the same event study on
   developed, emerging and frontier samples and test whether the frontier coefficient is
   significantly larger. The literature's "stronger in EM" finding is the hypothesis; the
   frontier comparison is the contribution.
4. Second test: relate the abnormal return to the float (proxy: log of market
   capitalisation, or the MSCI-reported foreign ownership limit) and to turnover
   pre-event. If the demand-shock story is right, the effect should scale with
   "float tightness".
5. Third test: does the pre-effective drift *reverse* after the effective date, as the
   reclassification literature implies? A fully-reverting effect is tradeable only in the
   announcement window; a persistent one is tradeable for months.
4. Convert to a strategy: a long-only event portfolio of frontier additions, financed with
   the Module 04 capital model, evaluated on CAGR, max drawdown and — critically — the
   *cost of the untradeable half*: illiquidity. Half the position cannot be exited on the
   effective date in a market with a few hundred trades a month.

### 6. Validation and risks

- **Benchmark**: the MSCI Frontier Markets Index over the same calendar window; and a
  "matched non-event" portfolio of frontier stocks of similar size and liquidity.
- **Falsification**: if frontier additions do not show a larger CAR than developed-market
  additions, the "thin float" thesis is rejected; if the CAR is entirely concentrated on
  the announcement day and reverses immediately, there is no tradeable edge.
- **Risks**: the sample is small (a handful of names per review); announcement dates can
  shift; **MSCI has since moved Pakistan into Frontier Markets and keeps Nigeria as a
  Standalone market, so classification itself is a live variable that must be handled as a
  treatment, not a fixed label**; transaction costs in frontier markets are large relative
  to the expected edge; and short-selling is rated "−" (not permitted) in most frontier
  markets under MSCI's own accessibility review, so deletions cannot be expressed as a
  short.

---

## Question 3 — Is the October/November 30-day-growth effect real, or a calendar artifact?

### 1. Market and scope

The course's shared dataset: the 33 largest listed companies of the US, the EU and India,
daily 2000-01-03 → 2026-09-18. The target is the course's own `is_positive_growth_30d_future`
binary, and the focal features are the `month_wom` dummies built in Homework 3 — of which
`October_w4` (corr **+0.0246**) and `November_w3` (**+0.0233**) are the most correlated
with the target, with `February_w1` (−0.0209) the most negative. Horizon: **30 trading
days forward**, matching the target definition.

### 2. Investment thesis

A correlation of ~0.025 between a calendar dummy and a 30-day forward growth binary is
tiny in isolation but non-trivial relative to everything else in the model: in Homework 3
the best decision tree was dominated by macro variables (`cpi_core_yoy`, `DGS10`,
`FEDFUNDS`), and the calendar dummies were the only *structural* feature to show up.

The thesis is that **the October/November signal is not a genuine seasonal premium but a
proxy for information-event timing**, and the specific candidate mechanisms are:

1. **Earnings clustering.** Q3 reporting is concentrated in mid-to-late October for both
   European and US issuers. The literature is explicit that stocks earn a premium in
   *scheduled earnings-announcement months* (Frazzini & Lamont, 2006) — so "October" may
   be a proxy for "the market is full of fresh fundamental information".
2. **Macro release clustering.** CPI prints in the middle of the month and other key
   releases land late in the month. Research on the equity premium finds that
   macroeconomic announcement days account for roughly **60% of the equity premium** once
   day-of-month fixed effects are included (AEA, "More than 100% of the equity premium"),
   i.e. announcement timing — not announcement content — is what generates the effect.
   The turn-of-the-month effect has been shown to be *entirely* attributable to
   announcement days (Fraser et al., Euronext).
3. **Data-mining artifact.** The literature on calendar effects is a warning as much as a
   result: across European markets the number of statistically significant calendar
   coefficients is "very similar to the number we would expect to find in random data", and
   the effects are country-specific rather than universal (de Peço/ISEG critique).

So the falsifiable claim is: **the month-of-year and week-of-month effects should shrink
sharply once announcement-date composition is controlled for, and should not survive
out-of-sample.** If they do survive, the seasonal signal is real and is tradeable with
the Homework 4 simulator.

### 3. Potential edge

1. **It is a self-refuting but testable claim**, and the test is cheap: the course dataset
   already contains the outcome variable and the calendar features; what is missing is the
   *event calendar*.
2. **Negative results are as valuable as positive ones here.** A demonstrated artifact
   would stop a whole class of "seasonal factor" strategies from being built on 33 stocks
   over 25 years — precisely the overfitting risk the Homework 3 model already exhibits
   (depth-10 tree with 299 features on ~33k test rows).
3. **The macro variables are already in the dataset** (`DGS10`, `FEDFUNDS`, `cpi_core_yoy`),
   so a partial control is possible immediately, and the full control needs only two
   additional calendars (earnings dates and macro release dates) — cheap, well-documented
   public data.

### 4. Data and sources

| Source | What it gives | Access |
|---|---|---|
| FRED / ALFRED | **ALFRED release dates** for CPI, PCE, payrolls, FOMC funds rate — the macro announcement calendar | `pandas_datareader` / `fredapi` (public, no key for `fredgraph.csv`) |
| Nasdaq / Yahoo earnings calendar | Announcement dates per ticker, so earnings clustering can be controlled for | `yf.Ticker.get_earnings_dates()` — **verified working in this environment** |
| Course dataset `stocks_df_combined_2026_09_18.parquet.brotli` | The 33-stock panel with 115 dummies, macro variables, and the target | Google Drive (course link) |
| The homework's own Wikipedia S&P 500 constituent list | Index-membership events, a third source of non-random event timing | `pandas.read_html` — **verified working** |

**Verified**: the earnings-dates API, the FRED/ALFRED release calendar and the Wikipedia
constituent list are all reachable and were exercised while producing this repo.
**To be verified**: (a) the exact ALFRED endpoint and whether release *dates* (not just
values) can be pulled programmatically in bulk; (b) historical earnings dates before the
yfinance ~24-quarter lookback, which will need a paid source for a 25-year panel — this is
the binding constraint on the design.

### 5. Research approach

1. **Baseline.** Reproduce the Homework 3 result: the `month_wom` correlation table, the
   decision-tree feature importances, and the precision of the depth-tuned tree.
2. **Control for announcement intensity.** Build, for each date, three counters:
   `n_earnings_releases`, `is_macro_release_day` (weighted by release importance), and
   `is_index_event_day`. Regress the target on the `month_wom` dummies *with* these
   counters and observe how the dummy coefficients move.
3. **Stratify instead of control.** Bin the panel by "high announcement-intensity" vs
   "low announcement-intensity" months and re-estimate the seasonal effect within each
   stratum. If the October premium only exists in high-intensity months, the artifact
   hypothesis survives; if it survives in low-intensity months, the seasonal hypothesis
   survives.
4. **Out-of-sample discipline.** Use the Homework 3 temporal split (train ≤2018,
   validation 2018–2022, test 2022–2026) and additionally re-estimate the dummy
   coefficients on the test window alone. A seasonal factor whose coefficient is
   estimated on the full sample and evaluated on the full sample is not a trading signal.
5. **Economic test.** For the surviving dummy, run the Module 04 trading simulation
   ($100 per positive prediction, 0.2% round-trip cost) on the test period and compare
   CAGR against `pred5_clf_10` and `pred10_rf_best_rule_55`. Report the trade-off: the
   dummy may add precision but at a much higher trade count.
6. **Trading calendar vs. event calendar.** A clean identification: compare a fixed-calendar
   bucket (week 4 of October) with an event-aligned bucket (the 10 trading days after a
   CPI release). If the event-aligned version dominates, the calendar effect was a proxy.

### 6. Validation and risks

- **Benchmark**: the macro-only decision tree from Homework 3 (`cpi_core_yoy`, `DGS10`,
  `FEDFUNDS`), and a random-permutation placebo in which the `month_wom` labels are
  shuffled across dates within each month — if the real dummies beat the placebo by no
  margin, the effect is noise.
- **Falsification**: (a) if the `month_wom` coefficients are unchanged after controlling for
  announcement counters, the artifact hypothesis is rejected and the seasonality is real;
  (b) if the effect is present in the test window with similar magnitude, it is not
  overfitting; (c) if the placebo performs equally well, it was always noise.
- **Risks**: **multiple-testing and data-mining risk is the central risk** — with 60
  `month_wom` dummies and a 2.5% correlation threshold, a spurious pick is close to
  guaranteed, and the course's own model already shows the cost of that (299 features,
  33k test rows); the macro variables are US-only while the tickers are US/EU/India, so
  "macro release timing" is not a common shock across the panel; earnings-date history
  before ~2021 is not available from free sources; and the whole framework assumes a
  stable data-generating process, which the 2008 and 2020 regimes violate.
- **What would make this a good project even if the seasonality is fake**: a rigorous
  demonstration that calendar features in this dataset are an artifact of event timing
  would be a clean, publishable-quality negative result — and it directly informs how the
  capstone model should be built (drop high-dimensional calendar features, control for
  event clustering).

---

## Note on the three ideas

Each idea satisfies the assignment's requirement of a **different economic mechanism**,
and each is deliberately falsifiable with a stated benchmark and a stated rejection
criterion. The common thread is the course's own stack: the Homework 4 trading simulator
provides the economic evaluation harness, and the Homework 2/3 feature-engineering
patterns provide the data plumbing. The third idea additionally feeds directly back into
the capstone model design — if calendar features are shown to be artifacts, the capstone
should not include them.
