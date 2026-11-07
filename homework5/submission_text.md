# Homework 5 — paste-ready submission text

Paste the three blocks below into the matching DataTalks form boxes. Each follows the
assignment's required structure: market and scope, investment thesis, potential edge,
data and sources, research approach, and validation and risks.

---

## Question 1 — Foreign-flow exhaustion and domestic absorption on the Nairobi Stock Exchange

**Market and scope.** Kenya's Nairobi Stock Exchange, focused on the liquid large caps
(Safaricom, KCB, EABL, Equity) with the NSE 20 / NASI as benchmark. Horizon is 5 to 20
trading days, measured from the date a flow signal becomes observable. This is a
short-horizon, flow-driven study, not a fundamental one.

**Investment thesis.** Kenya's equity market has changed who the marginal buyer is.
Average foreign participation fell to 25.58% in Q2 2026 from 32.27% in Q1 2026, and
foreign investors have been net sellers with cumulative net outflows of about KSh 15.65
billion since January 2026 (in Q3 2026 alone foreign investors bought KSh 21.7bn while
selling KSh 27.4bn). Yet the market rallied hard: NASI returned 26.8% in H1 2026, the
fourth best-performing stock market in the world in that half-year, and market
capitalisation crossed KSh 4 trillion in August 2026. The thesis is that this divergence
is a regime change in the marginal buyer rather than a contradiction. The single-unit
trading change introduced in August 2025, which lets retail buy one share instead of a
full board lot, widened domestic depth enough to absorb foreign supply. Once domestic
depth replaces foreign marginality, a large foreign net outflow should predict positive
subsequent returns rather than negative ones: the classic "flow drives price"
relationship inverts.

**Potential edge.** Three sources of edge, none of which require forecasting Kenyan GDP.
First, an information lag: the CMA publishes foreign participation and net flows monthly
and quarterly, so a model built on published flow data is trading on information that is
already public but not yet in the price. Second, slow repricing of a structural change:
the single-unit trading regime altered the shape of the demand curve, and the market's own
price discovery has not fully absorbed it. Investors keep applying the old "foreign selling
equals bearish" heuristic, which is a level effect rather than a timing effect. Third,
flow concentration: foreign selling concentrates in exactly the two most liquid counters
(KCB and Safaricom supplied about 54% of session turnover on 22 July 2026), so a
market-level flow signal has a specific, tradeable transmission channel rather than being
diffuse noise.

**Data and sources.** CMA Quarterly Statistical Bulletin (Q2 and Q3 2026) for monthly
foreign participation percentages, net foreign purchases and sales, turnover, market
capitalisation and investor counts; CMA Capital Markets Soundness Report Q1 FY2026/27 for
monthly net foreign flows and concentration ratios; NSE daily OHLCV and turnover, with
yfinance covering the Kenyan listings directly; Central Bank of Kenya interbank rate, FX
and T-bill yields as the macro regime the signal is conditioned on; NASI and the NSE 20 as
benchmarks. The CMA bulletins were opened and the figures read directly from them. To be
verified: whether NSE publishes a machine-readable daily investor-flow file, since if not,
flows must be interpolated at monthly frequency, which materially reduces the number of
independent observations.

**Research approach.** Build a monthly panel from the earliest CMA bulletin to the present:
NASI return, foreign participation percentage, net foreign flow in KSh bn, total turnover
and market capitalisation. Form the signal FlowShock = -(net foreign flow / market cap)
bucketed into deciles and interacted with the change in foreign participation, which
separates "foreign selling that domestic money absorbed" from "foreign selling that
dominated". Test by regressing monthly and weekly index and stock-level returns on the
signal with Newey-West standard errors, and run a placebo on pre-2025 data, before
single-unit trading, to see whether the effect exists only in the new regime. Convert to a
trade: after a flow-shock decile, buy an NSE 20 equal-weight basket for 10 trading days,
funded with the Homework 4 capital model (capital = 100 x 30 x q75 investments per day) and
evaluated for CAGR and max drawdown with the Module 04 simulator. Finally test the
transmission channel directly by splitting the liquid counters from the illiquid ones: if
the thesis is about domestic absorption of foreign supply, the effect must be strongest
where foreign flow is concentrated.

**Validation and risks.** Benchmarks are buy-and-hold NASI and a zero-flow strategy that
trades the same number of days, which separates "the signal works" from "it merely avoids
bad months". Falsification: if the return following large foreign outflows is still
negative in the post-August-2025 sample, the thesis is rejected; if the effect disappears
once turnover is controlled for, it is a liquidity artefact rather than a flow signal.
Risks: the signal is slow by construction, so it competes on capital efficiency rather
than turnover; the flow data are monthly, so the sample is small and statistical power is
genuinely limited; Kenyan market concentration of up to 82.9% of capitalisation in ten
names means single-stock risk dominates; the honest benchmark is the post-publication
return, not the post-flow return, because the signal may already be priced when it
becomes observable; and CMA methodology changes can break the series.

---

## Question 2 — MSCI index-review passive flows in frontier markets

**Market and scope.** Single stocks entering or already inside the MSCI Frontier Markets
Index, which spans 28 countries and 246 constituents and covers roughly 85% of
free-float-adjusted market capitalisation per country. Concretely testable 2026 cases are
already visible in the public record: the May 2026 review's additions (Ho Chi Minh City
Development Joint Stock Commercial Bank and Binh Son Refining and Petrochemical in Vietnam,
BICICI in Ivory Coast) and the August 2026 additions in Nigeria, alongside Pakistan's
addition to Frontier Markets. Horizon runs from announcement to effective date, typically
11 to 17 days, plus a 60-day post-effective window.

**Investment thesis.** Index inclusion is not an information event, it is a scheduled
demand shock. When a security is added to a benchmark institutions track, the mechanical
buying that follows is large relative to the float in small markets. The academic record is
specific: the announcement-to-implementation median cumulative abnormal return for
additions is around 5.5%, with foreign investors buying on the order of 2.7% of shares
outstanding and roughly 74% of the trades occurring on the effective date itself
(Williams, The Anatomy of Index Rebalancings). Earlier work found about +3.4% on the day
after announcement and a further +4.5% into the effective date, with the effect markedly
stronger in emerging markets (Chakrabarti, Huang, Jayaraman and Lee, JFQA, 2005). The
thesis is that the effect scales inversely with free float and analyst coverage, so it
should be larger in frontier markets than the developed-market estimates the literature
reports, and that frontier liquidity is too thin for arbitrageurs to fully absorb the
demand shock, leaving a persistent post-effective drift rather than a one-day pop.

**Potential edge.** The dates are public and calendar-driven: MSCI publishes the
additions and deletions list and the announcement date well in advance, so there is no
forecasting required and the research question is about magnitude and persistence rather
than prediction. Frontier markets are structurally under-covered: MSCI rates Kenya,
Bangladesh, Pakistan, Sri Lanka and Vietnam as Frontier or Standalone, meaning limited
sell-side coverage, wide spreads and small free floats, so the same demand shock should
produce a larger percentage move and unwind more slowly here. The reversion side is
under-studied as well: the NBER work on MSCI market reclassifications from Frontier to
Emerging shows overshoots of roughly +15% between announcement and effective date that
fully revert within a year, and the mirror image for downgrades, at -9.2% in the
announcement-to-effective window for markets moving to less-benchmarked status.

**Data and sources.** MSCI Index Reviews, published per Feb/May/Aug/Nov cycle, for exact
constituent lists, announcement dates and effective dates; MSCI Market Classification
together with the MSCI 2026 Global Market Accessibility Review for which markets are
Frontier versus Standalone and for free-float and foreign-ownership constraints; the MSCI
Frontier Markets Index factsheet for the 28-country, 246-constituent universe; local
exchange price data (Yahoo Finance for the larger names, exchange feeds for the rest) for
daily OHLCV and turnover around the event; and MSCI country index series as the benchmark
for abnormal-return computation. The classification, review cadence and the specific 2026
additions are all from MSCI's own published documents. To be verified: whether historical
frontier constituent lists can be reconstructed for a long enough sample, since public
review files cover only recent years; actual free-float percentages per name, which are
not public; and price coverage for the smaller frontier names.

**Research approach.** For each review between 2019 and 2026, collect additions and
deletions with announcement and effective dates. For each affected stock compute daily
abnormal returns against its local MSCI country index, or the local broad index where the
country index is unavailable, and build cumulative abnormal returns over windows AD-5 to
AD+1, AD+1 to CD, and CD+1 to CD+60. The core test is cross-sectional: run the same event
study on developed, emerging and frontier samples and test whether the frontier
coefficient is significantly larger, since "stronger in emerging markets" is the
literature's finding and the frontier comparison is the contribution. The second test
relates the abnormal return to float tightness, proxied by the log of market
capitalisation or the MSCI-reported foreign ownership limit, and to pre-event turnover: if
the demand-shock story is right, the effect should scale with float tightness. The third
test asks whether the pre-effective drift reverses after the effective date, which decides
whether the effect is tradeable only in the announcement window or for months. Convert to
a strategy: a long-only event portfolio of frontier additions, financed with the Module 04
capital model and evaluated on CAGR and max drawdown, and critically costed for the
untradeable half, since half the position cannot be exited on the effective date in a
market with a few hundred trades a month.

**Validation and risks.** Benchmarks are the MSCI Frontier Markets Index over the same
calendar window and a matched non-event portfolio of frontier stocks of similar size and
liquidity. Falsification: if frontier additions do not show a larger cumulative abnormal
return than developed-market additions, the thin-float thesis is rejected; if the entire
effect is concentrated on the announcement day and reverses immediately, there is no
tradeable edge. Risks: the sample is small, with a handful of names per review, and
announcement dates can shift; MSCI has moved Pakistan into Frontier Markets while keeping
Nigeria as a Standalone market, so classification itself is a live variable that must be
handled as a treatment rather than a fixed label; transaction costs in frontier markets
are large relative to the expected edge; and short selling is rated as not permitted in
most frontier markets under MSCI's own accessibility review, so the deletion side of the
trade cannot be expressed as a short.

---

## Question 3 — Is the October/November 30-day growth effect real, or a calendar artefact?

**Market and scope.** The course's shared dataset: the 33 largest listed companies of the
US, the EU and India, daily from 2000-01-03 to 2026-09-18. The target is the course's own
is_positive_growth_30d_future binary and the focal features are the month_wom dummies
built in Homework 3, of which October_w4 (correlation +0.0246) and November_w3 (+0.0233)
are the most correlated with the target, with February_w1 (-0.0209) the most negative.
Horizon is 30 trading days forward, matching the target definition.

**Investment thesis.** A correlation of about 0.025 between a calendar dummy and a 30-day
forward growth binary is tiny in isolation but non-trivial relative to everything else in
the model: in Homework 3 the best decision tree was dominated by macro variables
(cpi_core_yoy, DGS10, FEDFUNDS), and the calendar dummies were the only structural
feature to surface. The thesis is that the October and November signal is not a genuine
seasonal premium but a proxy for information-event timing, with three candidate
mechanisms. Earnings clustering: Q3 reporting is concentrated in mid-to-late October for
both European and US issuers, and stocks earn a premium in scheduled
earnings-announcement months (Frazzini and Lamont, 2006), so October may be a proxy for
"the market is full of fresh fundamental information". Macro release clustering: CPI prints
in the middle of the month and other key releases land late in the month, and research on
the equity premium finds that macroeconomic announcement days account for roughly 60% of
the equity premium once day-of-month fixed effects are included (AEA, "More than 100% of
the equity premium"), meaning announcement timing rather than announcement content
generates the effect; the turn-of-the-month effect has been shown to be entirely
attributable to announcement days (Fraser et al., Euronext). Data-mining artefact: across
European markets the number of statistically significant calendar coefficients is very
similar to the number expected in random data, and the effects are country-specific rather
than universal (de Peço/ISEG critique). The falsifiable claim is that the month-of-year
and week-of-month effects should shrink sharply once announcement-date composition is
controlled for, and should not survive out of sample.

**Potential edge.** The claim is self-refuting but testable, and the test is cheap: the
course dataset already contains the outcome variable and the calendar features, and what
is missing is only the event calendar. Negative results are as valuable as positive ones
here, because a demonstrated artefact would stop a whole class of seasonal factor
strategies from being built on 33 stocks over 25 years, which is precisely the
overfitting risk the Homework 3 model already exhibits with a depth-10 tree on 299
features and about 33,000 test rows. The macro variables are already in the dataset, so a
partial control is possible immediately, and the full control needs only two additional
calendars, earnings dates and macro release dates, both cheap and well-documented public
data.

**Data and sources.** FRED and ALFRED for the release dates of CPI, PCE, payrolls and the
FOMC funds rate, which together form the macro announcement calendar, accessible via
pandas_datareader or fredapi without a key for fredgraph.csv; the Nasdaq and Yahoo
earnings calendar for announcement dates per ticker, via yf.Ticker.get_earnings_dates(),
which was verified working in this environment; the course dataset
stocks_df_combined_2026_09_18.parquet.brotli for the 33-stock panel with its 115 dummies,
macro variables and the target; and the Wikipedia S&P 500 constituent list used in
Homework 1 for index-membership events, a third source of non-random event timing. The
earnings-dates API, the FRED and ALFRED release calendar and the Wikipedia constituent list
were all reachable and exercised while producing this repository. To be verified: whether
historical earnings dates before the roughly 24-quarter yfinance lookback are obtainable
from a free source, which is the binding constraint on the design, and whether ALFRED
release dates can be pulled programmatically in bulk.

**Research approach.** Start from the baseline: reproduce the Homework 3 month_wom
correlation table, the decision-tree feature importances and the precision of the
depth-tuned tree. Then control for announcement intensity by building, for each date, three
counters: n_earnings_releases, is_macro_release_day weighted by release importance, and
is_index_event_day, and regressing the target on the month_wom dummies with those
counters included to observe how the dummy coefficients move. Next, stratify instead of
controlling: bin the panel into high and low announcement-intensity months and re-estimate
the seasonal effect within each stratum, since if the October premium only exists in
high-intensity months the artefact hypothesis survives, and if it survives in
low-intensity months the seasonal hypothesis survives. Apply out-of-sample discipline
using the Homework 3 temporal split and re-estimating the dummy coefficients on the test
window alone, because a seasonal factor estimated and evaluated on the full sample is not
a trading signal. Run the economic test with the Module 04 simulation on the test period,
comparing CAGR against pred5_clf_10 and pred10_rf_best_rule_55 and reporting the
trade-off, since the dummy may add precision at a much higher trade count. Finally make
the identification clean by comparing the fixed-calendar bucket of week 4 of October with
an event-aligned bucket covering the ten trading days after a CPI release: if the
event-aligned version dominates, the calendar effect was a proxy.

**Validation and risks.** Benchmarks are the macro-only decision tree from Homework 3 and
a random-permutation placebo in which the month_wom labels are shuffled across dates
within each month; if the real dummies beat the placebo by no margin, the effect is
noise. Falsification: if the month_wom coefficients are unchanged after controlling for
announcement counters, the artefact hypothesis is rejected and the seasonality is real; if
the effect is present in the test window at similar magnitude it is not overfitting; if the
placebo performs equally well, it was always noise. Risks: multiple-testing and
data-mining risk is the central risk, because with 60 month_wom dummies and a 2.5%
correlation threshold a spurious pick is close to guaranteed, and the course's own model
already shows the cost of that; the macro variables are US-only while the tickers span the
US, EU and India, so macro release timing is not a common shock across the panel;
historical earnings dates before roughly 2021 are not available from free sources; and the
framework assumes a stable data-generating process, which the 2008 and 2020 regimes
violate. What would make this a good project even if the seasonality turns out to be
fake: a rigorous demonstration that calendar features in this dataset are an artefact of
event timing would be a clean, publishable-quality negative result, and it directly
informs how the capstone model should be built, namely by dropping high-dimensional
calendar features and controlling for event clustering.
