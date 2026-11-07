# Stock Markets Analytics Zoomcamp 2026 — Homework Solutions

Solutions for the five homeworks of the
[DataTalks.Club Stock Markets Analytics Zoomcamp 2026](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp)
cohort, with the working code that produces every number.

Everything here was computed from live data (Yahoo Finance, Wikipedia, iposcoop, the
course's shared parquet) rather than transcribed, and each answer is cross-checked
against an independent anchor wherever the assignment provides one.

**Author:** Cavin Otieno ([OumaCavin](https://github.com/OumaCavin))

---

## Answer summary

### Homework 1 — Intro and Data Sources

| # | Question | Answer |
|---|----------|--------|
| 1 | Year with the most S&P 500 additions (from 2020) | **2025** (17 additions) |
| 2 | Indexes with better YTD returns than the S&P 500 (as of 2026-08-21) | **2** (Nikkei, TSX) |
| 3 | Median drawdown of ≥ 5% S&P 500 corrections | **8** (7.99%) |
| 4 | Median 2-day return after positive AMZN earnings surprises | **0.35** (0.353%) |

### Homework 2 — One Dataframe

| # | Question | Answer |
|---|----------|--------|
| 1 | Company class with the highest total withdrawn IPO value | **500** (Acquisition Corp, $485M) |
| 2 | Median Sharpe ratio of pre-Sep-2025 IPOs (at 2026-09-11) | **0.04** (0.048) |
| 3 | Optimal IPO holding period (1–12 months) | **1** (median −7.2%) |
| 4 | Net income of the RSI < 30 strategy | **65** ($65.8K) |

### Homework 3 — The Model

| # | Question | Answer |
|---|----------|--------|
| 1 | Highest \|correlation\| of a `month_wom` dummy with the target | **0.025** (`October_w4`) |
| 2 | Precision of the best new hand rule (`pred3`/`pred4`) | **0.588** (`pred3_manual_dgs10_5`) |
| 3 | TEST rows where `pred5_clf_10` is the only correct positive | **1171** |
| 4 | Optimal `max_depth` (1–12) | **4** (precision 0.6292) |

### Homework 4 — Trading System

| # | Question | Answer |
|---|----------|--------|
| 1 | CAGR of `pred10_rf_best_rule_55` with reduced dummies | **1.136** (1.135860) |
| 2 | RF threshold for < 5 average trades/day | **0.56** (CAGR 1.1177 at 4.40 trades/day) |
| 3 | Trades for `pred5_clf_10` with the redefined target (> 1.022) | **8** (8.01 trades/day) |

### Homework 5 — Automation & New Ideas

Three exploratory research ideas, each stated as market and scope, investment thesis,
potential edge, data and sources, research approach, and validation and risks. Every one
carries an explicit falsification criterion, so each can be rejected by a stated test
rather than argued about.

| # | Idea | Mechanism | Market | Horizon |
|---|------|-----------|--------|---------|
| 1 | Foreign-flow exhaustion and domestic absorption | Order-flow imbalance vs. retail absorption | Kenya (Nairobi Stock Exchange) | 5–20 trading days |
| 2 | MSCI index-review passive flows | Benchmarked passive demand | Global frontier markets | Announcement → effective date |
| 3 | Is the October/November seasonality real? | Calendar artefact vs. true seasonality | US / EU / India large caps | 30-day forward |

The three ideas deliberately differ in economic mechanism, market, data source and holding
period, so no two of them depend on the same bet. Idea 1 is a flow-reversal hypothesis
built on Kenya's shift to a domestic marginal buyer, idea 2 is a scheduled demand shock
with a published calendar, and idea 3 is a falsification test of the course's own
strongest calendar feature, whose outcome feeds straight back into how the capstone model
should be built.

Full write-ups with sources: [`homework5/README.md`](homework5/README.md).
Paste-ready version of the three form answers: [`homework5/submission_text.md`](homework5/submission_text.md).

---

## Repository layout

```
homework1/          Intro and data sources   -> src/hw1_questions.py,  output/hw1_answers.json
homework2/          One dataframe             -> src/hw2_questions.py,  output/hw2_answers.json
homework3/          The model                 -> src/hw3_questions.py,  output/hw3_answers.json
homework4/          Trading system            -> src/hw4_questions.py,  src/merge_results.py
homework5/          Three research ideas      -> README.md,  submission_text.md
scripts/            download_data.py - fetches the two Google Drive parquet files
```

Each homework directory has a `README.md` with the full answer, the method, the numbers
and the reasoning behind the exploratory questions.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py        # ~270 MB of parquet from Google Drive

python homework1/src/hw1_questions.py
python homework2/src/hw2_questions.py
python homework3/src/hw3_questions.py

python homework4/src/hw4_questions.py baseline    # each scenario in its own process:
python homework4/src/hw4_questions.py reduced     # the Random Forest needs a lot of RAM
python homework4/src/hw4_questions.py redefined
python homework4/src/merge_results.py
```

Runtime: roughly 2 minutes for Homework 1, 6 for Homework 2, 4 for Homework 3 and
about 5 minutes per Homework 4 scenario on 2 cores.

## Three things that decide whether these answers are right

Getting a number that *looks* plausible is easy here. Three specific traps had to be
avoided, and each of them silently produces a wrong answer:

**1. The Module 03 and Module 04 pipelines are not the same.** Homework 3 uses the
Module 03 notebook (299 features: 184 numerical + 115 dummies) and Homework 4 uses
Module 04 (301 features — `div_payout` and `stock_split` are added; the reduced setup
drops to 208). Mixing them changes the decision trees and therefore the counts.

**2. `scikit-learn` 1.8.0 silently changes the Random Forest.** With 1.7.x the
Homework 4 baseline reproduces the Module 04 notebook's published table *exactly*
(`pred10_rf_best_rule_55`: 3,111 trades, $27,000 capital, CAGR 1.093571). With 1.8.0+
the same code gives 3,130 trades and CAGR 1.110799. `requirements.txt` pins
`scikit-learn==1.7.2` for that reason. Every deterministic prediction matches in every
version, so this only shows up in the Random Forest.

**3. `pandas.get_dummies` column order changes the tree.** The one-hot columns are
sorted alphabetically, so the `Month` encoding determines the feature indices. A
Decision Tree breaks impurity ties by feature index, which changes which month dummy is
selected and, through that, the number of "uniquely correct" predictions: 1171 with
`Month` as the month name (the encoding that also produces the `October_w1` dummies the
assignment asks for) versus 1187 with `Month` as a number.

Two further details worth knowing if you re-run this:

- `yfinance` treats `end` as **exclusive**, so `end='2026-08-21'` returns data through
  2026-08-20. The answer to Homework 1 Q2 is the same under either convention.
- `DataFrame.corr()` on a frame that contains the target returns the target's
  self-correlation (1.0) as a row. It has to be dropped before ranking the dummies.

## Data sources

| Source | Used for | Access |
|---|---|---|
| [Yahoo Finance](https://finance.yahoo.com/) via `yfinance` | Prices, index levels, AMZN earnings dates | Free, no key |
| [Wikipedia S&P 500 constituents](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) | Index addition dates | Free (needs a browser-like TLS fingerprint) |
| [iposcoop.com](https://www.iposcoop.com/) | IPO filings and pricings | Free |
| [Course Google Drive parquet](https://drive.google.com/file/d/1oQSUMCs2DyQQIh8Y62UhrT00cIsE9Sr5/view) | Shared dataset for HW3/HW4 | Course link |
| [CMA Kenya bulletins](https://www.cma.or.ke/) | Foreign participation and net flows (HW5 idea 1) | Free PDF |
| [MSCI index reviews](https://app2.msci.com/eqb/fm/index_review.html) | Frontier index additions and deletions (HW5 idea 2) | Free PDF |
| [FRED / ALFRED](https://fred.stlouisfed.org/) | Macro release calendar (HW5 idea 3) | Free, no key |

---

Course materials are © [DataTalks.Club](https://datatalks.club/); this repository contains
my own solutions to the assignments.
