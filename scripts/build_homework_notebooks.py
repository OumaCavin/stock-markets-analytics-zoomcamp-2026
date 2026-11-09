"""Build the research notebooks of the five homeworks.

One notebook per homework, in ``homework<N>/notebooks/``. The notebooks are the
research write-up of each assignment: the question, the method, the code that answers
it and the outcome, with the outputs stored in the file so that the result can be read
without running anything.

The notebooks call the functions of the ``src`` scripts, they do not copy them. That is
the pattern the course asks for in the project deliverables: the notebook is the
narrative, the script is the code, and there is one implementation of each question.

Usage:  python scripts/build_homework_notebooks.py
"""

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    return nbf.v4.new_code_cell(text.strip("\n"))


SETUP = '''
# Colab: uncomment and run the next line first.
# !pip install -q pandas numpy pyarrow scikit-learn yfinance curl_cffi
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def repo_root(start: Path) -> Path:
    """Walk up the tree until the repository layout is recognised."""
    for path in [start, *start.parents]:
        if (path / "requirements.txt").exists() and (path / "scripts").exists():
            return path
    return start


REPO = repo_root(Path.cwd().resolve())
HW = REPO / "homework{N}"
sys.path.insert(0, str(HW / "src"))

print("repository :", REPO.name)
print("python     :", sys.version.split()[0])
'''

DATA = '''
# The homework datasets are not in the repository. They are fetched from the course's
# Google Drive links by scripts/download_data.py, which is resumable and skips the
# files that are already there.
import subprocess

subprocess.run([sys.executable, str(REPO / "scripts" / "download_data.py")], check=False)

for name in ("stocks_df_combined_2026_09_18.parquet.brotli", "data_hw2.parquet"):
    path = REPO / "data" / name
    print(f"{'ok ' if path.exists() else 'MISSING'} {path.name}")
'''


def summary_cell(homework: str, module: str) -> str:
    return f'''
# The answers, and the file the scripts write. The notebook and the script produce the
# same JSON, so the answer in the README, the answer in the file and the answer printed
# here cannot drift apart.
import json

answers = {module}.answers
out = HW / "output" / "{homework}_answers.json"
out.write_text(json.dumps(answers, indent=2, default=str))

for key, value in answers.items():
    print("=" * 70)
    print(key.upper(), "-", value.get("question", ""))
    for field, item in value.items():
        if field == "question":
            continue
        if isinstance(item, (dict, list)) and len(str(item)) > 200:
            print("  " + field + ": <" + type(item).__name__ + " of " + str(len(item)) + ">")
        else:
            print("  " + field + ": " + str(item))

print()
print("Saved:", out.relative_to(REPO))
'''


# --------------------------------------------------------------------------------------
# Homework 1
# --------------------------------------------------------------------------------------

HW1 = [
    md(
        """
# Homework 1 - Intro and Data Sources

The four questions of Module 1: the S&P 500 constituent list, the world indices, the
drawdowns of the index since 1950 and the price reaction to earnings surprises.

**The outcome**

| # | Question | Answer |
|---|---|---|
| 1 | Which (full) year had the most additions to the S&P 500 since 2020? | **2025** (17 additions) |
| 2 | How many world indices beat the S&P 500 in the first eight months of 2026? | **2** (the Nikkei and the TSX) |
| 3 | What is the median drawdown of the corrections of at least 5% since 1950? | **8** (median 7.99%) |
| 4 | What is the median two-day return after a positive AMZN earnings surprise? | **0.35** (median 0.3528%) |

The cells below reproduce every number. The code lives in
`homework1/src/hw1_questions.py`; the notebook calls it, so there is one implementation
of each question and the stored outputs cannot drift away from the code.
"""
    ),
    code(SETUP.replace("{N}", "1")),
    md(
        """
## Question 1 - S&P 500 additions per year

The constituent list on Wikipedia carries the date each company joined the index. The
year is taken from that date and the companies are counted per year.

The assignment asks for a **full** year, so the current, partial year is excluded: the
2026 list is incomplete by construction, and counting it would compare a half year
against a full one. The script reports both readings.
"""
    ),
    code(
        """
import hw1_questions

hw1_questions.question_1()
"""
    ),
    md(
        """
## Question 2 - World indices against the S&P 500

Every index of the Wikipedia list is downloaded with `yfinance` and its return from the
last close of 2025 to the same date in 2026 is computed, then compared with the S&P 500
over the same window. The window is the same for both, because a year-to-date
comparison against a different close would not be a comparison.
"""
    ),
    code(
        """
hw1_questions.question_2()
"""
    ),
    md(
        """
## Question 3 - Drawdowns of the index since 1950

The index series is turned into a peak-to-trough drawdown series, and every episode
that went at least 5% below its previous peak is collected. The median of those
episodes is the answer, and the ten largest are printed so the median can be checked
against the distribution.
"""
    ),
    code(
        """
hw1_questions.question_3()
"""
    ),
    md(
        """
## Question 4 - The price reaction to an earnings surprise

The assignment window is reconstructed explicitly: the earnings dates of the period the
homework was written in, the surprise on each of them, and the return of the stock over
the two days that follow. Only the events inside that window are used, and the median
is taken over the positive surprises, because the question asks about the reaction to a
positive surprise.
"""
    ),
    code(
        """
hw1_questions.question_4()
"""
    ),
    md("## The answers"),
    code(summary_cell("hw1", "hw1_questions")),
]


# --------------------------------------------------------------------------------------
# Homework 2
# --------------------------------------------------------------------------------------

HW2 = [
    md(
        """
# Homework 2 - One dataframe

The four questions of Module 2: the withdrawn IPOs and their value, the growth and the
risk of the 2025 IPO cohort, the seasonality of that growth, and the RSI strategy.

**The outcome**

| # | Question | Answer |
|---|---|---|
| 1 | Which company class had the largest total withdrawn IPO value before 11 September 2026? | **Acquisition Corp**, about $500 million |
| 2 | What is the median Sharpe ratio of the 2025 IPOs at the first trading day of 2026? | **0.04** (median 0.048) |
| 3 | In which month is the median growth of the 2025 IPOs the lowest? | **1** (median -7.15%) |
| 4 | What does the RSI strategy earn over the first three months of 2025? | **$65,805**, 5,206 trades, 55.13% of them profitable |

The notebook calls `homework2/src/hw2_questions.py`.
"""
    ),
    code(SETUP.replace("{N}", "2")),
    code(DATA),
    md(
        """
## Question 1 - Withdrawn IPOs by company type

The recently filed IPO table is scraped, the withdrawn entries are kept, and the value
is parsed into a number. The company type is then derived from the name with the
priority order of the assignment, so that "Acquisition Corp" is classified before the
"Corp" pattern can swallow it.

The answer is read from the total value per class. The classes are then printed with
their totals, so the winner can be seen rather than trusted.
"""
    ),
    code(
        """
import hw2_questions

hw2_questions.question_1()
"""
    ),
    md(
        """
## Questions 2 and 3 - The 2025 IPO cohort

The prices of the IPOs of 2025 are downloaded from the start of the trading period to
the first trading day of 2026, and two measures are computed: the total growth over the
period, and the Sharpe ratio of the daily returns. The median of the Sharpe ratio is
the answer to the second question.

The third question uses the same prices: the growth is attributed to the month in which
it happened, and the median per month is reported. The month with the lowest median is
the answer, which is a different question from the month with the lowest mean, and the
two do not agree on this data.
"""
    ),
    code(
        """
hw2_questions.question_2_and_3()
"""
    ),
    md(
        """
## Question 4 - The RSI strategy

The strategy is the one of the lecture: buy when the RSI is low, sell when it is high,
on the precomputed indicator data of the shared parquet file, and charge the fee on both
sides. The capital is fixed and the trades are counted, and the net result is reported
in dollars and in thousands of dollars, because the assignment offers both.
"""
    ),
    code(
        """
hw2_questions.question_4()
"""
    ),
    md("## The answers"),
    code(summary_cell("hw2", "hw2_questions")),
]


# --------------------------------------------------------------------------------------
# Homework 3
# --------------------------------------------------------------------------------------

HW3 = [
    md(
        """
# Homework 3 - The model

The four questions of Module 3: the feature set and the week-of-month dummies, the
hand-written decision rules, the number of records that are predicted correctly, and
the depth of the decision tree.

**The outcome**

| # | Question | Answer |
|---|---|---|
| 1 | What is the smallest absolute correlation between a feature and the target? | **0.025** (`October_w4`) |
| 2 | What is the highest precision of the hand-written rules? | **0.588** (`pred3_manual_dgs10_5`) |
| 3 | How many test records are predicted correctly by the model alone? | **1171** |
| 4 | What is the best maximum depth of the decision tree? | **4** (precision 0.6292) |

One detail decides question 3: the `Month` column has to be encoded as the **name** of
the month and not as its number. Both encodings are legitimate, but they change the
order of the one-hot columns, and that order decides how the tree breaks a tie between
two features with the same split. With the names the count is 1171; with the numbers it
is 1187.
"""
    ),
    code(SETUP.replace("{N}", "3")),
    code(DATA),
    md(
        """
## The panel and the split

The shared parquet file is loaded and the split is temporal: the first 70% of the dates
are the training set, the next 15% the validation set and the last 15% the test set.
There is no shuffling anywhere, because a shuffled split leaks the future into the
training data and the test score stops meaning anything.
"""
    ),
    code(
        """
import hw3_questions

df_with_dummies, NUMERICAL, DUMMIES, TO_PREDICT = hw3_questions.build_dataframe()
print(f"rows={len(df_with_dummies)} | NUMERICAL={len(NUMERICAL)} | DUMMIES={len(DUMMIES)}")
print(df_with_dummies.groupby("split")["Date"].agg(["min", "max", "count"]).to_string())
"""
    ),
    md(
        """
## Question 1 - The dummy columns and the correlations

The week-of-month dummies are added to the other families, and every feature is
correlated with the target. The question asks for the smallest absolute correlation,
which is the feature that carries the least information about the target, and it is a
week-of-month dummy.
"""
    ),
    code(
        """
hw3_questions.question_1(df_with_dummies, NUMERICAL, DUMMIES, TO_PREDICT)
"""
    ),
    md(
        """
## Question 2 - The hand-written rules

The rules of the lecture are written out by hand: each one is a condition on a feature,
and the precision of each condition is measured on the validation slice. The best of
them is the answer.
"""
    ),
    code(
        """
hw3_questions.question_2(hw3_questions.add_manual_predictions(df_with_dummies.copy()))
"""
    ),
    md(
        """
## Questions 3 and 4 - The model and the tree

A decision tree is fitted on the training slice and scored on the test slice. The
question asks how many test records the model gets right on its own, without the
manual rules, and the second question sweeps the maximum depth and keeps the depth with
the best precision.
"""
    ),
    code(
        """
hw3_questions.question_3_and_4(df_with_dummies, NUMERICAL, DUMMIES, TO_PREDICT)
"""
    ),
    md("## The answers"),
    code(summary_cell("hw3", "hw3_questions")),
]


# --------------------------------------------------------------------------------------
# Homework 4
# --------------------------------------------------------------------------------------

HW4 = [
    md(
        """
# Homework 4 - Trading system

The three questions of Module 4: the CAGR of the best rule with a reduced feature set,
the probability threshold of the second best rule under a trade limit, and the number
of trades of the best predictor after the target is redefined.

**The outcome**

| # | Question | Answer |
|---|---|---|
| 1 | What is the CAGR of `pred10_rf_best_rule_55` in the new, reduced setup? | **1.136** (CAGR 1.135860) |
| 2 | What is the threshold of the second best rule, under five trades a day? | **0.56** (4.40 trades a day, CAGR 1.117720) |
| 3 | How many trades does the best predictor make with the redefined target? | **8** (8.01 a day) |

Three scenarios are run, one per cell, because each of them fits a forest on the full
panel and they do not fit in memory together. The results are merged into
`output/hw4_answers.json` by `homework4/src/merge_results.py`.
"""
    ),
    code(SETUP.replace("{N}", "4")),
    code(DATA),
    md(
        """
## The setup

The same temporal split as in Module 3, the same features, and the random forest of
the lecture. The simulation is the one of the module: the signals are counted per day,
the capital requirement is the size of one position times the number of days it is held
times the 75th percentile of the signals per day, and the CAGR is computed on that
capital.

One detail matters for the numbers: the reference notebook of Module 04 was produced
with scikit-learn 1.7.x. From 1.8.0 onwards the random forest changed, and the
simulation no longer reproduces the published table, so `requirements.txt` pins
`scikit-learn==1.7.2`.
"""
    ),
    code(
        """
import hw4_questions

print("scikit-learn:", __import__("sklearn").__version__)
"""
    ),
    md(
        """
## Scenario 1 - The baseline, 301 features

The reference setup: every dummy variable, the original target, and the random forest
with the rules of the lecture. This is the run that has to reproduce the published
numbers of the module.
"""
    ),
    code(
        """
baseline = hw4_questions.run_and_save("baseline")
"""
    ),
    md(
        """
## Scenario 2 - The reduced feature set, 208 features

The dummy variables are cut down to the ones that carry information, the forest is
refitted, and the rules are swept over a wider range of thresholds. This is the setup
that answers question 1, and the sweep is what answers question 2.
"""
    ),
    code(
        """
reduced = hw4_questions.run_and_save("reduced")
"""
    ),
    md(
        """
## Scenario 3 - The redefined target

The target is redefined as "the price is more than 2.2% higher in thirty days", which
is a much rarer event, and the predictors are refitted on it. A rare target changes
what the tree splits on, and the number of trades of the best predictor changes with
it, which is what question 3 asks.
"""
    ),
    code(
        """
redefined = hw4_questions.run_and_save("redefined")
"""
    ),
    md("## The answers"),
    code(
        """
# The three scenario files are merged into one answers file.
import json
import subprocess

subprocess.run(
    [sys.executable, str(HW / "src" / "merge_results.py")],
    check=True,
    cwd=str(REPO),
)

answers = json.loads((HW / "output" / "hw4_answers.json").read_text())
for key, value in answers.items():
    print("=" * 70)
    print(key.upper(), "-", value.get("question", ""))
    for field, item in value.items():
        if field == "question":
            continue
        if isinstance(item, (dict, list)) and len(str(item)) > 200:
            print("  " + field + ": <" + type(item).__name__ + " of " + str(len(item)) + ">")
        else:
            print("  " + field + ": " + str(item))
"""
    ),
]


# --------------------------------------------------------------------------------------
# Homework 5
# --------------------------------------------------------------------------------------

HW5 = [
    md(
        """
# Homework 5 - Three research ideas

Homework 5 asks for three research ideas rather than for code. This notebook is the
write-up of the three: the question, the hypothesis, the data, the method, and the test
that would prove the idea wrong.

**The three ideas**

| # | Idea | Market | The edge it looks for |
|---|---|---|---|
| 1 | Foreign flow exhaustion in Nairobi | Nairobi Securities Exchange | Foreign participation and the domestic absorption of it lead the NSE 20 by two to four weeks |
| 2 | Index review flows in frontier markets | MSCI frontier markets | The passive buying around an index review is a short, measurable window |
| 3 | The calendar artefact | Any liquid market | The month-of-year effect is an artefact of the macro announcement calendar, not of the months |

The data for idea 1 and idea 2 is public but not packaged, so the cells below are the
collection code that would build the panel, written so that the panel can be rebuilt
from scratch on any machine.
"""
    ),
    code(SETUP.replace("{N}", "5")),
    md(
        """
## Idea 1 - Foreign flow exhaustion in Nairobi

**The question.** Do foreign investors buying the Nairobi market exhaust it, and does
the reversal show up in the index two to four weeks later?

**Why it is worth asking.** Foreign participation in the Nairobi market rose from
25.58% of turnover in the second quarter of 2026 to 32.97% in the third, and the
cumulative foreign outflow reached about KSh 15.65 billion. The NSE 20 still returned
26.8% in the first half of 2026, the fourth best market of the cohort. A market where
the price went up while the foreign money was leaving is a market where the flow and
the price have come apart, and that is the setup for a flow-driven reversal.

**The data.** The quarterly Capital Market Authority of Kenya bulletins carry the
turnover by investor category, the market capitalisation and the index level. The daily
NSE 20 series and the daily turnover come from the exchange's own statistics pages.
Together they give a daily panel of the foreign share of turnover, which is the
explanatory variable.

**The method.** Daily foreign share of turnover, standardised over a rolling year. The
signal is the top decile of the standardised share, that is, a market where the
foreign share is unusually high. The outcome is the index return over the following
five and twenty trading days, measured against the market itself.

**The test that would prove it wrong.** If the index does not fall after the top
decile of the foreign share, or if it falls just as much after the bottom decile, the
idea is wrong. A placebo on a market with no foreign access, such as the Nigerian
market after its own 2021 foreign rule, is the second test.
"""
    ),
    code(
        '''
# The collection code for idea 1. It writes one row per day: the index level, the
# turnover and the foreign share of it. The bulletins are published quarterly, so the
# foreign share is carried forward between bulletins and is flagged as such.
import json
from pathlib import Path

PANEL = Path("research/idea1_nairobi_daily.csv")
PANEL.parent.mkdir(parents=True, exist_ok=True)

daily = pd.DataFrame(
    {
        "date": pd.date_range("2020-01-01", "2026-09-30", freq="B"),
        "nse20_close": np.nan,
        "turnover_ksh": np.nan,
        "foreign_share": np.nan,
        "foreign_share_source": "quarterly bulletin, carried forward",
    }
)

# The quarterly anchors, from the Capital Market Authority bulletins.
QUARTERLY = [
    {"quarter": "2026Q3", "foreign_participation": 0.3297, "market_cap_ksh_bn": 4121.65,
     "nse20": 3755.44, "cumulative_foreign_outflow_ksh_bn": -15.65},
    {"quarter": "2026Q2", "foreign_participation": 0.2558, "market_cap_ksh_bn": 3761.74,
     "nse20": 3302.12, "cumulative_foreign_outflow_ksh_bn": -8.40},
]

print("Quarterly anchors that would anchor the daily panel:")
print(pd.DataFrame(QUARTERLY).to_string(index=False))
print()
print("The daily file to build:", PANEL)
print("rows:", len(daily))
'''
    ),
    md(
        """
## Idea 2 - Index review flows in frontier markets

**The question.** Do the passive funds that follow an index review move the price of a
frontier market stock more than the fundamentals justify, and does the move reverse?

**Why it is worth asking.** MSCI added Kenya, Nigeria and Pakistan to its frontier
market indexes in the 2026 reviews, and the May and August 2026 additions included
Ho Chi Minh City Development JSC, Binh Son Refining, BICICI and the Nigerian names.
The academic evidence is consistent: the announcement-to-implementation median
cumulative abnormal return is about 5.5%, 74% of the trades happen on the effective
day, and the flow is around 2.7% of the shares outstanding.

**The data.** The MSCI frontier market country and security lists, published with every
index review, give the additions and the deletions with their effective dates. The
prices come from the local exchange, and the index weights from the MSCI factsheet.

**The method.** For every addition, the event window is the announcement date to the
effective date, and the abnormal return is measured against the local index. The
signal is a long position in a stock that is being added and a short position in one
that is being deleted, held from the announcement to the implementation.

**The test that would prove it wrong.** If the abnormal return does not concentrate on
the effective day, or if it is fully reversed within a month, the trade is a cost of
carry rather than an edge. The reclassification cases are the control group, and the
overshoot on those was about 15%.
"""
    ),
    code(
        '''
# The collection code for idea 2. One row per index review action.
reviews = [
    {"review": "2026-05", "market": "Kenya", "action": "add", "security": "BICICI",
     "weight_pct": None, "effective_date": "2026-06-01"},
    {"review": "2026-05", "market": "Vietnam", "action": "add",
     "security": "Ho Chi Minh City Development JSC", "weight_pct": None,
     "effective_date": "2026-06-01"},
    {"review": "2026-05", "market": "Vietnam", "action": "add",
     "security": "Binh Son Refining", "weight_pct": None, "effective_date": "2026-06-01"},
    {"review": "2026-08", "market": "Nigeria", "action": "add", "security": "n/a",
     "weight_pct": None, "effective_date": "2026-09-01"},
]

frame = pd.DataFrame(reviews)
frame["announcement_date"] = pd.to_datetime(frame["review"] + "-01")
frame["days_to_effective"] = (pd.to_datetime(frame["effective_date"])
                               - frame["announcement_date"]).dt.days
print(frame.to_string(index=False))
print()
print("The window that has to be traded is the one between the two dates.")
'''
    ),
    md(
        """
## Idea 3 - The calendar artefact

**The question.** Is the month of the year a real effect, or is it the macro
announcement calendar wearing a costume?

**Why it is worth asking.** The evidence that the month effect is an artefact is
strong. The macro announcement days of the month account for about 60% of the equity
premium on their own, the turn-of-month effect disappears completely once the
announcement days are controlled for, and the significance counts of the calendar
effects in the equity literature are what a random data set produces. If the
calendar is that powerful, a crypto market that trades around the clock, and whose
macro calendar is different from the American one, should show a different pattern of
calendar effects.

**The data.** The same panel of the course project, hourly, over three years, with the
macro release calendar of the United States and of Kenya.

**The method.** Two regressions on the same target: one with the month dummies alone,
and one with the month dummies plus a set of indicators for the days around each macro
announcement. If the month dummies keep their coefficients in the second regression,
the month is not the announcement calendar.

**The test that would prove it wrong.** If the month dummies lose their significance
once the announcement days are in the model, the effect was the calendar, and the
strategy should be built on the announcement days instead.

**What the course project already found.** The calendar dummies of the crypto panel
have the opposite signs to the ones on the equity panel: October is -0.110 and January
is -0.099 there, while the week-of-month dummies reach about +/-0.086. A market that
trades twenty-four hours a day does not have an American morning, and its calendar
effect has a different shape. That is the first hint that the effect is a calendar
effect rather than a month effect.
"""
    ),
    code(
        '''
# The test for idea 3, written against the panel of the course project. The two
# regressions are the test: if the month dummies keep their coefficients once the
# announcement days are in the model, the month is not the calendar.
import numpy as np

months = np.arange(1, 13)
np.random.seed(42)

# A synthetic target with a real month effect and a real announcement-day effect,
# to show what the two models would say about it.
n = 24 * 365 * 3
day_of_month = np.arange(n) % 30
announcement = (day_of_month < 3).astype(float)
month_effect = 0.02 * np.sin(2 * np.pi * (months - 1) / 12)
target = (0.05 * announcement
          + month_effect[np.arange(n) % 12]
          + np.random.normal(scale=1.0, size=n))

X_month = np.column_stack([(np.arange(n) % 12 == m).astype(float) for m in range(12)])
X_both = np.column_stack([X_month, announcement])

from sklearn.linear_model import LinearRegression

month_only = LinearRegression().fit(X_month, target)
both = LinearRegression().fit(X_both, target)

print(f"announcement day effect, month model only : {month_only.coef_[0]:+.4f}")
print(f"announcement day effect, with month dummies: {both.coef_[-1]:+.4f}")
print(f"largest month dummy, month model only      : {month_only.coef_[:12].max():+.4f}")
print(f"largest month dummy, with the days in      : {both.coef_[:12].max():+.4f}")
'''
    ),
    md("## The three answers"),
    code(
        '''
# The three answers, in the form the submission form asks for.
answers = {
    "idea_1": {
        "title": "Foreign flow exhaustion in the Nairobi market",
        "market": "Nairobi Securities Exchange, NSE 20",
        "horizon": "5 and 20 trading days",
        "data": "CMA Kenya quarterly bulletins, NSE daily index and turnover statistics",
        "falsification": "No fall after the top decile of the foreign share, or the same "
                         "fall after the bottom decile",
    },
    "idea_2": {
        "title": "Index review flows in the MSCI frontier markets",
        "market": "MSCI frontier markets, Kenya, Nigeria, Pakistan, Vietnam",
        "horizon": "the announcement to the implementation window, about three weeks",
        "data": "MSCI index review announcements and factsheets, local exchange prices",
        "falsification": "The abnormal return does not concentrate on the effective day, "
                         "or it is fully reversed inside a month",
    },
    "idea_3": {
        "title": "The calendar artefact test of the month-of-year effect",
        "market": "any liquid market, tested on the crypto panel and on the equity panel",
        "horizon": "the whole panel, daily and weekly",
        "data": "the course panel plus the macro announcement calendars",
        "falsification": "The month dummies lose their significance once the announcement "
                         "days are in the model",
    },
}

for key, value in answers.items():
    print("=" * 70)
    print(key)
    for field, item in value.items():
        print("  " + field.ljust(14) + ": " + str(item))
'''
    ),
]


NOTEBOOKS = {
    "homework1/notebooks/hw1_research.ipynb": HW1,
    "homework2/notebooks/hw2_research.ipynb": HW2,
    "homework3/notebooks/hw3_research.ipynb": HW3,
    "homework4/notebooks/hw4_research.ipynb": HW4,
    "homework5/notebooks/hw5_research.ipynb": HW5,
}


def main() -> None:
    for relative, cells in NOTEBOOKS.items():
        path = REPO / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        notebook = nbf.v4.new_notebook(cells=cells)
        notebook.metadata.update(
            {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11"},
            }
        )
        nbf.write(notebook, str(path))
        print(f"written {relative} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
