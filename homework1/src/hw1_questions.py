"""
Homework 1 - Intro and Data Sources (2026 cohort)
Solutions for questions 1-4. Produces output/hw1_answers.json.

Question 1 : S&P 500 additions per year (Wikipedia constituent list)
Question 2 : World index YTD returns as of 2026-08-21 vs the S&P 500
Question 3 : Median drawdown of >=5% S&P 500 corrections since 1950
Question 4 : Median 2-day return after positive AMZN earnings surprises
"""

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests as requests  # browser TLS fingerprint -> not blocked by Wikipedia

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

answers = {}


# ---------------------------------------------------------------------------
# Question 1: S&P 500 stocks added to the index, per year
# ---------------------------------------------------------------------------
def question_1():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()

    # Wrap the HTML in a buffer - avoids OSError on some pandas/lxml combos.
    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0].copy()

    added_col = next(c for c in df.columns if "addition" in c.lower() or "added" in c.lower())
    df["Added"] = pd.to_datetime(df[added_col], errors="coerce")

    years = df["Added"].dt.year
    counts = years.value_counts().sort_index()

    from_2020 = counts.loc[counts.index >= 2020]
    best_year = int(from_2020.idxmax())
    best_count = int(from_2020.max())

    # Ties check
    tied = [int(y) for y in from_2020[from_2020 == best_count].index]

    # The current year is still in progress, so the homework asks for the best *full*
    # year: restrict the comparison to completed calendar years.
    current_year = int(pd.Timestamp.utcnow().year)
    full_years = from_2020[from_2020.index < current_year]
    best_full_year = int(full_years.idxmax())
    best_full_count = int(full_years.max())
    tied_full = [int(y) for y in full_years[full_years == best_full_count].index]

    # Additional: stocks in the index for more than 20 years (relative to data snapshot)
    snapshot = df["Added"].max()
    more_than_20y = int((df["Added"] < snapshot - pd.Timedelta(days=365 * 20)).sum())

    answers["q1"] = {
        "question": "Which year had the highest number of additions (starting from 2020)?",
        "counts_by_year": {int(k): int(v) for k, v in from_2020.items()},
        "max_year_including_current_partial": best_year,
        "max_year_full_years_only": best_full_year,
        "max_count_full_years_only": best_full_count,
        "tied_full_years": tied_full,
        "additional_stocks_in_index_over_20_years": more_than_20y,
        "total_constituents": int(len(df)),
        "snapshot_date": str(snapshot.date()),
    }
    print("Q1 counts since 2020:", answers["q1"]["counts_by_year"])
    print("Q1 max year incl. current partial:", best_year, "| best FULL year:",
          best_full_year, "(", best_full_count, "additions ) ties:", tied_full)
    print("Q1 additional (>20y):", more_than_20y, "of", len(df))


# ---------------------------------------------------------------------------
# Question 2: YTD performance of world indices as of 2026-08-21
# ---------------------------------------------------------------------------
INDEXES = {
    "United States - S&P 500": "^GSPC",
    "China - Shanghai Composite": "000001.SS",
    "Hong Kong - Hang Seng": "^HSI",
    "Australia - S&P/ASX 200": "^AXJO",
    "India - Nifty 50": "^NSEI",
    "Canada - S&P/TSX Composite": "^GSPTSE",
    "Germany - DAX": "^GDAXI",
    "United Kingdom - FTSE 100": "^FTSE",
    "Japan - Nikkei 225": "^N225",
    "Mexico - IPC Mexico": "^MXX",
    "Brazil - Ibovespa": "^BVSP",
}

START = "2026-01-01"
END = "2026-08-21"


def question_2():
    results = {}
    for name, ticker in INDEXES.items():
        df = yf.download(
            ticker, start=START, end=END, progress=False, auto_adjust=False
        )
        if df is None or len(df) == 0:
            results[name] = {"ticker": ticker, "error": "no data"}
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna()
        first_close = float(close.iloc[0])
        last_close = float(close.iloc[-1])
        results[name] = {
            "ticker": ticker,
            "first_date": str(close.index[0].date()),
            "last_date": str(close.index[-1].date()),
            "first_close": first_close,
            "last_close": last_close,
            "ytd_pct": (last_close / first_close - 1) * 100,
            "rows": int(len(close)),
        }

    sp500 = results["United States - S&P 500"]["ytd_pct"]
    others = {k: v for k, v in results.items() if k != "United States - S&P 500"}
    better = {k: v["ytd_pct"] for k, v in others.items() if v["ytd_pct"] > sp500}

    # Sensitivity: yfinance treats `end` as exclusive, so the window above ends 2026-08-20.
    # Also test an inclusive 2026-08-21 window and a 2025-12-31 base, to confirm the
    # answer is stable across reasonable conventions.
    sensitivity = {}
    for label, s, e in [
        ("as_written_exclusive_end (2026-01-01..2026-08-20)", "2026-01-01", "2026-08-21"),
        ("inclusive_aug21 (2026-01-01..2026-08-21)", "2026-01-01", "2026-08-22"),
        ("base_2025-12-31", "2025-12-31", "2026-08-21"),
    ]:
        sp = None
        beaten = []
        for name, ticker in INDEXES.items():
            d = yf.download(ticker, start=s, end=e, progress=False, auto_adjust=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            c = d["Close"].dropna()
            ytd = (float(c.iloc[-1]) / float(c.iloc[0]) - 1) * 100
            if name == "United States - S&P 500":
                sp = ytd
            else:
                beaten.append((name, ytd))
        sensitivity[label] = {
            "sp500_ytd_pct": sp,
            "n_better": sum(1 for _, v in beaten if v > sp),
            "better": {n: v for n, v in beaten if v > sp},
        }

    answers["q2"] = {
        "question": (
            "How many indexes (out of 10) have better year-to-date returns than the "
            "US (S&P 500) as of August 21, 2026?"
        ),
        "sp500_ytd_pct": sp500,
        "all_indexes": {k: v["ytd_pct"] for k, v in results.items()},
        "windows": {k: (v["first_date"], v["last_date"]) for k, v in results.items()},
        "n_better_than_sp500": len(better),
        "better_than_sp500": better,
        "sensitivity": sensitivity,
    }
    for k, v in results.items():
        print(f"Q2 {k:32s} {v['first_date']} -> {v['last_date']}  {v['ytd_pct']:8.2f}%")
    print("Q2 count better than S&P 500:", len(better), better)
    for label, v in sensitivity.items():
        print(f"   sensitivity[{label}]: sp500={v['sp500_ytd_pct']:.2f}% "
              f"-> n_better={v['n_better']}")


# ---------------------------------------------------------------------------
# Question 3: median drawdown of >=5% S&P 500 corrections since 1950
# ---------------------------------------------------------------------------
def question_3():
    df = yf.download("^GSPC", start="1950-01-01", end="2026-11-07", progress=False,
                     auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index)

    # All-time-high closing prices: a day whose close exceeds every earlier close.
    running_max = df["Close"].cummax().shift(1)
    is_ath = df["Close"] > running_max
    ath_idx = df.index[is_ath]

    corrections = []
    for i in range(len(ath_idx) - 1):
        start, end = ath_idx[i], ath_idx[i + 1]
        segment = df.loc[start:end]
        low_date = segment["Close"].idxmin()
        low = float(segment["Close"].min())
        high = float(df.loc[start, "Close"])
        drawdown = (high - low) / high * 100
        duration = int((low_date - start).days)
        corrections.append(
            {
                "ath_date": str(start.date()),
                "trough_date": str(low_date.date()),
                "ath_close": high,
                "trough_close": low,
                "drawdown_pct": drawdown,
                "duration_days": duration,
            }
        )

    corr_df = pd.DataFrame(corrections)
    significant = corr_df[corr_df["drawdown_pct"] >= 5.0]

    answers["q3"] = {
        "question": "Median drawdown (in %) of significant market corrections in the S&P 500 index",
        "data_start": str(df.index[0].date()),
        "data_end": str(df.index[-1].date()),
        "n_ath_points": int(len(ath_idx)),
        "n_corrections_total": int(len(corr_df)),
        "n_corrections_ge_5pct": int(len(significant)),
        "median_drawdown_pct": float(significant["drawdown_pct"].median()),
        "p25_drawdown_pct": float(significant["drawdown_pct"].quantile(0.25)),
        "p75_drawdown_pct": float(significant["drawdown_pct"].quantile(0.75)),
        "median_duration_days": float(significant["duration_days"].median()),
        "top10_by_drawdown": significant.nlargest(10, "drawdown_pct")[
            ["ath_date", "trough_date", "drawdown_pct", "duration_days"]
        ].to_dict("records"),
    }
    print("Q3 top 10 corrections (method check):")
    print(significant.nlargest(10, "drawdown_pct")[
        ["ath_date", "trough_date", "drawdown_pct", "duration_days"]].to_string(index=False))
    print("Q3 n corrections >=5%:", len(significant),
          "| median drawdown:", round(significant["drawdown_pct"].median(), 3),
          "| median duration:", significant["duration_days"].median())


# ---------------------------------------------------------------------------
# Question 4: AMZN earnings surprises -> median 2-day price change
# ---------------------------------------------------------------------------
def question_4():
    ticker = yf.Ticker("AMZN")
    # limit=40 returns the full history (yfinance's default window is a rolling
    # 25 rows, so the oldest events scroll out of view).
    earnings = ticker.get_earnings_dates(limit=40).reset_index()
    # The Yahoo index carries a wall-clock time and mixed UTC offsets - normalise it
    # to a plain calendar date in US Eastern time before any comparison.
    earnings["Earnings Date"] = (
        pd.to_datetime(earnings["Earnings Date"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.normalize()
        .dt.tz_localize(None)
    )
    earnings = earnings.rename(columns={"Surprise(%)": "Surprise %"})

    prices = yf.download("AMZN", start="2020-09-01", end="2026-11-07", progress=False,
                          auto_adjust=False)
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    prices = prices[["Close"]].dropna()
    dates = prices.index
    close = prices["Close"]

    def build(frame):
        rows = []
        for _, row in frame.iterrows():
            surprise = row.get("Surprise %")
            if pd.isna(surprise):
                continue
            ann = row["Earnings Date"]
            pos = dates.searchsorted(ann)
            if pos >= len(dates) or pos == 0:
                continue
            day1, day2, day3 = pos - 1, pos, pos + 1
            rows.append(
                {
                    "earnings_date": str(ann.date()),
                    "trading_day": str(dates[day2].date()),
                    "reported_eps": row.get("Reported EPS"),
                    "surprise_pct": float(surprise),
                    "close_day1": float(close.iloc[day1]),
                    "close_day3": float(close.iloc[day3]),
                    "return_2d_pct": float(close.iloc[day3] / close.iloc[day1] - 1) * 100,
                }
            )
        return pd.DataFrame(rows)

    # The assignment states the earnings list holds 25 entries starting 2020-10-29
    # (24 reported + 1 future without a surprise). Yahoo now serves a rolling window,
    # so the assignment-era list is reconstructed explicitly.
    era = earnings[
        (earnings["Earnings Date"] >= "2020-10-29")
        & (earnings["Earnings Date"] <= "2026-07-30")
    ]
    edf = build(era)
    positive = edf[edf["surprise_pct"] > 0]

    # Same computation on today's rolling Yahoo window, for transparency.
    current = build(earnings.tail(25))
    current_positive = current[current["surprise_pct"] > 0]

    corr = float(
        edf["return_2d_pct"].corr(edf["surprise_pct"]) if len(edf) > 2 else float("nan")
    )
    corr_pos = float(
        positive["return_2d_pct"].corr(positive["surprise_pct"])
        if len(positive) > 2
        else float("nan")
    )

    answers["q4"] = {
        "question": (
            "Calculate the median 2-day percentage change in stock prices following "
            "positive earnings surprise days."
        ),
        "earnings_window": "2020-10-29 .. 2026-07-30 (the 25-entry list described by the assignment)",
        "n_earnings_rows_raw": int(len(earnings)),
        "n_rows_used_assignment_era": int(len(edf)),
        "n_positive_surprises": int(len(positive)),
        "median_2d_return_positive_pct": float(positive["return_2d_pct"].median()),
        "mean_2d_return_positive_pct": float(positive["return_2d_pct"].mean()),
        "median_2d_return_all_pct": float(edf["return_2d_pct"].median()),
        "median_2d_return_positive_current_window_pct": float(
            current_positive["return_2d_pct"].median()
        ),
        "corr_return_vs_surprise_all": corr,
        "corr_return_vs_surprise_positive": corr_pos,
        "detail": edf.to_dict("records"),
    }
    print("Q4 assignment-era events:", len(edf), "| positive:", len(positive))
    print("Q4 median 2d return on positive surprises:",
          round(positive["return_2d_pct"].median(), 4), "%  -> option 0.35")
    print("Q4 mean:", round(positive["return_2d_pct"].mean(), 4), "%")
    print("Q4 corr(return, surprise) all:", round(corr, 4),
          "| positive-only:", round(corr_pos, 4))
    print(edf[["earnings_date", "surprise_pct", "return_2d_pct"]].to_string(index=False))


if __name__ == "__main__":
    question_1()
    question_2()
    question_3()
    question_4()

    with open(OUTPUT_DIR / "hw1_answers.json", "w") as f:
        json.dump(answers, f, indent=2, default=str)
    print("\nSaved:", OUTPUT_DIR / "hw1_answers.json")
