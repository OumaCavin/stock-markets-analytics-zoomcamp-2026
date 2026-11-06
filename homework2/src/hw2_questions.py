"""
Homework 2 - One Dataframe (2026 cohort)
Solutions for questions 1-4. Produces output/hw2_answers.json.

Question 1 : iposcoop "Recently Filed" -> withdrawn IPOs grouped by company type
Question 2 : Median Sharpe ratio of 2025 IPOs, measured on 2026-09-11
Question 3 : Optimal IPO holding period (1-12 months) by median growth
Question 4 : Net income of the RSI<30 oversold strategy
"""

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
AS_OF = "2026-09-11"

answers = {}


def scrape(url):
    response = requests.get(url, impersonate="chrome", timeout=90)
    response.raise_for_status()
    return pd.read_html(io.StringIO(response.text))[0]


# ---------------------------------------------------------------------------
# Question 1: withdrawn IPOs by company type
# ---------------------------------------------------------------------------
# Ordered rules - the first matching rule wins (order matters!).
COMPANY_TYPE_RULES = [
    ("Technologies", ["Technologies"]),
    ("Acquisition Corp", ["Acquisition Corp", "Acquisition Corporation", "Corp"]),
    ("Inc.", ["Inc", "Incorporated"]),
    ("Group", ["Group"]),
    ("Limited", ["Ltd", "Limited"]),
    ("Holdings", ["Holdings", "Holding"]),
]


def classify_company(name):
    for label, patterns in COMPANY_TYPE_RULES:
        for pattern in patterns:
            if pattern in name:
                return label
    return "Other"


def parse_money(value):
    """'$8.00' -> 8.0 ; '-' / NaN / '' -> NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if text in {"-", "", "nan", "None"}:
        return np.nan
    text = text.replace("$", "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


def question_1():
    df = scrape("https://www.iposcoop.com/ipos-recently-filed/")
    df["File Date"] = pd.to_datetime(df["File Date"], errors="coerce")

    withdrawn = df[df["Expected To Trade"].astype(str).str.strip().str.lower() == "withdrawn"].copy()
    withdrawn = withdrawn[withdrawn["File Date"] < pd.Timestamp("2026-09-11")]

    withdrawn["Company Type"] = withdrawn["Company"].map(classify_company)
    withdrawn["Price Low Num"] = withdrawn["Price Low"].map(parse_money)
    withdrawn["Price High Num"] = withdrawn["Price High"].map(parse_money)
    withdrawn["Avg_price"] = withdrawn[["Price Low Num", "Price High Num"]].mean(axis=1)
    withdrawn["Shares (millions)"] = pd.to_numeric(
        withdrawn["Shares (millions)"], errors="coerce")
    withdrawn["Est Vol (millions)"] = withdrawn["Est $ Vol (millions)"].map(parse_money)

    withdrawn["Shares_offered_value"] = withdrawn["Shares (millions)"] * withdrawn["Avg_price"]
    withdrawn["Shares_offered_value"] = withdrawn["Shares_offered_value"].fillna(
        withdrawn["Est Vol (millions)"])

    grouped = (
        withdrawn.groupby("Company Type")["Shares_offered_value"]
        .agg(["count", "sum"])
        .sort_values("sum", ascending=False)
    )
    top_class = grouped.index[0]
    top_value = float(grouped["sum"].iloc[0])

    answers["q1"] = {
        "question": (
            "What is the total withdrawn IPO value (in $ millions) for the company "
            "class with the highest total withdrawal value?"
        ),
        "n_withdrawn_rows_before_2026_09_11": int(len(withdrawn)),
        "top_class": top_class,
        "top_class_total_value_musd": top_value,
        "by_class": {
            k: {"count": int(v["count"]), "total_musd": float(v["sum"])}
            for k, v in grouped.iterrows()
        },
        "rows": withdrawn[
            ["File Date", "Company", "Company Type", "Shares (millions)", "Avg_price",
             "Est Vol (millions)", "Shares_offered_value"]
        ].to_dict("records"),
    }
    print("Q1 withdrawn rows (< 2026-09-11):", len(withdrawn))
    print(grouped.to_string())
    print("Q1 top class:", top_class, "->", round(top_value, 2), "$M")


# ---------------------------------------------------------------------------
# Question 2 & 3: 2025 IPO cohort - Sharpe ratio and holding period
# ---------------------------------------------------------------------------
def load_ipo_stocks():
    """Download the 2025 IPO cohort and the daily price history for it."""
    df = scrape("https://www.iposcoop.com/2025-pricings/")
    df["Offer Date"] = pd.to_datetime(df["Offer Date"], errors="coerce")
    # 'Return' arrives as a percentage string, e.g. '48.69%'.
    df["Return"] = pd.to_numeric(
        df["Return"].astype(str).str.replace("%", "", regex=False), errors="coerce")
    n_all = len(df)

    filtered = df[(df["Offer Date"] < "2025-09-01") & (df["Return"] != 0)].copy()
    n_filtered = len(filtered)

    tickers = filtered["Symbol"].astype(str).str.strip()
    tickers = [t for t in tickers if t and t.lower() != "nan"]
    # de-duplicate while keeping order
    seen, unique = set(), []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    prices = yf.download(unique, start="2025-01-01", end="2026-11-07", progress=False,
                         auto_adjust=False, threads=True)
    if isinstance(prices.columns, pd.MultiIndex):
        # default group_by='column' -> (Price, Ticker)
        closes = prices.xs("Close", axis=1, level=0)
    else:
        closes = prices[["Close"]]
        closes.columns = closes.columns.get_level_values(0)
    # keep only tickers Yahoo actually returned data for
    available = [c for c in unique if c in closes.columns and closes[c].notna().any()]
    closes = closes[available].dropna(how="all")
    n_available = len(closes.columns)

    stocks_df = closes.stack().reset_index()
    stocks_df.columns = ["Date", "Ticker", "Close"]
    stocks_df = stocks_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return df, filtered, stocks_df, n_all, n_filtered, n_available


def question_2_and_3():
    raw, filtered, stocks_df, n_all, n_filtered, n_available = load_ipo_stocks()

    # Feature engineering, computed per ticker (the Module 02 notebook computes every
    # feature inside the per-ticker loop, before the frames are concatenated).
    grouped = stocks_df.groupby("Ticker", sort=False)["Close"]
    stocks_df["growth_252d"] = grouped.transform(lambda s: s / s.shift(252))
    stocks_df["growth_1d"] = grouped.transform(lambda s: s / s.shift(1))
    stocks_df["volatility"] = grouped.transform(lambda s: s.rolling(30).std() * np.sqrt(252))
    # Module 02 variant: volatility measured on daily returns instead of the price level.
    stocks_df["volatility_ret"] = stocks_df.groupby("Ticker", sort=False)["growth_1d"].transform(
        lambda s: s.rolling(30).std() * np.sqrt(252))
    stocks_df["Sharpe"] = (stocks_df["growth_252d"] - 0.05) / stocks_df["volatility"]
    stocks_df["Sharpe_ret"] = (stocks_df["growth_252d"] - 0.05) / stocks_df["volatility_ret"]

    as_of = stocks_df[stocks_df["Date"] == pd.Timestamp(AS_OF)]
    stats = as_of[["growth_252d", "volatility", "Sharpe"]].describe()
    median_sharpe = float(stats.loc["50%", "Sharpe"])
    median_sharpe_ret = float(as_of["Sharpe_ret"].describe().loc["50%"])

    # ---- Question 3: fixed months holding strategy -------------------------
    wide = stocks_df.pivot(index="Date", columns="Ticker", values="Close")
    growth_long = None
    for m in range(1, 13):
        days = 21 * m
        growth = (wide.shift(-days) / wide - 1)
        long_m = growth.reset_index().melt(
            id_vars=["Date"], value_name=f"future_growth_{m}_m")
        growth_long = long_m if growth_long is None else growth_long.merge(
            long_m, on=["Date", "Ticker"], how="outer")

    # Entry point = the first trading day on which the stock actually has a price.
    # (The pivot keeps pre-IPO rows as NaN, so the minimum date alone is wrong.)
    traded = stocks_df[stocks_df["Close"].notna()].sort_values(["Ticker", "Date"])
    entry = (
        traded.groupby("Ticker")
        .agg(min_date=("Date", "min"), first_close=("Close", "first"))
        .reset_index()
        .rename(columns={"min_date": "Date"})
    )
    joined = entry.merge(growth_long, on=["Date", "Ticker"], how="inner")
    month_cols = [f"future_growth_{m}_m" for m in range(1, 13)]
    medians = joined[month_cols].median()
    best_month = int(medians.idxmax().split("_")[2])
    best_value = float(medians.max())

    answers["q2"] = {
        "question": (
            "What is the median Sharpe ratio (as of 11 September 2026) for companies "
            "that went public before 1 September 2025?"
        ),
        "n_ipos_2025": n_all,
        "n_after_filter": n_filtered,
        "n_tickers_downloaded": n_available,
        "as_of": AS_OF,
        "median_sharpe": median_sharpe,
        "median_sharpe_vol_on_returns_variant": median_sharpe_ret,
        "mean_sharpe": float(stats.loc["mean", "Sharpe"]),
        "describe": {k: {kk: float(vv) for kk, vv in v.items()}
                     for k, v in stats.to_dict().items()},
        "n_reached_252d": int(as_of["growth_252d"].notna().sum()),
        "n_rows_on_as_of_date": int(len(as_of)),
        "median_growth_252d": float(stats.loc["50%", "growth_252d"]),
        "mean_growth_252d": float(stats.loc["mean", "growth_252d"]),
    }
    answers["q3"] = {
        "question": (
            "What is the optimal number of months (1 to 12) to hold a newly IPO'd "
            "stock in order to maximize the median growth value?"
        ),
        "median_by_month": {f"month_{m}": float(medians[f"future_growth_{m}_m"])
                            for m in range(1, 13)},
        "mean_by_month": {f"month_{m}": float(joined[f"future_growth_{m}_m"].mean())
                          for m in range(1, 13)},
        "n_by_month": {f"month_{m}": int(joined[f"future_growth_{m}_m"].notna().sum())
                       for m in range(1, 13)},
        "optimal_month": best_month,
        "max_median_growth": best_value,
        "n_stocks_used": int(len(joined)),
    }
    print("Q2 IPOs 2025:", n_all, "| after filter:", n_filtered,
          "| downloaded:", n_available)
    print("Q2 rows on", AS_OF, ":", len(as_of))
    print(stats.to_string())
    print("Q2 median Sharpe:", round(median_sharpe, 4),
          "| returns-vol variant:", round(median_sharpe_ret, 4))
    print("Q3 medians by month:")
    print(medians.to_string())
    print("Q3 n by month:", {m: int(joined[f'future_growth_{m}_m'].notna().sum())
                             for m in range(1, 13)})
    print("Q3 optimal month:", best_month, "median growth:", round(best_value, 4))


# ---------------------------------------------------------------------------
# Question 4: RSI < 30 oversold strategy
# ---------------------------------------------------------------------------
def question_4():
    path = DATA_DIR / "data_hw2.parquet"
    df = pd.read_parquet(path, engine="pyarrow")
    print("Q4 parquet shape:", df.shape)
    print("Q4 columns sample:", [c for c in df.columns if not c.startswith("cdl")][:40])

    df["Date"] = pd.to_datetime(df["Date"])
    window = df[(df["Date"] >= "2000-01-01") & (df["Date"] <= "2025-06-01")]
    signals = window[window["rsi"] < 30]

    net_income = 1000 * (signals["growth_future_30d"] - 1).sum()
    avg_return = signals["growth_future_30d"].mean()
    win_rate = (signals["growth_future_30d"] > 1).mean()

    answers["q4"] = {
        "question": (
            "What is the total profit (in $ thousands) you would have earned by "
            "investing $1000 every time a stock was oversold (RSI < 30)?"
        ),
        "n_trades": int(len(signals)),
        "net_income_usd": float(net_income),
        "net_income_usd_thousands": float(net_income / 1000),
        "avg_30d_return_pct": float(avg_return * 100),
        "win_rate_pct": float(win_rate * 100),
        "window": "2000-01-01 .. 2025-06-01",
    }
    print("Q4 trades:", len(signals),
          "| avg 30d return:", round(avg_return * 100, 3), "%",
          "| win rate:", round(win_rate * 100, 3), "%")
    print("Q4 net income: $", round(net_income, 2),
          "=", round(net_income / 1000, 2), "thousand")


if __name__ == "__main__":
    question_1()
    question_2_and_3()
    question_4()

    with open(OUTPUT_DIR / "hw2_answers.json", "w") as f:
        json.dump(answers, f, indent=2, default=str)
    print("\nSaved:", OUTPUT_DIR / "hw2_answers.json")
