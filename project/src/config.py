"""Central configuration for the Binance perpetuals trading project.

Every path, window and constant that the rest of the pipeline depends on lives here so
that the scripts stay importable from notebooks, cron and Docker without argument
parsing in each module.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent

DATA_DIR = Path(os.environ.get("PROJECT_DATA_DIR", PROJECT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
KLINE_DIR = RAW_DIR / "klines"
FUNDING_DIR = RAW_DIR / "funding"
PREMIUM_DIR = RAW_DIR / "premium"
DAILY_DIR = RAW_DIR / "daily"
MACRO_DIR = RAW_DIR / "macro"
SENTIMENT_DIR = RAW_DIR / "sentiment"

CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = Path(os.environ.get("PROJECT_OUTPUT_DIR", PROJECT_DIR / "output"))
MODEL_DIR = Path(os.environ.get("PROJECT_MODEL_DIR", PROJECT_DIR / "models"))

def display_path(path) -> str:
    """Print a path relative to the repository root.

    The repository is cloned at a different place on every machine, so an absolute
    path in the output says nothing to a reader. The relative path is the same
    everywhere and is the one that exists in the repository.
    """
    try:
        return str(Path(path).resolve().relative_to(REPO_DIR))
    except ValueError:
        return str(path)


for _d in (DATA_DIR, RAW_DIR, MACRO_DIR, SENTIMENT_DIR, CACHE_DIR, OUTPUT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------------------

BINANCE_MONTHLY = "https://data.binance.vision/data/futures/um/monthly"
BINANCE_DAILY = "https://data.binance.vision/data/futures/um/daily"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=0"
COINGECKO_MARKETS = "https://api.coingecko.com/api/v3/coins/markets"
COINGECKO_LIST = "https://api.coingecko.com/api/v3/coins/list"

# Binance symbol -> CoinGecko id. The bulk archive uses the 1000x prefix for large-token
# denominations, which has to be stripped before the id can be resolved.
COINGECKO_ID_OVERRIDES = {
    "1000SHIBUSDT": "shiba-inu",
    "1000XECUSDT": "ecash",
    "1000LUNCUSDT": "terra-luna",
    "1000BTTCUSDT": "bittcoin",
    "1000FLOKIUSDT": "floki",
    "1000PEPEUSDT": "pepe",
    "1000BONKUSDT": "bonk",
    "1000RATSUSDT": "rat-my-balls",
    "1000SATSUSDT": "satellite-coin",
    "1000MOGUSDT": "mog-coin",
    "1000000BOBUSDT": "bob",
    "1000000MOGUSDT": "mog-coin",
    "1000CATUSDT": "cat-in-a-dogs-world",
    "1000CHEEMSUSDT": "cheems-token",
    "1000WHYUSDT": "why",
    "1MBABYDOGEUSDT": "baby-doge",
}

# FRED series used as macro regime features. VIX, HY spreads and oil are the ones that
# actually move crypto, on top of the lecture's rate and CPI variables.
FRED_SERIES = {
    "DGS10": "10-year Treasury yield",
    "FEDFUNDS": "effective federal funds rate",
    "CPILFESL": "core CPI, year-on-year",
    "VIXCLS": "CBOE volatility index",
    "BAMLH0A0HYM2": "ICE BofA US high yield option-adjusted spread",
    "DCOILWTICO": "WTI crude oil price",
    "DEXUSEU": "US dollar to euro exchange rate",
}

# --------------------------------------------------------------------------------------
# Panel definition
# --------------------------------------------------------------------------------------

# Full calendar months of history. The window ends on the last complete month so that
# every symbol has a full set of bars.
START_MONTH = "2023-11"
END_MONTH = "2026-10"

# Hours between the feature timestamp and the future return that the model predicts.
# A 7-day horizon is the shortest one at which the panel shows a usable signal: the
# correlation between the momentum features and the forward return is an order of
# magnitude larger at 168 hours than at 24 hours.
HORIZON_HOURS = 168

# How many concurrent positions may be open in the same contract. One is what a real
# trader would do, and it stops the simulator from stacking twenty overlapping bets on
# the same coin.
MAX_POSITIONS_PER_SYMBOL = 1

# The features are computed on every hourly bar, but only every fourth hour is kept for
# modelling. A one-week horizon means a new signal is not needed every hour, and the
# step keeps the design matrix inside the memory budget of a small machine while still
# leaving more than a million modelling rows.
MODEL_FREQUENCY_HOURS = 4

# Trading simulator settings. The lecture used 0.2% round trip for equities; Binance
# charges 0.05% per side on USD-M perpetuals, so the project uses 0.1% round trip and
# the comparison is documented.
TRADE_SIZE_USD = 100.0
FEE_RATE = 0.001
DAYS_PER_YEAR = 365.0

# Liquidity filter: a symbol must clear this median daily quote volume over the panel to
# be tradable with a $100 ticket.
MIN_MEDIAN_DAILY_QUOTE_VOLUME = 10_000_000.0

# Hard cap on the number of contracts in the panel. The cap keeps the feature matrix
# inside the memory budget of a small machine while still leaving millions of rows.
MAX_UNIVERSE_SIZE = 150

# --------------------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------------------

# Temporal split: train / validation / test, in that order, on the panel timeline.
TRAIN_END = "2025-03-31"
VALID_END = "2025-12-31"

RANDOM_STATE = 42
N_JOBS = 1  # the sandbox has two cores and the random forest is memory bound

# --------------------------------------------------------------------------------------
# Feature sets
# --------------------------------------------------------------------------------------

TARGET = "is_positive_growth_24h_future"

# The lecture's feature families, reproduced on the crypto panel.
NUMERIC = [
    "close",
    "volume",
    "quote_volume",
    "rsi_14",
    "cci_14",
    "sma_20_ratio",
    "sma_50_ratio",
    "volatility_24h",
    "volatility_168h",
    "growth_1h",
    "growth_24h",
    "growth_168h",
    "prev_growth_24h",
    "div_payout",
    "stock_split",
    "dgs10",
    "fedfunds",
    "cpi_core_yoy",
    "vix",
    "hy_spread",
    "oil",
    "eurusd",
]

# Feature families that are one-hot encoded before the model sees them.
DUMMIES = [
    "hour",
    "weekday",
    "month",
    "ticker",
    "ticker_type",
    "month_wom",
]

# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------

DOWNLOAD_WORKERS = int(os.environ.get("DOWNLOAD_WORKERS", "12"))
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
