"""Turn the raw Binance archive into one modelling frame.

The panel is built one symbol at a time so that the rolling windows never have to hold
the whole dataset in memory at once. Everything the model sees is computed here:

* the lecture's feature families (RSI, CCI, moving averages, realised volatility,
  momentum) reproduced on an hourly crypto panel;
* crypto-specific features that only exist because of the new data sources: the
  funding rate, the perpetual basis, and the taker-side order flow that the exchange
  publishes with every bar;
* macro and sentiment regime variables from FRED, CoinGecko and the Fear & Greed index;
* calendar dummies, including the week-of-month dummies that Homework 3 found.

The target is the lecture's own target, rescaled to the panel's horizon: whether the
close 24 hours ahead is above the close now.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402

# --------------------------------------------------------------------------------------
# Technical indicators
# --------------------------------------------------------------------------------------


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's relative strength index."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def cci(close: pd.Series, high: pd.Series, low: pd.Series, window: int = 14) -> pd.Series:
    """Commodity channel index."""
    tp = (high + low + close) / 3
    sma = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0.0, np.nan))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average true range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def bollinger_pct_b(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.Series:
    """Position of the close inside its Bollinger band, 0 at the lower and 1 at the upper."""
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return (close - mid) / (2 * k * sd.replace(0.0, np.nan))


# --------------------------------------------------------------------------------------
# Per-symbol assembly
# --------------------------------------------------------------------------------------


def load_symbol(symbol: str, kind: str = "klines") -> pd.DataFrame | None:
    """Read one contract's stored parquet and return it indexed by open time."""
    folder = {"klines": C.KLINE_DIR, "premium": C.PREMIUM_DIR}[kind]
    path = folder / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    return df


def load_funding(symbol: str) -> pd.DataFrame | None:
    path = C.FUNDING_DIR / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    df = df.rename(columns={"last_funding_rate": "funding_rate"})
    return df[["funding_rate"]]


def _rolling(df: pd.DataFrame, col: str, window: int, fn: str) -> pd.Series:
    """Rolling aggregation of one column of a single symbol's frame."""
    rolled = getattr(df[col].rolling(window), fn)()
    return rolled


def build_symbol_frame(symbol: str) -> pd.DataFrame | None:
    """Build the full feature frame for one contract."""
    raw = load_symbol(symbol)
    if raw is None:
        return None
    df = raw.copy()
    df = df[~df.index.duplicated(keep="last")]

    out = pd.DataFrame(index=df.index)
    out["ticker"] = symbol
    out["open"] = df["open"].astype("float32")
    out["high"] = df["high"].astype("float32")
    out["low"] = df["low"].astype("float32")
    out["close"] = df["close"].astype("float32")
    out["volume"] = df["volume"].astype("float32")
    out["quote_volume"] = df["quote_volume"].astype("float32")
    out["n_trades"] = df["count"].astype("float32")

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    volume = df["volume"].astype("float64")
    quote_volume = df["quote_volume"].astype("float64")
    trades = df["count"].astype("float64")
    taker_buy_quote = df["taker_buy_quote_volume"].astype("float64")
    taker_buy_vol = df["taker_buy_volume"].astype("float64")

    # ---- lecture feature families, reproduced on hourly crypto bars -----------------
    out["rsi_14"] = rsi(close, 14).astype("float32")
    out["cci_14"] = cci(close, high, low, 14).astype("float32")
    out["sma_20_ratio"] = (close / close.rolling(20).mean()).astype("float32")
    out["sma_50_ratio"] = (close / close.rolling(50).mean()).astype("float32")
    out["sma_200_ratio"] = (close / close.rolling(200).mean()).astype("float32")

    hourly_ret = close.pct_change()
    out["volatility_24h"] = (hourly_ret.rolling(24).std() * np.sqrt(24)).astype("float32")
    out["volatility_168h"] = (hourly_ret.rolling(168).std() * np.sqrt(24)).astype("float32")
    out["growth_1h"] = (close.pct_change(1)).astype("float32")
    out["growth_24h"] = (close.pct_change(24)).astype("float32")
    out["growth_168h"] = (close.pct_change(168)).astype("float32")
    out["prev_growth_24h"] = out["growth_24h"].shift(24).astype("float32")

    # ---- new features: order flow published with every bar -------------------------
    # The exchange splits each bar's volume into the taker-buy side and the resting
    # side, which gives a signed order-flow imbalance that no price-only source has.
    denominator = (2 * quote_volume - taker_buy_quote).replace(0.0, np.nan)
    out["taker_buy_ratio"] = (taker_buy_quote / denominator).astype("float32")
    out["order_flow_imbalance"] = (
        (2 * taker_buy_vol - volume) / volume.replace(0.0, np.nan)
    ).astype("float32")
    out["taker_imbalance_z"] = (
        (out["order_flow_imbalance"] - out["order_flow_imbalance"].rolling(24).mean())
        / out["order_flow_imbalance"].rolling(24).std().replace(0.0, np.nan)
    ).astype("float32")
    out["avg_trade_size"] = (quote_volume / trades.replace(0.0, np.nan)).astype("float32")
    out["n_trades_log"] = np.log1p(trades).astype("float32")
    out["volume_z_24h"] = (
        (quote_volume - quote_volume.rolling(24).mean())
        / quote_volume.rolling(24).std().replace(0.0, np.nan)
    ).astype("float32")
    out["amihud_illiq"] = (
        hourly_ret.abs() / quote_volume.replace(0.0, np.nan) * 1e9
    ).astype("float32")
    out["close_position_in_bar"] = ((close - low) / (high - low).replace(0.0, np.nan)).astype(
        "float32"
    )
    out["bar_range_pct"] = ((high - low) / close.replace(0.0, np.nan)).astype("float32")

    # ---- new features: derivatives structure ---------------------------------------
    premium = load_symbol(symbol, kind="premium")
    if premium is not None and not premium.empty:
        prem_close = premium["close"].astype("float64").reindex(out.index).ffill()
        out["basis"] = (prem_close / close - 1.0).astype("float32")
        out["basis_mean_24h"] = out["basis"].rolling(24).mean().astype("float32")
    else:
        out["basis"] = np.nan
        out["basis_mean_24h"] = np.nan

    funding = load_funding(symbol)
    if funding is not None and not funding.empty:
        rate = funding["funding_rate"].astype("float64").reindex(out.index).ffill()
        out["funding_rate"] = rate.astype("float32")
        out["funding_annualised"] = (rate * 3 * 365).astype("float32")
        out["funding_mean_24h"] = rate.rolling(24).mean().astype("float32")
        out["funding_z_168h"] = (
            (rate - rate.rolling(168).mean()) / rate.rolling(168).std().replace(0.0, np.nan)
        ).astype("float32")
    else:
        out["funding_rate"] = np.nan
        out["funding_annualised"] = np.nan
        out["funding_mean_24h"] = np.nan
        out["funding_z_168h"] = np.nan

    # ---- new features: shape of the recent path ------------------------------------
    out["atr_14"] = atr(high, low, close, 14).astype("float32")
    out["bollinger_pct_b"] = bollinger_pct_b(close, 20, 2.0).astype("float32")
    out["vol_regime"] = (out["volatility_24h"] / out["volatility_168h"].replace(0.0, np.nan)).astype(
        "float32"
    )
    high_90d = close.rolling(90 * 24, min_periods=24 * 10).max()
    low_90d = close.rolling(90 * 24, min_periods=24 * 10).min()
    out["distance_to_90d_high"] = (close / high_90d.replace(0.0, np.nan) - 1.0).astype("float32")
    out["distance_to_90d_low"] = (close / low_90d.replace(0.0, np.nan) - 1.0).astype("float32")
    out["momentum_accel"] = (out["growth_24h"] - out["growth_24h"].shift(24)).astype("float32")
    out["reversal_1h"] = (-out["growth_1h"]).astype("float32")
    out["range_expansion"] = (out["bar_range_pct"] / out["bar_range_pct"].rolling(24).mean()).astype(
        "float32"
    )

    # ---- target ---------------------------------------------------------------------
    horizon = C.HORIZON_HOURS
    future_close = close.shift(-horizon)
    out["future_return"] = (future_close / close - 1.0).astype("float32")
    out[C.TARGET] = (out["future_return"] > 0).fillna(False).astype("int8")

    # ---- calendar ------------------------------------------------------------------
    out["hour"] = out.index.hour.astype("int8")
    out["weekday"] = out.index.dayofweek.astype("int8")
    out["month"] = out.index.month.astype("int8")
    out["day"] = out.index.day.astype("int8")
    out["week_of_month"] = ((out.index.day - 1) // 7 + 1).astype("int8")
    out["quarter"] = out.index.quarter.astype("int8")

    out["hours_since_listing"] = np.arange(len(out), dtype="int32")
    out["ticker_type"] = _ticker_type(symbol)
    out.index.name = "open_time"

    # The rolling windows above were computed on every hourly bar. Only every Nth bar
    # is kept for modelling, and the thinning happens here, per contract, so the panel
    # never has to hold the full hourly history of every contract at the same time.
    if C.MODEL_FREQUENCY_HOURS > 1:
        out = out.iloc[:: C.MODEL_FREQUENCY_HOURS]
    return out


def _ticker_type(symbol: str) -> str:
    """Bucket a contract into a coarse type, used as a categorical feature."""
    base = symbol[:-4]
    for prefix, label in (
        ("1000000", "meme_micro"),
        ("1000", "meme"),
        ("1MBABY", "meme"),
        ("10000", "meme"),
    ):
        if base.startswith(prefix):
            return label
    stable = {
        "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK",
        "TRX", "MATIC", "LTC", "BCH", "ATOM", "NEAR", "UNI", "APT", "ARB", "OP",
        "ETC", "FIL", "ICP", "HBAR", "VET", "AAVE", "MKR", "ALGO", "SUI", "TON",
        "XLM", "EOS", "NEO", "IOTA", "XTZ", "THETA", "FLOW", "CHZ", "DASH", "ZEC",
    }
    return "major" if base in stable else "alt"


# --------------------------------------------------------------------------------------
# Cross-sectional and macro features
# --------------------------------------------------------------------------------------


def add_cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
    """Add features that only exist relative to the rest of the panel.

    The market-wide state matters for a directional strategy: a coin that is down 5%
    while the rest of the market is up 5% is a different trade from one that is down 5%
    in a market that is down 5%. Breadth captures the regime directly.
    """
    panel = panel.sort_index()
    by_time = panel.groupby(level=0)

    market = by_time["growth_24h"].transform("median").rename("market_growth_24h")
    breadth = by_time["growth_24h"].transform(lambda s: (s > 0).mean()).rename("breadth_24h")
    rank_ret = by_time["growth_24h"].rank(pct=True).rename("cs_return_rank")
    rank_vol = by_time["quote_volume"].rank(pct=True).rename("cs_volume_rank")
    rel = (panel["growth_24h"] - market).rename("relative_to_market")

    out = pd.DataFrame(index=panel.index)
    out["market_growth_24h"] = market.astype("float32")
    out["breadth_24h"] = breadth.astype("float32")
    out["cs_return_rank"] = rank_ret.astype("float32")
    out["cs_volume_rank"] = rank_vol.astype("float32")
    out["relative_to_market"] = rel.astype("float32")
    return pd.concat([panel, out], axis=1)


def add_btc_features(panel: pd.DataFrame, btc: str = "BTCUSDT") -> pd.DataFrame:
    """Add the market benchmark's own path, plus each coin's beta to it."""
    btc_rows = panel[panel["ticker"] == btc][["growth_24h", "growth_1h"]]
    btc_rows = btc_rows[~btc_rows.index.duplicated(keep="last")]
    panel = panel.join(
        btc_rows.rename(columns={"growth_24h": "btc_growth_24h", "growth_1h": "btc_growth_1h"}),
        how="left",
    )
    panel["btc_growth_24h"] = panel["btc_growth_24h"].astype("float32")
    panel["btc_growth_1h"] = panel["btc_growth_1h"].astype("float32")
    panel["alt_minus_btc"] = (panel["growth_24h"] - panel["btc_growth_24h"]).astype("float32")
    return panel


def load_macro() -> pd.DataFrame:
    """Read the FRED series and turn them into a regime table."""
    frames = {}
    for series, description in C.FRED_SERIES.items():
        path = C.MACRO_DIR / f"{series}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        raw.columns = ["date", series]
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.set_index("date")[series].astype("float64")
        raw.index = raw.index.tz_localize("UTC")
        frames[series] = raw

    if not frames:
        return pd.DataFrame()

    macro = pd.DataFrame(frames)
    macro = macro.ffill()
    if "CPILFESL" in macro:
        macro["cpi_core_yoy"] = (macro["CPILFESL"] / macro["CPILFESL"].shift(12) - 1.0) * 100
    macro = macro.rename(
        columns={
            "DGS10": "dgs10",
            "FEDFUNDS": "fedfunds",
            "VIXCLS": "vix",
            "BAMLH0A0HYM2": "hy_spread",
            "DCOILWTICO": "oil",
            "DEXUSEU": "eurusd",
        }
    )
    keep = ["dgs10", "fedfunds", "cpi_core_yoy", "vix", "hy_spread", "oil", "eurusd"]
    keep = [k for k in keep if k in macro]
    macro = macro[keep]
    macro["vix_change_5d"] = macro["vix"].diff(5 * 21)
    macro["hy_spread_change_5d"] = macro["hy_spread"].diff(5 * 21)
    return macro


def load_fear_greed() -> pd.DataFrame:
    path = C.SENTIMENT_DIR / "fear_and_greed.csv"
    if not path.exists():
        return pd.DataFrame()
    fg = pd.read_csv(path, parse_dates=["date"])
    fg["date"] = fg["date"].dt.tz_localize("UTC")
    fg = fg.set_index("date")
    fg["fear_greed"] = fg["value"].astype("float32")
    fg["fear_greed_change_7d"] = fg["fear_greed"].diff(7)
    return fg[["fear_greed", "fear_greed_change_7d"]]


def load_coingecko() -> pd.DataFrame:
    path = C.SENTIMENT_DIR / "coingecko_market_data.csv"
    if not path.exists():
        return pd.DataFrame()
    gecko = pd.read_csv(path)
    return gecko.set_index("ticker")[
        ["market_cap", "circulating_supply", "market_cap_rank", "volume_24h"]
    ]


def add_market_context(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach macro, sentiment and fundamental columns to the panel."""
    macro = load_macro()
    if not macro.empty:
        panel = panel.join(macro, how="left")
        for col in macro.columns:
            panel[col] = panel[col].astype("float32")

    fear = load_fear_greed()
    if not fear.empty:
        panel = panel.join(fear, how="left")
        panel["fear_greed"] = panel["fear_greed"].astype("float32")
        panel["fear_greed_change_7d"] = panel["fear_greed_change_7d"].astype("float32")

    gecko = load_coingecko()
    if not gecko.empty:
        # The fundamental data is keyed by contract, not by timestamp, so it is mapped
        # onto the panel with a lookup rather than joined on the index.
        for col in gecko.columns:
            panel[col] = panel["ticker"].map(gecko[col]).astype("float32")
    return panel


# --------------------------------------------------------------------------------------
# Panel assembly
# --------------------------------------------------------------------------------------


def _trim_memory() -> None:
    """Ask the allocator to return the freed per-contract temporaries to the OS."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def build_panel(
    symbols: list[str],
    start: str | None = None,
    end: str | None = None,
    cross_sectional: bool = True,
) -> pd.DataFrame:
    """Build the unified modelling frame for a list of contracts."""
    frames = []
    for i, sym in enumerate(symbols, 1):
        frame = build_symbol_frame(sym)
        if frame is None or frame.empty:
            continue
        frames.append(frame)
        if i % 25 == 0:
            # Each contract needs the full hourly history while its rolling windows are
            # computed. Returning that memory to the operating system between contracts
            # keeps the peak inside the budget of a small machine.
            _trim_memory()
            print(f"  features built for {i}/{len(symbols)} symbols", flush=True)
    if not frames:
        raise RuntimeError("no symbols produced data")

    panel = pd.concat(frames)
    del frames
    _trim_memory()
    panel = panel.sort_index()
    panel.index.name = "open_time"
    panel["open_time"] = panel.index
    panel["timestamp"] = panel.index

    if start is not None:
        panel = panel[panel.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        panel = panel[panel.index <= pd.Timestamp(end, tz="UTC")]

    if cross_sectional:
        panel = add_cross_sectional(panel)
        if "BTCUSDT" in set(panel["ticker"]):
            panel = add_btc_features(panel)
    panel = add_market_context(panel)
    panel = panel.reset_index(drop=True)
    return panel


def feature_columns(panel: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return the numeric feature list and the dummy list for a built panel."""
    numeric = [
        c
        for c in panel.columns
        if c
        not in {
            C.TARGET,
            "future_return",
            "open_time",
            "ticker",
            "ticker_type",
            "hour",
            "weekday",
            "month",
            "day",
            "week_of_month",
            "quarter",
        }
        and pd.api.types.is_numeric_dtype(panel[c])
    ]
    dummies = [
        c
        for c in ["hour", "weekday", "month", "ticker", "ticker_type", "month_wom"]
        if c in panel.columns
    ]
    return numeric, dummies


def add_month_wom(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the week-of-month label used by Homework 3."""
    if "month_wom" not in panel.columns:
        panel["month_wom"] = panel["month"].astype(str) + "_w" + panel["week_of_month"].astype(str)
    return panel


def symbol_liquidity() -> pd.DataFrame:
    """Median daily quote volume per contract, used to pick a tradable universe."""
    rows = []
    for path in C.KLINE_DIR.glob("*.parquet"):
        df = pd.read_parquet(path, columns=["open_time", "quote_volume"])
        if df.empty:
            continue
        ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        daily = df.groupby(ts.dt.floor("D"))["quote_volume"].sum()
        rows.append(
            {
                "ticker": path.stem,
                "median_daily_quote_volume": float(daily.median()),
                "total_daily_quote_volume": float(daily.sum()),
                "first_timestamp": ts.min(),
                "last_timestamp": ts.max(),
                "n_hours": int(len(df)),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the unified crypto feature panel")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=str(C.CACHE_DIR / "panel.parquet"))
    args = ap.parse_args()

    symbols = args.symbols
    if not symbols:
        symbols = sorted(p.stem for p in C.KLINE_DIR.glob("*.parquet"))
    print(f"building panel for {len(symbols)} symbols", flush=True)
    panel = build_panel(symbols, start=args.start, end=args.end)
    panel = add_month_wom(panel)
    print(panel.shape)
    print(panel[C.TARGET].value_counts())
    panel.to_parquet(args.out)
    print(f"written to {args.out}")
