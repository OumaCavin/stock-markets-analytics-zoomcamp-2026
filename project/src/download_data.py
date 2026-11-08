"""Download every data source the project needs.

The heavy lifting is done against the Binance bulk archive (``data.binance.vision``),
which serves the same monthly zip files that the exchange API publishes. Working from the
archive rather than the REST API means the whole panel can be rebuilt with plain HTTP
requests, no API key and no rate limit, and it gives access to two datasets the public
REST endpoints do not expose historically: the funding rate history and the premium
index (the basis) series.

Every function is idempotent and resumable. A month that is already present in the
per-symbol parquet file is skipped, so an interrupted run can simply be restarted.

Usage::

    python -m src.download_data --stage all
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402

# A requests.Session is not safe to share between threads, and the archive refuses
# connections that reuse a session across threads. One session per worker thread keeps
# the connection pool healthy under the download concurrency.
_LOCAL = threading.local()


def session() -> requests.Session:
    """Return the calling thread's session, creating it on first use."""
    s = getattr(_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "sma-zoomcamp-project/1.0"})
        _LOCAL.session = s
    return s

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def month_range(start: str = C.START_MONTH, end: str = C.END_MONTH) -> list[str]:
    """Return every calendar month between two inclusive ``YYYY-MM`` strings."""
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out, y, m = [], sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def _get(url: str, retries: int = C.MAX_RETRIES) -> requests.Response | None:
    """GET with backoff.

    The archive answers bursts of traffic with a spurious 404 rather than a 429, so a
    404 is retried before it is believed. Only after several attempts does the helper
    report the resource as missing.
    """
    for attempt in range(retries):
        try:
            r = session().get(url, timeout=C.REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
        except requests.RequestException:
            pass
        time.sleep(0.4 * (attempt + 1))
    return None


def _get_text(url: str, params: dict | None = None, retries: int = C.MAX_RETRIES) -> str:
    """GET a text resource with retries. Returns an empty string when it cannot be read."""
    for attempt in range(retries):
        try:
            r = session().get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.text
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
        except requests.RequestException:
            pass
        time.sleep(0.6 * (attempt + 1))
    return ""


def list_available_months(symbol: str, kind: str) -> list[str]:
    """Ask the archive which months exist for one symbol.

    The object listing is far cheaper than probing month by month: one request returns
    every month that was ever published for the contract, including the gaps left by
    contracts that were listed part way through the window.
    """
    if kind == "fundingRate":
        prefix = f"data/futures/um/monthly/fundingRate/{symbol}/"
    elif kind == "premium":
        prefix = f"data/futures/um/monthly/premiumIndexKlines/{symbol}/1h/"
    else:
        prefix = f"data/futures/um/monthly/klines/{symbol}/1h/"

    s3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
    months: set[str] = set()
    token = None
    for _ in range(10):
        params = {"prefix": prefix}
        if token:
            params["continuation-token"] = token
        text = _get_text(s3, params=params)
        for key in re.findall(r"<Key>([^<]+\.zip)</Key>", text):
            stamp = key.rsplit("-", 2)[-2] + "-" + key.rsplit("-", 2)[-1].replace(".zip", "")
            if re.fullmatch(r"\d{4}-\d{2}", stamp):
                months.add(stamp)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", text)
        if not m:
            break
        token = m.group(1)
    return sorted(months)


def _read_zip_csv(payload: bytes) -> pd.DataFrame:
    """Read the single CSV member of a Binance zip archive into a DataFrame."""
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            return pd.read_csv(fh)


def _months_already_stored(path: Path, time_col: str = "open_time") -> set[str]:
    """Return the ``YYYY-MM`` set already present in a per-symbol parquet file."""
    if not path.exists():
        return set()
    try:
        df = pd.read_parquet(path, columns=[time_col])
    except Exception:
        return set()
    if df.empty:
        return set()
    ts = pd.to_datetime(df[time_col], unit="ms", utc=True)
    return set(ts.dt.strftime("%Y-%m"))


def _append_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Append rows to a parquet file, de-duplicating on ``open_time``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        frame = pd.concat([old, frame], ignore_index=True)
    frame = frame.drop_duplicates(subset="open_time").sort_values("open_time")
    frame.to_parquet(path, index=False)


# --------------------------------------------------------------------------------------
# Stage 1: symbol universe
# --------------------------------------------------------------------------------------


def list_symbols() -> list[str]:
    """Every USDT-quoted perpetual future that has monthly klines in the archive."""
    s3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
    found: set[str] = set()
    token = None
    while True:
        params = {"delimiter": "/", "prefix": "data/futures/um/monthly/klines/"}
        if token:
            params["continuation-token"] = token
        text = _get_text(s3, params=params)
        found |= set(re.findall(r"<Prefix>data/futures/um/monthly/klines/([^<]+)/</Prefix>", text))
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", text)
        if not m:
            break
        token = m.group(1)
    return sorted(s for s in found if s.endswith("USDT"))


def probe_coverage(symbols: list[str], first: str = C.START_MONTH, last: str = C.END_MONTH) -> dict:
    """Split the symbol list into those alive across the whole window and those not.

    A symbol is a *survivor* when it has klines in both the first and the last month of
    the window. Symbols present only at the start were delisted during the window, and
    symbols present only at the end were listed after the window opened. Both groups are
    reported so the size of the survivorship effect is measured rather than assumed.
    """
    def check(sym: str) -> tuple[str, bool, bool]:
        months = set(list_available_months(sym, "klines"))
        return sym, first in months, last in months

    with ThreadPoolExecutor(max_workers=C.DOWNLOAD_WORKERS) as ex:
        results = list(ex.map(check, symbols))

    return {
        "survivors": [s for s, a, b in results if a and b],
        "delisted": [s for s, a, b in results if a and not b],
        "listed_later": [s for s, a, b in results if b and not a],
    }


# --------------------------------------------------------------------------------------
# Stage 2: bulk downloads
# --------------------------------------------------------------------------------------


def _iter_symbol_months(sym: str, months: list[str], kind: str):
    """Yield ``(month, frame)`` for every month that exists for one symbol.

    The month list comes from the archive index rather than from probing, so a contract
    that was listed part way through the window only costs the months it actually
    traded. A month that the index lists but that has already been stored locally is
    skipped, which is what makes the download resumable.
    """
    available = set(list_available_months(sym, kind))
    for month in months:
        if month not in available:
            continue
        if kind == "fundingRate":
            url = f"{C.BINANCE_MONTHLY}/fundingRate/{sym}/{sym}-fundingRate-{month}.zip"
        elif kind == "premium":
            url = f"{C.BINANCE_MONTHLY}/premiumIndexKlines/{sym}/1h/{sym}-1h-{month}.zip"
        else:
            url = f"{C.BINANCE_MONTHLY}/klines/{sym}/1h/{sym}-1h-{month}.zip"
        r = _get(url)
        if r is None:
            continue
        try:
            df = _read_zip_csv(r.content)
        except Exception:
            continue
        yield month, df


def download_klines(symbols: list[str], months: list[str], kind: str = "klines") -> int:
    """Download monthly 1h klines for every symbol. Returns files written."""
    out_dir = {"klines": C.KLINE_DIR, "premium": C.PREMIUM_DIR}[kind]
    written = 0

    def one(sym: str) -> int:
        target = out_dir / f"{sym}.parquet"
        stored = _months_already_stored(target)
        frames = []
        for month, df in _iter_symbol_months(sym, months, kind):
            if month in stored:
                continue
            df = df[KLINE_COLUMNS].copy()
            df["open_time"] = df["open_time"].astype("int64")
            df["close_time"] = df["close_time"].astype("int64")
            frames.append(df)
        if not frames:
            return 0
        _append_parquet(target, pd.concat(frames, ignore_index=True))
        return len(frames)

    with ThreadPoolExecutor(max_workers=C.DOWNLOAD_WORKERS) as ex:
        for i, n in enumerate(ex.map(one, symbols), 1):
            written += n
            if i % 50 == 0:
                print(f"  [{kind}] {i}/{len(symbols)} symbols, {written} months written", flush=True)
    return written


def download_funding(symbols: list[str], months: list[str]) -> int:
    """Download the perpetual funding rate history for every symbol."""
    written = 0

    def one(sym: str) -> int:
        target = C.FUNDING_DIR / f"{sym}.parquet"
        stored = _months_already_stored(target, time_col="calc_time")
        frames = []
        for month, df in _iter_symbol_months(sym, months, "fundingRate"):
            if month in stored:
                continue
            df = df.rename(columns={"calc_time": "open_time"})
            df["open_time"] = df["open_time"].astype("int64")
            frames.append(df)
        if not frames:
            return 0
        _append_parquet(target, pd.concat(frames, ignore_index=True))
        return len(frames)

    with ThreadPoolExecutor(max_workers=C.DOWNLOAD_WORKERS) as ex:
        for i, n in enumerate(ex.map(one, symbols), 1):
            written += n
            if i % 50 == 0:
                print(f"  [funding] {i}/{len(symbols)} symbols, {written} months written", flush=True)
    return written


def download_daily(symbols: list[str], days: list[str]) -> int:
    """Download daily 1h klines, used by the incremental update path."""
    written = 0

    def one(job: tuple[str, str]) -> bool:
        sym, day = job
        target = C.DAILY_DIR / f"{sym}.parquet"
        if day in _months_already_stored(target):
            return False
        url = f"{C.BINANCE_DAILY}/klines/{sym}/1h/{sym}-1h-{day}.zip"
        r = _get(url)
        if r is None:
            return False
        try:
            df = _read_zip_csv(r.content)
        except Exception:
            return False
        df = df[KLINE_COLUMNS].copy()
        df["open_time"] = df["open_time"].astype("int64")
        df["close_time"] = df["close_time"].astype("int64")
        _append_parquet(target, df)
        return True

    jobs = [(s, d) for s in symbols for d in days]
    with ThreadPoolExecutor(max_workers=C.DOWNLOAD_WORKERS) as ex:
        for i, ok in enumerate(ex.map(one, jobs), 1):
            written += int(ok)
            if i % 250 == 0:
                print(f"  [daily] {i}/{len(jobs)} files, {written} newly written", flush=True)
    return written


# --------------------------------------------------------------------------------------
# Stage 3: auxiliary sources
# --------------------------------------------------------------------------------------


def download_fred() -> int:
    """Download the FRED macro series used as regime features."""
    written = 0
    for series in C.FRED_SERIES:
        path = C.MACRO_DIR / f"{series}.csv"
        r = _get(C.FRED_URL.format(series=series))
        if r is None:
            print(f"  [fred] FAILED {series}")
            continue
        path.write_bytes(r.content)
        written += 1
    return written


def download_fear_greed() -> int:
    """Download the alternative.me crypto Fear & Greed index, the full daily history."""
    path = C.SENTIMENT_DIR / "fear_and_greed.csv"
    r = _get(C.FEAR_GREED_URL)
    if r is None:
        return 0
    payload = r.json()
    df = pd.DataFrame(payload["data"])
    df["date"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="s")
    df = df[["date", "value", "value_classification"]].sort_values("date")
    df.to_csv(path, index=False)
    return len(df)


def download_coingecko(symbols: list[str]) -> int:
    """Download market capitalisation and circulating supply for the panel's coins.

    Binance symbols do not match CoinGecko ids (``1000PEPEUSDT`` is ``pepe``), so the
    mapping is resolved through the CoinGecko coin list with a small override table for
    the prefixed denominations.
    """
    import json

    mapping_file = C.SENTIMENT_DIR / "coingecko_map.json"
    mapping: dict[str, str] = {}
    if mapping_file.exists():
        mapping = json.loads(mapping_file.read_text())

    missing = [s for s in symbols if s not in mapping and s not in C.COINGECKO_ID_OVERRIDES]
    if missing:
        r = _get(C.COINGECKO_LIST)
        if r is not None:
            catalogue = {c["symbol"].lower(): c["id"] for c in r.json()}
            for sym in missing:
                base = sym[:-4].lower()
                candidates = [base]
                for prefix in ("1000000", "1000", "10000"):
                    if base.startswith(prefix):
                        candidates.append(base[len(prefix):])
                for candidate in candidates:
                    if candidate in catalogue:
                        mapping[sym] = catalogue[candidate]
                        break
        mapping_file.write_text(json.dumps(mapping, indent=1))

    ids = [C.COINGECKO_ID_OVERRIDES.get(s, mapping.get(s)) for s in symbols]
    wanted = {i: s for s, i in zip(symbols, ids) if i}
    if not wanted:
        return 0

    # The markets endpoint is paginated and only the first pages cover the liquid coins,
    # so several pages are walked until the wanted ids stop appearing.
    rows: list[dict] = []
    for page in range(1, 6):
        r = _get(f"{C.COINGECKO_MARKETS}?vs_currency=usd&per_page=250&page={page}&order=market_cap_desc")
        if r is None:
            break
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        found = sum(1 for row in batch if row["id"] in wanted)
        if found == 0 and page > 1:
            break
        time.sleep(1.2)  # stay inside the public rate limit

    out = pd.DataFrame(
        [
            {
                "ticker": wanted[row["id"]],
                "coingecko_id": row["id"],
                "market_cap": row.get("market_cap"),
                "circulating_supply": row.get("circulating_supply"),
                "total_supply": row.get("total_supply"),
                "volume_24h": row.get("total_volume"),
                "price_usd": row.get("current_price"),
                "market_cap_rank": row.get("market_cap_rank"),
            }
            for row in rows
            if row["id"] in wanted
        ]
    )
    if out.empty:
        return 0
    out.to_csv(C.SENTIMENT_DIR / "coingecko_market_data.csv", index=False)
    return len(out)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def run_full() -> dict:
    """Run every download stage and return a manifest of what was fetched."""
    months = month_range()
    started = datetime.now()

    print("[1/5] listing Binance perpetual symbols", flush=True)
    symbols = list_symbols()
    print(f"      {len(symbols)} USDT perpetuals", flush=True)

    print("[2/5] probing window coverage", flush=True)
    coverage = probe_coverage(symbols)
    # The panel holds every contract that is alive at the end of the window, whether it
    # was listed on day one or three years later. Contracts that were alive at the start
    # and died during the window cannot be modelled without delisting returns, so they
    # are excluded and counted: that count is the survivorship-bias measure.
    panel = coverage["survivors"] + coverage["listed_later"]
    print(
        f"      panel {len(panel)} (survivors {len(coverage['survivors'])}, "
        f"listed later {len(coverage['listed_later'])}), "
        f"delisted mid-window {len(coverage['delisted'])}",
        flush=True,
    )

    print(f"[3/5] downloading 1h klines for {len(panel)} symbols", flush=True)
    klines = download_klines(panel, months)

    print("[4/5] downloading funding rates and premium index", flush=True)
    funding = download_funding(panel, months)
    premium = download_klines(panel, months, kind="premium")

    print("[5/5] downloading FRED, Fear & Greed and CoinGecko", flush=True)
    fred = download_fred()
    fear = download_fear_greed()
    gecko = download_coingecko(panel)

    manifest = {
        "started": started.isoformat(),
        "finished": datetime.now().isoformat(),
        "months": months,
        "n_symbols_available": len(symbols),
        "n_panel_symbols": len(panel),
        "n_alive_at_window_start": len(coverage["survivors"]) + len(coverage["delisted"]),
        "n_survivors_full_window": len(coverage["survivors"]),
        "n_listed_later": len(coverage["listed_later"]),
        "n_delisted_mid_window": len(coverage["delisted"]),
        "klines_files_written": klines,
        "funding_files_written": funding,
        "premium_files_written": premium,
        "fred_series": fred,
        "fear_greed_rows": fear,
        "coingecko_rows": gecko,
    }
    (C.DATA_DIR / "download_manifest.json").write_text(pd.Series(manifest).to_json(indent=2))
    return manifest


def run_incremental(days_back: int = 3) -> dict:
    """Fetch the most recent daily files so the panel can be extended to today."""
    symbols = [p.stem for p in C.KLINE_DIR.glob("*.parquet")]
    days = [
        (date.today() - pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(days_back, 0, -1)
    ]
    written = download_daily(symbols, days)
    return {"symbols": len(symbols), "days": days, "files_written": written}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download every data source used by the project")
    ap.add_argument("--stage", default="all", choices=["all", "incremental", "fred", "sentiment"])
    ap.add_argument("--days-back", type=int, default=3)
    args = ap.parse_args()

    if args.stage == "all":
        print(run_full())
    elif args.stage == "incremental":
        print(run_incremental(args.days_back))
    elif args.stage == "fred":
        print({"fred_series": download_fred(), "fear_greed_rows": download_fear_greed()})
    else:
        print({"coingecko_rows": download_coingecko([p.stem for p in C.KLINE_DIR.glob("*.parquet")])})
