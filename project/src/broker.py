"""Place orders with a broker and keep a ledger of what would have been traded.

Two brokers are implemented.

``BinanceBroker``
    A real client for the exchange's REST API. It signs every request with HMAC-SHA256
    exactly as the exchange expects, supports the spot and USD-M futures endpoints, and
    can be pointed at the testnet. Because the exchange restricts trading from some
    regions, the client refuses to run without an explicit opt-in, and it always keeps
    writing to the local ledger so that a dry run and a live run produce the same
    audit trail.

``PaperBroker``
    A simulator that records orders, tracks positions and marks them to the panel's
    prices. This is the broker the cron job uses by default, and it is what produces
    the "latest trades" table without needing exchange credentials.

Both write to ``output/order_ledger.jsonl``, one JSON object per order.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402

LEDGER_PATH = C.OUTPUT_DIR / "order_ledger.jsonl"


@dataclass
class Order:
    """One order, in the shape both brokers understand."""

    symbol: str
    side: str  # BUY or SELL
    order_type: str  # MARKET or LIMIT
    quantity: float
    price: float | None = None
    strategy: str = ""
    probability: float | None = None
    timestamp: str = field(default_factory=lambda: pd.Timestamp.utcnow().isoformat())
    broker: str = "paper"
    status: str = "NEW"
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class PaperBroker:
    """Records orders and marks a position book against the panel's prices."""

    def __init__(self, ledger: Path = LEDGER_PATH):
        self.ledger = Path(ledger)
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.positions: dict[str, float] = {}

    def submit(self, order: Order) -> dict:
        order.broker = "paper"
        order.status = "FILLED"
        signed = order.side.upper() == "BUY"
        self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + (
            order.quantity if signed else -order.quantity
        )
        with self.ledger.open("a") as fh:
            fh.write(json.dumps(order.to_dict()) + "\n")
        return order.to_dict()

    def position_book(self) -> pd.DataFrame:
        if not self.positions:
            return pd.DataFrame(columns=["symbol", "quantity"])
        return (
            pd.DataFrame(self.positions.items(), columns=["symbol", "quantity"])
            .query("quantity != 0")
            .reset_index(drop=True)
        )


class BinanceBroker:
    """A signed client for the exchange's REST API.

    The signing scheme is the standard one: the query string is signed with the API
    secret using HMAC-SHA256 and the hexadecimal digest is appended as the signature.
    """

    SPOT = "https://api.binance.com"
    SPOT_TESTNET = "https://testnet.binance.vision"
    FUTURES = "https://fapi.binance.com"
    FUTURES_TESTNET = "https://testnet.binancefuture.vision"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        market: str = "spot",
        testnet: bool = True,
        live: bool = False,
        ledger: Path = LEDGER_PATH,
    ):
        self.api_key = api_key or os.environ.get("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("BINANCE_API_SECRET", "")
        self.market = market
        self.testnet = testnet
        self.live = live or os.environ.get("BINANCE_LIVE", "").lower() in ("1", "true", "yes")
        self.ledger = Path(ledger)
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    @property
    def base_url(self) -> str:
        if self.market == "futures":
            return self.FUTURES_TESTNET if self.testnet else self.FUTURES
        return self.SPOT_TESTNET if self.testnet else self.SPOT

    def _sign(self, params: dict) -> str:
        query = urlencode(params)
        return hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()

    def signed_request(self, method: str, path: str, params: dict | None = None) -> dict:
        """Send a signed request to the exchange."""
        if not self.api_key or not self.api_secret:
            raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET must be set")
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        params["signature"] = self._sign(params)
        url = f"{self.base_url}{path}"
        response = self.session.request(method, url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def market_price(self, symbol: str) -> float:
        path = "/fapi/v1/ticker/price" if self.market == "futures" else "/api/v3/ticker/price"
        return float(self.signed_request("GET", path, {"symbol": symbol})["price"])

    def submit(self, order: Order) -> dict:
        """Send an order to the exchange, unless the broker is in dry-run mode."""
        if not self.live:
            order.broker = "binance-dry-run"
            order.status = "DRY_RUN"
            with self.ledger.open("a") as fh:
                fh.write(json.dumps(order.to_dict()) + "\n")
            return order.to_dict()

        path = "/fapi/v1/order" if self.market == "futures" else "/api/v3/order"
        payload = {
            "symbol": order.symbol,
            "side": order.side.upper(),
            "type": order.order_type.upper(),
            "quantity": f"{order.quantity:.6f}".rstrip("0").rstrip("."),
        }
        if order.order_type.upper() == "LIMIT":
            payload["timeInForce"] = "GTC"
            payload["price"] = f"{order.price:.8f}"
        result = self.signed_request("POST", path, payload)
        order.status = result.get("status", "NEW")
        order.broker = f"binance-{'futures' if self.market == 'futures' else 'spot'}"
        order.note = str(result.get("orderId", ""))
        with self.ledger.open("a") as fh:
            fh.write(json.dumps(order.to_dict()) + "\n")
        return order.to_dict()


# --------------------------------------------------------------------------------------
# Turning model signals into orders
# --------------------------------------------------------------------------------------


def orders_from_signals(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    signal_column: str,
    broker,
    notional: float = 100.0,
    max_orders: int = 10,
    side_column: str | None = None,
) -> list[dict]:
    """Create orders from the most recent signals and submit them.

    Only the last bar is acted on: a prediction made an hour ago has already been
    traded, and re-submitting every historical signal would be a backtest, not a
    trading bot.
    """
    if signals.empty:
        return []
    # The panel is ordered by contract, so the last row is not the last bar. The most
    # recent bar is the set of rows that share the latest timestamp.
    last_ts = panel["timestamp"].max()
    rows = panel.index[panel["timestamp"] == last_ts]
    if len(rows) == 0:
        return []
    recent = signals.loc[rows].copy()
    recent["timestamp"] = panel.loc[rows, "timestamp"].to_numpy()
    recent["ticker"] = panel.loc[rows, "ticker"].to_numpy()
    recent["close"] = panel.loc[rows, "close"].to_numpy()
    if side_column and side_column in panel.columns:
        recent["side_flag"] = panel.loc[rows, side_column].to_numpy()

    fired = recent[recent[signal_column] != 0]
    if fired.empty:
        return []
    fired = fired.reindex(fired["close"].abs().sort_values(ascending=False).index)
    fired = fired.head(max_orders)

    results = []
    for _, row in fired.iterrows():
        price = float(row["close"])
        if not price or price != price:
            continue
        side_flag = int(row.get("side_flag", 1) or 1)
        side = "BUY" if side_flag >= 0 else "SELL"
        order = Order(
            symbol=str(row["ticker"]),
            side=side,
            order_type="MARKET",
            quantity=round(notional / price, 6),
            price=price,
            strategy=signal_column,
        )
        results.append(broker.submit(order))
    return results


def read_ledger(path: Path = LEDGER_PATH) -> pd.DataFrame:
    """Read the order ledger back into a frame."""
    if not Path(path).exists():
        return pd.DataFrame()
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Submit the latest signals through a broker")
    ap.add_argument("--signal", default="pred_ensemble")
    ap.add_argument("--notional", type=float, default=100.0)
    ap.add_argument("--max-orders", type=int, default=10)
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()

    if not (C.CACHE_DIR / "predictions.parquet").exists():
        raise SystemExit("run the pipeline first: python -m src.pipeline")

    panel = pd.read_parquet(C.CACHE_DIR / "panel.parquet")
    predictions = pd.read_parquet(C.CACHE_DIR / "predictions.parquet")
    from src.simulate import build_strategy_columns

    strategies = build_strategy_columns(panel, predictions)
    broker = BinanceBroker(live=args.live) if args.live else PaperBroker()
    submitted = orders_from_signals(
        panel, strategies, args.signal, broker, notional=args.notional, max_orders=args.max_orders
    )
    print(json.dumps(submitted, indent=2))
