"""Monitoring: a static dashboard and a Telegram bot that reports the day's activity.

The dashboard is a single self-contained HTML file with no external assets, so it can be
opened from a disk, served from any static host, or attached to a cron report.

The Telegram bot posts the same summary to a chat. Credentials come from the
environment (``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID``); without them the module
still runs and simply reports that no channel is configured.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------


def daily_summary(
    panel: pd.DataFrame,
    strategies: pd.DataFrame,
    exact: pd.DataFrame | None = None,
    top_n: int = 8,
) -> str:
    """A short text summary of the latest signals and the best strategies."""
    latest_ts = panel["timestamp"].max()
    rows = []
    for column in strategies.columns:
        if not column.startswith("pred_"):
            continue
        fired = strategies.loc[strategies[column] != 0]
        # The panel is not re-indexed, so the age of a signal is taken from the panel's
        # own timestamp column rather than from the row index.
        stamps = panel["timestamp"].reindex(fired.index)
        recent = fired[(stamps >= latest_ts - pd.Timedelta(hours=24)).to_numpy()]
        if recent.empty:
            continue
        tickers = (
            panel.loc[recent.index]
            .groupby("ticker")["close"]
            .last()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
        rows.append((column, len(recent), tickers))

    lines = [
        f"*Crypto perpetuals strategy update* - {latest_ts:%Y-%m-%d %H:%M} UTC",
        "",
        f"Signals fired in the last 24 hours:",
    ]
    if rows:
        for column, count, tickers in rows:
            names = ", ".join(t.replace("USDT", "") for t in tickers)
            lines.append(f"- `{column}`: {count} signals - {names}")
    else:
        lines.append("- none")

    if exact is not None and len(exact):
        lines += ["", "*Strategy leaderboard (exact simulation)*"]
        for _, row in exact.head(5).iterrows():
            lines.append(
                f"- `{row['prediction']}`: CAGR {row['CAGR']:.2%}, "
                f"Sharpe {row['Sharpe']:.2f}, max drawdown {row['max_drawdown']:.2%}"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------------------

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Crypto perpetuals strategy dashboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 2rem;
         background: #f7f8fa; color: #16202c; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.2rem; }}
  p.sub {{ color: #5b6b7c; margin-top: 0; }}
  .card {{ background: #fff; border: 1px solid #e3e8ef; border-radius: 8px;
           padding: 1rem 1.25rem; margin-bottom: 1.25rem; }}
  h2 {{ font-size: 1rem; text-transform: uppercase; letter-spacing: .06em;
        color: #5b6b7c; margin: 0 0 .75rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .88rem; }}
  th, td {{ text-align: left; padding: .35rem .6rem; border-bottom: 1px solid #eef1f5; }}
  th {{ color: #5b6b7c; font-weight: 600; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .pos {{ color: #0f7b45; }} .neg {{ color: #b4232c; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: .75rem; }}
  .kpi {{ background: #fff; border: 1px solid #e3e8ef; border-radius: 8px; padding: .8rem 1rem; }}
  .kpi .v {{ font-size: 1.35rem; font-variant-numeric: tabular-nums; }}
  .kpi .l {{ color: #5b6b7c; font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; }}
</style>
</head>
<body>
<h1>Crypto perpetuals strategy dashboard</h1>
<p class="sub">Generated {generated}</p>
<div class="grid">{kpis}</div>
<div class="card"><h2>Strategy leaderboard</h2>{strategies}</div>
<div class="card"><h2>Model scores</h2>{models}</div>
<div class="card"><h2>Latest signals</h2>{signals}</div>
</body>
</html>
"""


def _table(rows: list[dict], numeric_cols: tuple[str, ...] = ()) -> str:
    if not rows:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{c}</th>" for c in rows[0])
    body = []
    for row in rows:
        cells = []
        for col, value in row.items():
            if col in numeric_cols and isinstance(value, (int, float)):
                cls = "num"
                if "cagr" in col.lower() or "return" in col.lower() or "sharpe" in col.lower():
                    cls += " pos" if value > 0 else " neg"
                cells.append(f'<td class="{cls}">{value:,.4f}</td>')
            else:
                cells.append(f"<td>{value}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_dashboard(
    panel: pd.DataFrame,
    strategies: pd.DataFrame,
    exact: pd.DataFrame | None = None,
    model_scores: dict | None = None,
    path: Path = C.OUTPUT_DIR / "dashboard.html",
) -> Path:
    """Write the self-contained HTML dashboard."""
    latest_ts = panel["timestamp"].max()
    stamps_all = panel["timestamp"].reindex(strategies.index)
    fired_recent = (
        (stamps_all >= latest_ts - pd.Timedelta(hours=24)).to_numpy()
        & (strategies.loc[:, [c for c in strategies.columns if c.startswith("pred_")]] != 0).any(axis=1).to_numpy()
    )
    kpis = [
        ("Latest bar", f"{latest_ts:%Y-%m-%d %H:%M} UTC"),
        ("Contracts", f"{panel['ticker'].nunique():,}"),
        ("Panel rows", f"{len(panel):,}"),
        (
            "Signals (24h)",
            f"{int(fired_recent.sum()):,}",
        ),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="v">{v}</div><div class="l">{label}</div></div>'
        for label, v in kpis
    )

    strategy_rows = []
    if exact is not None and len(exact):
        for _, row in exact.head(12).iterrows():
            strategy_rows.append(
                {
                    "strategy": row["prediction"],
                    "CAGR": float(row["CAGR"]),
                    "Sharpe": float(row["Sharpe"]),
                    "max_drawdown": float(row["max_drawdown"]),
                    "trades": int(row["trades"]),
                    "win_rate": float(row["win_rate"]),
                }
            )
    strategy_html = _table(strategy_rows, ("CAGR", "Sharpe", "max_drawdown", "trades", "win_rate"))

    model_rows = []
    if model_scores:
        for name, res in model_scores.items():
            model_rows.append(
                {
                    "model": name,
                    "valid_precision": float(res["scores"]["valid"]["precision"]),
                    "valid_auc": float(res["scores"]["valid"]["auc"]),
                    "test_precision": float(res["scores"]["test"]["precision"]),
                    "test_auc": float(res["scores"]["test"]["auc"]),
                }
            )
    model_html = _table(model_rows, ("valid_precision", "valid_auc", "test_precision", "test_auc"))

    recent = panel.loc[panel["timestamp"] >= latest_ts - pd.Timedelta(hours=24)]
    signal_rows = []
    for column in strategies.columns:
        if not column.startswith("pred_"):
            continue
        fired = strategies.loc[strategies[column] != 0]
        fired = fired[fired.index.isin(recent.index)]
        if fired.empty:
            continue
        signal_rows.append({"strategy": column, "signals_24h": int(len(fired))})
    signal_html = _table(signal_rows, ("signals_24h",))

    html = HTML_TEMPLATE.format(
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        kpis=kpi_html,
        strategies=strategy_html,
        models=model_html,
        signals=signal_html,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path


# --------------------------------------------------------------------------------------
# Telegram
# --------------------------------------------------------------------------------------


def send_telegram(message: str, token: str | None = None, chat_id: str | None = None) -> dict:
    """Post a message to Telegram. Returns the API response or an explanation."""
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return {"sent": False, "reason": "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not set"}
    response = requests.post(
        TELEGRAM_API.format(token=token),
        json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
        timeout=20,
    )
    return {"sent": response.ok, "status_code": response.status_code, "body": response.text[:200]}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the dashboard and post the daily summary")
    ap.add_argument("--dashboard", action="store_true")
    ap.add_argument("--telegram", action="store_true")
    args = ap.parse_args()

    if not (C.CACHE_DIR / "panel.parquet").exists():
        raise SystemExit("run the pipeline first: python -m src.pipeline")

    from src.simulate import build_strategy_columns

    panel = pd.read_parquet(C.CACHE_DIR / "panel.parquet")
    predictions = pd.read_parquet(C.CACHE_DIR / "predictions.parquet")
    strategies = build_strategy_columns(panel, predictions)

    exact = None
    exact_path = C.OUTPUT_DIR / "strategy_exact.csv"
    if exact_path.exists():
        exact = pd.read_csv(exact_path)

    model_scores = None
    scores_path = C.OUTPUT_DIR / "model_scores.json"
    if scores_path.exists():
        model_scores = json.loads(scores_path.read_text())

    if args.dashboard or not args.telegram:
        path = build_dashboard(panel, strategies, exact, model_scores)
        print(f"dashboard written to {C.display_path(path)}")

    if args.telegram:
        result = send_telegram(daily_summary(panel, strategies, exact))
        print(json.dumps(result, indent=2))
