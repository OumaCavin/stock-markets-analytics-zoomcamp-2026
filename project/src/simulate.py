"""Trading simulation: vectorised estimate, exact account simulation and benchmarks.

Three levels of detail are provided on purpose.

``simulate_vectorized``
    The lecture's approach. It counts how many signals fire per day, sizes the capital
    requirement with the same formula the Module 04 notebook uses, and multiplies the
    forward return of every signal by the position size. It is fast and it is
    approximate: it assumes every trade can be filled at the close of the signal bar and
    that the capital is never actually short.

``simulate_exact``
    The account-level simulation. It walks the bars in order, opens and closes
    positions on a 30-day-style overlapping schedule, deducts the trading fee on both
    sides, tracks cash and open positions, and produces an equity curve that can be
    turned into a CAGR, a Sharpe ratio and a maximum drawdown. Positions can be sized
    as a fraction of the running equity, which is what reinvestment means.

``benchmarks``
    Buy and hold on BTC, and an equal-weight basket of every contract in the panel, so
    that "the strategy made money" can be separated from "the market went up".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402


# --------------------------------------------------------------------------------------
# Vectorised simulation (lecture style)
# --------------------------------------------------------------------------------------


def simulate_vectorized(
    df: pd.DataFrame,
    prediction_column: str,
    forward_return_column: str = "future_return",
    trade_size: float = C.TRADE_SIZE_USD,
    fee_rate: float = C.FEE_RATE,
    horizon_days: int | None = None,
    side_column: str | None = None,
) -> dict:
    """Approximate returns on capital for one signal column.

    The capital requirement follows the Module 04 notebook: the capital has to cover
    the overlapping positions, so it is the position size times the number of days a
    position is held times the 75th percentile of the signals per day.

    A signal column may hold -1 as well as 1, which is how the long and short legs of
    the market-neutral strategy are expressed. The sign is applied to the forward
    return so that a short leg earns when the price falls.
    """
    if horizon_days is None:
        horizon_days = max(1, C.HORIZON_HOURS // 24)

    trades = df[df[prediction_column] != 0].copy()
    if trades.empty:
        return {
            "prediction": prediction_column,
            "count_investments": 0,
            "capital": 0.0,
            "CAGR": 0.0,
            "avg_net_rev_per_investment": 0.0,
        }

    if side_column and side_column in trades.columns:
        signed = trades[forward_return_column].astype("float64") * trades[side_column].astype("float64")
    else:
        signed = trades[forward_return_column].astype("float64")

    per_day = trades.groupby(trades["timestamp"].dt.floor("D")).size()
    q75 = float(np.percentile(per_day, 75))
    capital = trade_size * horizon_days * q75

    gross = (signed * trade_size).sum()
    fees = -fee_rate * trade_size * len(trades)
    net = gross + fees
    avg_net = net / len(trades)

    days = (df["timestamp"].max() - df["timestamp"].min()).days
    if days > 0 and capital > 0:
        cagr = (1 + net / capital) ** (C.DAYS_PER_YEAR / days) - 1
    else:
        cagr = 0.0

    wins = (signed > 0).mean()
    return {
        "prediction": prediction_column,
        "count_investments": int(len(trades)),
        "avg_investments_per_day": float(len(trades) / max(1, len(per_day))),
        "q75_investments_per_day": q75,
        "gross_rev": float(gross),
        "fees": float(fees),
        "net_rev": float(net),
        "avg_net_rev_per_investment": float(avg_net),
        "win_rate": float(wins),
        "capital": float(capital),
        "CAGR": float(cagr),
    }


# --------------------------------------------------------------------------------------
# Exact account simulation
# --------------------------------------------------------------------------------------


def simulate_exact(
    df: pd.DataFrame,
    prediction_column: str,
    forward_return_column: str = "future_return",
    side_column: str | None = None,
    trade_size: float = C.TRADE_SIZE_USD,
    fee_rate: float = C.FEE_RATE,
    horizon_hours: int = C.HORIZON_HOURS,
    equity_fraction: float | None = None,
    max_concurrent: int | None = None,
    max_per_symbol: int = C.MAX_POSITIONS_PER_SYMBOL,
) -> dict:
    """Walk the account over the signals and produce an equity curve.

    The simulator is event driven. Only the bars that carry a signal are visited, which
    keeps a multi-million row panel tractable, and each position is closed exactly
    ``horizon_hours`` later at the price of that bar.

    ``equity_fraction`` switches between the two sizing modes. With it set, every new
    position is a fraction of the running equity, which reinvests the gains and lets the
    account compound. Without it, every position is a fixed dollar amount and the
    function reports the capital that such a book would have needed.

    ``max_per_symbol`` limits how many positions may be open in the same contract at
    once, which stops the simulator from stacking overlapping bets on one coin.
    """
    work = df[["timestamp", "ticker", "close", forward_return_column, prediction_column]].copy()
    if side_column and side_column in df.columns:
        work["side"] = df[side_column].astype("int8").to_numpy()
    else:
        work["side"] = 1

    by_symbol = work.groupby("ticker", sort=False)
    # The horizon is a number of hours, but the shift counts rows, and the panel is
    # thinned to one row every MODEL_FREQUENCY_HOURS. The step between two rows is
    # measured from the data itself, so the position is closed after the same number of
    # hours whatever the frequency of the panel is.
    step = work.groupby("ticker")["timestamp"].diff().dt.total_seconds().median()
    if not step or step <= 0:
        step = 3600.0 * max(1, C.MODEL_FREQUENCY_HOURS)
    horizon_rows = max(1, int(round(horizon_hours * 3600.0 / float(step))))
    work["exit_price"] = by_symbol["close"].shift(-horizon_rows)
    work["exit_timestamp"] = by_symbol["timestamp"].shift(-horizon_rows)

    entries = work[work[prediction_column] != 0].dropna(subset=["exit_price", "exit_timestamp"])
    if entries.empty:
        return {
            "prediction": prediction_column,
            "trades": 0,
            "win_rate": 0.0,
            "gross_pnl": 0.0,
            "fees": 0.0,
            "net_pnl": 0.0,
            "required_capital": 0.0,
            "final_equity": 0.0,
            "CAGR": 0.0,
            "Sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_equity": 0.0,
            "equity_curve": np.array([]),
            "equity_curve_index": pd.DatetimeIndex([]),
        }

    entries = entries.sort_values("timestamp")
    # Two accounting modes share one code path.
    #   equity_fraction is None: every position is a fixed dollar amount and the
    #     function reports the capital the book would have needed, measured as the
    #     deepest point of the cumulative cash flow.
    #   equity_fraction is set: every position is a fraction of the running equity, so
    #     the gains are reinvested and the account compounds. The equity curve records
    #     realised equity, which is what a cash account shows.
    cash = 0.0
    equity = 1.0 if equity_fraction is not None else 0.0
    open_positions: dict[str, list[dict]] = {}
    open_count = 0
    events: list[tuple[pd.Timestamp, float]] = []
    trades = 0
    wins = 0
    gross_pnl = 0.0
    fees_paid = 0.0
    cash_flow = 0.0
    peak_capital = 0.0

    def close_due(now: pd.Timestamp) -> None:
        nonlocal cash, equity, open_count, trades, wins, gross_pnl, fees_paid
        nonlocal cash_flow, peak_capital
        for symbol, positions in list(open_positions.items()):
            still = []
            for pos in positions:
                if pos["exit_timestamp"] > now:
                    still.append(pos)
                    continue
                pnl = pos["side"] * pos["qty"] * (pos["exit_price"] - pos["entry_price"])
                exit_fee = abs(pos["qty"]) * pos["exit_price"] * fee_rate
                cash += pnl - exit_fee
                cash_flow += pnl - exit_fee
                equity += pnl - exit_fee
                fees_paid += exit_fee
                gross_pnl += pnl
                wins += int(pos["side"] * (pos["exit_price"] - pos["entry_price"]) > 0)
                trades += 1
                open_count -= 1
            if still:
                open_positions[symbol] = still
            else:
                open_positions.pop(symbol, None)
        peak_capital = max(peak_capital, -cash_flow)

    for _, row in entries.iterrows():
        now = row["timestamp"]
        close_due(now)

        symbol = row["ticker"]
        held = len(open_positions.get(symbol, []))
        if held >= max_per_symbol:
            continue
        if max_concurrent is not None and open_count >= max_concurrent:
            continue

        side = int(row["side"])
        price = float(row["close"])
        if equity_fraction is None:
            notional = trade_size
        else:
            notional = equity * equity_fraction
        if notional <= 0:
            continue
        qty = notional / price
        entry_fee = abs(qty) * price * fee_rate
        cash -= entry_fee
        cash_flow -= entry_fee
        equity -= entry_fee
        fees_paid += entry_fee
        open_positions.setdefault(symbol, []).append(
            {
                "qty": qty,
                "entry_price": price,
                "exit_price": float(row["exit_price"]),
                "exit_timestamp": row["exit_timestamp"],
                "side": side,
            }
        )
        open_count += 1
        events.append((now, equity if equity_fraction is not None else cash_flow))

    # Close whatever is still open so that the final equity is complete.
    close_due(entries["exit_timestamp"].max() + pd.Timedelta(hours=1))

    curve_index = pd.DatetimeIndex([])
    if not events:
        equity_curve = np.array([])
    else:
        series = pd.Series(
            [v for _, v in events], index=pd.DatetimeIndex([t for t, _ in events])
        ).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        equity_curve = series.to_numpy()
        curve_index = series.index

    if equity_fraction is not None:
        # The compounding book starts with one unit of equity and ends with the value
        # of the account, so the ratio of the two is the growth factor.
        final_equity = float(equity)
        start_equity = 1.0
        ending_capital = final_equity
    else:
        # The fixed size book reports the profit separately. The capital a book like
        # this needs is the size of one position times the number of days it is held
        # times the 75th percentile of the signals per day: the positions overlap, so
        # the capital has to cover all of them at once. This is the Module 04
        # notebook's measure and it is what the CAGR is computed on.
        per_day = entries.groupby(entries["timestamp"].dt.floor("D")).size()
        q75 = float(np.percentile(per_day, 75)) if len(per_day) else 0.0
        start_equity = trade_size * max(1, horizon_hours // 24) * q75
        if start_equity <= 0:
            start_equity = peak_capital if peak_capital > 0 else float(trade_size * 100)
        final_equity = cash
        # The growth factor of the fixed size book is the capital it needed plus the
        # profit it made, divided by the capital it needed. The profit on its own is
        # not a growth factor, and dividing it by the capital would report a loss for
        # a book that in fact grew.
        ending_capital = start_equity + cash
        if len(equity_curve):
            equity_curve = equity_curve + start_equity

    days = max(1, (entries["exit_timestamp"].max() - entries["timestamp"].min()).days)
    if start_equity > 0 and ending_capital > 0:
        cagr = (ending_capital / start_equity) ** (C.DAYS_PER_YEAR / days) - 1
    elif start_equity > 0 and equity_fraction is None:
        # The book lost more than the capital it needed. A compound growth rate is not
        # defined for a negative balance, so the loss is reported as a return on the
        # capital instead of being hidden behind a zero.
        cagr = float("nan")
    else:
        cagr = 0.0

    if len(equity_curve):
        daily = pd.Series(equity_curve, index=curve_index).resample("D").last().dropna()
        daily_ret = daily.pct_change().dropna()
        sharpe = (
            float(daily_ret.mean() / daily_ret.std() * np.sqrt(C.DAYS_PER_YEAR))
            if len(daily_ret) > 2 and daily_ret.std() > 0
            else 0.0
        )
        running_max = pd.Series(equity_curve).cummax()
        max_dd = float((pd.Series(equity_curve) / running_max - 1).min())
        avg_equity = float(np.mean(equity_curve))
    else:
        sharpe, max_dd, avg_equity = 0.0, 0.0, 0.0

    return {
        "prediction": prediction_column,
        "trades": trades,
        "win_rate": float(wins / trades) if trades else 0.0,
        "gross_pnl": float(gross_pnl),
        "fees": float(fees_paid),
        "net_pnl": float(cash if equity_fraction is None else final_equity - 1.0),
        "ending_capital": float(ending_capital),
        "return_on_capital": float(ending_capital / start_equity - 1) if start_equity else 0.0,
        "required_capital": float(start_equity),
        "peak_cash_need": float(peak_capital),
        "final_equity": float(ending_capital),
        "CAGR": float(cagr),
        "Sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_equity": avg_equity,
        "equity_curve": equity_curve,
        "equity_curve_index": curve_index,
    }


def buy_and_hold(df: pd.DataFrame, ticker: str, forward_return_column: str = "future_return") -> dict:
    """Buy and hold one contract over the whole panel."""
    rows = df[df["ticker"] == ticker].sort_values("timestamp")
    if rows.empty:
        return {}
    price = rows["close"].to_numpy(dtype="float64")
    equity = price / price[0]
    stamps = rows["timestamp"].to_numpy()
    days = max(1, (pd.Timestamp(stamps[-1]) - pd.Timestamp(stamps[0])).days)
    cagr = (equity[-1]) ** (C.DAYS_PER_YEAR / days) - 1
    daily = pd.Series(equity, index=pd.to_datetime(stamps)).resample("D").last().dropna()
    dr = daily.pct_change().dropna()
    sharpe = (
        float(dr.mean() / dr.std() * np.sqrt(C.DAYS_PER_YEAR))
        if len(dr) > 2 and dr.std() > 0
        else 0.0
    )
    running_max = pd.Series(equity).cummax()
    return {
        "benchmark": f"buy_and_hold_{ticker}",
        "CAGR": float(cagr),
        "Sharpe": sharpe,
        "max_drawdown": float((pd.Series(equity) / running_max - 1).min()),
        "total_return": float(equity[-1] - 1),
        "equity_curve": equity,
    }


def equal_weight_basket(df: pd.DataFrame, forward_return_column: str = "future_return") -> dict:
    """Buy and hold an equal-weighted basket of every contract in the panel."""
    rows = df.sort_values(["timestamp", "ticker"]).copy()
    # Each contract is normalised by its own first price before the average, otherwise
    # a coin quoted at 100,000 would swamp the basket.
    normalised = rows.groupby("ticker")["close"].transform(lambda s: s / s.iloc[0])
    per_hour = normalised.groupby(rows["timestamp"]).mean()
    equity = per_hour / per_hour.iloc[0]
    stamps = per_hour.index.to_numpy()
    days = max(1, (pd.Timestamp(stamps[-1]) - pd.Timestamp(stamps[0])).days)
    cagr = equity.iloc[-1] ** (C.DAYS_PER_YEAR / days) - 1
    daily = equity.resample("D").last().dropna()
    dr = daily.pct_change().dropna()
    sharpe = (
        float(dr.mean() / dr.std() * np.sqrt(C.DAYS_PER_YEAR))
        if len(dr) > 2 and dr.std() > 0
        else 0.0
    )
    return {
        "benchmark": "equal_weight_basket",
        "CAGR": float(cagr),
        "Sharpe": sharpe,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "total_return": float(equity.iloc[-1] - 1),
        "equity_curve": equity.to_numpy(),
    }


def build_strategy_columns(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
    top_k: int = 3,
    top_k_floor: float = 0.5,
) -> pd.DataFrame:
    """Create one signal column per strategy.

    The strategies deliberately differ in how much they trade and in what they
    condition on, because the interesting question is not "which model is best" but
    "which combination of model, threshold and market state produces a book that
    survives its own costs".

    ``thresholds`` maps a model name to the probability cut-off that was selected on
    the validation slice. A strategy that uses no threshold at all trades the top
    ranked signals of each day instead.
    """
    thresholds = thresholds or {}
    out = pd.DataFrame(index=panel.index)

    prob_columns = [c for c in predictions.columns if c.startswith("prob_")]
    for col in prob_columns:
        name = col[5:]
        cut = thresholds.get(name, 0.5)
        out[f"pred_{name}"] = (predictions[col] >= cut).astype("int8").to_numpy()

    if not prob_columns:
        return out

    frame = pd.DataFrame({c[5:]: predictions[c] for c in prob_columns})
    frame["timestamp"] = panel["timestamp"].to_numpy()
    frame["ticker"] = panel["ticker"].to_numpy()

    # Rank based strategies. At every decision point the contracts are ranked by the
    # model and the best few are traded. A rank does not depend on the calibration of
    # the model, so the rate of signals survives a change in the distribution of the
    # predicted probabilities, which an absolute cut-off does not: the absolute
    # thresholds chosen on the validation slice fire on nothing at all in the test
    # slice, which is reported as a finding rather than hidden.
    for name in [c[5:] for c in prob_columns]:
        ordered = frame.sort_values(["timestamp", name], ascending=[True, False])
        for k in (1, 3):
            keep = ordered.groupby("timestamp").head(k)
            signal = pd.Series(0, index=frame.index)
            signal.loc[keep.index] = 1
            out[f"pred_{name}_top{k}"] = signal.astype("int8").to_numpy()

    # Ensemble of the tree families.
    tree_cols = [
        c for c in frame.columns if c in ("decision_tree", "random_forest", "xgboost", "lightgbm")
    ]
    if tree_cols:
        frame["ensemble"] = frame[tree_cols].mean(axis=1)
        cut = thresholds.get("ensemble", 0.5)
        out["pred_ensemble"] = (frame["ensemble"] >= cut).astype("int8").to_numpy()
    else:
        frame["ensemble"] = frame[frame.columns[0]]
        out["pred_ensemble"] = out[f"pred_{frame.columns[0]}"].to_numpy()

    # The ensemble is also traded as a rank, which is the version that survives out of
    # sample.
    ordered = frame.sort_values(["timestamp", "ensemble"], ascending=[True, False])
    for k in (1, 3, 5):
        keep = ordered.groupby("timestamp").head(k)
        signal = pd.Series(0, index=frame.index)
        signal.loc[keep.index] = 1
        out[f"pred_ensemble_top{k}"] = signal.astype("int8").to_numpy()

    base = out["pred_ensemble_top3"].to_numpy()

    # Market regime filter: only take the model's side when the market agrees.
    if "market_growth_24h" in panel.columns:
        aligned = panel["market_growth_24h"].to_numpy()
        out["pred_regime_aligned"] = (base * (aligned > 0)).astype("int8")
        out["pred_regime_contrarian"] = (base * (aligned < 0)).astype("int8")

    # Low volatility filter: the model's edge is larger when the market is calm.
    if "volatility_168h" in panel.columns:
        quiet = (panel["volatility_168h"] < panel["volatility_168h"].quantile(0.6)).to_numpy()
        out["pred_quiet_market"] = (base * quiet).astype("int8")

    # Long and short legs: long the most bullish contracts, short the least.
    if top_k * 2 <= len(frame):
        ordered = frame.sort_values(["timestamp", "ensemble"], ascending=[True, False])
        longs = ordered.groupby("timestamp").head(top_k).index
        shorts = ordered.groupby("timestamp").tail(top_k).index
        long_short = pd.Series(0, index=frame.index)
        long_short.loc[longs] = 1
        long_short.loc[shorts] = -1
        out["pred_long_short"] = long_short.astype("int8").to_numpy()

    # A hand-written macro rule, the kind the lecture used as a manual benchmark.
    if {"dgs10", "rsi_14", "growth_24h"}.issubset(panel.columns):
        rule = (
            (panel["dgs10"] < panel["dgs10"].quantile(0.60))
            & (panel["rsi_14"] < 45)
            & (panel["growth_24h"] < 0)
        ).astype("int8")
        out["pred_manual_macro"] = rule.to_numpy()

    # A hand rule built on the derivatives structure rather than on prices.
    if {"funding_rate", "rsi_14"}.issubset(panel.columns):
        rule = ((panel["funding_rate"] < 0) & (panel["rsi_14"] < 40)).astype("int8")
        out["pred_manual_funding"] = rule.to_numpy()

    return out


def strategy_summary(
    df: pd.DataFrame,
    strategies: pd.DataFrame,
    columns: list[str] | None = None,
    forward_return_column: str = "future_return",
) -> pd.DataFrame:
    """Run the vectorised simulation for every strategy and return one table."""
    columns = columns or [c for c in strategies.columns if c.startswith("pred_")]
    rows = []
    for col in columns:
        merged = df[["timestamp", "ticker", forward_return_column]].copy()
        merged["signal"] = strategies[col].to_numpy()
        if "pred_long_short" in columns:
            merged["side"] = strategies[col].to_numpy()
        result = simulate_vectorized(
            merged, "signal", forward_return_column, side_column="side" if "side" in merged else None
        )
        result["prediction"] = col
        rows.append(result)
    return pd.DataFrame(rows).sort_values("CAGR", ascending=False).reset_index(drop=True)


def exact_summaries(
    df: pd.DataFrame,
    strategies: pd.DataFrame,
    columns: list[str] | None = None,
    forward_return_column: str = "future_return",
    equity_fraction: float | None = None,
    max_concurrent: int | None = None,
) -> pd.DataFrame:
    """Run the exact account simulation for every strategy."""
    columns = columns or [c for c in strategies.columns if c.startswith("pred_")]
    rows = []
    for col in columns:
        merged = df[["timestamp", "ticker", "close", forward_return_column]].copy()
        merged["signal"] = strategies[col].to_numpy()
        merged["side"] = strategies[col].to_numpy()
        res = simulate_exact(
            merged,
            "signal",
            forward_return_column,
            side_column="side" if "pred_long_short" in columns else None,
            trade_size=C.TRADE_SIZE_USD,
            equity_fraction=equity_fraction,
            max_concurrent=max_concurrent,
        )
        # The equity curve and its index are dropped: they are long arrays, they are
        # used for the chart, and they do not belong in a summary table.
        res.pop("equity_curve", None)
        res.pop("equity_curve_index", None)
        res["prediction"] = col
        rows.append(res)
    return pd.DataFrame(rows).sort_values("CAGR", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print(__doc__)
