"""Tests for the trading simulation and the threshold selection.

The simulator is the part of the project that is easiest to get subtly wrong, so it is
checked against cases with a hand-computed answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C, models, simulate  # noqa: E402


def make_panel(n_hours: int = 48, n_symbols: int = 2, seed: int = 0) -> pd.DataFrame:
    """A small panel with a known, constant price path."""
    rng = np.random.default_rng(seed)
    stamps = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    rows = []
    for symbol in [f"T{i}USDT" for i in range(n_symbols)]:
        for i, ts in enumerate(stamps):
            rows.append(
                {
                    "timestamp": ts,
                    "ticker": symbol,
                    "close": 100.0 + i,
                    "future_return": 0.01,
                }
            )
    frame = pd.DataFrame(rows)
    frame["future_return"] = rng.uniform(-0.02, 0.02, len(frame))
    return frame


def test_vectorized_counts_signals_and_capital():
    panel = make_panel()
    panel["signal"] = 0
    panel.loc[panel.index[:10], "signal"] = 1
    result = simulate.simulate_vectorized(panel, "signal", trade_size=100.0, fee_rate=0.0)
    assert result["count_investments"] == 10
    assert result["capital"] > 0
    assert result["avg_net_rev_per_investment"] == pytest.approx(
        panel.loc[panel.index[:10], "future_return"].mean() * 100
    )


def test_vectorized_fees_reduce_the_net_result():
    panel = make_panel()
    panel["signal"] = 0
    panel.loc[panel.index[:10], "signal"] = 1
    free = simulate.simulate_vectorized(panel, "signal", fee_rate=0.0)
    with_fee = simulate.simulate_vectorized(panel, "signal", fee_rate=0.001)
    assert with_fee["net_rev"] < free["net_rev"]
    assert with_fee["fees"] == pytest.approx(-0.001 * 100 * 10)


def test_exact_matches_a_hand_computed_trade():
    """One position, a known exit price, the fee deducted on both sides."""
    panel = make_panel(n_hours=11, n_symbols=1)
    panel["signal"] = 0
    panel.loc[panel.index[0], "signal"] = 1
    # The price rises by one unit over the holding period.
    panel.loc[panel.index[10:], "close"] = 101.0

    result = simulate.simulate_exact(
        panel, "signal", trade_size=100.0, fee_rate=0.0, horizon_hours=10
    )
    # Bought one unit at 100, sold at 101, so the position is worth 101 and the
    # profit is 1 on a capital of 100.
    assert result["trades"] == 1
    assert result["net_pnl"] == pytest.approx(1.0)
    assert result["win_rate"] == 1.0


def test_exact_charges_the_fee_on_both_sides():
    panel = make_panel(n_hours=11, n_symbols=1)
    panel["signal"] = 0
    panel.loc[panel.index[0], "signal"] = 1
    panel.loc[panel.index[10:], "close"] = 101.0

    result = simulate.simulate_exact(
        panel, "signal", trade_size=100.0, fee_rate=0.001, horizon_hours=10
    )
    assert result["fees"] == pytest.approx(0.001 * 100 + 0.001 * 101, rel=1e-6)
    assert result["net_pnl"] == pytest.approx(1.0 - 0.001 * 100 - 0.001 * 101, rel=1e-6)


def test_exact_respects_one_position_per_symbol():
    """A second signal for the same contract while the first is open is ignored."""
    panel = make_panel(n_hours=12, n_symbols=1)
    panel["signal"] = 0
    panel.loc[panel.index[[0, 1]], "signal"] = 1
    result = simulate.simulate_exact(
        panel, "signal", trade_size=100.0, fee_rate=0.0, horizon_hours=10, max_per_symbol=1
    )
    assert result["trades"] == 1


def test_exact_returns_nothing_when_there_are_no_signals():
    panel = make_panel()
    panel["signal"] = 0
    result = simulate.simulate_exact(panel, "signal")
    assert result["trades"] == 0
    assert result["CAGR"] == 0.0


def test_buy_and_hold_returns_the_total_price_change():
    panel = make_panel(n_hours=48, n_symbols=1)
    panel.loc[panel.index[-1], "close"] = 150.0
    result = simulate.buy_and_hold(panel, "T0USDT")
    assert result["total_return"] == pytest.approx(0.5)
    assert result["CAGR"] > 0


def test_threshold_is_chosen_on_precision_and_respects_the_minimum_coverage():
    prob = pd.Series([0.9] * 50 + [0.55] * 950)
    y = pd.Series([1] * 50 + [0] * 950)
    best = models.choose_threshold(prob, y, min_share=0.02, min_signals=10)
    # Only the 50 high-probability rows are ever correct, so the search has to find a
    # cut-off that excludes the 950 low-probability rows.
    assert best["threshold"] > 0.55
    assert best["precision"] == pytest.approx(1.0)
    assert best["n_signals"] == 50


def test_threshold_falls_back_to_a_low_cut_off_when_nothing_is_precise():
    prob = pd.Series([0.51] * 1000)
    y = pd.Series([1] * 500 + [0] * 500)
    best = models.choose_threshold(prob, y, min_share=0.02, min_signals=10)
    assert 0.5 <= best["threshold"] <= 0.90


def test_strategy_columns_include_the_expected_families():
    panel = make_panel(n_hours=24, n_symbols=3)
    panel["market_growth_24h"] = 0.01
    panel["volatility_168h"] = 0.05
    panel["rsi_14"] = 30.0
    panel["dgs10"] = 3.0
    panel["growth_24h"] = -0.01
    panel["funding_rate"] = -0.0001
    predictions = pd.DataFrame(
        {
            "prob_xgboost": np.linspace(0.3, 0.9, len(panel)),
            "pred_xgboost": np.ones(len(panel), dtype="int8"),
        },
        index=panel.index,
    )
    strategies = simulate.build_strategy_columns(panel, predictions, thresholds={"xgboost": 0.5})
    assert "pred_xgboost" in strategies.columns
    assert "pred_manual_macro" in strategies.columns
    assert "pred_manual_funding" in strategies.columns
    assert strategies["pred_manual_funding"].sum() == len(panel)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
