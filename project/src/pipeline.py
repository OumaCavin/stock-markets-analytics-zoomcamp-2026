"""End-to-end pipeline: raw archives to a ranked table of strategies.

The pipeline is the single entry point used by the notebook, by the command line and by
the cron job, so that all three run exactly the same code path.

    python -m src.pipeline --regime cache      # rebuild from what is already stored
    python -m src.pipeline --regime download   # refresh the archives first
    python -m src.pipeline --regime incremental # extend the panel to the latest bar

Two regimes are supported on purpose. The *download* regime refreshes everything from
the exchange and therefore produces the latest predictions, while the *cache* regime
rebuilds from the stored parquet files, which is fast, reproducible and works even when
the network is unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import build_dataset, config as C, models, simulate  # noqa: E402


# --------------------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------------------


def select_universe(
    min_median_daily_quote_volume: float = C.MIN_MEDIAN_DAILY_QUOTE_VOLUME,
    min_hours: int = 24 * 60,
) -> pd.DataFrame:
    """Pick the tradable contracts and report why the others were dropped.

    The liquidity filter is computed on the *training* part of the panel only. Using the
    whole panel to decide which contracts are tradable would let the future decide the
    universe, which is the same look-ahead error the temporal split exists to prevent.
    """
    liquidity = build_dataset.symbol_liquidity()
    if liquidity.empty:
        raise RuntimeError("no raw data found; run the download stage first")

    train_end = pd.Timestamp(C.TRAIN_END, tz="UTC")
    liquidity["median_daily_quote_volume"] = liquidity["median_daily_quote_volume"].astype(float)

    # Re-derive the training-window liquidity from the stored bars.
    keep_rows = []
    for path in sorted(C.KLINE_DIR.glob("*.parquet")):
        sym = path.stem
        df = pd.read_parquet(path, columns=["open_time", "quote_volume"])
        if df.empty:
            continue
        ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        mask = ts <= train_end
        if mask.sum() < 24 * 20:
            continue
        daily = df.loc[mask].groupby(ts[mask].dt.floor("D"))["quote_volume"].sum()
        keep_rows.append(
            {
                "ticker": sym,
                "train_median_daily_quote_volume": float(daily.median()),
                "n_hours_total": int(len(df)),
                "first_timestamp": ts.min(),
                "last_timestamp": ts.max(),
            }
        )

    stats = pd.DataFrame(keep_rows)
    stats["liquid_enough"] = stats["train_median_daily_quote_volume"] >= min_median_daily_quote_volume
    stats["long_enough"] = stats["n_hours_total"] >= min_hours
    stats["selected"] = stats["liquid_enough"] & stats["long_enough"]
    stats = stats.sort_values("train_median_daily_quote_volume", ascending=False)
    stats["selected"] = stats["selected"] & (stats["selected"].cumsum() <= C.MAX_UNIVERSE_SIZE)
    stats.to_csv(C.OUTPUT_DIR / "universe_selection.csv", index=False)
    return stats


# --------------------------------------------------------------------------------------
# Report helpers
# --------------------------------------------------------------------------------------


def correlation_report(panel: pd.DataFrame, numeric: list[str], top: int = 25) -> pd.DataFrame:
    """Correlation of every numeric feature with the target, plus the dummy families."""
    cols = [c for c in numeric if c in panel.columns]
    corr = panel[cols].corrwith(panel[C.TARGET].astype("float64")).dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(top)
    return corr.to_frame("corr_with_target").assign(abs_corr=lambda d: d["corr_with_target"].abs())


def dummy_correlation_report(panel: pd.DataFrame, dummies: list[str], top: int = 20) -> pd.DataFrame:
    """Correlation of the one-hot columns with the target."""
    from src.models import make_dummies

    encoded = make_dummies(panel, dummies)
    if encoded.empty:
        return pd.DataFrame(columns=["corr_with_target"])
    corr = encoded.corrwith(panel[C.TARGET].astype("float64")).dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(top)
    return corr.to_frame("corr_with_target").assign(abs_corr=lambda d: d["corr_with_target"].abs())


def plot_equity_curves(equity_curves: dict, path: Path) -> None:
    """Save the equity curves of the strategies and the benchmarks on one chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, curve in equity_curves.items():
        if curve is None or len(curve) == 0:
            continue
        values = np.asarray(curve, dtype="float64")
        index = None
        if isinstance(curve, pd.Series) and isinstance(curve.index, pd.DatetimeIndex):
            index = curve.index
        series = pd.Series(values, index=index)
        series = series / series.iloc[0]
        # The curves are drawn against the calendar, not against the number of trades,
        # otherwise two strategies with different trade counts are not comparable.
        ax.plot(series.index if index is not None else range(len(series)),
                series.to_numpy(), label=name, linewidth=1.1)
    ax.set_title("Equity curves of the simulated strategies")
    ax.set_ylabel("growth of capital")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def precision_by_bucket(panel: pd.DataFrame, predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    """How the precision of a prediction varies across tickers, months and hours."""
    merged = panel[["timestamp", "ticker", C.TARGET, "future_return"]].copy()
    merged["prediction"] = predictions[column].to_numpy()
    merged["correct"] = predictions[column].to_numpy() == panel[C.TARGET].to_numpy()
    merged["hit"] = (predictions[column].to_numpy() == 1) & (panel[C.TARGET].to_numpy() == 1)

    rows = []
    for key, group in merged.groupby("ticker"):
        flagged = group[group["prediction"] == 1]
        if len(flagged) < 50:
            continue
        rows.append(
            {
                "ticker": key,
                "n_predictions": int(len(flagged)),
                "precision": float(flagged["hit"].mean()),
                "mean_forward_return": float(flagged["future_return"].mean()),
            }
        )
    columns = ["ticker", "n_predictions", "precision", "mean_forward_return"]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values("precision", ascending=False)


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------


def _rss_mb() -> float:
    """The resident memory of this process, in megabytes."""
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS"):
                return round(int(line.split()[1]) / 1024)
    return 0.0


def _cache_dataset(dataset, panel) -> None:
    """Write the design matrix and the split to the cache."""
    # The target is added to the frame in place and removed again, so the cache is
    # written without a second copy of the design matrix.
    dataset.X["__target__"] = dataset.y.to_numpy()
    dataset.X.to_parquet(C.CACHE_DIR / "dataset_X.parquet")
    dataset.X.drop(columns=["__target__"], inplace=True)
    meta = pd.DataFrame(
        {
            "timestamp": dataset.timestamp.to_numpy(),
            "ticker": dataset.ticker.to_numpy(),
            "close": panel["close"].to_numpy(),
            "future_return": dataset.target.to_numpy(),
            "train": dataset.train.to_numpy(),
            "valid": dataset.valid.to_numpy(),
            "test": dataset.test.to_numpy(),
        }
    )
    meta.to_parquet(C.CACHE_DIR / "dataset_meta.parquet")


def _load_cached_dataset():
    """Read the cached design matrix back and rebuild the dataset object."""
    X = pd.read_parquet(C.CACHE_DIR / "dataset_X.parquet")
    y = X.pop("__target__").astype("int8")
    meta = pd.read_parquet(C.CACHE_DIR / "dataset_meta.parquet")
    return models.Dataset(
        X=X,
        columns=list(X.columns),
        y=y,
        target=meta["future_return"].astype("float32"),
        timestamp=meta["timestamp"],
        ticker=meta["ticker"],
        train=meta["train"],
        valid=meta["valid"],
        test=meta["test"],
        meta={"n_numeric": 0, "n_dummies": 0},
    )


def run(
    regime: str = "cache",
    families: list[str] | None = None,
    tune: bool = True,
    top_k: int = 3,
    equity_fraction: float | None = 0.01,
    max_concurrent: int | None = 100,
    stages: str = "all",
    reuse_panel: bool = False,
) -> dict:
    """Run the project and write every artefact to ``output/``.

    The pipeline is split into four stages, and each stage is meant to run in its own
    process. The panel, the design matrix and the model zoo do not fit in the memory
    of the machine at the same time, so the stages are separated by the cache:

    ``panel``      build the feature panel and the exploratory analysis
    ``dataset``    turn the panel into the design matrix and cache it
    ``train``      fit the model zoo on the cached design matrix
    ``strategies`` build the strategies and simulate them on the cached panel
    """
    started = time.time()
    wanted = [s.strip() for s in stages.split(",")] if isinstance(stages, str) else list(stages)
    if "all" in wanted:
        wanted = ["panel", "dataset", "train", "strategies"]
    log: dict = {
        "regime": regime,
        "stages": wanted,
        "started": pd.Timestamp.now("UTC").isoformat(),
    }

    if regime in ("download", "incremental"):
        from src import download_data

        if regime == "download":
            log["download"] = download_data.run_full()
        else:
            log["incremental"] = download_data.run_incremental()

    # ---- universe -------------------------------------------------------------------
    print("[1/6] selecting the tradable universe", flush=True)
    stats = select_universe()
    selected = sorted(stats.loc[stats["selected"], "ticker"])
    log["universe"] = {
        "considered": int(len(stats)),
        "selected": len(selected),
        "dropped_illiquid": int((~stats["liquid_enough"]).sum()),
        "dropped_short_history": int((~stats["long_enough"]).sum()),
    }
    print(
        f"      {len(selected)} of {len(stats)} contracts selected "
        f"({log['universe']['dropped_illiquid']} illiquid, "
        f"{log['universe']['dropped_short_history']} too young)",
        flush=True,
    )

    cached_panel = C.CACHE_DIR / "panel.parquet"
    panel = None
    numeric: list[str] = []
    dummies: list[str] = []

    # ---- panel and exploratory analysis ---------------------------------------------
    if "panel" in wanted:
        print("[2/6] building the feature panel", flush=True)
        if reuse_panel and cached_panel.exists():
            # The cached panel is a few hundred megabytes against more than a gigabyte
            # of raw archives, so reusing it also keeps the page cache small.
            panel = pd.read_parquet(cached_panel)
            print("      reusing the cached panel", flush=True)
        else:
            panel = build_dataset.build_panel(selected)
            panel = build_dataset.add_month_wom(panel)
        numeric, dummies = build_dataset.feature_columns(panel)
        panel.to_parquet(cached_panel)
        log["panel"] = {
            "rows": int(len(panel)),
            "n_numeric": len(numeric),
            "n_dummies_families": len(dummies),
            "tickers": int(panel["ticker"].nunique()),
            "from": str(panel["timestamp"].min()),
            "to": str(panel["timestamp"].max()),
        }
        print(
            f"      {len(panel):,} rows x {len(numeric)} numeric features "
            f"({_rss_mb()} MB in use)",
            flush=True,
        )

        print("[3/6] exploratory analysis", flush=True)
        corr_num = correlation_report(panel, numeric)
        corr_dum = dummy_correlation_report(panel, dummies)
        corr_num.to_csv(C.OUTPUT_DIR / "eda_numeric_correlations.csv")
        corr_dum.to_csv(C.OUTPUT_DIR / "eda_dummy_correlations.csv")
        log["eda"] = {
            "top_numeric": corr_num.head(10).round(5).to_dict()["corr_with_target"],
            "top_dummy": corr_dum.head(5).round(5).to_dict()["corr_with_target"],
        }

    if "dataset" in wanted:
        if panel is None:
            panel = pd.read_parquet(cached_panel)
            numeric, dummies = build_dataset.feature_columns(panel)
        print("[2/6] assembling the design matrix", flush=True)
        dataset = models.build_dataset(panel, numeric, dummies)
        _cache_dataset(dataset, panel)
        log["dataset"] = {"shape": list(dataset.shape), "sizes": dataset.sizes}
        print(
            f"      {dataset.shape[0]:,} rows x {dataset.shape[1]} columns, "
            f"cached ({_rss_mb()} MB in use)",
            flush=True,
        )
        del dataset

    # The panel is not needed by the training stage, and holding it while the models
    # are fitted is what pushes the process over the memory limit.
    del panel
    build_dataset._trim_memory()

    # ---- models ---------------------------------------------------------------------
    if "train" in wanted:
        print("[4/6] training and tuning the models", flush=True)
        dataset = _load_cached_dataset()
        results = models.train_all(dataset, families=families, tune=tune)
        # The score file of a family that was trained in an earlier process is kept, so
        # the zoo is described by one file however many processes it took.
        scores_path = C.OUTPUT_DIR / "model_scores.json"
        scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
        scores.update(models.result_payload(results))
        scores_path.write_text(json.dumps(scores, indent=2, default=str))
        # The per-family prediction files are merged into one frame. A family that was
        # trained in an earlier process is kept, so the model zoo can be trained in
        # batches without losing anything.
        frames = []
        for path in sorted(C.CACHE_DIR.glob("predictions_*.parquet")):
            if path.name == "predictions.parquet":
                continue
            frames.append(pd.read_parquet(path))
        if frames:
            merged = pd.concat(frames, axis=1)
            merged = merged.loc[:, ~merged.columns.duplicated()]
            merged.to_parquet(C.CACHE_DIR / "predictions.parquet")
        del dataset, results
        build_dataset._trim_memory()

    # ---- strategies and simulation --------------------------------------------------
    if "strategies" in wanted:
        print("[5/6] selecting thresholds and simulating the strategies", flush=True)
        panel = pd.read_parquet(cached_panel)
        predictions = pd.read_parquet(C.CACHE_DIR / "predictions.parquet")
        log["model_families"] = sorted(
            c.replace("prob_", "") for c in predictions.columns if c.startswith("prob_")
        )
        # The panel description is rebuilt here, so that the manifest of a run that
        # only rebuilds the strategies still describes the panel it ran on.
        numeric, dummies = build_dataset.feature_columns(panel)
        log["panel"] = {
            "rows": int(len(panel)),
            "n_numeric": len(numeric),
            "n_dummies_families": len(dummies),
            "tickers": int(panel["ticker"].nunique()),
            "from": str(panel["timestamp"].min()),
            "to": str(panel["timestamp"].max()),
        }
        # The threshold is chosen on the validation window: the training window is in
        # sample for the models and the test window must stay untouched.
        stamps = pd.to_datetime(panel["open_time"], utc=True)
        train_mask = (stamps <= pd.Timestamp(C.TRAIN_END, tz="UTC")).to_numpy()
        valid_mask = (
            (stamps > pd.Timestamp(C.TRAIN_END, tz="UTC"))
            & (stamps <= pd.Timestamp(C.VALID_END, tz="UTC"))
        ).to_numpy()
        test_mask = (stamps > pd.Timestamp(C.VALID_END, tz="UTC")).to_numpy()
        valid_target = panel[C.TARGET][valid_mask]
        log["dataset"] = {
            "sizes": {
                "train": int(train_mask.sum()),
                "valid": int(valid_mask.sum()),
                "test": int(test_mask.sum()),
            }
        }

        # The probability cut-off is a parameter of the strategy, not of the model. It is
        # chosen on the validation slice and then applied unchanged to the test slice.
        thresholds = {}
        for column in predictions.columns:
            if not column.startswith("prob_"):
                continue
            thresholds[column.replace("prob_", "")] = models.choose_threshold(
                predictions.loc[valid_mask, column], valid_target
            )["threshold"]
        tree_cols = [
            f"prob_{n}"
            for n in ("decision_tree", "random_forest", "xgboost", "lightgbm")
            if f"prob_{n}" in predictions.columns
        ]
        if tree_cols:
            thresholds["ensemble"] = models.choose_threshold(
                predictions.loc[valid_mask, tree_cols].mean(axis=1), valid_target
            )["threshold"]
        log["thresholds"] = thresholds

        strategies = simulate.build_strategy_columns(
            panel, predictions, thresholds=thresholds, top_k=top_k
        )

        vectorised = simulate.strategy_summary(panel, strategies)
        vectorised.to_csv(C.OUTPUT_DIR / "strategy_vectorised.csv", index=False)
        exact = simulate.exact_summaries(panel, strategies, equity_fraction=None)
        exact.to_csv(C.OUTPUT_DIR / "strategy_exact.csv", index=False)

        # The thresholds were chosen on the validation slice, so the test slice is the
        # honest measure of what the strategy does out of sample.
        exact_test = simulate.exact_summaries(
            panel[test_mask], strategies[test_mask], equity_fraction=None
        )
        exact_test.to_csv(C.OUTPUT_DIR / "strategy_exact_test.csv", index=False)

        # Compounding book: every position is a fraction of the running equity, so the
        # gains are reinvested. The concurrency limit keeps the book from levering up.
        equity_book = simulate.exact_summaries(
            panel, strategies, equity_fraction=equity_fraction, max_concurrent=max_concurrent
        )
        equity_book.to_csv(C.OUTPUT_DIR / "strategy_equity.csv", index=False)
        equity_test = simulate.exact_summaries(
            panel[test_mask],
            strategies[test_mask],
            equity_fraction=equity_fraction,
            max_concurrent=max_concurrent,
        )
        equity_test.to_csv(C.OUTPUT_DIR / "strategy_equity_test.csv", index=False)

        bench_btc = simulate.buy_and_hold(panel, "BTCUSDT")
        bench_basket = simulate.equal_weight_basket(panel)
        benchmarks = pd.DataFrame([bench_btc, bench_basket]).drop(
            columns=["equity_curve"], errors="ignore"
        )
        benchmarks.to_csv(C.OUTPUT_DIR / "benchmarks.csv", index=False)

        curves = {}
        for col in list(strategies.columns)[:6]:
            merged = panel[["timestamp", "ticker", "close", "future_return"]].copy()
            merged["signal"] = strategies[col].to_numpy()
            res = simulate.simulate_exact(merged, "signal", equity_fraction=equity_fraction)
            curves[col] = pd.Series(
                res["equity_curve"], index=pd.DatetimeIndex(res["equity_curve_index"])
            )
        if bench_btc:
            btc_rows = panel[panel["ticker"] == "BTCUSDT"].sort_values("timestamp")
            curves["buy_and_hold_BTC"] = pd.Series(
                bench_btc["equity_curve"],
                index=pd.DatetimeIndex(btc_rows["timestamp"].to_numpy()[: len(bench_btc["equity_curve"])]),
            )
        plot_equity_curves(curves, C.OUTPUT_DIR / "equity_curves.png")

        # ---- diagnostics ---------------------------------------------------------------
        # The diagnostic is done on the strategy with the most signals, because the
        # best performing strategy is often a small sample and a precision table for
        # a hundred trades would say nothing about anything.
        if len(exact):
            busiest = exact.sort_values("trades", ascending=False).iloc[0]["prediction"]
            # The strategy name carries the model and the number of positions it takes,
            # and the precision table needs the model's own prediction column.
            model_column = busiest
            for suffix in ("_top1", "_top3", "_top5"):
                if model_column.endswith(suffix):
                    model_column = model_column[: -len(suffix)]
            if model_column in predictions.columns:
                precision_by_bucket(panel, predictions, model_column).to_csv(
                    C.OUTPUT_DIR / "precision_by_ticker.csv", index=False
                )
                log["diagnostics"] = {"precision_table": busiest, "model": model_column}

        # ---- latest signals -----------------------------------------------------------
        latest = panel[panel["timestamp"] >= panel["timestamp"].max() - pd.Timedelta(hours=24)]
        latest_signals = strategies.loc[latest.index]
        recent = latest.assign(
            **{col: latest_signals[col].to_numpy() for col in strategies.columns}
        )
        recent.to_csv(C.OUTPUT_DIR / "latest_signals.csv", index=False)

        log["strategies"] = {
            "n_strategies": int(len(strategies.columns)),
            "best_vectorised": vectorised.iloc[0].to_dict() if len(vectorised) else {},
            "best_exact": exact.iloc[0].to_dict() if len(exact) else {},
            "best_equity": equity_book.iloc[0].to_dict() if len(equity_book) else {},
            "benchmarks": benchmarks.to_dict(orient="records"),
        }

    log["finished"] = pd.Timestamp.now("UTC").isoformat()
    log["seconds"] = round(time.time() - started, 1)
    # The manifest is merged rather than overwritten, so that a run of one stage keeps
    # the description of the stages that were not part of it.
    manifest_path = C.OUTPUT_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(log)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[6/6] done in {log['seconds']}s -> {C.display_path(C.OUTPUT_DIR)}", flush=True)
    return log

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the whole trading project end to end")
    ap.add_argument("--regime", default="cache", choices=["cache", "download", "incremental"])
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--reuse-panel", action="store_true")
    ap.add_argument(
        "--stages",
        default="all",
        help="comma separated list: panel,dataset,train,strategies",
    )
    args = ap.parse_args()

    summary = run(
        regime=args.regime,
        families=args.families,
        tune=not args.no_tune,
        top_k=args.top_k,
        stages=args.stages,
        reuse_panel=args.reuse_panel,
    )
    if "strategies" in summary:
        print(json.dumps(summary["strategies"]["best_exact"], indent=2, default=str))
    else:
        brief = {k: v for k, v in summary.items() if k not in ("eda", "thresholds")}
        print(json.dumps(brief, indent=2, default=str)[:2000])
