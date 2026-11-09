"""Generate the end-to-end notebook.

The notebook is written by a script rather than by hand so that it stays in step with
the modules: every cell calls the same functions that the command line and the cron job
call, which is the point the assignment asks for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_DIR / "notebooks" / "project_pipeline.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Crypto perpetuals strategy: end-to-end workflow

This notebook calls the functions of the `src` package one after another and shows the
result of every stage:

1. **Download** the raw archives (Binance bulk archive, FRED, Fear & Greed, CoinGecko).
2. **Transform** them into one modelling frame.
3. **Exploratory analysis** of the target and of the features.
4. **Train** the model zoo and tune it on the validation slice.
5. **Simulate** the strategies and compare them with the benchmarks.
6. **Trade** the latest signals through the broker and show the orders.

The same code is available as scripts, so anything done here can be reproduced with
`make pipeline`. The notebook is the narrative version, the scripts are the repeatable
one.""",
    ),
    (
        "code",
        """import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd

from src import build_dataset, config as C, models, simulate

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

print("project directory:", C.display_path(C.PROJECT_DIR))
print("panel window:", C.START_MONTH, "->", C.END_MONTH)
print("horizon:", C.HORIZON_HOURS, "hours | fee:", C.FEE_RATE, "| trade size:", C.TRADE_SIZE_USD)""",
    ),
    (
        "markdown",
        """## 1. Download

The heavy lift is the Binance bulk archive, which serves the same monthly zip files as
the exchange API but without a key and without a rate limit. The download is
incremental: a month that is already stored is skipped, so the stage can be re-run at
any time.""",
    ),
    (
        "code",
        """from src import download_data

manifest = download_data.run_full()
print(manifest)""",
    ),
    (
        "markdown",
        """## 2. Transform

The universe is the most liquid perpetuals, fixed on the training window so that the
future cannot decide which contracts are tradable. Every feature is computed per
contract on the hourly bars and then thinned to every fourth hour for modelling.""",
    ),
    (
        "code",
        """from src import pipeline

stats = pipeline.select_universe()
selected = sorted(stats.loc[stats["selected"], "ticker"])
print(f"{len(selected)} contracts selected of {len(stats)} considered")
stats.head(10)[["ticker", "train_median_daily_quote_volume", "n_hours_total", "selected"]]""",
    ),
    (
        "code",
        """panel = build_dataset.build_panel(selected)
panel = build_dataset.add_month_wom(panel)
numeric, dummies = build_dataset.feature_columns(panel)

print(f"panel: {panel.shape[0]:,} rows x {panel.shape[1]} columns")
print(f"tickers: {panel['ticker'].nunique()}  from {panel['timestamp'].min()} to {panel['timestamp'].max()}")
print(f"numeric features: {len(numeric)}  dummy families: {dummies}")
print(f"target base rate: {panel[C.TARGET].mean():.4f}")
panel.head()""",
    ),
    (
        "markdown",
        """## 3. Exploratory analysis

The target is whether the close a week later is above the close now. The correlation
table shows which features carry information about it, and the dummy table shows which
calendar positions do.""",
    ),
    (
        "code",
        """numeric_corr = pipeline.correlation_report(panel, numeric, top=20)
numeric_corr""",
    ),
    (
        "code",
        """dummy_corr = pipeline.dummy_correlation_report(panel, dummies, top=15)
dummy_corr""",
    ),
    (
        "code",
        """import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

top = numeric_corr.head(12).sort_values("corr_with_target")
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(top.index, top["corr_with_target"], color=["#b4232c" if v < 0 else "#0f7b45" for v in top["corr_with_target"]])
ax.set_title("Correlation with the 7-day forward return")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()""",
    ),
    (
        "markdown",
        """## 4. Models

Seven model families are compared: the two from the lecture, two gradient-boosted tree
libraries that were not part of the lecture, a regularised logistic regression, a ridge
regression on the forward return and a multi-layer perceptron. The split is temporal,
never random.

The notebook runs the whole pipeline in one process, which needs about three gigabytes
of memory for the panel, the design matrix and the models at the same time. On a
smaller machine the same code is run as the four stages of `src/pipeline.py`, one model
family per process, and the results are identical.""",
    ),
    (
        "code",
        """dataset = models.build_dataset(panel, numeric, dummies)
print("design matrix:", dataset.X.shape)
print("split:", dataset.sizes)

results = models.train_all(dataset, tune=True)""",
    ),
    (
        "code",
        """scores = pd.DataFrame(
    {
        name: {
            "valid_precision": res["scores"]["valid"]["precision"],
            "valid_auc": res["scores"]["valid"]["auc"],
            "test_precision": res["scores"]["test"]["precision"],
            "test_auc": res["scores"]["test"]["auc"],
            "params": str(res["params"]),
        }
        for name, res in results.items()
    }
).T
scores""",
    ),
    (
        "markdown",
        """## 5. Strategies and simulation

The probability cut-off is treated as a parameter of the strategy and chosen on the
validation slice. The exact simulation then walks the account bar by bar, with the fee
charged on both sides and at most one open position per contract.""",
    ),
    (
        "code",
        """predictions = models.predict_frame(results, dataset)

thresholds = {}
valid_probs = predictions[dataset.valid]
valid_target = dataset.y[dataset.valid]
for name in results:
    thresholds[name] = models.choose_threshold(valid_probs[f"prob_{name}"], valid_target)["threshold"]

tree_cols = [f"prob_{n}" for n in ("decision_tree", "random_forest", "xgboost", "lightgbm") if f"prob_{n}" in valid_probs]
if tree_cols:
    thresholds["ensemble"] = models.choose_threshold(valid_probs[tree_cols].mean(axis=1), valid_target)["threshold"]

print("thresholds chosen on the validation slice:")
print(thresholds)""",
    ),
    (
        "code",
        """strategies = simulate.build_strategy_columns(panel, predictions, thresholds=thresholds, top_k=3)

vectorised = simulate.strategy_summary(panel, strategies)
vectorised[["prediction", "count_investments", "capital", "CAGR", "win_rate"]].head(12)""",
    ),
    (
        "code",
        """exact = simulate.exact_summaries(panel, strategies, equity_fraction=0.02)
exact[["prediction", "trades", "win_rate", "CAGR", "Sharpe", "max_drawdown", "required_capital"]].head(12)""",
    ),
    (
        "code",
        """benchmarks = pd.DataFrame(
    [
        {k: v for k, v in simulate.buy_and_hold(panel, "BTCUSDT").items() if k != "equity_curve"},
        {k: v for k, v in simulate.equal_weight_basket(panel).items() if k != "equity_curve"},
    ]
)
benchmarks""",
    ),
    (
        "markdown",
        """## 6. Trade the latest signals

Only the most recent bar is acted on: a prediction made a week ago has already been
traded. The paper broker writes every order to the ledger; the same code with
credentials signs and sends real orders.""",
    ),
    (
        "code",
        """from src.broker import PaperBroker, orders_from_signals

broker = PaperBroker()
submitted = orders_from_signals(panel, strategies, "pred_ensemble", broker, notional=100.0, max_orders=10)
pd.DataFrame(submitted)[["symbol", "side", "quantity", "price", "strategy", "status"]]""",
    ),
    (
        "code",
        """from src.monitor import build_dashboard, daily_summary

model_scores = {name: res for name, res in results.items()}
path = build_dashboard(panel, strategies, exact, model_scores)
print("dashboard:", path)
print()
print(daily_summary(panel, strategies, exact))""",
    ),
    (
        "markdown",
        """## 7. Running the whole thing

```bash
make setup        # virtual environment and pinned dependencies
make download     # fetch every raw source
make pipeline     # features, models, strategies, simulations
make dashboard    # rebuild the HTML report
make orders       # turn the latest signals into orders
```

`run_pipeline.sh` chains the same steps and is what the cron entry points at, so the
system can be re-run unattended every day and produce predictions for the latest
available bar.""",
    ),
]


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    cells = []
    for kind, source in CELLS:
        if kind == "markdown":
            cells.append(nbf.v4.new_markdown_cell(source))
        else:
            cells.append(nbf.v4.new_code_cell(source))
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    return NOTEBOOK_PATH


if __name__ == "__main__":
    path = build()
    print(f"notebook written to {path}")
    sys.exit(0)
