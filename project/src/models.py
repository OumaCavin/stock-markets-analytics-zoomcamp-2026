"""Train, tune and compare the models that drive the trading strategies.

Five model families are used so that the project can answer the question the lecture
cannot: does a stronger learner actually turn into a better strategy, or does it only
turn into a better classification score?

* the two lecture models, DecisionTree and RandomForest, retuned on the crypto panel;
* two gradient-boosted tree libraries that were not part of the lecture, XGBoost and
  LightGBM;
* two different model shapes, a regularised logistic regression and a multi-layer
  perceptron.

Everything is fitted on a temporal split, never on a random split, because the panel is
a time series and a random split leaks the future into the training set.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402

def _trim_memory() -> None:
    """Return the freed temporaries to the operating system."""
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# Rows used for the hyper-parameter search, before the final refit on everything.
TUNING_SAMPLE_ROWS = 120_000

# Families whose cost per row is high are fitted on a capped number of rows. The
# gradient boosters and the forests use everything; the neural network converges long
# before the last row and takes minutes rather than seconds with it.
# The models that predict the forward return instead of the label.
REGRESSION_MODELS = {"ridge_regression"}

MAX_TRAIN_ROWS = {
    "mlp": 80_000,
    "logistic_regression": 80_000,
    "ridge_regression": 80_000,
    "xgboost": 150_000,
    "lightgbm": 150_000,
}


# --------------------------------------------------------------------------------------
# Feature matrix
# --------------------------------------------------------------------------------------


@dataclass
class Dataset:
    """The modelling view of the panel: design matrix, target and the split masks.

    The design matrix is kept as a frame with the numeric columns in float32 and the
    one-hot columns in int8. Scikit-learn and XGBoost both convert their input to a
    float matrix before they work on it, so the conversion is done here, once per
    family and only for the rows that family needs, rather than for the whole panel.
    """

    X: pd.DataFrame
    columns: list[str]
    y: pd.Series
    target: pd.Series
    timestamp: pd.Series
    ticker: pd.Series
    train: pd.Series
    valid: pd.Series
    test: pd.Series
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape

    def take(self, mask) -> np.ndarray:
        """Return the selected rows as a float32 matrix."""
        if isinstance(mask, pd.Series):
            mask = mask.to_numpy()
        return self.X[mask].to_numpy(dtype="float32")

    def row_blocks(self, mask, chunk_size: int = 50_000):
        """Yield the selected rows in blocks, without materialising the whole slice.

        The pipeline models copy their input twice while they transform it, so a
        validation slice of half a million rows has to be pushed through in pieces.
        """
        if isinstance(mask, pd.Series):
            mask = mask.to_numpy()
        index = np.flatnonzero(mask)
        for start in range(0, len(index), chunk_size):
            block = index[start : start + chunk_size]
            yield self.X.iloc[block].to_numpy(dtype="float32")

    @property
    def sizes(self) -> dict:
        return {
            "train": int(self.train.sum()),
            "valid": int(self.valid.sum()),
            "test": int(self.test.sum()),
        }


def make_dummies(panel: pd.DataFrame, dummies: list[str]) -> pd.DataFrame:
    """One-hot encode the categorical families.

    ``pd.get_dummies`` sorts the columns alphabetically, which fixes the feature
    ordering. That ordering decides the tie-breaking inside the decision trees, so it is
    pinned here once and reused everywhere rather than being left to chance.
    """
    parts = []
    for col in dummies:
        if col not in panel.columns:
            continue
        if panel[col].dtype == object or panel[col].dtype.name == "category":
            parts.append(pd.get_dummies(panel[col].astype(str), prefix=col, dtype="int8"))
        else:
            parts.append(pd.get_dummies(panel[col], prefix=col, dtype="int8"))
    if not parts:
        return pd.DataFrame(index=panel.index)
    return pd.concat(parts, axis=1)


def build_dataset(
    panel: pd.DataFrame,
    numeric: list[str],
    dummies: list[str],
    train_end: str = C.TRAIN_END,
    valid_end: str = C.VALID_END,
) -> Dataset:
    """Assemble the design matrix and the temporal split."""
    numeric = [c for c in numeric if c in panel.columns and panel[c].notna().any()]
    X_num = panel[numeric].astype("float32")
    X_dum = make_dummies(panel, dummies)
    X = pd.concat([X_num, X_dum], axis=1)
    columns = list(X.columns)

    timestamp = pd.to_datetime(panel["open_time"], utc=True)
    train = timestamp <= pd.Timestamp(train_end, tz="UTC")
    valid = (timestamp > pd.Timestamp(train_end, tz="UTC")) & (
        timestamp <= pd.Timestamp(valid_end, tz="UTC")
    )
    test = timestamp > pd.Timestamp(valid_end, tz="UTC")

    # Missing values are left in place. The tree ensembles and the gradient boosters
    # learn a default direction for a missing value, which is better than any fixed
    # imputation, and the linear models and the network carry an imputer inside their
    # own pipeline so that the statistics come from the training rows only.
    X = X.replace([np.inf, -np.inf], np.nan)
    # The numeric columns stay float32 and the one-hot columns stay int8, which keeps
    # the frame at half the size of a float32 matrix of the same shape.
    for column in numeric:
        X[column] = X[column].astype("float32")
    for column in X.columns:
        if column not in numeric:
            X[column] = X[column].astype("int8")

    y = panel[C.TARGET].astype("int8")
    return Dataset(
        X=X,
        columns=columns,
        y=y,
        target=panel["future_return"].astype("float32"),
        timestamp=timestamp,
        ticker=panel["ticker"],
        train=train,
        valid=valid,
        test=test,
        meta={"n_numeric": len(numeric), "n_dummies": int(X_dum.shape[1])},
    )


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


def model_families() -> dict:
    """The model zoo. Every entry is a factory so that each fit is independent."""
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "decision_tree": lambda: DecisionTreeClassifier(
            random_state=C.RANDOM_STATE, class_weight="balanced"
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=50,
            max_samples=0.5,
            n_jobs=C.N_JOBS,
            random_state=C.RANDOM_STATE,
            class_weight="balanced_subsample",
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.6,
            colsample_bytree=0.6,
            min_child_weight=20,
            reg_lambda=2.0,
            n_jobs=C.N_JOBS,
            random_state=C.RANDOM_STATE,
            eval_metric="logloss",
        ),
        "lightgbm": lambda: LGBMClassifier(
            n_estimators=200,
            num_leaves=63,
            learning_rate=0.06,
            subsample=0.6,
            subsample_freq=1,
            colsample_bytree=0.6,
            min_child_samples=50,
            n_jobs=C.N_JOBS,
            random_state=C.RANDOM_STATE,
            verbose=-1,
        ),
        "logistic_regression": lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=400, n_jobs=C.N_JOBS, random_state=C.RANDOM_STATE
                    ),
                ),
            ]
        ),
        "ridge_regression": lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", Ridge(alpha=10.0)),
            ]
        ),
        "mlp": lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        alpha=1e-3,
                        max_iter=25,
                        early_stopping=True,
                        n_iter_no_change=5,
                        random_state=C.RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def _is_regressor(model) -> bool:
    """True when the model predicts the forward return instead of the label."""
    return not hasattr(model, "predict_proba")


def _scores_from(y, scores) -> dict:
    """Turn a raw model output into precision, recall, accuracy and AUC."""
    scores = np.asarray(scores, dtype="float64")
    continuous = not set(np.unique(scores)).issubset({0.0, 1.0})
    yhat = (scores > 0).astype(int) if continuous else scores.astype(int)
    out = {
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "accuracy": accuracy_score(y, yhat),
    }
    try:
        out["auc"] = roc_auc_score(y, scores)
    except ValueError:
        out["auc"] = np.nan
    return out


def score_model(model, X, y) -> dict:
    """Precision, recall, AUC and accuracy of a fitted model on one slice."""
    if hasattr(model, "predict_proba"):
        return _scores_from(y, model.predict_proba(X)[:, 1])
    return _scores_from(y, model.predict(X))


def score_blocks(model, blocks, y) -> dict:
    """Score a model on a generator of row blocks.

    The predictions of every block are collected and the scores are computed once at
    the end, so the memory cost is one block rather than one split.
    """
    predictions = []
    probabilities = []
    for block in blocks:
        predictions.append(model.predict(block))
        if hasattr(model, "predict_proba"):
            probabilities.append(model.predict_proba(block)[:, 1])
    yhat = np.concatenate(predictions)
    proba = np.concatenate(probabilities) if probabilities else yhat
    return _scores_from(y, proba)


def choose_threshold(
    prob: pd.Series,
    y: pd.Series,
    min_share: float = 0.02,
    min_signals: int = 200,
    coverages: np.ndarray | None = None,
) -> dict:
    """Pick the probability threshold that maximises precision on the given slice.

    The cut-off is chosen as a *quantile* of the validation probabilities rather than
    as an absolute probability. An absolute threshold does not survive a change in the
    distribution of the model's output: the validation slice may put a fifth of its
    rows above 0.88 while the test slice never goes above 0.62, and a strategy tuned on
    the absolute number then does nothing at all out of sample. The quantile rule keeps
    the rate of signals stable, which is what the strategy actually depends on.

    The absolute threshold that corresponds to the chosen coverage is reported as well,
    so the rule can be applied to any new bar.
    """
    coverages = coverages if coverages is not None else np.array(
        [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]
    )
    minimum = max(min_signals, int(min_share * len(prob)))
    best = {
        "threshold": 0.5,
        "coverage_target": 0.1,
        "precision": float("nan"),
        "n_signals": 0,
        "coverage": 0.0,
    }
    for target in coverages:
        threshold = float(np.quantile(prob, 1.0 - target))
        selected = prob >= threshold
        n = int(selected.sum())
        if n < minimum:
            continue
        precision = float(y[selected].mean())
        if np.isnan(best["precision"]) or precision > best["precision"]:
            best = {
                "threshold": threshold,
                "coverage_target": float(target),
                "precision": precision,
                "n_signals": n,
                "coverage": n / max(1, len(prob)),
            }
    return best


# --------------------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------------------


def tune_model(
    name: str,
    dataset: Dataset,
    param_grid: list[dict],
    n_splits: int = 2,
) -> tuple[dict, pd.DataFrame]:
    """Expanding-window tuning on the training slice only.

    The grid is scored with a time-series split inside the training window, so the
    validation block of the panel stays untouched until the final evaluation.
    """
    from sklearn.base import clone

    X_train = dataset.take(dataset.train)
    y_train = dataset.y[dataset.train]
    # A regression model is fitted on the forward return rather than on the label.
    y_fit = dataset.target[dataset.train] if name in REGRESSION_MODELS else y_train

    # The panel has millions of rows. The grid search runs on a deterministic stride of
    # the training rows so that the search stays affordable; the winning configuration
    # is then refitted on the whole training set.
    if len(X_train) > TUNING_SAMPLE_ROWS:
        stride = len(X_train) // TUNING_SAMPLE_ROWS + 1
        X_train = X_train[::stride]
        y_train = y_train.iloc[::stride]
        y_fit = y_fit.iloc[::stride]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    records = []
    best = {"score": -np.inf, "params": None, "model": None}

    for params in param_grid:
        folds = []
        for tr_idx, va_idx in tscv.split(X_train):
            if len(va_idx) < 1000:
                continue
            model = model_families()[name]()
            model.set_params(**params)
            model.fit(X_train[tr_idx], y_fit.iloc[tr_idx])
            if hasattr(model, "predict_proba"):
                score = model.predict_proba(X_train[va_idx])[:, 1]
            else:
                score = model.predict(X_train[va_idx])
            yhat = (score > 0).astype(int)
            folds.append(precision_score(y_train.iloc[va_idx], yhat, zero_division=0))
        if not folds:
            continue
        score = float(np.mean(folds))
        records.append({**params, "cv_precision": score, "folds": len(folds)})
        if score > best["score"]:
            model = model_families()[name]()
            model.set_params(**params)
            best = {"score": score, "params": params, "model": model}

    history = pd.DataFrame(records).sort_values("cv_precision", ascending=False)
    return {"name": name, "params": best["params"], "cv_precision": best["score"]}, history


def grid_for(name: str) -> list[dict]:
    """Hyper-parameter grids, kept small so that the whole search fits in a sandbox."""
    if name == "decision_tree":
        return [{"max_depth": d, "min_samples_leaf": leaf} for d in (6, 10) for leaf in (50, 200)]
    if name == "random_forest":
        return [
            {"max_depth": d, "min_samples_leaf": leaf, "n_estimators": 150}
            for d in (6, 10)
            for leaf in (50, 150)
        ]
    if name == "xgboost":
        return [
            {"max_depth": d, "learning_rate": lr, "n_estimators": 300}
            for d in (4, 6)
            for lr in (0.03, 0.08)
        ]
    if name == "lightgbm":
        return [
            {"num_leaves": leaves, "learning_rate": lr, "n_estimators": 300}
            for leaves in (31, 63)
            for lr in (0.03, 0.08)
        ]
    if name == "logistic_regression":
        return [{"clf__C": c} for c in (0.01, 0.1)]
    if name == "ridge_regression":
        return [{"clf__alpha": a} for a in (1.0, 10.0, 100.0)]
    return [
        {"clf__hidden_layer_sizes": h, "clf__alpha": a}
        for h in [(64, 32)]
        for a in (1e-3,)
    ]


# --------------------------------------------------------------------------------------
# Training entry point
# --------------------------------------------------------------------------------------


def train_all(dataset: Dataset, families: list[str] | None = None, tune: bool = True) -> dict:
    """Tune and fit every model family, then score each on train, validation and test."""
    families = families or list(model_families())
    results = {}
    for name in families:
        started = time.time()
        if tune:
            best, history = tune_model(name, dataset, grid_for(name))
            params = best["params"]
        else:
            params = {}
            history = pd.DataFrame()
        model = model_families()[name]()
        model.set_params(**params)

        X_train, y_train = dataset.take(dataset.train), dataset.y[dataset.train]
        y_fit = dataset.target[dataset.train] if name in REGRESSION_MODELS else y_train
        limit = MAX_TRAIN_ROWS.get(name)
        if limit and len(X_train) > limit:
            stride = len(X_train) // limit + 1
            model.fit(X_train[::stride], y_fit.iloc[::stride])
        else:
            model.fit(X_train, y_fit)

        # The slices are scored one at a time. Materialising the train, validation and
        # test blocks together would need several times the memory of the panel.
        scores = {"train": score_model(model, X_train, y_train)}
        del X_train, y_train
        scores["valid"] = score_blocks(model, dataset.row_blocks(dataset.valid), dataset.y[dataset.valid])
        scores["test"] = score_blocks(model, dataset.row_blocks(dataset.test), dataset.y[dataset.test])
        _trim_memory()
        # The fitted estimator is written to disk, so that the model can be used and
        # downloaded without training it again.
        estimator_path = save_estimator(name, model, params, scores, dataset.columns)
        print(f"                   saved to {estimator_path.name}", flush=True)

        # Every family writes its own predictions file, so the model zoo can be
        # trained in several processes and merged afterwards.
        family_frame = predict_family(model, dataset)
        family_frame = family_frame.rename(columns={"prob_family": f"prob_{name}"})
        family_frame[f"pred_{name}"] = (family_frame[f"prob_{name}"] > 0.5).astype("int8")
        family_frame.to_parquet(C.CACHE_DIR / f"predictions_{name}.parquet")

        # The freed rows of the design matrix are returned to the operating system
        # between families, otherwise the allocator holds the peak of every fit.
        _trim_memory()
        results[name] = {
            "params": params,
            "scores": scores,
            "fit_seconds": round(time.time() - started, 1),
            "history": history,
            "model": model,
        }
        print(
            f"  {name:20s} valid precision {scores['valid']['precision']:.4f} "
            f"auc {scores['valid']['auc']:.4f} | test precision {scores['test']['precision']:.4f} "
            f"auc {scores['test']['auc']:.4f} ({results[name]['fit_seconds']}s)",
            flush=True,
        )
    return results


def predict_family(model, dataset: Dataset, chunk_size: int = 50_000) -> pd.DataFrame:
    """Predict one fitted model over the whole panel, in chunks."""
    has_proba = hasattr(model, "predict_proba")
    scores = np.empty(len(dataset.X), dtype="float32")
    for start in range(0, len(dataset.X), chunk_size):
        block = dataset.X.iloc[start : start + chunk_size].to_numpy(dtype="float32")
        if has_proba:
            scores[start : start + len(block)] = model.predict_proba(block)[:, 1]
        else:
            # A regression model returns the predicted forward return. It is used for
            # its ordering, which is what the rank based strategies need.
            scores[start : start + len(block)] = model.predict(block)
    frame = pd.DataFrame(index=dataset.y.index)
    frame["prob_family"] = scores
    return frame


def predict_frame(results: dict, dataset: Dataset, chunk_size: int = 50_000) -> pd.DataFrame:
    """Probability and hard prediction for every model, for every row of the panel.

    The panel is predicted in chunks. Scikit-learn converts a frame to a float64
    matrix before it predicts, and doing that for the whole panel at once needs more
    memory than the machine has, so the rows are pushed through in blocks instead.
    """
    out = pd.DataFrame(index=dataset.y.index)
    for name, res in results.items():
        model = res["model"]
        probabilities = np.empty(len(dataset.X), dtype="float32")
        has_proba = hasattr(model, "predict_proba")
        for start in range(0, len(dataset.X), chunk_size):
            block = dataset.X.iloc[start : start + chunk_size].to_numpy(dtype="float32")
            if has_proba:
                probabilities[start : start + len(block)] = model.predict_proba(block)[:, 1]
            else:
                # A regression model returns the predicted forward return. It is used
                # for its ordering, which is what the rank based strategies need.
                probabilities[start : start + len(block)] = model.predict(block)
        out[f"prob_{name}"] = probabilities
        out[f"pred_{name}"] = (probabilities > 0.5).astype("int8")
    return out


def result_payload(results: dict) -> dict:
    """The scores and the parameters of the trained families, without the models."""
    payload = {}
    for name, res in results.items():
        payload[name] = {
            "params": res.get("params", {}),
            "scores": res.get("scores", {}),
            "fit_seconds": round(float(res.get("fit_seconds", 0.0)), 1),
        }
    return payload


def save_estimator(name: str, model, params: dict, scores: dict, columns: list[str]) -> Path:
    """Persist a fitted estimator, so it can be reloaded without refitting it.

    The payload carries the estimator itself, the hyper-parameters it was fitted with,
    its scores, the feature columns in the order they were built, and the dates of the
    temporal split. That is everything a reader needs to apply the model to a new bar,
    and it is what makes the model downloadable rather than merely reproducible.
    """
    path = C.MODEL_DIR / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "name": name,
            "model": model,
            "params": params,
            "scores": scores,
            "columns": list(columns),
            "train_end": C.TRAIN_END,
            "valid_end": C.VALID_END,
            "horizon_hours": C.HORIZON_HOURS,
            "target": C.TARGET,
            "trained_at": pd.Timestamp.now("UTC").isoformat(),
        },
        path,
        compress=3,
    )
    return path


def load_estimator(name: str, path: Path | None = None) -> dict:
    """Load a persisted estimator back."""
    path = path or (C.MODEL_DIR / f"{name}.joblib")
    if not path.exists():
        raise FileNotFoundError(f"no saved model at {path}; run the training stage first")
    return joblib.load(path)


def model_files() -> list[Path]:
    """Every persisted estimator, with its size on disk."""
    return sorted(C.MODEL_DIR.glob("*.joblib"))


def save_results(results: dict, path: Path) -> None:
    """Persist the scores and parameters, without the fitted estimators."""
    payload = {
        name: {
            "params": res["params"],
            "scores": res["scores"],
            "fit_seconds": res["fit_seconds"],
        }
        for name, res in results.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train the model zoo on the crypto panel")
    ap.add_argument("--panel", default=str(C.CACHE_DIR / "panel.parquet"))
    ap.add_argument("--out", default=str(C.OUTPUT_DIR / "model_scores.json"))
    ap.add_argument(
        "--list",
        action="store_true",
        help="list the persisted estimators and their scores, then exit",
    )
    args = ap.parse_args()

    if args.list:
        for path in model_files():
            payload = load_estimator(path.stem, path)
            scores = payload.get("scores", {})
            valid = scores.get("valid", {})
            test = scores.get("test", {})
            size_mb = path.stat().st_size / (1024 * 1024)
            print(
                f"{payload['name']:22s} valid auc {valid.get('auc', float('nan')):.4f} "
                f"test auc {test.get('auc', float('nan')):.4f} "
                f"{size_mb:6.1f} MB  {path}"
            )
    else:
        from src.build_dataset import add_month_wom, feature_columns

        panel = pd.read_parquet(args.panel)
        panel = add_month_wom(panel)
        numeric, dummies = feature_columns(panel)
        dataset = build_dataset(panel, numeric, dummies)
        print(f"features: {dataset.meta}, rows {dataset.sizes}")
        results = train_all(dataset)
        save_results(results, Path(args.out))
        print(f"written to {args.out}")
