"""
Homework 3 - The Model (2026 cohort)
Solutions for questions 1-4. Produces output/hw3_answers.json.

Follows the Module 03 Colab notebook ([2026]_Module_03_Colab_Time_Series_Modeling.ipynb)
and adds the month-of-week dummy variables required by the homework.

Question 1 : highest |correlation| between a month_wom dummy and the target
Question 2 : precision of the best NEW hand rule (pred3 / pred4) on TEST
Question 3 : TEST records where pred5_clf_10 is the only correct positive predictor
Question 4 : optimal DecisionTree max_depth (1..12) by TEST precision
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier, export_text

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PARQUET = DATA_DIR / "stocks_df_combined_2026_09_18.parquet.brotli"

TO_PREDICT_TARGET = "is_positive_growth_30d_future"

TECHNICAL_INDICATORS = [
    "adx", "adxr", "apo", "aroon_1", "aroon_2", "aroonosc",
    "bop", "cci", "cmo", "dx", "macd", "macdsignal", "macdhist", "macd_ext",
    "macdsignal_ext", "macdhist_ext", "macd_fix", "macdsignal_fix",
    "macdhist_fix", "mfi", "minus_di", "mom", "plus_di", "dm", "ppo",
    "roc", "rocp", "rocr", "rocr100", "rsi", "slowk", "slowd", "fastk",
    "fastd", "fastk_rsi", "fastd_rsi", "trix", "ultosc", "willr",
    "ad", "adosc", "obv", "atr", "natr", "ht_dcperiod", "ht_dcphase",
    "ht_phasor_inphase", "ht_phasor_quadrature", "ht_sine_sine", "ht_sine_leadsine",
    "ht_trendmod", "avgprice", "medprice", "typprice", "wclprice",
]
MACRO = ["gdppot_us_yoy", "gdppot_us_qoq", "cpi_core_yoy", "cpi_core_mom", "FEDFUNDS",
         "DGS1", "DGS5", "DGS10"]

answers = {}


def temporal_split(df, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    """Temporal 70/15/15 split on the Date column (Module 03 notebook)."""
    train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
    val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)
    df["split"] = np.where(
        df["Date"] <= train_end, "train",
        np.where(df["Date"] <= val_end, "validation", "test"),
    )
    return df


def build_dataframe():
    df_full = pd.read_parquet(PARQUET)

    GROWTH = [g for g in df_full.columns if g.startswith("growth_") and "future" not in g]
    OHLCV = ["Open", "High", "Low", "Close", "Adj Close_x", "Volume"]
    CATEGORICAL = ["Month", "Weekday", "Ticker", "ticker_type"]
    TO_PREDICT = [g for g in df_full.columns if "future" in g]
    TO_DROP = ["Year", "Date", "index_x", "index_y", "index", "Quarter", "Adj Close_y"] \
        + CATEGORICAL + OHLCV

    # ln(Volume) - a small hack to avoid log(0)
    df_full["ln_volume"] = df_full.Volume.replace(0, np.nan).fillna(1e-9).apply(np.log)

    CUSTOM_NUMERICAL = ["SMA10", "SMA20", "growing_moving_average",
                        "high_minus_low_relative", "volatility", "ln_volume"]
    TECHNICAL_PATTERNS = [g for g in df_full.columns if "cdl" in g]
    NUMERICAL = (GROWTH + TECHNICAL_INDICATORS + TECHNICAL_PATTERNS
                 + CUSTOM_NUMERICAL + MACRO)

    # 25 years of history with the growth variables defined
    df = df_full[df_full.Date >= "2000-01-01"].copy()

    # Module 03 encodings. `Month` keeps the full month name so that the
    # month-of-week dummies read 'October_w1' exactly as the homework specifies
    # (this is also how the Module 04 notebook builds `month_wom`).
    df["Weekday"] = df["Weekday"].astype(str)
    df["Month"] = df["Month"].dt.strftime("%B")

    # Homework 3, question 1: month + week-of-month dummies
    df["wom"] = df.Date.apply(lambda d: (d.day - 1) // 7 + 1)
    df["month_wom"] = df["Month"] + "_w" + df["wom"].astype(str)
    del df["wom"]

    CATEGORICAL = CATEGORICAL + ["month_wom"]

    dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype="int32")
    DUMMIES = list(dummy_variables.keys())
    df_with_dummies = pd.concat([df, dummy_variables], axis=1)

    df_with_dummies = temporal_split(
        df_with_dummies, df_with_dummies.Date.min(), df_with_dummies.Date.max())

    return df_with_dummies, NUMERICAL, DUMMIES, TO_PREDICT


def question_1(df_with_dummies, NUMERICAL, DUMMIES, TO_PREDICT):
    dummies_month_wom = [k for k in DUMMIES if k.startswith("month_wom")]
    corr = df_with_dummies[dummies_month_wom + TO_PREDICT].corr()[TO_PREDICT_TARGET]
    corr_df = pd.DataFrame(corr).rename(columns={TO_PREDICT_TARGET: "corr"})
    # Keep only the month_wom dummies (the target and growth_future_30d are not dummies).
    corr_df = corr_df[corr_df.index.isin(dummies_month_wom)]
    corr_df["abs_corr"] = corr_df["corr"].abs()
    corr_df = corr_df.sort_values(by="abs_corr", ascending=False)

    answers["q1"] = {
        "question": (
            "What is the ABSOLUTE CORRELATION VALUE of the most correlated dummy "
            "variable _w<week_of_month> with the binary outcome "
            "is_positive_growth_30d_future?"
        ),
        "n_dummies_total": len(DUMMIES),
        "n_month_wom_dummies": len(dummies_month_wom),
        "n_features_total": len(NUMERICAL) + len(DUMMIES),
        "top10": corr_df.head(10).reset_index().rename(
            columns={"index": "dummy"}).to_dict("records"),
        "max_abs_corr": float(corr_df["abs_corr"].max()),
        "max_abs_corr_rounded": round(float(corr_df["abs_corr"].max()), 3),
        "top_dummy": corr_df.index[0],
    }
    print("Q1 total dummies:", len(DUMMIES), "| month_wom dummies:", len(dummies_month_wom))
    print("Q1 features (NUMERICAL+DUMMIES):", len(NUMERICAL) + len(DUMMIES))
    print(corr_df.head(8).to_string())
    print("Q1 max |corr| =", round(float(corr_df["abs_corr"].max()), 5),
          "-> rounded", round(float(corr_df["abs_corr"].max()), 3))


def add_manual_predictions(new_df):
    new_df["pred0_manual_cci"] = (new_df.cci > 200).astype(int)
    new_df["pred1_manual_prev_g1"] = (new_df.growth_30d > 1).astype(int)
    new_df["pred2_manual_prev_g1_and_snp"] = (
        (new_df["growth_30d"] > 1) & (new_df["growth_snp500_30d"] > 1)).astype(int)
    # New hand rules from the homework
    new_df["pred3_manual_dgs10_5"] = (
        (new_df["DGS10"] <= 4.5) & (new_df["DGS5"] <= 4)).astype(int)
    new_df["pred4_manual_dgs10_fedfunds"] = (
        (new_df["DGS10"] > 4) & (new_df["FEDFUNDS"] <= 4.795)).astype(int)
    return new_df


def question_2(new_df):
    test = new_df[new_df.split == "test"]
    results = {}
    for pred in ["pred3_manual_dgs10_5", "pred4_manual_dgs10_fedfunds"]:
        flagged = test[test[pred] == 1]
        if len(flagged) == 0:
            results[pred] = {"n_flagged_test": 0, "precision": None}
            continue
        tp = int((flagged[TO_PREDICT_TARGET] == 1).sum())
        results[pred] = {
            "n_flagged_test": int(len(flagged)),
            "true_positives": tp,
            "precision": tp / len(flagged),
            "precision_rounded": round(tp / len(flagged), 3),
        }
    best = max((k for k, v in results.items() if v["precision"] is not None),
               key=lambda k: results[k]["precision"])
    answers["q2"] = {
        "question": (
            "What is the precision score for the best of the NEW predictions "
            "(pred3 or pred4), rounded to 3 digits after the comma?"
        ),
        "per_rule": results,
        "best_rule": best,
        "best_precision": results[best]["precision"],
        "best_precision_rounded": results[best]["precision_rounded"],
    }
    for k, v in results.items():
        print("Q2", k, "-> flagged:", v["n_flagged_test"],
              "precision:", None if v["precision"] is None else round(v["precision"], 6))
    print("Q2 best:", best, "=", results[best]["precision_rounded"])


def question_3_and_4(df_with_dummies, NUMERICAL, DUMMIES, TO_PREDICT):
    new_df = df_with_dummies.copy()
    new_df = add_manual_predictions(new_df)

    features_list = NUMERICAL + DUMMIES
    to_predict = TO_PREDICT_TARGET

    train_valid = new_df[new_df.split.isin(["train", "validation"])].copy(deep=True)
    X_train_valid = train_valid[features_list + [to_predict]].copy(deep=True)
    X_all = new_df[features_list + [to_predict]].copy(deep=True)

    for frame in (X_train_valid, X_all):
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        frame.fillna(0, inplace=True)

    y_train_valid = X_train_valid[to_predict]
    y_all = X_all[to_predict]
    del X_train_valid[to_predict]
    del X_all[to_predict]

    # ---- Question 4: hyper-parameter tuning over depth 1..12 ---------------
    test_df = new_df[new_df.split == "test"]
    X_test = test_df[features_list + [to_predict]].copy(deep=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.fillna(0, inplace=True)
    y_test = X_test[to_predict]
    del X_test[to_predict]

    precision_by_depth = {}
    for depth in range(1, 13):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train_valid, y_train_valid)
        p = precision_score(y_test, clf.predict(X_test), zero_division=0)
        precision_by_depth[depth] = round(float(p), 4)
    best_depth = max(precision_by_depth, key=precision_by_depth.get)

    # ---- Question 3: pred5_clf_10 (depth 10, random_state 42) -------------
    clf_10 = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf_10.fit(X_train_valid, y_train_valid)
    y_pred_all = clf_10.predict(X_all)
    new_df["pred5_clf_10"] = y_pred_all

    hand = ["pred0_manual_cci", "pred1_manual_prev_g1", "pred2_manual_prev_g1_and_snp",
            "pred3_manual_dgs10_5", "pred4_manual_dgs10_fedfunds"]
    condition = (
        (new_df[TO_PREDICT_TARGET] == 1)
        & (new_df["pred5_clf_10"] == 1)
    )
    for h in hand:
        condition &= (new_df[h] == 0)
    new_df["only_pred5_is_correct"] = condition.astype(int)

    n_test = int((new_df.split == "test").sum())
    only_pred5_test = int(new_df[(new_df.split == "test")]["only_pred5_is_correct"].sum())
    only_pred5_all = int(new_df["only_pred5_is_correct"].sum())

    # A generic (non hard-coded) version of the same test
    PREDICTIONS = [k for k in new_df.columns if k.startswith("pred")]
    hand_only = [p for p in PREDICTIONS if p != "pred5_clf_10"]
    generic = new_df[TO_PREDICT_TARGET] == 1
    for p in hand_only:
        generic &= new_df[p] == 0
    generic &= new_df["pred5_clf_10"] == 1
    generic_test = int(generic[new_df.split == "test"].sum())

    tree_text = export_text(
        DecisionTreeClassifier(max_depth=10, random_state=42).fit(
            X_train_valid, y_train_valid),
        feature_names=list(X_train_valid.columns), max_depth=3)

    answers["q3"] = {
        "question": (
            "What is the total number of records in the TEST dataset where the new "
            "prediction pred5_clf_10 is correct, while all 'hand' rule predictions "
            "(pred0 to pred4) are incorrect?"
        ),
        "n_test_records": n_test,
        "only_pred5_is_correct_test": only_pred5_test,
        "only_pred5_is_correct_all_splits": only_pred5_all,
        "generic_function_result_test": generic_test,
    }
    answers["q4"] = {
        "question": "What is the optimal tree depth (from 1 to 12) for a DecisionTreeClassifier?",
        "precision_by_depth": {str(k): v for k, v in precision_by_depth.items()},
        "best_max_depth": best_depth,
        "best_precision": precision_by_depth[best_depth],
        "n_features": len(features_list),
    }
    print("Q3 TEST rows:", n_test, "| only pred5 correct (TEST):", only_pred5_test,
          "| generic fn:", generic_test, "| all splits:", only_pred5_all)
    print("Q4 precision by depth:", precision_by_depth)
    print("Q4 best depth:", best_depth, "precision:", precision_by_depth[best_depth])
    print("\nTree head (depth 3):\n", tree_text[:2500])


if __name__ == "__main__":
    df_with_dummies, NUMERICAL, DUMMIES, TO_PREDICT = build_dataframe()
    print(f"rows={len(df_with_dummies)} | NUMERICAL={len(NUMERICAL)} | DUMMIES={len(DUMMIES)}"
          f" | features={len(NUMERICAL) + len(DUMMIES)}")
    print(df_with_dummies.groupby("split")["Date"].agg(["min", "max", "count"]).to_string())

    question_1(df_with_dummies, NUMERICAL, DUMMIES, TO_PREDICT)
    question_2(add_manual_predictions(df_with_dummies.copy()))
    question_3_and_4(df_with_dummies, NUMERICAL, DUMMIES, TO_PREDICT)

    with open(OUTPUT_DIR / "hw3_answers.json", "w") as f:
        json.dump(answers, f, indent=2, default=str)
    print("\nSaved:", OUTPUT_DIR / "hw3_answers.json")
