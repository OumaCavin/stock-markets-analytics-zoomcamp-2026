"""
Homework 4 - Trading System (2026 cohort)
Solutions for questions 1-3. Produces output/hw4_answers.json.

Follows the Module 04 Colab notebook ([2026]_Module_04_Colab_Trading_Simulations.ipynb).

Question 1 : CAGR of pred10_rf_best_rule_55 with the reduced dummy-variable setup
Question 2 : probability threshold of the second-best Random Forest rule with < 5 trades/day
Question 3 : number of trades for pred5_clf_10 after redefining the target (>1.022)

Each scenario runs in its own process (`python hw4_questions.py <scenario>`) because the
sandbox has a small memory budget: the three scenarios are executed one after another and
their partial results are merged by `merge_results.py`.
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PARQUET = DATA_DIR / "stocks_df_combined_2026_09_18.parquet.brotli"
TGT = "is_positive_growth_30d_future"

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

BEST_CLF_DEPTH = 7          # precision_by_depth peak in the Module 04 notebook
RF_MAX_DEPTH = 13
RF_N_ESTIMATORS = 100


def temporal_split(df, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
    val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)
    df["split"] = np.where(
        df["Date"] <= train_end, "train",
        np.where(df["Date"] <= val_end, "validation", "test"),
    )
    return df


def build_dataframe(reduced_dummies=False, redefine_target=False):
    """Module 04 pipeline. `reduced_dummies` drops Ticker and month_wom (HW4 Q1)."""
    df_full = pd.read_parquet(PARQUET)

    GROWTH = [g for g in df_full.columns if g.startswith("growth_") and "future" not in g]
    CATEGORICAL = ["Month", "Weekday", "Ticker", "ticker_type"]

    df_full["ln_volume"] = df_full.Volume.apply(lambda x: np.log(x + 1e-6))
    df_full["div_payout"] = (df_full.Dividends > 0).astype(int)
    df_full["stock_split"] = (df_full["Stock Splits"] > 0).astype(int)
    CUSTOM_NUMERICAL = ["SMA10", "SMA20", "growing_moving_average",
                        "high_minus_low_relative", "volatility", "ln_volume",
                        "div_payout", "stock_split"]
    TECHNICAL_PATTERNS = [g for g in df_full.columns if "cdl" in g]
    NUMERICAL = (GROWTH + TECHNICAL_INDICATORS + TECHNICAL_PATTERNS
                 + CUSTOM_NUMERICAL + MACRO)

    df = df_full[df_full.Date >= "2000-01-01"].copy()
    del df_full
    gc.collect()

    df["Month"] = pd.to_datetime(df["Month"], format="%m").dt.strftime("%B")
    df["Weekday"] = df["Weekday"].astype(str)
    df["wom"] = df.Date.apply(lambda d: (d.day - 1) // 7 + 1)
    df["month_wom"] = df["Month"] + "_w" + df["wom"].astype(str)
    del df["wom"]
    CATEGORICAL = CATEGORICAL + ["month_wom"]

    if reduced_dummies:
        # HW4 Q1: drop the two most granular categorical features.
        CATEGORICAL = [c for c in CATEGORICAL if c not in ("Ticker", "month_wom")]

    dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype="int32")
    DUMMIES = list(dummy_variables.keys())
    df_with_dummies = pd.concat([df, dummy_variables], axis=1)
    del df, dummy_variables
    gc.collect()

    df_with_dummies = temporal_split(
        df_with_dummies, df_with_dummies.Date.min(), df_with_dummies.Date.max())

    if redefine_target:
        # HW4 Q3: a positive outcome now means growth above 2.2%.
        df_with_dummies[TGT] = np.where(df_with_dummies["growth_future_30d"] > 1.022, 1, 0)

    return df_with_dummies, NUMERICAL, DUMMIES


def add_manual_predictions(df):
    df["pred0_manual_cci"] = (df.cci > 200).astype(int)
    df["pred1_manual_prev_g1"] = (df.growth_30d > 1).astype(int)
    df["pred2_manual_prev_g1_and_snp"] = (
        (df["growth_30d"] > 1) & (df["growth_snp500_30d"] > 1)).astype(int)
    df["pred3_manual_dgs10_5"] = ((df["DGS10"] <= 4.5) & (df["DGS5"] <= 4)).astype(int)
    df["pred4_manual_dgs10_fedfunds"] = (
        (df["DGS10"] > 4) & (df["FEDFUNDS"] <= 4.795)).astype(int)
    return df


def fit_models(df_with_dummies, NUMERICAL, DUMMIES):
    """Train the three models of the Module 04 notebook and attach their predictions."""
    new_df = add_manual_predictions(df_with_dummies)
    features_list = NUMERICAL + DUMMIES

    train_valid = new_df[new_df.split.isin(["train", "validation"])]
    X_train_valid = train_valid[features_list + [TGT]].astype(np.float32)
    del train_valid
    X_train_valid.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train_valid.fillna(0, inplace=True)
    y_train_valid = X_train_valid[TGT]
    del X_train_valid[TGT]
    gc.collect()

    X_all = new_df[features_list + [TGT]].astype(np.float32)
    X_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_all.fillna(0, inplace=True)
    del X_all[TGT]
    gc.collect()

    clf_10 = DecisionTreeClassifier(max_depth=10, random_state=42).fit(
        X_train_valid, y_train_valid)
    new_df["pred5_clf_10"] = clf_10.predict(X_all)
    del clf_10

    clf_4 = DecisionTreeClassifier(max_depth=4, random_state=42).fit(
        X_train_valid, y_train_valid)
    new_df["pred6_clf_4"] = clf_4.predict(X_all)
    del clf_4

    clf_best = DecisionTreeClassifier(max_depth=BEST_CLF_DEPTH, random_state=42).fit(
        X_train_valid, y_train_valid)
    new_df["pred7_clf_second_best"] = clf_best.predict(X_all)
    clf_proba = clf_best.predict_proba(X_all)[:, 1]
    new_df["proba_pred8"] = clf_proba
    new_df["pred8_clf_second_best_rule_56"] = (clf_proba >= 0.56).astype(int)
    new_df["proba_pred9"] = clf_proba
    new_df["pred9_clf_second_best_rule_66"] = (clf_proba >= 0.66).astype(int)
    del clf_best
    gc.collect()

    rf_best = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                                     random_state=42, n_jobs=1)
    rf_best.fit(X_train_valid, y_train_valid)
    rf_proba = rf_best.predict_proba(X_all)[:, 1]
    new_df["proba_pred10"] = rf_proba
    new_df["pred10_rf_best_rule_55"] = (rf_proba >= 0.55).astype(int)
    new_df["proba_pred11"] = rf_proba
    new_df["pred11_rf_best_rule_65"] = (rf_proba >= 0.65).astype(int)
    del rf_best, X_train_valid, X_all
    gc.collect()

    return new_df


def simulate(new_df, predictions):
    """The Module 04 trading simulation, reproduced exactly."""
    rows = []
    for pred in predictions:
        mask = (new_df.split == "test") & (new_df[pred] == 1)
        selected = new_df[mask]
        count = len(selected)
        gross = float((selected[pred] * 100 * (selected["growth_future_30d"] - 1)).sum())
        fees = float((-selected[pred] * 100 * 0.002).sum())
        net = gross + fees
        if count:
            daily = selected.groupby("Date")[pred].count()
            avg_per_day = float(daily.mean())
            q75_per_day = float(daily.quantile(0.75))
        else:
            avg_per_day, q75_per_day = np.nan, np.nan
        capital = 100 * 30 * q75_per_day if count else np.nan
        cagr = ((capital + net) / capital) ** (1 / 4) if count else np.nan
        rows.append({
            "prediction": pred,
            "count_investments": count,
            "gross_rev": gross,
            "fees": fees,
            "net_rev": net,
            "avg_net_rev_per_investment": net / count if count else np.nan,
            "avg_investments_per_day": avg_per_day,
            "q75_investments_per_day": q75_per_day,
            "capital": capital,
            "CAGR": cagr,
        })
    return pd.DataFrame(rows)


def run_scenario(label, reduced_dummies=False, redefine_target=False,
                 extra_rf_thresholds=False):
    df_with_dummies, NUMERICAL, DUMMIES = build_dataframe(
        reduced_dummies=reduced_dummies, redefine_target=redefine_target)
    n_features = len(NUMERICAL) + len(DUMMIES)
    print(f"### {label}: rows={len(df_with_dummies)} features={n_features} "
          f"(NUMERICAL={len(NUMERICAL)} DUMMIES={len(DUMMIES)})", flush=True)

    new_df = fit_models(df_with_dummies, NUMERICAL, DUMMIES)
    del df_with_dummies
    gc.collect()

    predictions = [k for k in new_df.columns if k.startswith("pred")]
    if extra_rf_thresholds:
        proba = new_df["proba_pred10"].to_numpy()
        for thr in [0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64]:
            name = f"pred_rf_best_rule_{int(round(thr * 100)):02d}"
            new_df[name] = (proba >= thr).astype(int)
            predictions.append(name)

    sim = simulate(new_df, predictions).set_index("prediction")
    print(sim[["count_investments", "avg_investments_per_day",
               "q75_investments_per_day", "capital", "CAGR"]].to_string(), flush=True)
    return sim, n_features


def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = {}

    if scenario in ("all", "baseline"):
        sim, n = run_scenario("BASELINE (301 features)")
        out["baseline"] = {
            "n_features": n,
            "pred10_cagr": float(sim.loc["pred10_rf_best_rule_55", "CAGR"]),
            "simulation": json.loads(sim.reset_index().to_json(orient="records")),
        }
        print("BASELINE pred10 CAGR:", out["baseline"]["pred10_cagr"], flush=True)

    if scenario in ("all", "reduced"):
        sim, n = run_scenario("REDUCED (208 features)", reduced_dummies=True,
                              extra_rf_thresholds=True)
        out["reduced"] = {
            "n_features": n,
            "pred10_cagr": float(sim.loc["pred10_rf_best_rule_55", "CAGR"]),
            "simulation": json.loads(sim.reset_index().to_json(orient="records")),
        }
        print("REDUCED pred10 CAGR:", out["reduced"]["pred10_cagr"], flush=True)

    if scenario in ("all", "redefined"):
        sim, n = run_scenario("REDEFINED TARGET (>1.022), 301 features",
                              redefine_target=True)
        row = sim.loc["pred5_clf_10"]
        out["redefined"] = {
            "n_features": n,
            "pred5_clf_10": {k: (None if pd.isna(v) else float(v))
                              for k, v in row.to_dict().items()},
        }
        print("Q3 pred5_clf_10:", out["redefined"]["pred5_clf_10"], flush=True)

    path = OUTPUT_DIR / f"hw4_{scenario}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("Saved:", path, flush=True)


if __name__ == "__main__":
    main()
