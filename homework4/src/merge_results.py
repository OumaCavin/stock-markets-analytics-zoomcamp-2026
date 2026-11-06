"""Merge the three Homework 4 scenario files into one answers file.

Each scenario runs in its own process (see hw4_questions.py) because the sandbox that
produced this repo has a small memory budget. Run them first:

    python homework4/src/hw4_questions.py baseline
    python homework4/src/hw4_questions.py reduced
    python homework4/src/hw4_questions.py redefined
    python homework4/src/hw4_questions.py merge
"""

import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

OPTIONS_Q1 = [1.121, 1.126, 1.136, 1.141]
OPTIONS_Q2 = [0.54, 0.56, 0.57, 0.58]


def nearest(value, options):
    return min(options, key=lambda o: abs(o - value))


def main():
    baseline = json.loads((OUTPUT_DIR / "hw4_baseline.json").read_text())["baseline"]
    reduced = json.loads((OUTPUT_DIR / "hw4_reduced.json").read_text())["reduced"]
    redefined = json.loads((OUTPUT_DIR / "hw4_redefined.json").read_text())["redefined"]

    sim_reduced = {r["prediction"]: r for r in reduced["simulation"]}

    rf_rules = {}
    for name, row in sim_reduced.items():
        if not (name.startswith("pred_rf_best_rule_")
                or name in ("pred10_rf_best_rule_55", "pred11_rf_best_rule_65")):
            continue
        threshold = float(name.rsplit("_", 1)[1]) / 100
        rf_rules[name] = {
            "threshold": threshold,
            "count_investments": row["count_investments"],
            "avg_investments_per_day": row["avg_investments_per_day"],
            "q75_investments_per_day": row["q75_investments_per_day"],
            "capital": row["capital"],
            "CAGR": row["CAGR"],
            "under_5_trades_per_day": row["avg_investments_per_day"] < 5,
        }

    eligible = {k: v for k, v in rf_rules.items() if v["under_5_trades_per_day"]}
    best_name = max(eligible, key=lambda k: eligible[k]["CAGR"])
    best_threshold = eligible[best_name]["threshold"]

    p5 = redefined["pred5_clf_10"]

    answers = {
        "q1": {
            "question": (
                "Find the CAGR for the pred10_rf_best_rule_55 using the new setup "
                "(with fewer dummy variables)."
            ),
            "n_features_original_setup": baseline["n_features"],
            "n_features_new_setup": reduced["n_features"],
            "cagr_original_setup": baseline["pred10_cagr"],
            "cagr_new_setup": reduced["pred10_cagr"],
            "notebook_reference_original_cagr": 1.1195,
            "selected_option": nearest(reduced["pred10_cagr"], OPTIONS_Q1),
            "best_predictor_new_setup": max(
                (r for r in reduced["simulation"] if r["prediction"].startswith("pred")),
                key=lambda r: r["CAGR"])["prediction"],
        },
        "q2": {
            "question": (
                "What is the probability threshold for the second-best Random Forest "
                "decision rule, if you want to have the number of average trades per day "
                "BELOW 5?"
            ),
            "rf_rules": rf_rules,
            "best_under_5_trades_per_day": best_name,
            "selected_threshold": best_threshold,
            "selected_option": nearest(best_threshold, OPTIONS_Q2),
            "cagr_at_selected_threshold": eligible[best_name]["CAGR"],
            "avg_trades_per_day_at_selected_threshold":
                eligible[best_name]["avg_investments_per_day"],
            "cagr_of_unconstrained_best": rf_rules["pred10_rf_best_rule_55"]["CAGR"],
        },
        "q3": {
            "question": (
                "What is the number of trades for the best predictor pred5_clf_10 "
                "(after redefining the target variable)?"
            ),
            "n_features": redefined["n_features"],
            "count_investments": p5["count_investments"],
            "avg_investments_per_day": p5["avg_investments_per_day"],
            "q75_investments_per_day": p5["q75_investments_per_day"],
            "capital": p5["capital"],
            "CAGR": p5["CAGR"],
            "net_rev": p5["net_rev"],
            "selected_option": min([6, 7, 8, 9],
                                    key=lambda o: abs(o - p5["avg_investments_per_day"])),
        },
    }

    path = OUTPUT_DIR / "hw4_answers.json"
    path.write_text(json.dumps(answers, indent=2, default=str))
    print(json.dumps(answers, indent=2, default=str))
    print("\nSaved:", path)


if __name__ == "__main__":
    main()
