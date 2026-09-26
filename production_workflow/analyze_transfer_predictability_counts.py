#!/usr/bin/env python3
"""Count-based version of the transfer-predictability test.

Same design as the published analysis — leave-one-square-out over the ten
central squares, plus the PIG transfer test, six configurations, two success
criteria, three feature sets. Only the reported quantity changes: how many
held-out points the classifier labels correctly, against the score of simply
guessing whichever outcome is more common in that region.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

PREDICTORS = ["s", "h", "mag_s", "mag_h", "driving_stress", "surface_air_temp",
              "b", "mag_b", "heatflux", "gravity_disturbance", "mag_anomaly",
              "cos_theta_bs"]
CONFIGS = [f"CFG{i:02d}" for i in range(1, 7)]
FEATURESETS = {
    "speed_only": ["observed_speed"],
    "predictors_only": PREDICTORS,
    "predictors_and_speed": PREDICTORS + ["observed_speed"],
}
LABELS = {"improves": 1.0, "halves": 0.5}
SEED = 20260919
MODELS = {
    "published": dict(n_estimators=120, max_depth=14, min_samples_leaf=40, n_jobs=-1,
                      random_state=SEED, class_weight="balanced_subsample"),
    "deep": dict(n_estimators=400, max_depth=None, min_samples_leaf=5, n_jobs=-1,
                 random_state=SEED, class_weight="balanced_subsample"),
}


def load_fields(path: Path, predictors: pd.DataFrame) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as z:
        frame = pd.DataFrame({
            "row_id": z["row_id"].astype(str),
            "observed_speed": z["observed_speed"].astype(float),
            "model_error": z["model_error"].astype(float),
            "uniform_error": z["uniform_error"].astype(float),
        })
        if "square_number" in z.files:
            frame["square"] = z["square_number"].astype(int)
            frame["central"] = z["central_square"].astype(bool)
    frame = frame.join(predictors, on="row_id")
    for label, factor in LABELS.items():
        frame[label] = frame.model_error < factor * frame.uniform_error
    return frame


def fold(train, test, label, cols, params):
    y_train = train[label].to_numpy()
    y_test = test[label].to_numpy()
    majority_share = float(max(y_test.mean(), 1 - y_test.mean()))
    entry = {"test_rows": int(len(y_test)),
             "actually_yes": int(y_test.sum()), "actually_no": int((~y_test).sum()),
             "majority_rule_correct": int(round(majority_share * len(y_test))),
             "majority_rule_accuracy": majority_share,
             "uniform_outcome": bool(y_test.std() == 0)}
    if y_train.std() == 0 or y_test.std() == 0:
        entry.update({"correct": None, "wrong": None, "accuracy": None,
                      "f1_macro": None, "auc": None, "beats_majority": None})
        return entry
    model = RandomForestClassifier(**params)
    model.fit(train[cols].to_numpy(), y_train)
    pred = model.predict(test[cols].to_numpy())
    proba = model.predict_proba(test[cols].to_numpy())[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[False, True]).ravel()
    correct = int(tp + tn)
    entry.update({
        "correct": correct, "wrong": int(len(y_test) - correct),
        "accuracy": correct / len(y_test),
        "true_yes": int(tp), "false_yes": int(fp), "true_no": int(tn), "false_no": int(fn),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y_test, proba)),      # kept in the archive only
        "beats_majority": bool(correct / len(y_test) > majority_share),
    })
    return entry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--map-fields", type=Path, required=True)
    p.add_argument("--pig-fields", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    canon = pd.read_csv(a.dataset, usecols=["row_id"] + PREDICTORS, low_memory=False)
    canon["row_id"] = canon["row_id"].astype(str)
    predictors = canon.set_index("row_id")

    rows = []
    for config in CONFIGS:
        frame = load_fields(a.map_fields / f"{config}_ten_square_controlled_fields.npz", predictors)
        central = frame[frame.central]
        for label in LABELS:
            for fs, cols in FEATURESETS.items():
                for model_name, params in MODELS.items():
                    for held in range(1, 11):
                        train = central[central.square != held]
                        test = central[central.square == held]
                        rows.append({"test": f"SQ{held:02d}", "configuration": config,
                                     "label": label, "features": fs, "model": model_name,
                                     **fold(train, test, label, cols, params)})
        pd.DataFrame(rows).to_csv(a.output / "counts_folds.csv", index=False)
        print(f"done {config}", flush=True)

    # PIG: train on the ten central squares with rows inside PIG removed
    pig = load_fields(a.pig_fields, predictors)
    squares = load_fields(a.map_fields / "CFG02_ten_square_controlled_fields.npz", predictors)
    squares = squares[squares.central]
    train = squares[~squares.row_id.isin(set(pig.row_id))]
    for label in LABELS:
        for fs, cols in FEATURESETS.items():
            for model_name, params in MODELS.items():
                rows.append({"test": "REG_PIG", "configuration": "CFG02", "label": label,
                             "features": fs, "model": model_name,
                             **fold(train, pig, label, cols, params)})
    out = pd.DataFrame(rows)
    out.to_csv(a.output / "counts_folds.csv", index=False)

    sq = out[out.test != "REG_PIG"]
    scored = sq[sq.correct.notna()]
    summary = {
        "folds_total": int(len(sq)),
        "folds_uniform_outcome": int(sq.uniform_outcome.sum()),
        "folds_scored": int(len(scored)),
        "points_scored": int(scored.test_rows.sum()),
        "points_correct": int(scored.correct.sum()),
        "overall_accuracy": float(scored.correct.sum() / scored.test_rows.sum()),
        "median_accuracy": float(scored.accuracy.median()),
        "median_majority_rule_accuracy": float(scored.majority_rule_accuracy.median()),
        "folds_beating_majority": int(scored.beats_majority.sum()),
        "median_f1_macro": float(scored.f1_macro.median()),
        "median_auc_archived_only": float(scored.auc.median()),
    }
    (a.output / "counts_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nby model and feature set (medians over scored folds):")
    print(scored.groupby(["model", "features"])[
        ["accuracy", "majority_rule_accuracy", "f1_macro"]].median().round(3).to_string())
    print("\nfolds beating the majority rule, by model and feature set:")
    print(scored.groupby(["model", "features"]).beats_majority.sum().to_string())
    print("\nPIG:")
    print(out[out.test == "REG_PIG"][
        ["label", "features", "model", "test_rows", "correct", "accuracy",
         "majority_rule_accuracy", "beats_majority"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
