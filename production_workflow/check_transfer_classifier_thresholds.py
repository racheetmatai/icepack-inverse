#!/usr/bin/env python3
"""Does the 0.5 cut-off explain the classifier's poor labelling?

For every fold, three accuracies:
  * at 0.5, scikit-learn's default and what the published run implies;
  * at a threshold chosen on a held-out slice of the TRAINING squares, which
    is what a real user could do;
  * at the best possible threshold for the held-out square itself — an oracle
    no user could have, and therefore an upper bound.
Each is compared with guessing whichever outcome is more common in that region.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_transfer_predictability_counts import FEATURESETS, LABELS, PREDICTORS, load_fields  # noqa: E402

SEED = 20260919
PUBLISHED = dict(n_estimators=120, max_depth=14, min_samples_leaf=40, n_jobs=-1,
                 random_state=SEED, class_weight="balanced_subsample")
GRID = np.linspace(0.02, 0.98, 97)


def best_threshold(y, proba):
    acc = [( (proba > t) == y ).mean() for t in GRID]
    k = int(np.argmax(acc))
    return float(GRID[k]), float(acc[k])


def run_fold(train, test, label, cols):
    y_train = train[label].to_numpy()
    y_test = test[label].to_numpy()
    if y_train.std() == 0 or y_test.std() == 0:
        return None
    X_train = train[cols].to_numpy()
    X_test = test[cols].to_numpy()

    # threshold chosen on a held-out slice of the training squares
    Xa, Xb, ya, yb = train_test_split(X_train, y_train, test_size=0.1,
                                      random_state=SEED, stratify=y_train)
    inner = RandomForestClassifier(**PUBLISHED).fit(Xa, ya)
    t_user, _ = best_threshold(yb, inner.predict_proba(Xb)[:, 1])

    model = RandomForestClassifier(**PUBLISHED).fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    t_oracle, acc_oracle = best_threshold(y_test, proba)
    majority = float(max(y_test.mean(), 1 - y_test.mean()))
    return {
        "test_rows": int(len(y_test)),
        "majority_rule_accuracy": majority,
        "accuracy_at_0.5": float(((proba > 0.5) == y_test).mean()),
        "threshold_from_training": t_user,
        "accuracy_at_training_threshold": float(((proba > t_user) == y_test).mean()),
        "oracle_threshold": t_oracle,
        "accuracy_at_oracle_threshold": acc_oracle,
        "beats_majority_at_0.5": bool(((proba > 0.5) == y_test).mean() > majority),
        "beats_majority_at_training_threshold": bool(((proba > t_user) == y_test).mean() > majority),
        "beats_majority_at_oracle": bool(acc_oracle > majority),
    }


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
    for config in [f"CFG{i:02d}" for i in range(1, 7)]:
        frame = load_fields(a.map_fields / f"{config}_ten_square_controlled_fields.npz", predictors)
        central = frame[frame.central]
        for label in LABELS:
            for fs, cols in FEATURESETS.items():
                for held in range(1, 11):
                    result = run_fold(central[central.square != held],
                                      central[central.square == held], label, cols)
                    if result is None:
                        continue
                    rows.append({"test": f"SQ{held:02d}", "configuration": config,
                                 "label": label, "features": fs, **result})
        pd.DataFrame(rows).to_csv(a.output / "threshold_folds.csv", index=False)
        print(f"done {config}", flush=True)

    pig = load_fields(a.pig_fields, predictors)
    squares = load_fields(a.map_fields / "CFG02_ten_square_controlled_fields.npz", predictors)
    squares = squares[squares.central]
    train = squares[~squares.row_id.isin(set(pig.row_id))]
    for label in LABELS:
        for fs, cols in FEATURESETS.items():
            result = run_fold(train, pig, label, cols)
            if result:
                rows.append({"test": "REG_PIG", "configuration": "CFG02",
                             "label": label, "features": fs, **result})
    out = pd.DataFrame(rows)
    out.to_csv(a.output / "threshold_folds.csv", index=False)

    sq = out[out.test != "REG_PIG"]
    summary = {
        "folds": int(len(sq)),
        "median_accuracy_at_0.5": float(sq["accuracy_at_0.5"].median()),
        "median_accuracy_at_training_threshold": float(sq.accuracy_at_training_threshold.median()),
        "median_accuracy_at_oracle_threshold": float(sq.accuracy_at_oracle_threshold.median()),
        "median_majority_rule": float(sq.majority_rule_accuracy.median()),
        "folds_beating_majority_at_0.5": int(sq["beats_majority_at_0.5"].sum()),
        "folds_beating_majority_at_training_threshold": int(sq.beats_majority_at_training_threshold.sum()),
        "folds_beating_majority_at_oracle": int(sq.beats_majority_at_oracle.sum()),
    }
    (a.output / "threshold_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nPIG:")
    print(out[out.test == "REG_PIG"][
        ["label", "features", "accuracy_at_0.5", "accuracy_at_training_threshold",
         "accuracy_at_oracle_threshold", "majority_rule_accuracy", "beats_majority_at_oracle"]
    ].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
