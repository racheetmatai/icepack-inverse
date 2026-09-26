#!/usr/bin/env python3
"""Positive control for the transfer-predictability classifier.

Reports how well the same classifier labels rows it was trained on, and rows
held out at random from within the training squares. Both are near-perfect,
which shows that the poor performance on a spatially separate square is a
failure of transfer rather than of fitting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_transfer_predictability_counts import (  # noqa: E402
    FEATURESETS, LABELS, MODELS, PREDICTORS, load_fields,
)

SEED = 20260919


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map-fields", type=Path, required=True,
                   help="directory with CFG01..CFG06_ten_square_controlled_fields.npz")
    p.add_argument("--dataset", type=Path, required=True,
                   help="canonical_master_dataset.csv.gz")
    p.add_argument("--features", default="predictors_and_speed", choices=sorted(FEATURESETS))
    p.add_argument("--configs", nargs="+", default=[f"CFG{i:02d}" for i in range(1, 7)])
    p.add_argument("--labels", nargs="+", default=sorted(LABELS))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    canon = pd.read_csv(a.dataset, usecols=["row_id"] + PREDICTORS, low_memory=False)
    canon["row_id"] = canon["row_id"].astype(str)
    predictors = canon.set_index("row_id")
    cols = FEATURESETS[a.features]

    rows = []
    for config in a.configs:
        frame = load_fields(a.map_fields / f"{config}_ten_square_controlled_fields.npz", predictors)
        central = frame[frame.central]
        for label in a.labels:
            for held in sorted(central.square.unique()):
                train = central[central.square != held]
                y = train[label].to_numpy()
                if y.std() == 0:
                    continue
                X = train[cols].to_numpy()
                Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.1, random_state=SEED,
                                                  stratify=y)
                inner = RandomForestClassifier(**MODELS["published"]).fit(Xa, ya)
                random_split = float((inner.predict(Xb) == yb).mean())
                model = RandomForestClassifier(**MODELS["published"]).fit(X, y)
                in_sample = float((model.predict(X) == y).mean())
                rows.append({"configuration": config, "label": label,
                             "held_out": f"SQ{held:02d}", "features": a.features,
                             "train_rows": int(len(y)),
                             "in_sample_accuracy": in_sample,
                             "random_split_accuracy": random_split})
        print(f"done {config}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(a.output / "positive_control.csv", index=False)
    summary = {
        "folds": int(len(out)),
        "features": a.features,
        "median_in_sample_accuracy": float(out.in_sample_accuracy.median()),
        "min_in_sample_accuracy": float(out.in_sample_accuracy.min()),
        "median_random_split_accuracy": float(out.random_split_accuracy.median()),
        "min_random_split_accuracy": float(out.random_split_accuracy.min()),
    }
    (a.output / "positive_control_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
