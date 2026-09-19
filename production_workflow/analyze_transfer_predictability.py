#!/usr/bin/env python3
"""Can the local predictors indicate in advance where transfer succeeds?

For each observation-grid row in a withheld region, the label is whether the
median predicted control gives lower local velocity error than the uniform-C
reference, e_ML < e_uniform, and a stricter version, e_ML < 0.5 e_uniform.
A random-forest classifier is trained to predict that label from the twelve
predictors (optionally with observed speed) and is scored on a region it did
not see.

Squares: leave-one-square-out. The classifier is trained on the central
squares of the other nine tests and scored on the tenth. Every training label
therefore comes from a median-C field predicted without that location, the
same condition as the test square. All six configurations.

PIG: CFG02 only, the configuration with a controlled PIG simulation. The
classifier is trained on all ten central squares, excluding rows inside PIG
(all of SQ01 and part of SQ09 and SQ05 lie in PIG), and scored on PIG.

Observed speed is included as a separate feature set because it is available
to any user from the velocity map. If adding the twelve predictors does not
improve on speed alone, the predictors carry no additional information about
where transfer succeeds.

No inversion, MLP training or forward simulation is run; the inputs are
saved per-row error fields and the canonical dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

PREDICTORS = ["s", "h", "mag_s", "mag_h", "driving_stress", "surface_air_temp",
              "b", "mag_b", "heatflux", "gravity_disturbance", "mag_anomaly",
              "cos_theta_bs"]
CONFIGS = [f"CFG{i:02d}" for i in range(1, 7)]
FEATURESETS = {
    "predictors_and_speed": PREDICTORS + ["observed_speed"],
    "speed_only": ["observed_speed"],
    "predictors_only": PREDICTORS,
}
LABELS = {
    "improves": 1.0,   # e_ML < 1.0 * e_uniform
    "halves": 0.5,     # e_ML < 0.5 * e_uniform
}
SEED = 20260919
FOREST = dict(n_estimators=120, max_depth=14, min_samples_leaf=40,
              n_jobs=-1, random_state=SEED, class_weight="balanced_subsample")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if frame[PREDICTORS].isna().any().any():
        raise RuntimeError(f"Predictor join incomplete for {path.name}")
    for label, factor in LABELS.items():
        frame[label] = frame.model_error < factor * frame.uniform_error
    return frame


def score(train: pd.DataFrame, test: pd.DataFrame, label: str, cols: list[str]):
    y_train, y_test = train[label].to_numpy(), test[label].to_numpy()
    if y_test.std() == 0 or y_train.std() == 0:
        return np.nan, None
    model = RandomForestClassifier(**FOREST)
    model.fit(train[cols].to_numpy(), y_train)
    auc = roc_auc_score(y_test, model.predict_proba(test[cols].to_numpy())[:, 1])
    return float(auc), model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-fields", type=Path, required=True,
                        help="directory with CFG01..CFG06_ten_square_controlled_fields.npz")
    parser.add_argument("--pig-fields", type=Path, required=True,
                        help="REG_PIG_CFG02_controlled_fields.npz")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="canonical_master_dataset.csv.gz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    canon = pd.read_csv(args.dataset, usecols=["row_id", "region_code"] + PREDICTORS,
                        low_memory=False)
    canon["row_id"] = canon["row_id"].astype(str)
    canon = canon.set_index("row_id")
    predictors = canon[PREDICTORS]

    folds, importances = [], []
    for config in CONFIGS:
        frame = load_fields(args.map_fields / f"{config}_ten_square_controlled_fields.npz", predictors)
        central = frame[frame.central]
        for label in LABELS:
            for name, cols in FEATURESETS.items():
                for held in range(1, 11):
                    train, test = central[central.square != held], central[central.square == held]
                    auc, model = score(train, test, label, cols)
                    folds.append({"test": f"SQ{held:02d}", "configuration": config,
                                  "label": label, "features": name, "auc": auc,
                                  "test_fraction_true": float(test[label].mean()),
                                  "train_rows": len(train), "test_rows": len(test),
                                  "degenerate": bool(np.isnan(auc))})
                    if model is not None and name == "predictors_and_speed":
                        importances.append({"test": f"SQ{held:02d}", "configuration": config,
                                            "label": label,
                                            **dict(zip(cols, model.feature_importances_))})
        print(f"{config}: square folds complete", flush=True)

    # PIG: CFG02, trained on the ten central squares with PIG rows removed.
    pig = load_fields(args.pig_fields, predictors)
    squares = load_fields(args.map_fields / "CFG02_ten_square_controlled_fields.npz", predictors)
    squares = squares[squares.central]
    in_pig = squares.row_id.isin(set(pig.row_id))
    train = squares[~in_pig]
    pig_rows = []
    for label in LABELS:
        for name, cols in FEATURESETS.items():
            auc, _ = score(train, pig, label, cols)
            pig_rows.append({"test": "REG_PIG", "configuration": "CFG02", "label": label,
                             "features": name, "auc": auc,
                             "test_fraction_true": float(pig[label].mean()),
                             "train_rows": len(train), "test_rows": len(pig),
                             "degenerate": bool(np.isnan(auc))})
    print("PIG complete", flush=True)

    folds_df = pd.DataFrame(folds)
    pig_df = pd.DataFrame(pig_rows)
    folds_df.to_csv(args.output / "square_folds.csv", index=False)
    pig_df.to_csv(args.output / "pig_transfer.csv", index=False)
    pd.DataFrame(importances).to_csv(args.output / "square_feature_importances.csv", index=False)

    summary = {"squares": {}, "pig_cfg02": {},
               "square_training_rows_excluded_as_inside_pig": int(in_pig.sum())}
    for label in LABELS:
        usable = folds_df[(folds_df.label == label) & ~folds_df.degenerate]
        wide = usable.pivot_table(index=["configuration", "test"], columns="features",
                                  values="auc").dropna()
        full, speed = wide["predictors_and_speed"], wide["speed_only"]
        summary["squares"][label] = {
            "usable_folds": int(len(wide)),
            "degenerate_folds": int(folds_df[(folds_df.label == label)
                                             & (folds_df.features == "speed_only")].degenerate.sum()),
            "mean_auc": {k: float(wide[k].mean()) for k in FEATURESETS},
            "median_auc": {k: float(wide[k].median()) for k in FEATURESETS},
            "std_auc": {k: float(wide[k].std()) for k in FEATURESETS},
            "folds_predictors_and_speed_beats_speed_only": int((full > speed).sum()),
            "folds_predictors_and_speed_below_0p5": int((full < 0.5).sum()),
            "mean_auc_by_configuration": {
                c: {k: float(usable[(usable.configuration == c) & (usable.features == k)].auc.mean())
                    for k in FEATURESETS} for c in CONFIGS},
            "mean_test_fraction_true": float(folds_df[(folds_df.label == label)
                                                      & (folds_df.features == "speed_only")]
                                             .test_fraction_true.mean()),
        }
        summary["pig_cfg02"][label] = {
            r.features: r.auc for r in pig_df[pig_df.label == label].itertuples()}
        summary["pig_cfg02"][label]["fraction_true"] = float(pig[label].mean())
    summary["degenerate_square_cases"] = (
        folds_df[folds_df.degenerate & (folds_df.features == "speed_only")]
        [["label", "configuration", "test", "test_fraction_true"]].to_dict("records"))
    if importances:
        imp = pd.DataFrame(importances)
        imp = imp[imp.label == "improves"]
        summary["mean_feature_importance_improves"] = (
            imp[PREDICTORS + ["observed_speed"]].mean().sort_values(ascending=False).to_dict())
    (args.output / "transfer_predictability_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema": "jog-transfer-predictability-v1",
        "status": "complete",
        "classifier": {"type": "sklearn.ensemble.RandomForestClassifier", **FOREST},
        "labels": {"improves": "e_ML < e_uniform", "halves": "e_ML < 0.5 e_uniform"},
        "inputs": {
            "map_fields": {p.name: sha256(p) for p in sorted(args.map_fields.glob("CFG0*_ten_square_controlled_fields.npz"))},
            "pig_fields": {args.pig_fields.name: sha256(args.pig_fields)},
            "dataset": {args.dataset.name: sha256(args.dataset)},
        },
        "outputs": {p.name: sha256(p) for p in sorted(args.output.glob("*.csv"))
                    + [args.output / "transfer_predictability_summary.json"]},
    }
    (args.output / "transfer_predictability_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: summary["squares"][k]["mean_auc"] for k in LABELS}, indent=2))
    print(json.dumps(summary["pig_cfg02"], indent=2))


if __name__ == "__main__":
    main()
