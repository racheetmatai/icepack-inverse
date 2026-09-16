"""Evaluate the already-frozen L2 model once on the sealed calibration test rows."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import verify_inputs
from .integrity import canonical_manifest_id, file_sha256
from .spec import FEATURE_CONFIGURATIONS, FROZEN_POLICY
from .verify_run import verify


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--calibration-bundle", required=True); parser.add_argument("--selection", required=True)
    parser.add_argument("--runs-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); dataset = Path(args.dataset_dir).resolve(); bundle = Path(args.calibration_bundle).resolve()
    selection_path = Path(args.selection).resolve(); runs = Path(args.runs_root).resolve(); output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads(selection_path.read_text())
    if canonical_manifest_id(selection) != selection.get("manifest_id") or selection.get("test_population_accessed") is not False:
        raise ValueError("Selection is invalid or does not prove sealed-test selection")
    dataset_manifest, split_manifest = verify_inputs(dataset, bundle, full_hash_check=True)
    if selection.get("calibration_split_manifest_id") != split_manifest.get("manifest_id"):
        raise ValueError("Selection/calibration split identity mismatch")
    run = runs / selection["selected_job_id"]; verify(run)
    run_manifest = json.loads((run / "run_manifest.json").read_text())
    if float(run_manifest["lambda_L2"]) != float(selection["selected_lambda_L2"]):
        raise ValueError("Selected run/L2 mismatch")
    with gzip.open(bundle / "sorted_common_eligible_row_ids.txt.gz", "rt", encoding="utf-8") as stream:
        row_ids = np.asarray([line.rstrip("\n") for line in stream], dtype=object)
    with np.load(bundle / "population_masks/CAL_GLOBAL.npz", allow_pickle=False) as archive:
        test = np.unpackbits(archive["test"], bitorder="little")[:len(row_ids)].astype(bool)
    features = FEATURE_CONFIGURATIONS["CFG06"]
    raw = pd.read_csv(dataset / "canonical_master_dataset.csv.gz",
                      usecols=["row_id", "common_eligible", *features, FROZEN_POLICY["target"]], low_memory=False)
    frame = raw.loc[raw["common_eligible"].astype(bool)].drop(columns="common_eligible").set_index("row_id").reindex(row_ids)
    if frame.isna().any(axis=None):
        raise ValueError("Calibration test data are incomplete/non-finite")
    x = frame.loc[test, features].to_numpy(np.float64); reference = frame.loc[test, FROZEN_POLICY["target"]].to_numpy(np.float64)
    input_scaler = joblib.load(run / "input_scaler.joblib"); target_scaler = joblib.load(run / "target_scaler.joblib")
    import tensorflow as tf
    # Evaluate the exact minimum-val_data_mse checkpoint. In runs produced
    # before the exact-checkpoint reload fix, restored_best_model.keras can be
    # the EarlyStopping min_delta epoch rather than the exact minimum epoch.
    model = tf.keras.models.load_model(run / "best_model.keras")
    scaled_x = input_scaler.transform(x).astype(np.float32)
    scaled_reference = target_scaler.transform(reference.reshape(-1, 1)).reshape(-1)
    scaled_prediction = model.predict(scaled_x, batch_size=1024, verbose=0).reshape(-1)
    prediction = target_scaler.inverse_transform(scaled_prediction.reshape(-1, 1)).reshape(-1)
    residual = prediction - reference; ss_res = float(np.sum(residual ** 2)); ss_tot = float(np.sum((reference - reference.mean()) ** 2))
    metrics = {
        "rows": len(reference), "scaled_data_mse": float(np.mean((scaled_prediction - scaled_reference) ** 2)),
        "log_C_rmse": math.sqrt(float(np.mean(residual ** 2))), "log_C_mae": float(np.mean(np.abs(residual))),
        "log_C_bias": float(np.mean(residual)), "log_C_r2": 1.0 - ss_res / ss_tot,
    }
    pd.DataFrame({"row_id": row_ids[test], "reference_log_C": reference, "predicted_log_C": prediction}).to_csv(
        output / "calibration_test_predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n")
    manifest = {
        "schema": "jog-l2-calibration-test-v1", "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_manifest_id": selection["manifest_id"], "selection_sha256": file_sha256(selection_path),
        "selected_lambda_L2": selection["selected_lambda_L2"], "selected_run_manifest_id": run_manifest["manifest_id"],
        "dataset_manifest_id": dataset_manifest["manifest_id"], "calibration_split_manifest_id": split_manifest["manifest_id"],
        "test_rows": int(test.sum()), "metrics": metrics,
        "environment": {"python": sys.version, "platform": platform.platform(), "tensorflow": tf.__version__},
        "output_sha256": {name: file_sha256(output / name) for name in ("metrics.json", "calibration_test_predictions.csv.gz")},
    }
    manifest["manifest_id"] = canonical_manifest_id(manifest)
    (output / "test_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"manifest_id": manifest["manifest_id"], "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
