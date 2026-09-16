"""Freeze one global L2 value from the four matched CFG06 calibration fits."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .integrity import canonical_manifest_id, file_sha256
from .verify_run import verify


MINIMUM_ABSOLUTE_IMPROVEMENT = 1.0e-4


def select_candidate(records: list[dict]) -> tuple[dict, list[dict]]:
    by_l2 = {record["lambda_L2"]: record for record in records}
    if set(by_l2) != {0.0, 1e-6, 1e-5, 1e-4}:
        raise ValueError("Selection requires the exact four frozen candidates")
    zero = by_l2[0.0]["best_val_data_mse_scaled"]
    enriched = []
    for original in records:
        record = dict(original)
        record["absolute_improvement_vs_zero"] = zero - record["best_val_data_mse_scaled"]
        record["relative_change_vs_zero"] = (record["best_val_data_mse_scaled"] - zero) / zero
        record["meaningfully_improves_zero"] = (
            record["lambda_L2"] > 0 and record["absolute_improvement_vs_zero"] >= MINIMUM_ABSOLUTE_IMPROVEMENT)
        enriched.append(record)
    qualified = [record for record in enriched if record["meaningfully_improves_zero"]]
    selected = min(qualified, key=lambda record: (record["best_val_data_mse_scaled"], record["lambda_L2"])) if qualified else next(
        record for record in enriched if record["lambda_L2"] == 0.0)
    return selected, enriched


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--calibration-bundle", required=True)
    parser.add_argument("--runs-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); bundle = Path(args.calibration_bundle).resolve(); runs = Path(args.runs_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite selection: {output}")
    with (bundle / "l2_pilot_registry.csv").open(newline="", encoding="utf-8") as stream:
        registry = list(csv.DictReader(stream))
    if len(registry) != 4 or {float(row["lambda_L2"]) for row in registry} != {0.0, 1e-6, 1e-5, 1e-4}:
        raise ValueError("Calibration registry is not the frozen four-candidate design")
    records = []
    identities = set()
    for row in registry:
        root = runs / row["pilot_job_id"]
        verification = verify(root)
        manifest = json.loads((root / "run_manifest.json").read_text())
        summary = json.loads((root / "training_summary.json").read_text())
        if manifest.get("mode") != "model-training" or manifest.get("job_id") != row["pilot_job_id"]:
            raise ValueError(f"Wrong training artifact for {row['pilot_job_id']}")
        if float(manifest["lambda_L2"]) != float(row["lambda_L2"]):
            raise ValueError(f"L2 mismatch for {row['pilot_job_id']}")
        if not summary.get("all_history_finite"):
            raise ValueError(f"Non-finite training history: {row['pilot_job_id']}")
        identity = manifest["data_identity"]
        identities.add((identity["dataset_manifest_id"], identity["split_bundle_manifest_id"],
                        identity["train_membership_id"], identity["validation_membership_id"],
                        tuple(identity["features"]), identity["job"]["model_seed"], identity["job"]["shuffle_seed"]))
        records.append({
            "job_id": row["pilot_job_id"], "lambda_L2": float(row["lambda_L2"]),
            "best_val_data_mse_scaled": float(summary["best_val_data_mse_scaled"]),
            "best_epoch_one_based": int(summary["best_epoch_one_based"]),
            "epochs_completed": int(summary["epochs_completed"]),
            "run_manifest_id": manifest["manifest_id"], "run_manifest_sha256": file_sha256(root / "run_manifest.json"),
            "verification_passed": verification["passed"],
        })
    if len(identities) != 1:
        raise ValueError("L2 candidates do not share the exact data/features/seeds")
    selected, records = select_candidate(records)
    zero = next(record["best_val_data_mse_scaled"] for record in records if record["lambda_L2"] == 0.0)
    calibration_manifest = json.loads((bundle / "split_bundle_manifest.json").read_text())
    result = {
        "schema": "jog-global-l2-selection-v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_split_manifest_id": calibration_manifest["manifest_id"],
        "configuration": "CFG06", "candidate_lambda_L2": [0.0, 1e-6, 1e-5, 1e-4],
        "minimum_absolute_improvement_vs_zero": MINIMUM_ABSOLUTE_IMPROVEMENT,
        "selection_rule": "lowest validation data_mse among nonzero candidates improving zero by >=1e-4; otherwise zero; exact tie smaller lambda",
        "selected_lambda_L2": selected["lambda_L2"], "selected_job_id": selected["job_id"],
        "zero_val_data_mse_scaled": zero, "candidates": sorted(records, key=lambda record: record["lambda_L2"]),
        "test_population_accessed": False,
    }
    result["manifest_id"] = canonical_manifest_id(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"selected_lambda_L2": result["selected_lambda_L2"], "selected_job_id": result["selected_job_id"],
                      "manifest_id": result["manifest_id"]}, indent=2))


if __name__ == "__main__":
    main()
