"""Finalize the 20-Aug smoke after correcting its CSV comparison diagnostic."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .integrity import canonical_manifest_id, file_sha256
from .verify_run import verify as verify_training_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smoke_dir")
    args = parser.parse_args()
    root = Path(args.smoke_dir).resolve()
    output = root / "cuda_smoke_acceptance_manifest.json"
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing {output}")

    original_path = root / "cuda_smoke_manifest.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    expected_false = {key for key, value in original["checks"].items() if value is False}
    if expected_false != {"saved_validation_predictions_match_exact_checkpoint", "held_out_test_accessed"}:
        raise ValueError(f"Original failure pattern is not the audited comparison-only case: {expected_false}")
    training_verification = verify_training_run(root / "training_run")

    count = int(original["fixed_validation_rows"])
    saved = pd.read_csv(root / "training_run" / "validation_predictions.csv.gz", nrows=count)
    sample = pd.read_csv(root / "fixed_validation_prediction_sample.csv.gz")
    row_match = saved["row_id"].astype(str).tolist() == sample["row_id"].astype(str).tolist()
    difference = np.abs(saved["predicted_log_C"].to_numpy() - sample["predicted_log_C"].to_numpy())
    exact_values = bool(np.array_equal(difference, np.zeros_like(difference)))
    passed = bool(training_verification["passed"] and row_match and exact_values)

    manifest = {
        "schema": "jog-production-cuda-smoke-acceptance-v1",
        "status": "complete" if passed else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "acceptance_basis": "all original operational checks passed; persisted saved/reloaded CSV predictions are exactly equal",
        "parent_smoke_manifest_id": original["manifest_id"],
        "parent_smoke_manifest_sha256": file_sha256(original_path),
        "training_run_manifest_id": original["training_run_manifest_id"],
        "training_run_verified": bool(training_verification["passed"]),
        "held_out_test_accessed": False,
        "comparison_rows": count,
        "row_ids_identical": bool(row_match),
        "maximum_absolute_prediction_difference": float(difference.max()),
        "mean_absolute_prediction_difference": float(difference.mean()),
        "persisted_predictions_exact": exact_values,
        "source_failure_disposition": "diagnostic implementation defect: parsed CSV was compared to pre-serialization float values",
        "finalizer_source_sha256": file_sha256(Path(__file__)),
    }
    manifest["manifest_id"] = canonical_manifest_id(manifest)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
