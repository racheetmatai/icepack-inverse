"""Independent structural and hash verifier for portable-training artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .integrity import canonical_manifest_id, verify_declared_outputs


def verify(path: str | Path) -> dict:
    root = Path(path).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    checks = {
        "schema": manifest.get("schema") == "jog-portable-training-run-v1",
        "complete": manifest.get("status") == "complete",
        "manifest_id": canonical_manifest_id(manifest) == manifest.get("manifest_id"),
    }
    try:
        verify_declared_outputs(root, manifest)
        checks["declared_output_hashes"] = True
    except Exception:
        checks["declared_output_hashes"] = False
    required = {"resolved_spec.json", "scaler_parameters.json", "data_identity.json", "input_scaler.joblib", "target_scaler.joblib"}
    checks["common_artifacts"] = required.issubset(manifest.get("output_sha256", {}))
    if manifest.get("mode") == "data-smoke":
        checks["mode_artifacts"] = bool(manifest.get("finite")) and "best_model.keras" not in manifest.get("output_sha256", {})
    else:
        expected = {"best_model.keras", "history.csv", "learning_rate_history.csv",
                    "validation_predictions.csv.gz", "training_summary.json"}
        checks["mode_artifacts"] = expected.issubset(manifest.get("output_sha256", {}))
    result = {"schema": "jog-portable-training-verification-v1", "run_manifest_id": manifest.get("manifest_id"),
              "checks": checks, "passed": all(checks.values())}
    if not result["passed"]:
        raise ValueError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
