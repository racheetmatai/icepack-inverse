"""Strict verifier for a JOG production inference bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from assemble_production_inference_bundle import canonical_id, sha256_file


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "inference_bundle_manifest.json").read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["schema"] = manifest.get("schema") == "jog-production-inference-bundle-v1"
    checks["status"] = manifest.get("status") == "complete"
    checks["manifest_id"] = canonical_id(manifest) == manifest.get("manifest_id")
    jobs = manifest.get("jobs", [])
    ensembles = manifest.get("ensembles", [])
    checks["counts"] = (
        len(jobs) == 660 and len(ensembles) == 66
        and manifest.get("counts") == {"jobs": 660, "ensembles": 66}
    )
    checks["unique_jobs"] = len({item.get("job_id") for item in jobs}) == 660
    checks["unique_ensembles"] = len({item.get("ensemble_id") for item in ensembles}) == 66
    checks["lambda_L2"] = all(item.get("lambda_L2") == 0.0 for item in jobs)
    checks["member_sets"] = all(
        len(item.get("member_job_ids", [])) == 10
        and len(set(item.get("member_job_ids", []))) == 10
        for item in ensembles
    )
    expected_files = {"inference_bundle_manifest.json"}
    failures = []
    for job in jobs:
        for relative, expected in job.get("files", {}).items():
            expected_files.add(relative.replace("/", "\\"))
            path = root / relative
            if not path.is_file() or sha256_file(path) != expected:
                failures.append(relative)
    registry = manifest.get("ensemble_registry", {})
    registry_path = root / registry.get("path", "")
    expected_files.add(str(registry_path.relative_to(root)))
    checks["registry_hash"] = registry_path.is_file() and sha256_file(registry_path) == registry.get("sha256")
    actual_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    checks["file_inventory"] = actual_files == expected_files
    checks["declared_hashes"] = not failures
    checks["frozen_rules"] = (
        manifest.get("frozen_rules", {}).get("combination")
        == "vertex-wise median of ten member log_C controls"
        and manifest.get("frozen_rules", {}).get("prediction_clipping") == "none"
    )
    result = {
        "schema": "jog-production-inference-bundle-verification-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "hash_failures": failures[:20],
        "manifest_id": manifest.get("manifest_id"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    verify(args.bundle)


if __name__ == "__main__":
    main()
