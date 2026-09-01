"""Independent array-level verifier for all full-mesh ensemble controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(payload: dict) -> str:
    body = dict(payload)
    body.pop("manifest_id", None)
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root: Path) -> dict:
    root = root.resolve()
    set_path = root / "prediction_set_manifest.json"
    prediction_set = read_json(set_path)
    checks: dict[str, bool] = {
        "set_schema": prediction_set.get("schema") == "jog-full-mesh-ensemble-prediction-set-v1",
        "set_status": prediction_set.get("status") == "complete" and prediction_set.get("scope") == "full",
        "set_manifest_id": canonical_id(prediction_set) == prediction_set.get("manifest_id"),
        "set_counts": prediction_set.get("counts", {}).get("ensembles") == 66
        and prediction_set.get("counts", {}).get("member_controls") == 660
        and prediction_set.get("counts", {}).get("median_controls") == 66,
    }
    references = []
    coordinates_all = []
    masks = []
    job_ids: list[str] = []
    failures = []
    for index, declared in enumerate(prediction_set.get("ensemble_manifests", []), start=1):
        item_path = root / declared["path"]
        if not item_path.is_file() or sha256_file(item_path) != declared["sha256"]:
            failures.append(f"manifest:{declared['path']}")
            continue
        item = read_json(item_path)
        if canonical_id(item) != item.get("manifest_id") or item.get("manifest_id") != declared["manifest_id"]:
            failures.append(f"manifest_id:{declared['path']}")
            continue
        npz_path = root / item["npz_path"]
        if not npz_path.is_file() or sha256_file(npz_path) != item["npz_sha256"]:
            failures.append(f"npz_hash:{item['npz_path']}")
            continue
        with np.load(npz_path, allow_pickle=False) as archive:
            coordinates = archive["coordinates"]
            eligible = archive["eligible_mask"].astype(bool)
            reference = archive["reference_log_C"]
            members = archive["member_log_C"]
            median = archive["median_log_C"]
            archived_ids = archive["member_job_ids"].astype(str).tolist()
        valid = (
            coordinates.shape == (35797, 2)
            and eligible.shape == (35797,)
            and int(eligible.sum()) == 32496
            and reference.shape == (35797,)
            and members.shape == (10, 35797)
            and median.shape == (35797,)
            and len(archived_ids) == 10
            and archived_ids == item["member_job_ids"]
            and np.isfinite(coordinates).all()
            and np.isfinite(reference).all()
            and np.isfinite(members).all()
            and np.isfinite(median).all()
            and np.array_equal(members[:, ~eligible], np.repeat(reference[None, ~eligible], 10, axis=0))
            and np.array_equal(median[~eligible], reference[~eligible])
            and np.array_equal(median, np.median(members, axis=0))
            and item.get("outside_mask_exact") is True
        )
        if not valid:
            failures.append(f"array_semantics:{item['ensemble_id']}")
            continue
        references.append(reference)
        coordinates_all.append(coordinates)
        masks.append(eligible)
        job_ids.extend(archived_ids)
        if index % 10 == 0 or index == 66:
            print(f"Verified {index}/66 ensemble arrays", flush=True)

    checks["all_ensemble_arrays"] = len(references) == 66 and not failures
    checks["common_coordinates"] = bool(references) and all(
        np.array_equal(coordinates_all[0], values) for values in coordinates_all[1:]
    )
    checks["common_reference_control"] = bool(references) and all(
        np.array_equal(references[0], values) for values in references[1:]
    )
    checks["common_eligible_mask"] = bool(references) and all(
        np.array_equal(masks[0], values) for values in masks[1:]
    )
    checks["exact_unique_job_population"] = len(job_ids) == 660 and len(set(job_ids)) == 660
    checks["root_manifest_population"] = len(prediction_set.get("ensemble_manifests", [])) == 66
    result = {
        "schema": "jog-full-mesh-ensemble-prediction-verification-v1",
        "passed": all(checks.values()),
        "prediction_set_manifest_id": prediction_set.get("manifest_id"),
        "checks": checks,
        "counts": {
            "ensemble_arrays_verified": len(references),
            "unique_member_controls": len(set(job_ids)),
            "median_controls": len(references),
            "total_forward_controls": len(set(job_ids)) + len(references),
        },
        "failures": failures,
    }
    result["manifest_id"] = canonical_id(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.prediction_root)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
