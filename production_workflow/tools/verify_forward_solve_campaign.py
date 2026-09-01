"""Independent verifier for the complete 726-control Icepack campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
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


def expected_controls(prediction_root: Path) -> tuple[dict[str, dict], dict]:
    root_manifest = read_json(prediction_root / "prediction_set_manifest.json")
    if (
        root_manifest.get("schema") != "jog-full-mesh-ensemble-prediction-set-v1"
        or root_manifest.get("status") != "complete"
        or canonical_id(root_manifest) != root_manifest.get("manifest_id")
    ):
        raise ValueError("Prediction-set root manifest is invalid")
    expected: dict[str, dict] = {}
    for declared in root_manifest["ensemble_manifests"]:
        item_path = prediction_root / declared["path"]
        if sha256_file(item_path) != declared["sha256"]:
            raise ValueError(f"Prediction ensemble-manifest hash mismatch: {item_path}")
        item = read_json(item_path)
        npz_path = prediction_root / item["npz_path"]
        if sha256_file(npz_path) != item["npz_sha256"]:
            raise ValueError(f"Prediction NPZ hash mismatch: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as archive:
            members = archive["member_log_C"]
            median = archive["median_log_C"]
        for index, control_id in enumerate(item["member_job_ids"]):
            expected[control_id] = {
                "ensemble_id": item["ensemble_id"], "kind": "member",
                "prediction_npz": item["npz_path"],
                "prediction_npz_sha256": item["npz_sha256"],
                "control_sha256": hashlib.sha256(members[index].tobytes(order="C")).hexdigest(),
            }
        control_id = f"{item['ensemble_id']}_MEDIAN"
        expected[control_id] = {
            "ensemble_id": item["ensemble_id"], "kind": "median",
            "prediction_npz": item["npz_path"],
            "prediction_npz_sha256": item["npz_sha256"],
            "control_sha256": hashlib.sha256(median.tobytes(order="C")).hexdigest(),
        }
    if len(expected) != 726:
        raise ValueError(f"Expected-control population is {len(expected)}, not 726")
    return expected, root_manifest


def verify(campaign_root: Path, prediction_root: Path) -> dict:
    campaign_root = campaign_root.resolve()
    prediction_root = prediction_root.resolve()
    expected, prediction_set = expected_controls(prediction_root)
    manifests = sorted((campaign_root / "solves").glob("*/forward_manifest.json"))
    observed_ids = {path.parent.name for path in manifests}
    checks = {
        "exact_control_directory_population": observed_ids == set(expected),
        "exact_manifest_count": len(manifests) == 726,
    }
    failures = []
    retry_controls = []
    legacy_primary_controls = []
    source_counts: dict[str, int] = {}
    elapsed = []
    speed_minima = []
    speed_maxima = []
    for index, manifest_path in enumerate(manifests, start=1):
        manifest = read_json(manifest_path)
        control_id = manifest_path.parent.name
        expected_item = expected.get(control_id)
        problems = []
        if expected_item is None:
            problems.append("unexpected_control")
        if manifest.get("schema") != "jog-icepack-forward-control-v1":
            problems.append("schema")
        if manifest.get("status") != "complete":
            problems.append("status")
        if canonical_id(manifest) != manifest.get("manifest_id"):
            problems.append("manifest_id")
        if manifest.get("control_id") != control_id:
            problems.append("control_id")
        if manifest.get("prediction_set_manifest_id") != prediction_set["manifest_id"]:
            problems.append("prediction_set_id")
        if expected_item is not None:
            for key in (
                "ensemble_id", "prediction_npz", "prediction_npz_sha256", "control_sha256"
            ):
                if manifest.get(key) != expected_item[key]:
                    problems.append(key)
            if manifest.get("control_kind") != expected_item["kind"]:
                problems.append("control_kind")
        velocity_path = manifest_path.parent / (manifest.get("velocity_path") or "")
        if not velocity_path.is_file() or sha256_file(velocity_path) != manifest.get("velocity_sha256"):
            problems.append("velocity_hash")
        else:
            velocity = np.load(velocity_path, allow_pickle=False)
            if velocity.shape != (35797, 2) or velocity.dtype != np.float64 or not np.isfinite(velocity).all():
                problems.append("velocity_array")
            else:
                speed = np.linalg.norm(velocity, axis=1)
                summary = manifest.get("velocity_summary") or {}
                if not (
                    summary.get("finite") is True
                    and summary.get("minimum_speed") == float(speed.min())
                    and summary.get("median_speed") == float(np.median(speed))
                    and summary.get("maximum_speed") == float(speed.max())
                ):
                    problems.append("velocity_summary")
                speed_minima.append(float(speed.min()))
                speed_maxima.append(float(speed.max()))

        attempts = manifest.get("attempts")
        if attempts:
            limits = [attempt.get("snes_max_it") for attempt in attempts]
            if 100 in limits:
                retry_controls.append(control_id)
                if attempts[-1].get("snes_max_it") != 100 or attempts[-1].get("status") != "complete":
                    problems.append("retry_terminal_attempt")
                prior = manifest.get("prior_failure")
                if prior:
                    archive = campaign_root / prior["archive_path"]
                    if not archive.is_file() or sha256_file(archive) != prior.get("manifest_sha256"):
                        problems.append("prior_failure_archive_hash")
                    else:
                        archived = read_json(archive)
                        if (
                            archived.get("manifest_id") != prior.get("manifest_id")
                            or archived.get("status") != "failed"
                            or "DIVERGED_MAX_IT" not in str(archived.get("exception"))
                        ):
                            problems.append("prior_failure_semantics")
                else:
                    if not (
                        len(attempts) == 2
                        and attempts[0].get("snes_max_it") == 50
                        and attempts[0].get("status") == "failed"
                        and "DIVERGED_MAX_IT" in str(attempts[0].get("exception"))
                    ):
                        problems.append("in_run_retry_lineage")
            elif not (len(attempts) == 1 and limits == [50] and attempts[0].get("status") == "complete"):
                problems.append("primary_attempt_semantics")
        else:
            legacy_primary_controls.append(control_id)
            policy = manifest.get("solver_policy") or {}
            if policy.get("snes_type") != "newtontr":
                problems.append("legacy_solver_policy")
        source = str((manifest.get("solver_policy") or {}).get("retry_snes_max_it", "legacy_primary_50"))
        source_counts[source] = source_counts.get(source, 0) + 1
        elapsed.append(float(manifest.get("elapsed_seconds", np.nan)))
        if problems:
            failures.append({"control_id": control_id, "problems": sorted(set(problems))})
        if index % 100 == 0 or index == 726:
            print(f"Verified {index}/726 forward controls", flush=True)

    workers = [read_json(path) for path in sorted(campaign_root.glob("worker_shard_*.json"))]
    checks.update({
        "all_control_manifests_and_arrays": not failures,
        "six_same_solver_retries": len(retry_controls) == 6,
        "terminal_worker_population": len(workers) == 2
        and {item.get("shard_index") for item in workers} == {0, 1}
        and all(item.get("status") == "complete" for item in workers),
        "worker_registered_population": sum(item.get("registered_to_worker", 0) for item in workers) == 726,
        "worker_no_new_failures": sum(item.get("newly_failed", 0) for item in workers) == 0,
    })
    archived_failures = sorted(campaign_root.glob("failed_attempts/*/*/forward_manifest.json"))
    checks["two_preserved_prepolicy_failures"] = len(archived_failures) == 2
    result = {
        "schema": "jog-icepack-forward-campaign-verification-v1",
        "status": "complete" if all(checks.values()) else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "prediction_set_manifest_id": prediction_set["manifest_id"],
        "checks": checks,
        "counts": {
            "controls": len(manifests), "members": sum("_MEDIAN" not in name for name in expected),
            "medians": sum("_MEDIAN" in name for name in expected),
            "same_solver_100_iteration_retries": len(retry_controls),
            "legacy_primary_50_controls": len(legacy_primary_controls),
            "preserved_prepolicy_failure_manifests": len(archived_failures),
        },
        "retry_controls": sorted(retry_controls),
        "solver_manifest_populations": source_counts,
        "elapsed_seconds_summary": {
            "minimum": float(np.nanmin(elapsed)), "median": float(np.nanmedian(elapsed)),
            "maximum": float(np.nanmax(elapsed)), "sum": float(np.nansum(elapsed)),
        },
        "velocity_speed_global_summary": {
            "minimum_of_minima": min(speed_minima), "maximum_of_maxima": max(speed_maxima),
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
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.campaign_root, args.prediction_root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
