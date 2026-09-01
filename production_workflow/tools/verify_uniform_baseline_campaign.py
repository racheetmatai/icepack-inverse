"""Independent structural and numerical audit of the 12 uniform-C baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EXPECTED = {*(f"SQ{i:02d}_UNIFORM_C" for i in range(1, 11)),
            "REG_INTER_UNIFORM_C", "REG_PIG_UNIFORM_C"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(payload: dict) -> str:
    body = dict(payload); body.pop("manifest_id", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root: Path) -> dict:
    root = root.resolve()
    campaign = read_json(root / "baseline_campaign_manifest.json")
    checks = {
        "campaign_schema": campaign.get("schema") == "jog-uniform-c-baseline-campaign-v1",
        "campaign_id": canonical_id(campaign) == campaign.get("manifest_id"),
        "declared_count": campaign.get("count") == 12,
    }
    declared = {record["control_id"]: record for record in campaign.get("controls", [])}
    checks["exact_control_population"] = set(declared) == EXPECTED
    failures, retry_controls, scalar_values = [], [], {}
    common_coordinates = common_mask = common_reference = None
    for control_id in sorted(EXPECTED):
        record = declared.get(control_id)
        if record is None:
            failures.append({"control_id": control_id, "problems": ["missing_declaration"]})
            continue
        problems = []
        control_path = root / record["control_path"]
        if not control_path.is_file() or sha256_file(control_path) != record.get("control_sha256"):
            problems.append("control_hash")
        else:
            with np.load(control_path, allow_pickle=False) as archive:
                coordinates = archive["coordinates"]
                mask = archive["eligible_mask"].astype(bool)
                reference = archive["reference_C"]
                value = float(archive["uniform_C"])
                control = archive["control_C"]
            if (coordinates.shape != (35797, 2) or mask.shape != (35797,)
                    or reference.shape != (35797,) or control.shape != (35797,)
                    or int(mask.sum()) != 32496):
                problems.append("control_shapes_or_mask")
            if not all(np.isfinite(x).all() for x in (coordinates, reference, control)) or not np.isfinite(value):
                problems.append("control_finiteness")
            if not np.all(control[mask] == value) or not np.array_equal(control[~mask], reference[~mask]):
                problems.append("uniform_or_retained_values")
            if hashlib.sha256(control.tobytes(order="C")).hexdigest() != record.get("control_values_sha256"):
                problems.append("control_values_hash")
            if not np.isclose(value, float(record["uniform_C"]), rtol=0.0, atol=0.0):
                problems.append("uniform_scalar")
            scalar_values[record["experiment"]] = value
            if common_coordinates is None:
                common_coordinates, common_mask, common_reference = coordinates.copy(), mask.copy(), reference.copy()
            elif not (np.array_equal(common_coordinates, coordinates)
                      and np.array_equal(common_mask, mask)
                      and np.array_equal(common_reference, reference)):
                problems.append("common_mesh_context")
        solve_path = root / "solves" / control_id / "forward_manifest.json"
        if not solve_path.is_file():
            problems.append("missing_forward_manifest")
        else:
            solve = read_json(solve_path)
            velocity_path = solve_path.parent / str(solve.get("velocity_path", ""))
            if (solve.get("schema") != "jog-icepack-uniform-c-forward-control-v1"
                    or solve.get("status") != "complete"
                    or solve.get("control_id") != control_id
                    or solve.get("baseline_campaign_manifest_id") != campaign.get("manifest_id")
                    or canonical_id(solve) != solve.get("manifest_id")):
                problems.append("forward_manifest")
            if not velocity_path.is_file() or sha256_file(velocity_path) != solve.get("velocity_sha256"):
                problems.append("velocity_hash")
            else:
                velocity = np.load(velocity_path, mmap_mode="r", allow_pickle=False)
                if velocity.shape != (35797, 2) or velocity.dtype != np.float64 or not np.isfinite(velocity).all():
                    problems.append("velocity_array")
            attempts = solve.get("attempts", [])
            if len(attempts) == 2:
                retry_controls.append(control_id)
            if not attempts or attempts[-1].get("status") != "complete":
                problems.append("attempt_terminal_status")
        if problems:
            failures.append({"control_id": control_id, "problems": problems})
    summary_path = root / "campaign_summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    checks["campaign_summary"] = bool(
        summary.get("status") == "complete" and summary.get("completed") == 12
        and summary.get("failed") == 0 and canonical_id(summary) == summary.get("manifest_id")
        and summary.get("baseline_campaign_manifest_id") == campaign.get("manifest_id")
    )
    checks["all_controls_valid"] = not failures
    result = {
        "schema": "jog-uniform-c-baseline-campaign-verification-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_manifest_id": campaign.get("manifest_id"),
        "checks": checks, "failed_controls": failures,
        "retry_controls": retry_controls, "uniform_C_by_experiment": scalar_values,
        "passed": all(checks.values()) and not failures,
    }
    result["manifest_id"] = canonical_id(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.campaign_root)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
